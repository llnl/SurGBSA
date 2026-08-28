import copy
from pathlib import Path
import argparse

import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sur_gbsa.ProtMD.egnn import EGNN_Network, Regressor
from sur_gbsa.egnn.egnn_clean import EGNN
from sur_gbsa.egnn.gnn import GNN
from sur_gbsa.semla.semla import EquiInvDynamics
from sur_gbsa.semla.semla_encoder import SemlaEncoderWrapper
from sur_gbsa.data_utils import filter_collate_fn, get_train_val_test_datasets
from sur_gbsa.ProtMD.utils.utils import Logger, set_seed
from sur_gbsa.objectives import (
    parse_loss_weights,
    atomic_token_prediction,
    edge_token_prediction,
    coordinate_denoising,
    next_timestep_coordinate_prediction,
    pairwise_snapshot_ordering_logits,
    rmsf_prediction,
)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [value]



def resolve_checkpoint_args(args, checkpoint=None):
    resolved = copy.deepcopy(args)

    if checkpoint is not None and "args" in checkpoint:
        resolved = copy.deepcopy(checkpoint["args"])

    resolved.model = args.model
    resolved.dataset = args.dataset
    resolved.train_split = args.train_split
    resolved.val_split = args.val_split
    resolved.test_split = args.test_split
    resolved.data_dir = args.data_dir
    resolved.pretrain = args.pretrain
    resolved.save_path = args.save_path
    resolved.seed = args.seed
    resolved.num_workers = args.num_workers
    resolved.tokens = getattr(args, "tokens", getattr(resolved, "tokens", 100))
    resolved.use_residue_features = getattr(args, "use_residue_features", False)
    resolved.linear_probe = getattr(args, "linear_probe", False)
    resolved.prompt = getattr(args, "prompt", "0")
    resolved.gbsa_label_process = getattr(args, "gbsa_label_process", "sign-flip")
    resolved.gbsa_max_len = getattr(args, "gbsa_max_len", 600)
    resolved.gbsa_rmsd_thresh = getattr(args, "gbsa_rmsd_thresh", 2.0)
    resolved.gbsa_train_frac = getattr(args, "gbsa_train_frac", 0.8)
    resolved.gbsa_max_frames = getattr(args, "gbsa_max_frames", 1000)

    resolved.train_objectives = _as_list(
        getattr(args, "train_objectives", None),
        fallback=_as_list(getattr(resolved, "train_objectives", None), fallback=[getattr(args, "objective", getattr(resolved, "objective", "mmgbsa"))]),
    )
    resolved.val_objectives = _as_list(
        getattr(args, "val_objectives", None),
        fallback=_as_list(getattr(resolved, "val_objectives", None), fallback=[getattr(args, "objective", getattr(resolved, "objective", "mmgbsa"))]),
    )

    if not hasattr(resolved, "dim"):
        resolved.dim = getattr(args, "dim", 128)
    if not hasattr(resolved, "depth"):
        resolved.depth = getattr(args, "depth", 6)
    if not hasattr(resolved, "num_nearest"):
        resolved.num_nearest = getattr(args, "num_nearest", 32)
    if not hasattr(resolved, "dropout"):
        resolved.dropout = getattr(args, "dropout", 0.15)

    if not hasattr(resolved, "global_batch_size"):
        resolved.global_batch_size = getattr(args, "global_batch_size", None)
    if not hasattr(resolved, "microbatch_size"):
        resolved.microbatch_size = getattr(args, "microbatch_size", None)
    if not hasattr(resolved, "accumulation_steps"):
        resolved.accumulation_steps = getattr(args, "accumulation_steps", 1)

    return resolved



def get_objectives(cfg, stage="train"):
    key = f"{stage}_objectives"
    vals = _as_list(getattr(cfg, key, []))
    if vals:
        return vals
    return _as_list(getattr(cfg, "objective", "mmgbsa"))


def normalize_checkpoint_args(ckpt_args):
    if ckpt_args is None:
        return None

    if isinstance(ckpt_args, dict):
        data = dict(ckpt_args)
    else:
        data = copy.deepcopy(vars(ckpt_args))

    defaults = {
        "model": "egmn",
        "objective": "mmgbsa",
        "train_objectives": ["mmgbsa"],
        "val_objectives": ["mmgbsa"],
        "test_objectives": ["mmgbsa"],
        "tokens": 100,
        "dim": 128,
        "depth": 6,
        "num_nearest": 32,
        "dropout": 0.15,
        "use_residue_features": False,
    }

    for k, v in defaults.items():
        data.setdefault(k, v)

    return argparse.Namespace(**data)


def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def extract_model_state_dict(checkpoint):
    if checkpoint is None:
        return None
    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return None


def build_semla_encoder(ckpt_cfg):
    dynamics = EquiInvDynamics(
        d_model=ckpt_cfg.dim,
        d_message=ckpt_cfg.dim,
        n_coord_sets=1,
        n_layers=ckpt_cfg.depth,
        n_attn_heads=ckpt_cfg.dim,
        d_message_hidden=None,
        d_edge=None,
        bond_refine=False,
        self_cond=False,
        coord_norm="length",
    )

    return SemlaEncoderWrapper(
        num_tokens=ckpt_cfg.tokens,
        dim=ckpt_cfg.dim,
        dynamics=dynamics,
        use_residue_features=ckpt_cfg.use_residue_features,
        num_residue_tokens=22 if ckpt_cfg.use_residue_features else None,
        residue_dim=32 if ckpt_cfg.use_residue_features else None,
    )


def build_egmn_encoder(ckpt_cfg):
    return EGNN_Network(
        num_tokens=ckpt_cfg.tokens,
        dim=ckpt_cfg.dim,
        depth=ckpt_cfg.depth,
        num_nearest_neighbors=ckpt_cfg.num_nearest,
        dropout=ckpt_cfg.dropout,
        global_linear_attn_every=1,
        norm_coors=True,
        coor_weights_clamp_value=2.0,
        aggregate=False,
        num_residue_tokens=22 if ckpt_cfg.use_residue_features else None,
        residue_dim=32 if ckpt_cfg.use_residue_features else None,
    )


def build_egnn_encoder(ckpt_cfg):
    return EGNN(
        in_node_nf=ckpt_cfg.dim,
        hidden_nf=ckpt_cfg.dim,
        out_node_nf=ckpt_cfg.dim,
        in_edge_nf=1,
        num_node_tokens=ckpt_cfg.tokens,
        num_residue_tokens=22 if ckpt_cfg.use_residue_features else None,
        residue_dim=32 if ckpt_cfg.use_residue_features else None,
    )


def build_gnn_encoder(ckpt_cfg):
    return GNN(
        input_dim=ckpt_cfg.dim,
        hidden_nf=ckpt_cfg.dim,
        out_node_nf=ckpt_cfg.dim,
        recurrent=True,
        num_node_tokens=ckpt_cfg.tokens,
        num_residue_tokens=22 if ckpt_cfg.use_residue_features else None,
        residue_dim=32 if ckpt_cfg.use_residue_features else None,
    )


def build_encoder(ckpt_cfg):
    if ckpt_cfg.model == "semla":
        return build_semla_encoder(ckpt_cfg)
    if ckpt_cfg.model == "egmn":
        return build_egmn_encoder(ckpt_cfg)
    if ckpt_cfg.model == "egnn":
        return build_egnn_encoder(ckpt_cfg)
    if ckpt_cfg.model == "gnn":
        return build_gnn_encoder(ckpt_cfg)
    raise ValueError(f"Unknown model: {ckpt_cfg.model}")


def build_objective_heads(ckpt_cfg, objectives=None):
    heads = torch.nn.ModuleDict()
    objectives = objectives or get_objectives(ckpt_cfg, stage="train")

    for objective in objectives:
        if objective == "coord_pred":
            heads["coord_head"] = torch.nn.Linear(ckpt_cfg.dim, 3)
        elif objective == "rmsf":
            heads["rmsf_head"] = torch.nn.Linear(ckpt_cfg.dim, 1)
        elif objective == "ordering":
            heads["order_head"] = torch.nn.Sequential(
                torch.nn.Linear(ckpt_cfg.dim * 2, ckpt_cfg.dim),
                torch.nn.ReLU(),
                torch.nn.Linear(ckpt_cfg.dim, 1),
            )
        elif objective == "atom":
            heads["atom_head"] = torch.nn.Linear(ckpt_cfg.dim, ckpt_cfg.tokens)
        elif objective == "edge":
            heads["edge_head"] = torch.nn.Linear(ckpt_cfg.dim, ckpt_cfg.tokens)

    return heads


def build_model(ckpt_cfg):
    model = torch.nn.ModuleDict()
    model["encoder"] = build_encoder(ckpt_cfg)

    objectives = get_objectives(ckpt_cfg, stage="train")
    if "mmgbsa" in objectives:
        model["mmgbsa"] = Regressor(ckpt_cfg.dim)

    extra_objectives = [obj for obj in objectives if obj != "mmgbsa"]
    if extra_objectives:
        model.update(build_objective_heads(ckpt_cfg, extra_objectives))

    return model


def load_model_from_checkpoint(args, ckpt_path, device, strict=False, mode="test"):
    checkpoint = torch.load(ckpt_path, map_location=f"cuda:{device}", weights_only=False)
    ckpt_args = checkpoint.get("args", None)
    ckpt_args = normalize_checkpoint_args(ckpt_args)

    if ckpt_args is None:
        raise ValueError(f"Checkpoint missing args: {ckpt_path}")

    model = build_model(ckpt_args).to(device)

    state_dict = extract_model_state_dict(checkpoint)
    if state_dict is None:
        raise ValueError(f"No model state dict found in checkpoint: {ckpt_path}")

    state_dict = strip_module_prefix(state_dict)

    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if strict:
            raise
        model.load_state_dict(state_dict, strict=False)

    return model, checkpoint, ckpt_args


def forward_encoder(model, cfg, x, pos, mask=None, res_feats=None):
    encoder = model["encoder"]

    if cfg.model in {"gnn", "egnn"}:
        edges, edge_attr = get_distance_edges_batch(pos, threshold=2.0)
        edges[0] = edges[0].to(pos.device)
        edges[1] = edges[1].to(pos.device)
        edge_attr = edge_attr.to(pos.device)

        x_flat = x.reshape(-1)
        pos_flat = pos.reshape(-1, pos.shape[-1])
        mask_flat = mask.reshape(-1) if mask is not None else None
        res_feats_flat = res_feats.reshape(-1) if res_feats is not None else None

        return encoder(
            x_flat,
            pos_flat,
            edges,
            edge_attr=edge_attr,
            res_feats=res_feats_flat,
            mask=mask_flat,
        )

    return encoder(x, pos, mask=mask, res_feats=res_feats)


def extract_encoder_features(enc_out, cfg):
    if cfg.model in {"gnn", "egnn"}:
        return enc_out[0]
    if cfg.model in {"egmn", "semla"}:
        return enc_out[1]
    return enc_out


def pool_graph_features(feat, x, mask=None):
    if feat.dim() != 2:
        raise ValueError(f"Expected 2D encoder output, got {feat.shape}")

    B, N = x.shape[:2]

    if feat.shape[0] == B * N:
        feat = feat.view(B, N, -1)
    elif feat.shape[0] == B:
        feat = feat.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected encoder output shape {feat.shape} for batch shape {x.shape}")

    if mask is not None:
        m = mask.unsqueeze(-1).float()
        feat = (feat * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    else:
        feat = feat.mean(dim=1)

    return feat


def aggregate_frame_embeddings(model, frame_embeddings, frame_mask, cfg):
    """Aggregate frame embeddings if aggregator exists in model.

    Args:
        model: dict containing model components (may include 'aggregator')
        frame_embeddings: [B, T, dim] - per-frame embeddings
        frame_mask: [B, T] - binary mask for valid frames
        cfg: config namespace

    Returns:
        aggregated: [B, dim] - pooled trajectory embedding
    """
    if 'aggregator' in model:
        return model['aggregator'](frame_embeddings, frame_mask)
    else:
        # Fallback to mean pooling if no aggregator
        mask = frame_mask.unsqueeze(-1).float()
        return (frame_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def parse_batch_for_objective(batch, cfg):
    if batch is None:
        return None

    if isinstance(batch, dict):
        return batch

    if len(batch) == 4:
        if getattr(cfg, "use_residue_features", False):
            x, res_feats, pos, y, idx = batch
            return {"x": x, "res_feats": res_feats, "pos": pos, "y": y, "idx": idx}
        x, pos, y, idx = batch
        return {"x": x, "pos": pos, "y": y, "idx": idx}

    if len(batch) == 7:
        x, pose_list, pos, y, file_list, rmsd, frame_num = batch
        return {
            "x": x,
            "pose_list": pose_list,
            "pos": pos,
            "y": y,
            "file_list": file_list,
            "rmsd": rmsd,
            "frame_num": frame_num,
        }

    return batch


def get_distance_edges_batch(xyz: torch.Tensor, threshold: float = 2.0):
    B, N, _ = xyz.shape
    dists = torch.cdist(xyz, xyz)
    mask = (dists < threshold) & (dists > 0)

    rows, cols, dvals = [], [], []

    for b in range(B):
        idx = mask[b].nonzero(as_tuple=False)
        src = idx[:, 0] + b * N
        dst = idx[:, 1] + b * N
        dist_vals = dists[b][idx[:, 0], idx[:, 1]]

        rows.append(src)
        cols.append(dst)
        dvals.append(dist_vals)

    edge_src = torch.cat(rows, dim=0).long()
    edge_dst = torch.cat(cols, dim=0).long()
    edge_attr = torch.cat(dvals, dim=0).unsqueeze(1)

    return [edge_src, edge_dst], edge_attr


def compute_objective_loss(model, batch, cfg, device, criterion, objective=None):
    objective = objective or getattr(cfg, "objective", "mmgbsa")

    x = batch["x"].long().to(device)
    pos = batch["pos"].float().to(device)
    y = batch.get("y", None)
    if y is not None:
        y = y.float().to(device)

    res_feats = batch.get("res_feats", None)
    if res_feats is not None:
        res_feats = res_feats.to(device)

    mask = x != 0

    def run_gnn_encoder(x_in, pos_in, res_feats_in=None, mask_in=None):
        edges, edge_attr = get_distance_edges_batch(pos_in, threshold=2.0)
        edges[0] = edges[0].to(device)
        edges[1] = edges[1].to(device)
        edge_attr = edge_attr.to(device)

        x_flat = x_in.reshape(-1)
        pos_flat = pos_in.reshape(-1, pos_in.shape[-1])
        mask_flat = mask_in.reshape(-1) if mask_in is not None else None
        res_feats_flat = res_feats_in.reshape(-1) if res_feats_in is not None else None

        enc_out = model["encoder"](
            x_flat,
            pos_flat,
            edges,
            edge_attr=edge_attr,
            res_feats=res_feats_flat,
            mask=mask_flat,
        )

        return pool_graph_features(extract_encoder_features(enc_out, cfg), x_in, mask_in)

    def run_encoder(x_in, pos_in, res_feats_in=None, mask_in=None):
        if cfg.model in {"gnn", "egnn"}:
            return run_gnn_encoder(x_in, pos_in, res_feats_in=res_feats_in, mask_in=mask_in)

        enc_out = model["encoder"](x_in, pos_in, res_feats=res_feats_in, mask=mask_in)
        return extract_encoder_features(enc_out, cfg)

    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        if objective == "mmgbsa":
            feat = run_encoder(x, pos, res_feats_in=res_feats, mask_in=mask)
            if feat.dim() == 3:
                feat = feat.mean(dim=1)
            pred = model["mmgbsa"](feat)
            loss = criterion(pred, y)

        elif objective == "coord_pred":
            if "pos_t1" not in batch:
                raise ValueError("coord_pred requires batch['pos_t1']")
            feat = run_encoder(x, pos, res_feats_in=res_feats, mask_in=mask)
            coord_pred = model["coord_head"](feat)
            loss = next_timestep_coordinate_prediction(
                coord_pred,
                batch["pos_t1"].float().to(device),
                mask=mask,
            )

        elif objective == "denoise":
            if "pos_clean" not in batch:
                raise ValueError("denoise requires batch['pos_clean']")
            feat = run_encoder(x, pos, res_feats_in=res_feats, mask_in=mask)
            denoise_pred = model["coord_head"](feat)
            loss = coordinate_denoising(
                denoise_pred,
                batch["pos_clean"].float().to(device),
                mask=mask,
            )

        elif objective == "ordering":
            if "x_b" not in batch or "pos_b" not in batch or "order_target" not in batch:
                raise ValueError("ordering requires x_b, pos_b, and order_target in batch")

            x_b = batch["x_b"].long().to(device)
            pos_b = batch["pos_b"].float().to(device)
            mask_b = x_b != 0
            res_feats_b = batch.get("res_feats_b", None)
            if res_feats_b is not None:
                res_feats_b = res_feats_b.to(device)

            feat_a = run_encoder(x, pos, res_feats_in=res_feats, mask_in=mask)
            feat_b = run_encoder(x_b, pos_b, res_feats_in=res_feats_b, mask_in=mask_b)

            order_logits = model["order_head"](torch.cat([feat_a, feat_b], dim=-1))
            loss = pairwise_snapshot_ordering_logits(
                order_logits,
                batch["order_target"].float().to(device),
            )

        elif objective == "rmsf":
            if "rmsf_target" not in batch:
                raise ValueError("rmsf requires batch['rmsf_target']")
            feat = run_encoder(x, pos, res_feats_in=res_feats, mask_in=mask)
            rmsf_pred = model["rmsf_head"](feat)
            loss = rmsf_prediction(
                rmsf_pred,
                batch["rmsf_target"].float().to(device),
            )

        elif objective == "atom":
            if "atom_targets" not in batch:
                raise ValueError("atom requires batch['atom_targets']")
            feat = run_encoder(x, pos, res_feats_in=res_feats, mask_in=mask)
            atom_logits = model["atom_head"](feat)
            atom_mask = batch.get("atom_mask", None)
            if atom_mask is not None:
                atom_mask = atom_mask.to(device)
            loss = atomic_token_prediction(
                atom_logits,
                batch["atom_targets"].long().to(device),
                atom_mask,
            )

        elif objective == "edge":
            if "edge_targets" not in batch:
                raise ValueError("edge requires batch['edge_targets']")
            feat = run_encoder(x, pos, res_feats_in=res_feats, mask_in=mask)
            edge_logits = model["edge_head"](feat)
            edge_mask = batch.get("edge_mask", None)
            if edge_mask is not None:
                edge_mask = edge_mask.to(device)
            loss = edge_token_prediction(
                edge_logits,
                batch["edge_targets"].long().to(device),
                edge_mask,
            )

        else:
            raise ValueError(f"Unknown objective: {objective}")

    return loss


compute_batch_loss = compute_objective_loss


def compute_multi_objective_loss(model, batch, cfg, device, criterion, objective_list=None):
    objective_list = objective_list or _as_list(getattr(cfg, "train_objectives", []))
    if not objective_list:
        objective_list = _as_list(getattr(cfg, "objective", "mmgbsa"))

    loss_weights = parse_loss_weights(getattr(cfg, "loss_weights", ""))

    total_loss = None
    loss_dict = {}

    for objective in objective_list:
        loss_i = compute_objective_loss(
            model=model,
            batch=batch,
            cfg=cfg,
            device=device,
            criterion=criterion,
            objective=objective,
        )
        weight = loss_weights.get(objective, 1.0) if isinstance(loss_weights, dict) else 1.0
        weighted = loss_i * weight
        total_loss = weighted if total_loss is None else total_loss + weighted
        loss_dict[objective] = loss_i.detach()

    if total_loss is None:
        total_loss = torch.tensor(0.0, device=device)

    return total_loss, loss_dict


def train_one_epoch(model, train_loader, optimizer, scaler, cfg, device, criterion, accumulation_steps=1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for step, batch in enumerate(train_loader):
        if batch is None:
            continue

        batch = parse_batch_for_objective(batch, cfg)
        loss, _ = compute_multi_objective_loss(
            model=model,
            batch=batch,
            cfg=cfg,
            device=device,
            criterion=criterion,
        )
        loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        is_update_step = ((step + 1) % accumulation_steps == 0) or (step + 1 == len(train_loader))
        if is_update_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.detach().float().item() * accumulation_steps

    return total_loss


def evaluate_one_epoch(model, val_loader, cfg, device, criterion):
    model.eval()
    total = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            batch = parse_batch_for_objective(batch, cfg)
            loss, _ = compute_multi_objective_loss(
                model=model,
                batch=batch,
                cfg=cfg,
                device=device,
                criterion=criterion,
                objective_list=_as_list(getattr(cfg, "val_objectives", [])),
            )
            bs = batch["x"].shape[0]
            total += loss.detach().float().item() * bs
            count += bs

    return total / max(1, count)


def train_one_trial(args, config, trial_dir, rank=0, device=0):
    args = copy.deepcopy(args)
    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, "train_objectives"):
        args.train_objectives = [getattr(args, "objective", "mmgbsa")]
    if not hasattr(args, "val_objectives"):
        args.val_objectives = [getattr(args, "objective", "mmgbsa")]

    set_seed(args.seed)

    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    log = Logger(trial_dir, "main.log", rank=rank)
    writer = SummaryWriter(trial_dir) if rank == 0 else None

    ckpt_args = normalize_checkpoint_args(resolve_checkpoint_args(args, None))
    model = build_model(ckpt_args).to(device)
    model = model.float()

    criterion = torch.nn.MSELoss()
    optimizer = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    datasets = get_train_val_test_datasets(args)
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]

    microbatch_size = int(args.microbatch_size or args.batch_size)
    accumulation_steps = int(getattr(args, "accumulation_steps", 1))
    if args.global_batch_size is not None and args.microbatch_size is not None:
        denom = microbatch_size * max(1, getattr(args, "world_size", 1))
        expected = args.global_batch_size / denom
        if abs(expected - round(expected)) > 1e-6:
            raise ValueError(
                f"global_batch_size={args.global_batch_size} is not divisible by "
                f"microbatch_size={microbatch_size} * world_size={getattr(args, 'world_size', 1)}"
            )
        accumulation_steps = int(round(expected))

    train_loader = DataLoader(
        train_dataset,
        batch_size=microbatch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=filter_collate_fn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=filter_collate_fn,
    )

    best_metric = float("inf")
    best_epoch = 0
    global_step = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=args,
            device=device,
            criterion=criterion,
            accumulation_steps=accumulation_steps,
        )

        val_loss = evaluate_one_epoch(
            model=model,
            val_loader=val_loader,
            cfg=args,
            device=device,
            criterion=criterion,
        )

        if rank == 0 and writer is not None:
            writer.add_scalar("val/loss", val_loss, epoch)

        if val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "global_step": global_step,
                    "args": args,
                    "config": config,
                },
                trial_dir / "best_model.pt",
            )

    if writer is not None:
        writer.close()

    return {
        "val_loss": best_metric,
        "best_epoch": best_epoch,
        "best_model_path": str(trial_dir / "best_model.pt"),
        "model": args.model,
        "config": config,
    }