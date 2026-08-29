################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################

import os
import socket
import json
import argparse
import hashlib
from time import time
from pathlib import Path
from datetime import timedelta
import copy

import torch
import torch.optim as opt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchinfo import summary

from sur_gbsa import PBSA_DATASET_LIST, GBSA_DATASET_LIST, FINETUNE_DATASET_LIST
from sur_gbsa.data_utils import filter_collate_fn, get_train_val_test_datasets
from sur_gbsa.ProtMD.utils.utils import Logger, set_seed
from sur_gbsa.test import run_eval
from sur_gbsa.train_core import (
    normalize_checkpoint_args,
    build_model,
    compute_multi_objective_loss,
    parse_batch_for_objective,
)

host = socket.gethostname()
warnings = None

if "tuo" in host:
    if "OMP_PLACES" in os.environ:
        del os.environ["OMP_PLACES"]
    if "OMP_PROC_BIND" in os.environ:
        del os.environ["OMP_PROC_BIND"]


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


def load_json_config(path):
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return json.load(f)


def merge_args(args, cfg):
    merged = copy.deepcopy(args)
    for k, v in cfg.items():
        setattr(merged, k, v)
    return merged


def make_run_hash(config, length=8):
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def split_tag(split_path):
    if not split_path:
        return "nosplit"
    p = Path(split_path)
    if str(split_path).endswith("_train.txt"):
        return p.name.replace("_train.txt", "")
    if "split-by-sequence-identity-30" in str(split_path):
        return "atom3d30all"
    if "split-by-sequence-identity-60" in str(split_path):
        return "atom3d60all"
    return p.stem


def build_run_name(args, pretrain_path):
    condition_tag = "lp" if getattr(args, "linear_probe", False) else ("ft" if pretrain_path.exists() and pretrain_path.name else "scratch")
    residue_tag = "res" if getattr(args, "use_residue_features", False) else "nores"

    train_objectives = _as_list(getattr(args, "train_objectives", getattr(args, "objective", "mmgbsa")))
    val_objectives = _as_list(getattr(args, "val_objectives", getattr(args, "objective", "mmgbsa")))

    hash_inputs = {
        "model": args.model,
        "train_objectives": train_objectives,
        "val_objectives": val_objectives,
        "dataset": args.dataset,
        "train_split": str(args.train_split),
        "val_split": str(args.val_split),
        "test_split": str(args.test_split),
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "global_batch_size": getattr(args, "global_batch_size", None),
        "microbatch_size": getattr(args, "microbatch_size", None),
        "pretrain": str(pretrain_path),
        "residue": getattr(args, "use_residue_features", False),
    }
    run_hash = make_run_hash(hash_inputs)
    run_name = f"{args.model}/train-{'+'.join(train_objectives)}__val-{'+'.join(val_objectives)}__{args.dataset}__{split_tag(args.train_split)}/{condition_tag}/{residue_tag}/seed{args.seed}-{run_hash}"
    return run_name, run_hash


def init_distributed():
    rank = 0
    world_size = 1
    local_rank = 0
    device = 0
    gpus_per_node = max(1, torch.cuda.device_count())
    timeout = timedelta(minutes=60)

    # Check for torch.distributed.run environment variables first (most portable)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print(f"[Rank {rank}] Initializing process group with NCCL backend, device={device}...")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
        print(f"[Rank {rank}] Process group initialized successfully")
    elif "tuo" in host:
        world_size = int(os.environ["FLUX_JOB_SIZE"])
        rank = int(os.environ["FLUX_TASK_RANK"])
        local_rank = int(rank % gpus_per_node)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=device,
            timeout=timeout,
        )
    elif "matrix" in host and "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(rank % gpus_per_node)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
    else:
        # Single GPU mode - no distributed training
        print(f"Running in single GPU mode (no distributed training detected)")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        elif torch.backends.mps.is_available():
            
            device = torch.device("mps")
            print(f"Metal Performance Shaders (MPS) enabled.")
        else:
            device = torch.device("cpu")
            print("No GPU detected, running on CPU.")

    return rank, world_size, local_rank, device


def parse_args():
    dataset_list = PBSA_DATASET_LIST + GBSA_DATASET_LIST + FINETUNE_DATASET_LIST

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)

    p.add_argument("--dataset", choices=dataset_list, default="gbsa")
    p.add_argument("--objective", type=str, default="mmgbsa")
    p.add_argument("--train-objectives", type=str, default="")
    p.add_argument("--val-objectives", type=str, default="")
    p.add_argument("--test-objectives", type=str, default="")
    p.add_argument("--train-split")
    p.add_argument("--val-split")
    p.add_argument("--test-split")
    p.add_argument("--data-dir", type=str)
    p.add_argument("--pretrain", default="")
    p.add_argument("--save_path", default="runs")

    p.add_argument("--model", choices=["gnn", "egmn", "egnn", "semla"], default="egmn")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_batch_size", type=int, default=None)
    p.add_argument("--microbatch_size", type=int, default=None)
    p.add_argument("--accumulation_steps", type=int, default=1)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--prompt", type=str, default="0")
    p.add_argument("--use_residue_features", action="store_true")
    p.add_argument("--linear_probe", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--device", type=int, default=0)

    p.add_argument("--gbsa_label_process", default="sign-flip")
    p.add_argument("--gbsa_max_len", type=int, default=600)
    p.add_argument("--gbsa_rmsd_thresh", type=float, default=2.0)
    p.add_argument("--gbsa_train_frac", type=float, default=0.8)
    p.add_argument("--gbsa_max_frames", type=int, default=1000)
    p.add_argument("--loss_weights", type=str, default="")

    return p.parse_args()


def validate_config(args):
    if not args.train_split or not args.val_split or not args.test_split:
        raise ValueError("train-split, val-split, and test-split are required")


def load_pretrained_weights(module, checkpoint, rank=0):
    if checkpoint is None:
        return
    state_dict = checkpoint.get("model")
    if state_dict is None:
        if rank == 0:
            print("Checkpoint does not contain a model state_dict.")
        return

    # Filter and strip prefixes from checkpoint keys
    # Only load encoder weights, skip finetune/task head weights (we train new head from scratch)
    new_state_dict = {}
    for key, value in state_dict.items():
        # Skip finetune head weights
        if key.startswith('finetune'):
            continue

        # Strip DDP wrapper prefixes
        if key.startswith('encoder.module.'):
            new_key = key.replace('encoder.module.', '')
        elif key.startswith('encoder.'):
            new_key = key.replace('encoder.', '')
        elif key.startswith('module.'):
            new_key = key.replace('module.', '')
        else:
            new_key = key

        new_state_dict[new_key] = value

    try:
        module.load_state_dict(new_state_dict)
    except RuntimeError as e:
        if rank == 0:
            print(f"Strict load failed: {e}")
            print("Retrying with strict=False.")
        module.load_state_dict(new_state_dict, strict=False)


def main():
    rank, world_size, local_rank, device = init_distributed()
    args = parse_args()
    args = merge_args(args, load_json_config(args.config))
    validate_config(args)

    if not getattr(args, "train_objectives", ""):
        args.train_objectives = args.objective
    if not getattr(args, "val_objectives", ""):
        args.val_objectives = args.objective
    if not getattr(args, "test_objectives", ""):
        args.test_objectives = args.val_objectives if hasattr(args, "val_objectives") else args.objective

    args.train_objectives = _as_list(args.train_objectives)
    args.val_objectives = _as_list(args.val_objectives)
    args.test_objectives = _as_list(args.test_objectives)

    set_seed(args.seed)

    pretrain_path = Path(args.pretrain)
    pretrain_exists = pretrain_path.exists() and pretrain_path.name != ""

    run_name, run_hash = build_run_name(args, pretrain_path)
    run_root = Path(args.save_path) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"torch version: {torch.__version__}")
        print(f"world size: {world_size}, global rank: {rank}, local rank: {local_rank}, device: {device}")
        print(f"run_name: {run_name}")
        print(f"run_root: {run_root}")

        with (run_root / "run_config.json").open("w") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "run_hash": run_hash,
                    "args": vars(args),
                    "pretrain_exists": pretrain_exists,
                },
                f,
                indent=2,
                default=str,
            )

    # Skip initial barrier - it seems to cause hangs with torch.distributed.run
    # if world_size > 1:
    #     print(f"[Rank {rank}] About to call first barrier, dist.is_initialized()={dist.is_initialized()}")
    #     dist.barrier()
    #     print(f"[Rank {rank}] First barrier passed")

    log = Logger(run_root, "main.log", rank=rank)
    writer = SummaryWriter(run_root) if rank == 0 else None

    checkpoint = None
    if pretrain_exists:
        if rank == 0:
            log.logger.info(f"Loading pretrained weights from {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location=f"{device}", weights_only=False)

    ckpt_args = normalize_checkpoint_args(checkpoint.get("args", None) if checkpoint is not None else None)
    if ckpt_args is None:
        ckpt_args = copy.deepcopy(args)
    else:
        # Override path-related arguments from command line (paths can change between systems)
        for path_arg in ['data_dir', 'train_split', 'val_split', 'test_split', 'save_path']:
            if hasattr(args, path_arg):
                cmd_value = getattr(args, path_arg)
                if cmd_value is not None:
                    setattr(ckpt_args, path_arg, cmd_value)

    model = build_model(ckpt_args).to(device)

    if pretrain_exists:
        load_pretrained_weights(model["encoder"], checkpoint, rank=rank)

    if getattr(args, "linear_probe", False):
        for p in model["encoder"].parameters():
            p.requires_grad = False
    elif world_size > 1:
        model["encoder"] = DDP(model["encoder"], device_ids=[device], find_unused_parameters=True)

    if world_size > 1:
        for name in list(model.keys()):
            if name != "encoder":
                model[name] = DDP(model[name], device_ids=[device], find_unused_parameters=True)

    model = model.float()

    criterion = torch.nn.MSELoss()
    optimizer = opt.AdamW(model.parameters(), lr=args.lr / max(1, world_size), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    datasets = get_train_val_test_datasets(ckpt_args)
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    # Synchronize after dataset loading to avoid desync in DDP
    if world_size > 1:
        dist.barrier()

    global_bs = args.global_batch_size or args.batch_size

    if args.microbatch_size is not None:
        micro_bs = args.microbatch_size
    else:
        if global_bs % max(1, world_size) != 0:
            raise ValueError(f"global_batch_size={global_bs} must be divisible by world_size={world_size}")
        micro_bs = global_bs // max(1, world_size)

    accumulation_steps = args.accumulation_steps
    if args.global_batch_size is None and args.microbatch_size is None:
        accumulation_steps = 1
    elif args.global_batch_size is not None and args.microbatch_size is not None:
        accumulation_steps = global_bs // (micro_bs * max(1, world_size))

    # Use DistributedSampler only for multi-GPU training
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_bs,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=filter_collate_fn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=micro_bs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=1,
        collate_fn=filter_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=micro_bs,
        shuffle=False,
        sampler=test_sampler,
        num_workers=1,
        collate_fn=filter_collate_fn,
    )

    best_metric = float("inf")
    best_epoch = 0
    global_step = 0

    if rank == 0:
        summary(model)

    try:
        for epoch in range(args.epochs):
            if world_size > 1:
                dist.barrier()
                train_sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            t1 = time()

            train_iter = tqdm(train_loader, desc=f"Train Epoch {epoch}") if rank == 0 else train_loader

            for step, batch in enumerate(train_iter):
                if batch is None:
                    continue

                batch = parse_batch_for_objective(batch, ckpt_args)

                loss, _ = compute_multi_objective_loss(
                    model=model,
                    batch=batch,
                    cfg=ckpt_args,
                    device=device,
                    criterion=criterion,
                    objective_list=args.train_objectives,
                )
                loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                is_update = ((step + 1) % accumulation_steps == 0) or (step + 1 == len(train_loader))
                if is_update:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss.detach().float().item() * accumulation_steps
                global_step += 1

            val_dict = run_eval(model, val_loader, ckpt_args, device=device)
            metrics = val_dict.get("metrics", val_dict)

            metric = metrics.get("rmse", metrics.get("mse"))
            pearson = metrics.get("pearson")
            spearman = metrics.get("spearman")
            pdb_pearson = metrics.get("pdb_pearson")
            pdb_spearman = metrics.get("pdb_spearman")

            if rank == 0:
                if writer is not None:
                    writer.add_scalar("loss/train", train_loss / max(1, len(train_loader.dataset)), epoch)
                    writer.add_scalar("metric/val", metric, epoch)
                    writer.add_scalar("pearson/val", pearson, epoch)
                    writer.add_scalar("spearman/val", spearman, epoch)
                    writer.add_scalar("pdb_pearson/val", pdb_pearson, epoch)
                    writer.add_scalar("pdb_spearman/val", pdb_spearman, epoch)

                log.logger.info(
                    f"Epoch {epoch} | time {time() - t1:.1f}s | "
                    f"train {train_loss / max(1, len(train_loader.dataset)):.5f} | "
                    f"val {metric:.5f} | pearson {pearson:.5f} | spearman {spearman:.5f} | "
                    f"pdb_pearson {pdb_pearson:.5f} | pdb_spearman {pdb_spearman:.5f}"
                )

                if metric < best_metric:
                    best_metric = metric
                    best_epoch = epoch + 1
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "best_epoch": best_epoch,
                            "best_metric": best_metric,
                            "global_step": global_step,
                            "args": ckpt_args,
                            "run_name": run_name,
                        },
                        run_root / "best_model.pt",
                    )

        if writer is not None:
            writer.close()

    finally:
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()

    if rank == 0:
        print("done.")


if __name__ == "__main__":
    main()