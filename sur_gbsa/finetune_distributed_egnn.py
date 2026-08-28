################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################

import socket

host = socket.gethostname()
import os

# if "tuo" in host:
# del os.environ['OMP_PLACES']
# del os.environ['OMP_PROC_BIND']
import copy
from time import time
from pathlib import Path
import numpy as np
import traceback
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as opt
from torch.nn.functional import mse_loss
from scipy import stats
import torch.nn as nn
from sur_gbsa.egnn.egnn_clean import EGNN
from sur_gbsa.ProtMD.utils.utils import Logger, set_seed
from tqdm import tqdm
from torchinfo import summary
from sur_gbsa.data_utils import filter_collate_fn, get_train_val_test_datasets
from sur_gbsa.distributed_utils import distributed_gather
from sur_gbsa.finetune_distributed import gather_to_rank0, get_metadata
from sur_gbsa.egnn.egnn_clean import EGNN
from sur_gbsa.egnn.gnn import GNN
from sur_gbsa.ProtMD.egnn import Regressor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datetime import timedelta


import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# from dtra_mpro.finetune_distributed import run_full_test_eval

DEV_IDS = list(range(torch.cuda.device_count()))
world_size = len(DEV_IDS)

from sur_gbsa import GBSA_DATASET_LIST, FINETUNE_DATASET_LIST


def get_distance_edges_batch(xyz: torch.Tensor, threshold: float = 2.0):
    """
    Compute batch-wise edge indices and distances using a distance threshold.

    Args:
        xyz (Tensor): shape (B, N, 3)
        threshold (float): distance cutoff

    Returns:
        edges (List[Tensor, Tensor]): [sources, targets] of shape [num_edges]
        edge_attr (Tensor): shape [num_edges, 1] with distances
    """
    B, N, _ = xyz.shape
    dists = torch.cdist(xyz, xyz)  # shape (B, N, N)
    mask = (dists < threshold) & (dists > 0)  # exclude self-loops

    rows, cols, dvals = [], [], []

    for b in range(B):
        idx = mask[b].nonzero(as_tuple=False)  # [E_b, 2]
        src = idx[:, 0] + b * N
        dst = idx[:, 1] + b * N
        dist_vals = dists[b][idx[:, 0], idx[:, 1]]

        rows.append(src)
        cols.append(dst)
        dvals.append(dist_vals)

    # Concatenate across batch
    edge_src = torch.cat(rows, dim=0).long()
    edge_dst = torch.cat(cols, dim=0).long()
    edge_attr = torch.cat(dvals, dim=0).unsqueeze(1)  # shape [E, 1]

    return [edge_src, edge_dst], edge_attr


# from dtra_mpro.finetune_distributed import run_eval


def run_gbsa_eval(model, loader, sampler, args, device="cpu"):
    model.eval()

    y_pred = []
    y_true = []

    idx_list = []
    latent_list = []
    frame_list = []

    metadata_list = []

    batch_start = 0
    rank_indices = list(sampler)  # indices this rank will see
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            else:
                h, x, y, idx = batch

                # todo: don't need to move each tensor to GPU
                h, x, y, idx = (
                    h.to(device),
                    x.float().to(device),
                    y.float().to(device),
                    idx.to(device),
                )

                h = model["token_emb"](h)

                edges, edge_attr = get_distance_edges_batch(x, threshold=2.0)
                edges[0] = edges[0].to(device)
                edges[1] = edges[1].to(device)

                h = h.reshape(-1, args.dim)
                if args.model == "egnn":
                    x = x.reshape(-1, 3)
                    h, x = model["encoder"](h=h, x=x, edges=edges, edge_attr=edge_attr)
                elif args.model == "gnn":
                    h = h.float()
                    h = model["encoder"](nodes=h, edges=edges, edge_attr=edge_attr)

                h = h.reshape(-1, args.gbsa_max_len, args.dim)
                h = h.mean(dim=1)
                latent_list.append(h)
                pred = model["finetune"](h).to(device)
                y_true.append(y.cpu())
                y_pred.append(pred.cpu())
                idx_list.append(idx.cpu())
                batch_size = len(y)  # or len(batch[0]) depending on your dataset output

                batch_indices = rank_indices[batch_start : batch_start + batch_size]
                batch_start += batch_size

                # metadata_list = [get_metadata(loader.dataset, idx) for idx in batch_indices]
                metadata = [get_metadata(loader.dataset, idx) for idx in batch_indices]

                # print(len(y), len(batch_indices), len(metadata[0]), len(metadata[1]))
                metadata_list.extend([x[0] for x in metadata])
                frame_list.extend([x[1] for x in metadata])
                # print(metadata_list)
                # Now call your custom method on each index

    y_pred = torch.cat(y_pred).reshape(-1, 1).cpu()
    y_true = torch.cat(y_true).reshape(-1, 1).cpu()
    idx = torch.cat(idx_list).reshape(-1, 1).cpu()
    latent = torch.cat(latent_list).reshape(-1, args.dim)
    # print(idx)
    pearson = stats.pearsonr(y_pred.numpy().flatten(), y_true.numpy().flatten())[0]
    spearman = stats.spearmanr(y_pred.numpy().flatten(), y_true.numpy().flatten())[0]
    rmse = torch.sqrt(
        mse_loss(y_true, y_pred, reduction="sum") / len(loader.dataset)
    ).item()

    return {
        "pearson": pearson,
        "spearman": spearman,
        "rmse": rmse,
        "y_pred": y_pred,
        "y_true": y_true,
        "idx": idx,
        "latent": latent,
        "metadata": metadata_list,
        "frame_num": frame_list,
    }


def run_tensor_dataset_eval(model, loader, args, device="cpu"):
    model.eval()
    metric = 0
    y_pred = []
    y_true = []
    pose_list = []
    path_list = []
    frame_list = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            else:
                x, pose, pos, y = batch
                # print(rmsd)
                y_true.append(y)
                x, pos, y = x.long().to(device), pos.float().to(device), y.to(device)
                mask = x != 0
                # import pdb
                # pdb.set_trace()
                out = model["encoder"](x, pos, mask=mask)[1]

                out = out.mean(dim=1)  # average over number of atoms

                pred = model["finetune"](out).mean(dim=1)
                # coors, out = model(x, pos, mask=mask)
                y_pred.append(pred)

                path_list.extend(
                    ["mpro"] * len(pred)
                )  # replace with the ID string for each molecule
                frame_list.append(torch.zeros_like(y))
                pose_list.append(torch.zeros_like(y))

    y_pred = torch.cat(y_pred).reshape(-1, 1)
    y_true = torch.cat(y_true).reshape(-1, 1)

    # import pdb
    # pdb.set_trace()
    pearson = stats.pearsonr(y_pred.cpu().numpy().flatten(), y_true.numpy().flatten())[
        0
    ]
    spearman = stats.spearmanr(
        y_pred.cpu().numpy().flatten(), y_true.numpy().flatten()
    )[0]
    rmse = torch.sqrt(
        mse_loss(y_true.cpu(), y_pred.cpu(), reduction="sum") / len(loader.dataset)
    ).item()
    frame = torch.cat(frame_list)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "rmse": rmse,
        "y_pred": y_pred,
        "y_true": y_true,
        "path_list": path_list,
        # "id": path_list,
        "idx": torch.arange(len(y)),  # idx = tensor([0, 1, 2]),
        "frame_num": frame,
    }


def run_eval(model, loader, sampler, args, device="cpu"):
    if args.dataset in GBSA_DATASET_LIST:
        return run_gbsa_eval(
            model=model, sampler=sampler, loader=loader, args=args, device=device
        )

    elif args.dataset in FINETUNE_DATASET_LIST:
        return run_tensor_dataset_eval(
            model=model, loader=loader, sampler=sampler, args=args, device=device
        )

    else:
        raise Exception


def run_full_test_eval(
    args,
    model_path,
    best_epoch,
    best_model,
    ckpt_root_dir,
    current_device,
    rank,
    writer,
    epoch,
    batch_size_per_gpu,
    train_loader,
    val_loader,
    test_loader,
    log,
):
    for dataset in GBSA_DATASET_LIST:
        # save the predictions and ground truth labels
        result_path = ckpt_root_dir / Path(
            f"test_result-{model_path.with_name(f'best_model-epoch-{best_epoch}').name}-{dataset}.pt"
        )
        if result_path.exists():
            print(f"{result_path} already exists, work is finished.")
            # pass
        else:
            # if True:
            config = args
            config.dataset = dataset

            _, _, eval_dataset = get_train_val_test_datasets(config)
            eval_sampler = DistributedSampler(eval_dataset)
            eval_sampler.set_epoch(0)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size_per_gpu,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=filter_collate_fn,
                sampler=eval_sampler,
                pin_memory=False,
                persistent_workers=False,
                drop_last=False,
            )

            eval_dict = run_eval(
                best_model,
                eval_loader,
                sampler=eval_sampler,
                args=config,
                device=current_device,
            )
            pearson, spearman, metric, pred, gt, idx = (
                eval_dict["pearson"],
                eval_dict["spearman"],
                eval_dict["rmse"],
                eval_dict["y_pred"],
                eval_dict["y_true"],
                eval_dict["idx"],
            )

            dist.barrier()

            pred = distributed_gather(pred.float())
            gt = distributed_gather(gt.float())
            idx = distributed_gather(idx)
            latent = distributed_gather(eval_dict["latent"])
            metadata = [None] * world_size
            # print(eval_dict["metadata"])
            # all_gather_object(eval_dict["metadata"], metadata)
            metadata = gather_to_rank0(eval_dict["metadata"])
            frame_num = gather_to_rank0(
                eval_dict["frame_num"]
            )  # todo: can use torch tensors instead
            # metadata = distributed_gather(eval_dict["metadata"])
            dist.barrier()

            if rank == 0:
                # stack the data on the rank dimension
                pred = pred.view(-1)
                gt = gt.view(-1)
                idx = idx.view(-1)
                latent = latent.view(-1, args.dim)
                metadata = [item for sublist in metadata for item in sublist]
                # frame_num = torch.cat([item for sublist in frame_num for item in sublist]).view(-1)
                frame_num = torch.cat(
                    [item.view(-1) for sublist in frame_num for item in sublist]
                ).view(-1)

                metric = mse_loss(pred, gt)
                pearson = stats.pearsonr(pred, gt)[0]
                spearman = stats.spearmanr(pred, gt)[0]

                writer.add_scalar(f"RMSE/{dataset}/test-{rank}", metric, epoch)
                writer.add_scalar(f"Pearsonr/{dataset}/test-{rank}", pearson, epoch)
                writer.add_scalar(f"Spearmanr/{dataset}/test-{rank}", spearman, epoch)

                writer.flush()

                torch.save(
                    {
                        "train_size": len(train_loader.dataset),
                        "val_size": len(val_loader.dataset),
                        "test_size": len(test_loader.dataset),
                        "y_pred": pred,
                        "y_true": gt,
                        "id": idx,
                        "latent": latent,
                        "metadata": metadata,
                        "frame_num": frame_num,
                        # "pdbid": pdbid_list,
                        # "pose": pose_list,
                        # "frame": frame_list,
                        # "id": path_list,
                        # "frame_num": frame_num,
                        "args": args,
                    },
                    result_path,
                )

                log.logger.info(
                    f"rank: {rank} | dataset: {dataset} | RMSE: {metric} | Test Pearson: {spearman} | Test Spearman: {pearson}"
                )
                log.logger.info(
                    f"Save the best model as {str(model_path)}\nBest result as {result_path}\nBest Epoch: {best_epoch}"
                )

                log.logger.info(f"{result_path.parent}")

                plot_path = result_path.with_name(
                    f"{result_path.stem}_scatterplot-{dataset}.png"
                )

                print(plot_path)
                if plot_path.exists():
                    pass
                else:
                    f, ax = plt.subplots(1, 2, figsize=(18, 8))
                    ax = ax.flatten()

                    sns.scatterplot(x=gt.numpy(), y=pred.numpy(), ax=ax[0])

                    sns.kdeplot(
                        x=pred,
                        color="red",
                        ax=ax[1],
                        shade=True,
                        label="Predicted MM/GBSA",
                    )
                    sns.kdeplot(
                        x=gt,
                        color="blue",
                        ax=ax[1],
                        shade=True,
                        label="Calculated MM/GBSA",
                    )

                    ax[1].legend()

                    pearson_score = pearsonr(pred.numpy(), gt.numpy())[0]
                    hparam_str = plot_path.parent.stem
                    f.suptitle(f"{hparam_str}\n(r={pearson_score:.2f})")
                    plt.tight_layout()
                    f.savefig(plot_path, dpi=450)
                    plt.close(f)


def main():
    print(f"torch version: {torch.__version__}")

    gpus_per_node = 4
    timeout = timedelta(minutes=60)
    if "tuo" in host:
        world_size = int(os.environ["FLUX_JOB_SIZE"])
        rank = int(os.environ["FLUX_TASK_RANK"])
        local_rank = int(rank % gpus_per_node)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            # timeout=timedelta(minutes=30)
            timeout=timeout,
        )
    elif "lassen" in host:
        # LC HACK:
        # init_method="end://" requires $MASTER_ADDR and $MASTER_PORT to be set
        # we pull world_size and rank values from variables set by jsrun
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(rank % gpus_per_node)

        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )

    elif "matrix" in host:
        # world_size = dist.get_world_size()
        world_size = int(os.environ["SLURM_NTASKS"])
        # rank = dist.get_rank()
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(rank % gpus_per_node)

        torch.cuda.set_device(0)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )

    # if "matix" in host:
    # torch.cuda.set_device(local_rank)
    # current_device = torch.cuda.current_device()

    current_device = 0
    print(
        f"world size: {world_size}, global rank: {rank}, local rank: {local_rank}, current_device: {current_device}"
    )
    # dist.barrier()

    scaler = None
    if "tuo" in host:
        scaler = torch.amp.GradScaler(enabled=True)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    dataset_list = GBSA_DATASET_LIST + FINETUNE_DATASET_LIST
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=dataset_list, default="gbsa"
    )  # todo: need to be able to specify pdbid and pose list for gbsa task
    # parser.add_argument("--gbsa-split-strat", choices=["temporal", "random", "dock-pose"], default="temporal")
    parser.add_argument("--train-split")
    parser.add_argument("--val-split")
    parser.add_argument("--test-split")
    parser.add_argument(
        "--train-loss-log-freq",
        type=int,
        default=1,
        help="frequency (num. steps) of logging mini-batch loss to tensorboard",
    )
    parser.add_argument(
        "--gbsa-label-process",
        choices=["sign-flip", "standardize"],
        default="sign-flip",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="path to directory containing the MD data with MM/GBSA scores",
    )
    parser.add_argument("--gbsa-max-len", type=int, default=600)
    parser.add_argument("--gbsa-rmsd-thresh", type=float, default=2.0)
    parser.add_argument(
        "--gbsa-train-frac",
        type=float,
        default=0.8,
        help="proportion of data to use for training, remaining proportion is used for testing",
    )
    parser.add_argument(
        "--gbsa-max-frames",
        type=int,
        default=1000,
        help="max frames to use from each simulation",
    )
    parser.add_argument(
        "--model", choices=["egmn", "egnn", "gnn"], required=True
    )  # may support SGCNN, other E3 models
    parser.add_argument(
        "--pretrain",
        default="",
        help="path to pretrained model to use for finetuning. if path does not exist, model is trained from scratch",
    )
    parser.add_argument("--save_path", default="debug")
    parser.add_argument(
        "--task", choices=["regress"], default="regress"
    )  # may support pretraining
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size to use for finetuning"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of workers to use for dataloading",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="1",
        help="Whether to use the snapshot ordering task.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        metavar="N",
        help="weight decay parameter",
    )
    parser.add_argument("--noise_scale", type=float, default=10)
    parser.add_argument(
        "--linear_probe", default=False, action="store_true"
    )  # todo: make sure this is implemented correctly below, i.e. turn off the gradient for the encoder
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    # parser.add_argument('--model', type=str, default="egmn", help="name of the model to use for training")
    parser.add_argument(
        "--noise",
        type=bool,
        default=True,
        help="Whether to add noise during pretraining.",
    )  # (derek) does this not cause a bug?
    parser.add_argument(
        "--tokens", type=int, default=100, help="The default number of atom classes."
    )
    args = parser.parse_args()

    pretrain_path = Path(args.pretrain)

    baseline_model = False

    # print(f"model path does not exist. using randomly initialized model.")
    ckpt_args = args
    # ckpt_args.gbsa_split_strat = "random"
    # ckpt_args.gbsa_label_process = "sign-flip"
    # ckpt_args.model = "egmn"
    # ckpt_args.depth = 6
    # ckpt_args.dim = 128
    # ckpt_args.num_nearest = 32
    # ckpt_args.dropout = 0.15
    # baseline_model = False
    # ckpt_args.pretrain_tasks = []

    # seed RNGs
    set_seed(args.seed)

    filtered_args = {
        k: (
            Path(v).stem
            if k == "train_split" and isinstance(v, str) and v.endswith(".csv")
            else v
        )
        for k, v in vars(args).items()
        if k in ["dataset", "batch_size", "lr", "train_split", "seed"]
    }

    # Format as string
    args_string = "-".join(
        f"{k}={'_'.join(map(str, v)) if isinstance(v, list) else v}"
        for k, v in filtered_args.items()
    )

    ckpt_root_dir = Path(f"{args.save_path}") / Path(f"{args.model}-{args_string}")
    # ckpt_root_dir = Path(f'{args.save_path}') / Path(f"{args.model}")
    print(ckpt_root_dir, ckpt_root_dir.exists())

    if not ckpt_root_dir.exists():
        print(f"creating checkpoint directory {ckpt_root_dir}")
        ckpt_root_dir.mkdir(parents=True, exist_ok=True)

    model_path = ckpt_root_dir / Path(f"best_model.pt")

    if model_path.exists():
        print(f"{model_path} already exists, work is finished.")
        dist.barrier()
        dist.destroy_process_group()
        return 0
    else:
        pass

    # scratch = True
    # baseline_model = False

    lr_per_gpu = args.lr / world_size  # scale the learning according to number of GPUs
    batch_size_per_gpu = int(args.batch_size / world_size)

    log = Logger(ckpt_root_dir, f"main.log", rank=rank)
    writer = SummaryWriter(ckpt_root_dir)

    log.logger.info(str(args))

    print(f"DEBUG: using {args.data_dir}")
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(args)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=filter_collate_fn,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=1,
        collate_fn=filter_collate_fn,
        sampler=val_sampler,
    )

    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=1,
        collate_fn=filter_collate_fn,
        sampler=test_sampler,
    )

    log.logger.info(
        f"using {args.dataset} for training/testing (train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)})"
    )

    # print("TODO: make sure all `forward` function outputs participate in calculating loss.")
    find_unused_parameters = True
    # find_unused_parameters=False

    if args.model == "egnn":
        encoder = EGNN(
            in_node_nf=args.dim, hidden_nf=args.dim, out_node_nf=args.dim, in_edge_nf=1
        ).to(current_device)

    elif args.model == "gnn":
        encoder = GNN(
            input_dim=args.dim,
            hidden_nf=args.dim,
            out_node_nf=args.dim,
            device=current_device,
            recurrent=True,
        )

    encoder.to(current_device)

    model = torch.nn.ModuleDict()

    # if not baseline_model:

    if not (pretrain_path.exists() and pretrain_path.name != ""):
        model["encoder"] = DDP(
            encoder.to(current_device),
            device_ids=[current_device],
            find_unused_parameters=find_unused_parameters,
        )
        model["token_emb"] = DDP(
            nn.Embedding(args.gbsa_max_len, args.dim).to(current_device),
            device_ids=[current_device],
        )

        if args.linear_probe:
            print(
                "training in linear-probe mode, turning off gradient update for encoder."
            )
            for param in encoder.parameters():
                param.requires_grad = False

        model["finetune"] = DDP(
            Regressor(ckpt_args.dim).to(current_device),
            device_ids=[current_device],
            find_unused_parameters=find_unused_parameters,
        )
    elif baseline_model:
        print("loading weights for baseline model")
        encoder.load_state_dict(checkpoint["model"])

        if args.linear_probe:
            print(
                "training in linear-probe mode, turning off gradient update for encoder."
            )
            for param in encoder.parameters():
                param.requires_grad = False

        model["encoder"] = DDP(
            encoder.to(current_device),
            device_ids=[current_device],
            find_unused_parameters=find_unused_parameters,
        )

        model["finetune"] = DDP(
            Regressor(ckpt_args.dim).to(current_device),
            device_ids=[current_device],
            find_unused_parameters=find_unused_parameters,
        )
    else:
        print("loading weights for pretrained model")
        model["encoder"] = DDP(
            encoder.to(current_device),
            device_ids=[current_device],
            find_unused_parameters=find_unused_parameters,
        )

        model["finetune"] = DDP(
            Regressor(ckpt_args.dim).to(current_device),
            device_ids=[current_device],
            find_unused_parameters=find_unused_parameters,
        )
        model.load_state_dict(checkpoint["model"])

        if args.linear_probe:
            print(
                "training in linear-probe mode, turning off gradient update for encoder."
            )
            for param in encoder.parameters():
                param.requires_grad = False

    model = model.float()

    if rank == 0:
        summary(model)
    if rank == 0 and args.model == "gnn":
        print(model)

    best_model = model
    best_epoch = 0
    criterion = torch.nn.MSELoss()
    best_metric = 1e9

    optimizer = opt.AdamW(
        model.parameters(), lr=lr_per_gpu, weight_decay=args.weight_decay
    )

    log.logger.info(
        f'{"=" * 40} {args.dataset} {"=" * 40}\n'
        f"Embed_dim: {ckpt_args.dim}; Train: {len(train_loader.dataset)}; Val: {len(val_loader.dataset)}; Test: {len(test_loader.dataset)}; Pre-train Model: {pretrain_path}"
        f'Dataset: {args.dataset}; Batch_size: {args.batch_size}; Batch_size_per_gpu: {batch_size_per_gpu}; Linear-probe: {args.linear_probe}\n{"=" * 40} Start Training {"=" * 40}'
    )

    t0 = time()

    global_step = 0

    resume_epoch = 0

    resume_path = ckpt_root_dir / Path("latest_model.pt")
    if resume_path.exists():
        print(f"resuming training from {resume_path}")
        checkpoint = torch.load(resume_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        resume_epoch = checkpoint["resume_epoch"]
        best_epoch = checkpoint["best_epoch"]
        global_step = checkpoint["global_step"]

        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        # lr_scheduler.load_state_dict(checkpoint['resume_lr_scheduler'])
        print(
            f"resuming training from {ckpt_root_dir}, epoch {resume_epoch}, global_step {global_step}"
        )

    else:
        print(f"{resume_path} does not exist, training from scratch")

    if resume_epoch >= args.epochs:
        print("training is already finished. exiting.")
        # return 0

    #

    # for i, (name, param) in enumerate(model.named_parameters()):
    # if param.grad is None:
    # print(f"Rank {rank}: No grad for param {i} ({name})")

    else:
        try:
            for epoch in range(resume_epoch, args.epochs):
                # in order to shuffle the training data with distributed sampler https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
                train_sampler.set_epoch(epoch)
                model.train()
                loss = 0.0
                t1 = time()

                n_feat = args.dim
                x_dim = 3

                #
                batch_ct = 0
                for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
                    if batch is None:
                        continue
                    else:
                        # x, pose_num, pos, y = None, None, None, None
                        # if args.dataset in GBSA_DATASET_LIST:
                        # x, pose_num, pos, y, _, _, _ = batch
                        # x, pos, y, idx = batch
                        # elif args.dataset in FINETUNE_DATASET_LIST:
                        # x, pose_num, pos, y = batch

                        optimizer.zero_grad()

                        h, x, y, idx = batch

                        h, x, y, idx = (
                            h.to(current_device),
                            x.float().to(current_device),
                            y.float().to(current_device),
                            idx.to(current_device),
                        )

                        h = model["token_emb"](h)

                        edges, edge_attr = get_distance_edges_batch(x, threshold=2.0)
                        edges[0] = edges[0].to(current_device)
                        edges[1] = edges[1].to(current_device)

                        h = h.reshape(-1, n_feat)
                        if args.model == "egnn":
                            x = x.reshape(-1, x_dim)
                            h, x = model["encoder"](
                                h=h, x=x, edges=edges, edge_attr=edge_attr
                            )
                        elif args.model == "gnn":
                            h = h.float()
                            h = model["encoder"](
                                nodes=h, edges=edges, edge_attr=edge_attr
                            )

                        h = h.reshape(-1, args.gbsa_max_len, n_feat)
                        h = h.mean(dim=1)
                        # pred = model["finetune"](h).mean(dim=1).to(current_device)
                        pred = model["finetune"](h).to(current_device)

                        loss_batch = criterion(pred, y)

                        if rank == 0:
                            if global_step % args.train_loss_log_freq == 0:
                                writer.add_scalar(
                                    f"Loss/train-batch-{rank}",
                                    loss_batch.detach().item(),
                                    global_step,
                                )

                        loss += (
                            loss_batch.detach().item()
                        )  # also detach here to avoid warning

                        scaler.scale(loss_batch).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        batch_ct += 1
                        global_step += 1

                # if rank == 0:

                # '''
                val_dict = run_eval(
                    model=model,
                    loader=val_loader,
                    sampler=val_sampler,
                    args=args,
                    device=current_device,
                )
                pearson, spearman, metric, y_pred, y_true = (
                    val_dict["pearson"],
                    val_dict["spearman"],
                    val_dict["rmse"],
                    val_dict["y_pred"],
                    val_dict["y_true"],
                )

                #
                if rank == 0:
                    writer.add_scalar(f"Loss/train-epoch-{rank}", loss, epoch)
                    writer.add_scalar(f"RMSE/val-{rank}", metric, epoch)
                    writer.add_scalar(f"Spearmanr/val-{rank}", spearman, epoch)
                    writer.add_scalar(f"Pearsonr/val-{rank}", pearson, epoch)

                log.logger.info(
                    f"Val. Epoch: {epoch} | Time: {time() - t1:.1f}s | Train RMSE: {loss/len(train_loader.dataset):.5f} | Val. RMSE: {metric:.5f} | Val. Pearson: {pearson:.5f} | Val. Spearman: {spearman:.5f}| Lr*1e-5: {optimizer.param_groups[0]['lr'] * 1e5:.3f} | mean pred. {torch.mean(y_pred, axis=1).mean().item():.5f} | std pred. {torch.mean(y_pred, axis=1).std().item():.5f} | mean gt. {torch.mean(y_true, axis=1).mean().item():.5f} | std gt. {torch.mean(y_true, axis=1).std().item():.5f}"
                )

                # lr_scheduler.step(metric)

                if rank == 0:
                    if metric < best_metric:
                        best_metric = metric
                        best_model = copy.deepcopy(model)  # deep copy model
                        best_epoch = epoch + 1

                        if not model_path.parent.exists():
                            model_path.parent.mkdir(parents=True, exist_ok=True)
                        # the model weights should be synchonized across all ranks, so don't need to save one for each
                        # torch.save(checkpoint, model_path)
                        torch.save(
                            {
                                "model": best_model.state_dict(),  # todo: should I keep track of the best optimizer and scaler as well?
                                "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict(),
                                "best_epoch": best_epoch,
                                "best_metric": best_metric,
                                "save_epoch": epoch,
                                "resume_epoch": epoch + 1,
                                "global_step": global_step,
                                "epochs": args.epochs,
                                "train_size": len(train_loader.dataset),
                                "val_size": len(val_loader.dataset),
                                "test_size": len(test_loader.dataset),
                                "args": args,
                            },
                            # model_path.with_name(f"best_model-epoch-{best_epoch}.pt"))
                            # model_path.with_name(f"latest_model.pt"))
                            model_path.with_name(f"best_model-epoch-{best_epoch}.pt"),
                        )

                    # each epoch will cache the current model

                    if not model_path.parent.exists():
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                    # the model weights should be synchonized across all ranks, so don't need to save one for each
                    # torch.save(checkpoint, model_path)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "best_epoch": best_epoch,
                            "best_metric": best_metric,
                            "save_epoch": epoch,
                            "resume_epoch": epoch + 1,
                            "global_step": global_step,
                            "epochs": args.epochs,
                            "train_size": len(train_loader.dataset),
                            "val_size": len(val_loader.dataset),
                            "test_size": len(test_loader.dataset),
                            "args": args,
                        },
                        # model_path.with_name(f"best_model-epoch-{best_epoch}.pt"))
                        model_path.with_name(f"latest_model-tmp.pt"),
                    )
                    # write to temporary file first to avoid corruption
                    # if model_path.exists():
                    os.replace(
                        model_path.with_name(f"latest_model-tmp.pt"),
                        model_path.with_name(f"latest_model.pt"),
                    )
            # '''

        except Exception as e:
            print("Exception occurred:", e)
            traceback.print_exc()
            log.logger.info(f"{e}\tTraining is interrupted.")
        log.logger.info(
            "{} End Training (Time: {:.2f}h) {}".format(
                "=" * 20, (time() - t0) / 3600, "=" * 20
            )
        )

        # '''
        test_dict = run_eval(
            model=best_model,
            loader=test_loader,
            sampler=test_sampler,
            args=args,
            device=current_device,
        )

        # pearson, spearman, metric, pred, gt, path_list, frame_num = test_dict["pearson"], test_dict["spearman"], test_dict["rmse"], test_dict["y_pred"], test_dict["y_true"], test_dict["path_list"], test_dict["frame_num"]

        pearson, spearman, metric, pred, gt, idx = (
            test_dict["pearson"],
            test_dict["spearman"],
            test_dict["rmse"],
            test_dict["y_pred"],
            test_dict["y_true"],
            test_dict["idx"],
        )

        dist.barrier()

        pred = distributed_gather(pred.float())
        gt = distributed_gather(gt.float())
        idx = distributed_gather(idx)

        dist.barrier()

        if rank == 0:
            pred = pred.view(-1)
            gt = gt.view(-1)
            idx = idx.view(-1)
            metric = mse_loss(pred, gt)
            pearson = stats.pearsonr(pred, gt)[0]
            spearman = stats.spearmanr(pred, gt)[0]

            writer.add_scalar(f"RMSE/test-{rank}", metric, epoch)
            writer.add_scalar(f"Pearsonr/test-{rank}", pearson, epoch)
            writer.add_scalar(f"Spearmanr/test-{rank}", spearman, epoch)

            writer.flush()

            # save the predictions and ground truth labels
            result_path = ckpt_root_dir / Path(
                f"test_result-{model_path.with_name(f'best_model-epoch-{best_epoch}').name}-rank={rank}.pt"
            )

            torch.save(
                {
                    "train_size": len(train_loader.dataset),
                    "val_size": len(val_loader.dataset),
                    "test_size": len(test_loader.dataset),
                    "y_pred": pred,
                    "y_true": gt,
                    "id": idx,
                    # "id": path_list,
                    # "frame_num": frame_num,
                    "args": args,
                },
                result_path,
            )

            log.logger.info(
                f"rank: {rank} | RMSE: {metric} | Test Pearson: {spearman} | Test Spearman: {pearson}"
            )
            log.logger.info(
                f"Save the best model as {str(model_path)}\nBest result as {result_path}\nBest Epoch: {best_epoch}"
            )

            log.logger.info(f"{result_path.parent}")

            plot_path = result_path.with_name(
                f"{result_path.stem}_scatterplot-{epoch}.png"
            )

            print(plot_path)
            if plot_path.exists():
                pass
            else:
                f, ax = plt.subplots(1, 2, figsize=(18, 8))
                ax = ax.flatten()

                sns.scatterplot(x=gt.numpy(), y=pred.numpy(), ax=ax[0])

                sns.kdeplot(
                    x=pred, color="red", ax=ax[1], shade=True, label="Predicted MM/GBSA"
                )
                sns.kdeplot(
                    x=gt, color="blue", ax=ax[1], shade=True, label="Calculated MM/GBSA"
                )

                ax[1].legend()

                pearson_score = pearsonr(pred.numpy(), gt.numpy())[0]
                hparam_str = plot_path.parent.stem
                f.suptitle(f"{hparam_str}\n(r={pearson_score:.2f})")
                plt.tight_layout()
                f.savefig(plot_path, dpi=450)
                plt.close(f)

        dist.barrier()
        # after model has been trained or if model has already been trained and we're loading up those weights
        # run model on the eval datasets

    dist.barrier()

    run_full_test_eval(
        args=args,
        model_path=model_path,
        best_epoch=best_epoch,
        best_model=best_model,
        ckpt_root_dir=ckpt_root_dir,
        current_device=current_device,
        rank=rank,
        writer=writer,
        epoch=resume_epoch,
        batch_size_per_gpu=batch_size_per_gpu,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        log=log,
    )

    # LC HACK:
    # Synchronize ranks before destroying process group.
    # It seems rank 0 can hang if the process group is destroyed
    # by some ranks before all ranks have reached this point.
    # see https://rzlc.llnl.gov/jira/browse/ELCAP-398
    dist.barrier()
    dist.destroy_process_group()
    print("done.")


if __name__ == "__main__":
    main()
