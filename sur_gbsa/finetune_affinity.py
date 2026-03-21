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

if "tuo" in host:
    del os.environ["OMP_PLACES"]
    del os.environ["OMP_PROC_BIND"]
import copy
from time import time
from pathlib import Path

# import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as opt
from torch.nn.functional import mse_loss
from scipy import stats
import torch.nn as nn
from sur_gbsa.ProtMD.egnn import EGNN_Network
from sur_gbsa.ProtMD.utils.utils import Logger, set_seed
from tqdm import tqdm
import pandas as pd
from torchinfo import summary
from sur_gbsa.data_utils import filter_collate_fn, get_train_val_test_datasets
from sur_gbsa.distributed_utils import distributed_gather
from sur_gbsa.ProtMD.egnn import EGNN_Network, Classifier, Regressor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import all_gather_object
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sur_gbsa import GBSA_DATASET_LIST, FINETUNE_DATASET_LIST
from bisect import bisect_right

import warnings

# Suppress the specific DDP warning
warnings.filterwarnings(
    "ignore", message=".*find_unused_parameters=True was specified.*"
)


DEV_IDS = list(range(torch.cuda.device_count()))
world_size = len(DEV_IDS)

import torch
import torch.distributed as dist
import pickle


def get_metadata(concat_dataset, global_idx):
    cs = concat_dataset.cumulative_sizes
    ds_idx = bisect_right(cs, global_idx)
    local_idx = global_idx if ds_idx == 0 else global_idx - cs[ds_idx - 1]
    return concat_dataset.datasets[ds_idx].__getfilename__(
        local_idx
    ), concat_dataset.datasets[ds_idx].__getframenum__(local_idx)


def run_eval(model, loader, sampler, args, device):
    """Run evaluation on the given dataset.
    
    Args:
        model: Model to evaluate
        loader: Data loader
        sampler: Distributed sampler
        args: Arguments
        device: Device to use for computation (should be current_device)
    """
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


                # x, pos, y, idx = batch

                if args.use_residue_features:
                    x, res_feats, pos, y, idx = batch
                else:
                    x, pos, y, idx = batch
                    res_feats = None

                y_true.append(y)
                x, pos, y = x.long().to(device), pos.float().to(device), y.to(device)
                mask = x != 0

                out = model["encoder"](x, pos, res_feats=res_feats, mask=mask)[1]
                # leave the atomistic dimension intact
                out = out.mean(dim=1)
                latent_list.append(out.cpu())
                pred = model["finetune"](out)
                y_pred.append(pred)
                idx_list.append(idx)
                batch_size = len(y)  # or len(batch[0]) depending on your dataset output

                batch_indices = rank_indices[batch_start : batch_start + batch_size]
                batch_start += batch_size

                # metadata_list = [get_metadata(loader.dataset, idx) for idx in batch_indices]
                # metadata = [get_metadata(loader.dataset, idx) for idx in batch_indices]

                # print(len(y), len(batch_indices), len(metadata[0]), len(metadata[1]))
                # metadata_list.extend([x[0] for x in metadata])
                # frame_list.extend([x[1] for x in metadata])
                # print(metadata_list)
                # Now call your custom method on each index

    y_pred = torch.cat(y_pred).reshape(-1, 1).cpu()
    y_true = torch.cat(y_true).reshape(-1, 1).cpu()
    idx = torch.cat(idx_list).reshape(-1, 1).cpu()
    latent = torch.cat(latent_list).reshape(-1, out.shape[-1])
    pearson = stats.pearsonr(y_pred.numpy().flatten(), y_true.numpy().flatten())[0]
    spearman = stats.spearmanr(y_pred.numpy().flatten(), y_true.numpy().flatten())[0]
    rmse = torch.sqrt(
        mse_loss(y_true, y_pred, reduction="sum") / len(loader.dataset)
    ).item()

    return {"pearson": pearson, "spearman":spearman, "rmse":rmse, 
            "y_pred": y_pred, "y_true": y_true, "idx": idx,
            "latent": latent, 
            # "metadata": metadata_list,  
            # "frame_num": frame_list,
            }

def main():
    # Get rank early for controlling output
    rank = None
    current_device = 0 # each process sees a single GPU available

    if rank == 0:
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
            device_id=current_device,
            timeout=timeout,
        )

    elif "matrix" in host:

        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(rank % gpus_per_node)

        print(rank, local_rank)
        # Fix: Use local_rank or 0, not global rank
        current_device = 0  # Consistent with torch.cuda.set_device(0) below
        torch.cuda.set_device(current_device)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )

    if rank == 0:
        print(
            f"world size: {world_size}, global rank: {rank}, local rank: {local_rank}, current_device: {current_device}"
        )

    scaler = None
    if "tuo" in host:
        scaler = torch.amp.GradScaler(enabled=True)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    dataset_list = FINETUNE_DATASET_LIST
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
        "--model", choices=["egmn", "egnn"], default="egmn"
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
        "--weight-decay",
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
    parser.add_argument(
        "--noise",
        type=bool,
        default=True,
        help="Whether to add noise during pretraining.",
    )  # (derek) does this not cause a bug?
    parser.add_argument(
        "--tokens", type=int, default=100, help="The default number of atom classes."
    )
    parser.add_argument(
        "--use_residue_features", default=False, action="store_true", help="Whether to use residue features."
    )
    args = parser.parse_args()

    pretrain_path = Path(args.pretrain)

    baseline_model = False
    if pretrain_path.exists() and pretrain_path.name != "":
        if rank == 0:
            print(f"loading weights from {pretrain_path}")
        checkpoint = torch.load(
            pretrain_path,
            map_location=f"cuda:{current_device}",
            weights_only=False,
        )
        scratch = False

        if "args" in checkpoint.keys():
            ckpt_args = checkpoint["args"]
            ckpt_args.gbsa_split_strat = "random"
            ckpt_args.gbsa_label_process = "sign-flip"

        else:
            # loading base checkpoint file, make surethe value I paste in are correct
            ckpt_args = args
            # ckpt_args.depth = 6
            ckpt_args.dim = 128
            ckpt_args.num_nearest = 32
            ckpt_args.dropout = 0.15
            # ckpt_args.gbsa_split_strat = "random"
            # ckpt_args.gbsa_label_process = "sign-flip"
            ckpt_args.model = "egmn"
            baseline_model = True
            # ckpt_args.pretrain_tasks = ["order", "coors"]
            ckpt_args.pretrain_tasks = []
    else:
        if rank == 0:
            print(f"model path does not exist. using randomly initialized model.")
        ckpt_args = args
        # ckpt_args.gbsa_split_strat = "random"
        # ckpt_args.gbsa_label_process = "sign-flip"
        ckpt_args.model = "egmn"
        # ckpt_args.depth = 6
        ckpt_args.dim = 128
        ckpt_args.num_nearest = 32
        ckpt_args.dropout = 0.15
        baseline_model = False
        ckpt_args.pretrain_tasks = []

    # seed RNGs
    set_seed(args.seed)

    # filtered_args = {
    #     k: (
    #         Path(v).stem
    #         if k == "train_split" and isinstance(v, str)  # Handle any file extension
    #         else v
    #     )
    #     for k, v in vars(args).items()
    #     if k in ["dataset", "batch_size", "lr", "train_split", "seed", "weight_decay"]
    # } 

    # # Format as string
    # args_string = "-".join(
    #     f"{k}={'_'.join(map(str, v)) if isinstance(v, list) else v}"
    #     for k, v in filtered_args.items()
    # )

    # ckpt_root_dir = Path(f"{args.save_path}") / Path(
    #     f"{args.model}-{args_string}-pretrain={pretrain_path.exists() and pretrain_path.name != ''}"
    # )

    filtered_args = {
        k: (
            Path(v).stem
            if k == "train_split" and isinstance(v, str)
            else v
        )
        for k, v in vars(args).items()
        if k in ["dataset", "batch_size", "lr", "train_split", "seed", "linear_probe"]
    }

    # Format as string
    args_string = "-".join(
        f"{k}={'_'.join(map(str, v)) if isinstance(v, list) else v}"
        for k, v in filtered_args.items()
    )

    # ADD THIS: Create residue feature indicator
    residue_suffix = "_res" if args.use_residue_features else ""

    # MODIFY THIS: Add residue_suffix to the checkpoint directory path
    ckpt_root_dir = Path(f"{args.save_path}") / Path(
        f"{args.model}-{args_string}-pretrain={pretrain_path.exists() and pretrain_path.name != ''}{residue_suffix}"
    )
    if rank == 0:
        print(ckpt_root_dir, ckpt_root_dir.exists())

    if not ckpt_root_dir.exists():
        if rank == 0:
            print(f"creating checkpoint directory {ckpt_root_dir}")
        ckpt_root_dir.mkdir(parents=True, exist_ok=True)

    model_path = ckpt_root_dir / Path(f"best_model.pt")

    if model_path.exists():
        if rank == 0:
            print(f"{model_path} already exists, work is finished.")
        # dist.barrier()
        # dist.destroy_process_group()
        # return 0
    else:
        pass

    # scratch = True
    # baseline_model = False

    lr_per_gpu = args.lr / world_size  # scale the learning according to number of GPUs
    batch_size_per_gpu = int(args.batch_size / world_size)

    log = Logger(ckpt_root_dir, f"main.log", rank=rank)
    writer = SummaryWriter(ckpt_root_dir) if rank == 0 else None

    if rank == 0:
        log.logger.info(str(args))

    if rank == 0:
        print(f"DEBUG: using {args.data_dir}")

    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(args)


    num_workers_train = max(
        1, (os.cpu_count() // gpus_per_node) - 3
    )  # # minus three to reserve the validation, test, and final output eval loop workers

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=16,
        collate_fn=filter_collate_fn,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True if num_workers_train > 0 else False,
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

    if rank == 0:
        log.logger.info(
            f"using {args.dataset} for training/testing (train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)})"
        )

    if rank == 0:
        print(
            "TODO: make sure all `forward` function outputs participate in calculating loss."
        )
    find_unused_parameters = True

    # a virtual atom is better than global pooling
    encoder = EGNN_Network(
        num_tokens=ckpt_args.tokens,
        dim=ckpt_args.dim,
        depth=ckpt_args.depth,
        num_nearest_neighbors=ckpt_args.num_nearest,
        dropout=ckpt_args.dropout,
        global_linear_attn_every=1,
        norm_coors=True,
        coor_weights_clamp_value=2.0,
        #  update_coors=False, # Derek: I put this here to see if it affects the output when the atoms are constant
        aggregate=False,
    )

    encoder.to(current_device)

    model = torch.nn.ModuleDict()

    if not (pretrain_path.exists() and pretrain_path.name != ""):

        model["encoder"] = DDP(
            encoder.to(current_device),
            device_ids=[current_device],
            find_unused_parameters=find_unused_parameters,
        )

        if "order" in ckpt_args.pretrain_tasks:

            model["order"] = DDP(
                Classifier(ckpt_args.dim).to(current_device),
                device_ids=[current_device],
                find_unused_parameters=find_unused_parameters,
            )

        if "rmsd" in ckpt_args.pretrain_tasks:

            model["rmsd"] = DDP(
                Regressor(ckpt_args.dim).to(current_device),
                device_ids=[current_device],
                find_unused_parameters=find_unused_parameters,
            )

        if "pdbid" in ckpt_args.pretrain_tasks:

            model["pdbid"] = DDP(
                torch.nn.Sequential(
                    nn.Linear(ckpt_args.dim, int(ckpt_args.dim * 2)),
                    nn.ReLU(),
                    nn.Linear(int(ckpt_args.dim * 2), ckpt_args.dim),
                ).to(current_device),
                device_ids=[current_device],
                find_unused_parameters=find_unused_parameters,
            )
        if "cluster" in ckpt_args.pretrain_tasks:

            model["cluster"] = DDP(
                torch.nn.Sequential(
                    nn.Linear(ckpt_args.dim, int(ckpt_args.dim * 2)),
                    nn.ReLU(),
                    nn.Linear(int(ckpt_args.dim * 2), ckpt_args.dim),
                ).to(current_device),
                device_ids=[current_device],
                find_unused_parameters=find_unused_parameters,
            )

        if "pose" in ckpt_args.pretrain_tasks:

            model["pose"] = DDP(
                torch.nn.Sequential(
                    nn.Linear(ckpt_args.dim, int(ckpt_args.dim * 2)),
                    nn.ReLU(),
                    nn.Linear(int(ckpt_args.dim * 2), ckpt_args.dim),
                ).to(current_device),
                device_ids=[current_device],
                find_unused_parameters=find_unused_parameters,
            )

        if args.linear_probe:
            if rank == 0:
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
        if rank == 0:
            print("loading weights for baseline model")
        encoder.load_state_dict(checkpoint["model"])

        if args.linear_probe:
            if rank == 0:
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

        if rank == 0:
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
        model.load_state_dict({key: value for key, value in checkpoint["model"].items() if key != "finetune"})

        if args.linear_probe:
            if rank == 0:
                print(
                    "training in linear-probe mode, turning off gradient update for encoder."
                )
            for param in encoder.parameters():
                param.requires_grad = False

    model = model.float()




    # Reinitialize the finetune layer parameters



    if rank == 0:
        summary(model)

    best_model = model
    best_epoch = 0
    criterion = torch.nn.MSELoss()
    best_metric = 1e9

    optimizer = opt.AdamW(
        model.parameters(), lr=lr_per_gpu, weight_decay=args.weight_decay
    )

    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=5e-6)


    if rank == 0:
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

        checkpoint = torch.load(resume_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        resume_epoch = checkpoint["resume_epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_metric = checkpoint["best_metric"]
        global_step = checkpoint["global_step"]

        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        
        # Load scheduler state if it exists
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if rank == 0:
            print(
                f"resuming training from {ckpt_root_dir}, epoch {resume_epoch}, global_step {global_step}, best_epoch {best_epoch}, best_metric {best_metric}, lr {optimizer.param_groups[0]['lr']:.2e}"
            )

    else:
        if rank == 0:
            print(f"{resume_path} does not exist, training from scratch")

    if resume_epoch >= args.epochs:
        if rank == 0:
            print("training is already finished. exiting.")

    else:

        # '''
        try:
            for epoch in range(resume_epoch, args.epochs):
                # Synchronize all processes at the start of each epoch
                dist.barrier()

                # in order to shuffle the training data with distributed sampler https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
                train_sampler.set_epoch(epoch)
                model.train()
                loss = 0.0
                t1 = time()

                batch_ct = 0
                # Only show progress bar on rank 0
                train_iter = (
                    tqdm(train_loader, desc=f"Train Epoch {epoch}")
                    if rank == 0
                    else train_loader
                )
                for batch in train_iter:

                    if batch is None:
                        continue
                    else:

                        # x, pos, y, idx = batch


                        if args.use_residue_features:
                            x, res_feats, pos, y, idx = batch
                        else:
                            x, pos, y, idx = batch
                            res_feats = None

                        optimizer.zero_grad()


                        x, pos, y = (
                            x.long().to(current_device),
                            pos.float().to(current_device),
                            y.float().to(current_device),
                        )

                        mask = x != 0

                        out = model["encoder"](x, pos, res_feats=res_feats, mask=mask)[1]
                        out = out.mean(dim=1)

                        pred = model["finetune"](out)

                        loss_batch = criterion(pred, y)

                        # if rank == 0 and writer:
                            # if global_step % args.train_loss_log_freq == 0:
                                # writer.add_scalar(
                                    # f"Loss/train-batch-{rank}", loss_batch, global_step
                                # )

                        loss += loss_batch.item()
                        scaler.scale(loss_batch).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        batch_ct += 1
                        global_step += 1
                        # break


                # prevent deadlocking by running on all ranks
                val_dict = run_eval(
                    model,
                    val_loader,
                    sampler=val_sampler,
                    args=args,
                    device=current_device,
                )               

                # Gather predictions from all ranks to compute global validation metric
                val_pred = val_dict["y_pred"]
                val_true = val_dict["y_true"]

                dist.barrier()
                val_pred_gathered = distributed_gather(val_pred.float())
                val_true_gathered = distributed_gather(val_true.float())
                dist.barrier()

                # Compute global validation metrics on all ranks (or just rank 0)
                if rank == 0:
                    val_pred_gathered = val_pred_gathered.view(-1)
                    val_true_gathered = val_true_gathered.view(-1)
                    
                    metric = torch.sqrt(mse_loss(val_true_gathered, val_pred_gathered)).item()
                    pearson = stats.pearsonr(val_pred_gathered.numpy(), val_true_gathered.numpy())[0]
                    spearman = stats.spearmanr(val_pred_gathered.numpy(), val_true_gathered.numpy())[0]
                else:
                    # Initialize placeholder values on non-zero ranks
                    metric = torch.tensor(0.0, device=current_device)
                    pearson = 0.0
                    spearman = 0.0

                # Broadcast the validation metric from rank 0 to all ranks
                # This ensures all ranks use the same metric for scheduler.step()
                metric_tensor = torch.tensor([metric], dtype=torch.float32, device=current_device)
                dist.broadcast(metric_tensor, src=0)
                metric = metric_tensor.item()

                # Broadcast other metrics for logging
                pearson_tensor = torch.tensor([pearson], dtype=torch.float32, device=current_device)
                spearman_tensor = torch.tensor([spearman], dtype=torch.float32, device=current_device)
                dist.broadcast(pearson_tensor, src=0)
                dist.broadcast(spearman_tensor, src=0)
                pearson = pearson_tensor.item()
                spearman = spearman_tensor.item()

                # Step the learning rate scheduler on ALL ranks with the same metric
                lr_scheduler.step(metric)

                # Log on rank 0
                if rank == 0 and writer:
                    writer.add_scalar(f"Loss/train-epoch-{rank}", loss, epoch)
                    writer.add_scalar(f"RMSE/val-{rank}", metric, epoch)
                    writer.add_scalar(f"Spearmanr/val-{rank}", spearman, epoch)
                    writer.add_scalar(f"Pearsonr/val-{rank}", pearson, epoch)
                    writer.add_scalar(f"Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
                    
                    log.logger.info(
                        f"Val. Epoch: {epoch} | Time: {time() - t1:.1f}s | "
                        f"Train RMSE: {loss/len(train_loader.dataset):.5f} | "
                        f"Val. RMSE: {metric:.5f} | "
                        f"Val. Pearson: {pearson:.5f} | "
                        f"Val. Spearman: {spearman:.5f} | "
                        f"Lr: {optimizer.param_groups[0]['lr']:.2e} | "
                        f"mean pred. {val_pred_gathered.mean().item():.5f} | "
                        f"std pred. {val_pred_gathered.std().item():.5f} | "
                        f"mean gt. {val_true_gathered.mean().item():.5f} | "
                        f"std gt. {val_true_gathered.std().item():.5f}"
                    )

                    if metric < best_metric:
                        print(f"New best model found at epoch {epoch}, val metric {metric:.5f}, "
                              f"previous best {best_metric:.5f}, saving model...")
                        best_metric = metric
                        best_model = copy.deepcopy(model)  # deep copy model
                        best_epoch = epoch + 1

                        if not model_path.parent.exists():
                            model_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        torch.save(
                            {
                                "model": best_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
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
                            model_path.with_name(f"best_model-epoch-{best_epoch}.pt"),
                        )

                    # each epoch will cache the current model
                    if not model_path.parent.exists():
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
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
                        model_path.with_name(f"latest_model-tmp.pt"),
                    )
                    
                    os.replace(
                        model_path.with_name(f"latest_model-tmp.pt"),
                        model_path.with_name(f"latest_model.pt"),
                    )
                
                dist.barrier()

        except Exception as e:
            if rank == 0:
                log.logger.info(f"{e}\tTraining is interrupted.")
        if rank == 0:
            log.logger.info(
                "{} End Training (Time: {:.2f}h) {}".format(
                    "=" * 20, (time() - t0) / 3600, "=" * 20
                )
            )

        # """

        # TODO: this is failing in large distributed training runs, need to figure out why
        test_dict = run_eval(best_model, test_loader, sampler=test_sampler, args=args, device=current_device)

        pearson, spearman, metric, pred, gt, idx = test_dict["pearson"], test_dict["spearman"], test_dict["rmse"], test_dict["y_pred"], test_dict["y_true"], test_dict["idx"]
        dist.barrier()
        pred = distributed_gather(pred.float())
        gt = distributed_gather(gt.float())
        idx = distributed_gather(idx)

        dist.barrier()



        if rank == 0:

            pred = pred.view(-1)
            gt = gt.view(-1)
            idx = idx.view(-1)

            metric = torch.sqrt(mse_loss(pred, gt)).item() # this isn't rmse YET
            pearson = stats.pearsonr(pred, gt)[0]
            spearman = stats.spearmanr(pred, gt)[0]       

            if writer:
                writer.add_scalar(f"RMSE/test-{rank}", metric, epoch)
                writer.add_scalar(f"Pearsonr/test-{rank}", pearson, epoch)
                writer.add_scalar(f"Spearmanr/test-{rank}", spearman, epoch)

                writer.flush()

            # save the predictions and ground truth labels
            result_path = ckpt_root_dir / Path(f"test_result-{model_path.with_name(f'best_model-epoch-{best_epoch}').name}-rank={rank}.pt")
            


            torch.save({"train_size": len(train_loader.dataset), 
                        "val_size": len(val_loader.dataset), 
                        "test_size": len(test_loader.dataset), 
                        "y_pred": pred,
                        "y_true": gt,
                        "id": idx, 
                        # "id": path_list, 
                        # "frame_num": frame_num, 
                        "args": args}, 
                        result_path)


            log.logger.info(f'rank: {rank} | RMSE: {metric} | Test Pearson: {spearman} | Test Spearman: {pearson}')
            log.logger.info(f'Save the best model as {str(model_path)}\nBest result as {result_path}\nBest Epoch: {best_epoch}')

            log.logger.info(f'{result_path.parent}')


            plot_path = result_path.with_name(f"{result_path.stem}_scatterplot-{epoch}.png")

            print(plot_path)
            if plot_path.exists():
                pass
            else:



                f, ax = plt.subplots(1, 2, figsize=(18, 8))
                ax = ax.flatten()

                sns.scatterplot(x=gt.numpy(), y=pred.numpy(), ax=ax[0])
                
                sns.kdeplot(x=pred, color="red", ax=ax[1], shade=True, label="Predicted MM/GBSA")
                sns.kdeplot(x=gt, color="blue", ax=ax[1], shade=True, label="Calculated MM/GBSA")
                
                ax[1].legend()

                pearson_score = pearsonr(pred.numpy(), gt.numpy())[0]
                hparam_str=plot_path.parent.stem
                f.suptitle(f"{hparam_str}\n(r={pearson_score:.2f})")
                plt.tight_layout()
                f.savefig(plot_path, dpi=450)
                plt.close(f)


        dist.barrier()
        # after model has been trained or if model has already been trained and we're loading up those weights


    # LC HACK:
    # Synchronize ranks before destroying process group.
    # It seems rank 0 can hang if the process group is destroyed
    # by some ranks before all ranks have reached this point.
    # see https://rzlc.llnl.gov/jira/browse/ELCAP-398
    # """
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("done.")


if __name__ == "__main__":

    main()