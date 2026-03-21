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
import traceback
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as opt
from torch.nn.functional import mse_loss
from tqdm import tqdm
from torchinfo import summary
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, ConcatDataset
from datetime import timedelta
from scipy.stats import pearsonr
from sur_gbsa.ProtMD.egnn import EGNN_Network, Classifier, Regressor
from sur_gbsa.test import run_eval
from sur_gbsa.data_utils import filter_collate_fn, get_train_val_test_datasets
from sur_gbsa.ProtMD.egnn import EGNN_Network
from sur_gbsa.ProtMD.utils.utils import Logger, set_seed
from sur_gbsa.distributed_utils import distributed_gather, get_gpu_partition_mode
from sur_gbsa import PBSA_DATASET_LIST, GBSA_DATASET_LIST, FINETUNE_DATASET_LIST
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


def gather_to_rank0(obj):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Serialize the object to bytes and convert to torch.uint8 tensor
    serialized = pickle.dumps(obj)
    byte_tensor = torch.tensor(list(serialized), dtype=torch.uint8, device="cuda")
    local_size = torch.tensor([byte_tensor.numel()], dtype=torch.int64, device="cuda")

    # Gather sizes on rank 0
    size_list = (
        [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(world_size)]
        if rank == 0
        else None
    )
    dist.gather(local_size, gather_list=size_list, dst=0)

    # Determine max size for padding
    if rank == 0:
        max_size = max(s.item() for s in size_list)
    else:
        max_size = torch.tensor([0], dtype=torch.int64, device="cuda")
    max_size = (
        torch.tensor([max_size], dtype=torch.int64, device="cuda")
        if rank == 0
        else max_size
    )
    dist.broadcast(max_size, src=0)
    max_size = max_size.item()

    # Pad byte tensor
    padded = torch.zeros(max_size, dtype=torch.uint8, device="cuda")
    padded[: byte_tensor.numel()] = byte_tensor

    # Gather padded tensors
    gathered = (
        [
            torch.zeros(max_size, dtype=torch.uint8, device="cuda")
            for _ in range(world_size)
        ]
        if rank == 0
        else None
    )
    dist.gather(padded, gather_list=gathered, dst=0)

    # Deserialize on rank 0
    if rank == 0:
        return [
            pickle.loads(bytes(g[: s.item()].tolist()))
            for g, s in zip(gathered, size_list)
        ]
    return None


def main():
    rank = None
    current_device = 0 # each process sees a single GPU available

    
    gpus_per_node = 4
    



    timeout = timedelta(minutes=60)
    if "tuo" in host:
        world_size = int(os.environ["FLUX_JOB_SIZE"])
        rank = int(os.environ["FLUX_TASK_RANK"])
        local_rank = int(rank % gpus_per_node)

        print(f"{socket.gethostname()}: FLUX_JOB_SIZE: {os.environ['FLUX_JOB_SIZE']}\tFLUX_TASK_RANK: {os.environ['FLUX_TASK_RANK']}, world_size: {world_size}, rank: {rank}, local_rank:{local_rank}, current_device: {current_device}")
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

        torch.cuda.set_device(0)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
    if rank == 0:

        print(f"torch version: {torch.__version__}")
        print(
            f"world size: {world_size}, global rank: {rank}, local rank: {local_rank}, current_device: {current_device}"
        )

    scaler = None
    if "tuo" in host:
        scaler = torch.amp.GradScaler(enabled=True)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    dataset_list = PBSA_DATASET_LIST + GBSA_DATASET_LIST + FINETUNE_DATASET_LIST
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="specify to print debugging info."
    )
    args = parser.parse_args()

    pretrain_path = Path(args.pretrain)

    baseline_model = False
    # if pretrain_path.exists() and pretrain_path.name != "":
    #     if rank == 0:
    #         print(f"loading weights from {pretrain_path}")
    #     checkpoint = torch.load(
    #         pretrain_path,
    #         map_location=f"cuda:{torch.cuda.current_device()}",
    #         weights_only=False,
    #     )
    #     scratch = False

    #     if "args" in checkpoint.keys():
    #         ckpt_args = checkpoint["args"]
    #         ckpt_args.gbsa_split_strat = "random"
    #         ckpt_args.gbsa_label_process = "sign-flip"

    #     else:
    #         # loading base checkpoint file, make surethe value I paste in are correct
    #         ckpt_args = args
    #         # ckpt_args.depth = 6
    #         ckpt_args.dim = 128
    #         ckpt_args.num_nearest = 32
    #         ckpt_args.dropout = 0.15
    #         # ckpt_args.gbsa_split_strat = "random"
    #         # ckpt_args.gbsa_label_process = "sign-flip"
    #         ckpt_args.model = "egmn"
    #         baseline_model = True
    #         # ckpt_args.pretrain_tasks = ["order", "coors"]
    #         ckpt_args.pretrain_tasks = []
    # Around line 280-310, update checkpoint loading:

    if pretrain_path.exists() and pretrain_path.name != "":
        if rank == 0:
            print(f"loading weights from {pretrain_path}")
        checkpoint = torch.load(
            pretrain_path,
            map_location=f"cuda:{torch.cuda.current_device()}",
            weights_only=False,
        )
        scratch = False

        if "args" in checkpoint.keys():
            ckpt_args = checkpoint["args"]
            ckpt_args.gbsa_split_strat = "random"
            ckpt_args.gbsa_label_process = "sign-flip"
            
            # NEW: Check if old checkpoint has residue feature info
            if "model_config" in checkpoint:
                model_config = checkpoint["model_config"]
                if rank == 0:
                    print(f"Loaded model config from checkpoint: {model_config}")
                    if args.use_residue_features and not model_config.get('use_residue_features', False):
                        print("WARNING: Current config uses residue features but checkpoint does not.")
                        print("Will load with strict=False and initialize residue layers randomly.")

        else:
            # loading base checkpoint file
            ckpt_args = args
            ckpt_args.dim = 128
            ckpt_args.num_nearest = 32
            ckpt_args.dropout = 0.15
            ckpt_args.model = "egmn"
            baseline_model = True
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


#     filtered_args = {
#         k: (
#             Path(v).stem
#             if k == "train_split" and isinstance(v, str)
#             else v
#         )
#         for k, v in vars(args).items()
#         if k in ["dataset", "batch_size", "lr", "train_split", "seed", "linear_probe"]
#     }



#    # Format as string
#     args_string = "-".join(
#         f"{k}={'_'.join(map(str, v)) if isinstance(v, list) else v}"
#         for k, v in filtered_args.items()
#     )

#     ckpt_root_dir = Path(f"{args.save_path}") / Path(
#         f"{args.model}-{args_string}-pretrain={pretrain_path.exists() and pretrain_path.name != ''}"
#     ) 
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



   # Step 1: Create directory FIRST (before Logger)
    if not ckpt_root_dir.exists():
        if rank == 0:
            print(f"creating checkpoint directory {ckpt_root_dir}")  # Use print, not log
        ckpt_root_dir.mkdir(parents=True, exist_ok=True)


    # Step 2: Barrier to ensure all ranks see the directory
    dist.barrier()


    # Step 3: NOW create the Logger (directory exists)
    log = Logger(ckpt_root_dir, f"main.log", rank=rank)
    writer = SummaryWriter(ckpt_root_dir) if rank == 0 else None

    if rank == 0:
        log.logger.info(str(args))

    if rank == 0:
        log.logger.info(f"DEBUG: using {args.data_dir}")

    # Step 4: Now you can use log
    if rank == 0:
        log.logger.info(f"{ckpt_root_dir} exists: {ckpt_root_dir.exists()}")

    model_path = ckpt_root_dir / Path(f"best_model.pt")
    if model_path.exists():
        if rank == 0:
            log.logger.info(f"{model_path} already exists, work is finished.")

    # model_path = ckpt_root_dir / Path(f"best_model.pt")



    # if model_path.exists():
        # if rank == 0:
            # print(f"{model_path} already exists, work is finished.")
        # dist.barrier()
        # dist.destroy_process_group()
        # return 0
    # else:
        # pass

    # scratch = True
    # baseline_model = False

    lr_per_gpu = args.lr / world_size  # scale the learning according to number of GPUs
    batch_size_per_gpu = int(args.batch_size / world_size)


    # log = Logger(ckpt_root_dir, f"main.log", rank=rank)
    # writer = SummaryWriter(ckpt_root_dir) if rank == 0 else None

    # if rank == 0:
        # log.logger.info(str(args))

    # if rank == 0:
        # print(f"DEBUG: using {args.data_dir}")
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(args)

    num_workers_train = args.num_workers
    # num_workers_train = max(
        # 1, (os.cpu_count() // gpus_per_node) - 3
    # )  # # minus three to reserve the validation, test, and final output eval loop workers

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    # import pdb; pdb.set_trace()
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=num_workers_train,
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

    if args.verbose and rank == 0:
        print(f"Rank {rank}: Sampler will provide {len(train_sampler)} samples")

    if args.verbose:
        print(f"\n{'='*50}")
        print(f"DEBUG Rank {rank}:")
        print(f"  train_dataset type: {type(train_dataset)}")
        print(f"  train_dataset length: {len(train_dataset)}")

        if isinstance(train_dataset, ConcatDataset):
            print(f"  ConcatDataset with {len(train_dataset.datasets)} sub-datasets")
            for i, ds in enumerate(train_dataset.datasets[:3]):  # Check first 3
                print(f"    Sub-dataset {i}:")
                print(f"      type: {type(ds).__name__}")
                print(f"      length: {len(ds)}")
                print(f"      has filter_dataset_for_rank: {hasattr(ds, 'filter_dataset_for_rank')}")
                if hasattr(ds, 'x_0'):
                    print(f"      x_0 shape: {ds.x_0.shape}")
        print(f"{'='*50}\n")



    if rank == 0 and args.verbose:
        print(f"\n{'='*60}")
        print(f"POST-FILTERING CHECK (Mode: {os.environ.get('GPU_MODE', 'unknown')})")
        print(f"  World size: {world_size}")
        print(f"  Batch size per GPU: {batch_size_per_gpu}")
        print(f"  train_dataset length: {len(train_dataset)}")
        print(f"  Expected batches: {len(train_dataset) // (batch_size_per_gpu * world_size)}")
        
        if isinstance(train_dataset, ConcatDataset):
            total_from_subdatasets = sum(len(ds) for ds in train_dataset.datasets)
            print(f"  ConcatDataset sub-datasets sum: {total_from_subdatasets}")
            print(f"  cumulative_sizes[-1]: {train_dataset.cumulative_sizes[-1]}")
            print(f"  Match: {total_from_subdatasets == len(train_dataset)}")
        print(f"{'='*60}\n")




    if rank == 0:
        log.logger.info(
            f"using {args.dataset} for training/testing (train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)})"
        )

    if rank == 0:
        log.logger.info(
            "TODO: make sure all `forward` function outputs participate in calculating loss."
        )
    find_unused_parameters = True

    # # a virtual atom is better than global pooling
    # encoder = EGNN_Network(
    #     num_tokens=ckpt_args.tokens,
    #     dim=ckpt_args.dim,
    #     depth=ckpt_args.depth,
    #     num_nearest_neighbors=ckpt_args.num_nearest,
    #     dropout=ckpt_args.dropout,
    #     global_linear_attn_every=1,
    #     norm_coors=True,
    #     coor_weights_clamp_value=2.0,
    #     #  update_coors=False, # Derek: I put this here to see if it affects the output when the atoms are constant
    #     aggregate=False,
    #     num_residue_tokens=22,
    #     residue_dim=32
    # )
    # Around line 480-495, replace encoder initialization:

    encoder = EGNN_Network(
        num_tokens=ckpt_args.tokens,
        dim=ckpt_args.dim,
        depth=ckpt_args.depth,
        num_nearest_neighbors=ckpt_args.num_nearest,
        dropout=ckpt_args.dropout,
        global_linear_attn_every=1,
        norm_coors=True,
        coor_weights_clamp_value=2.0,
        aggregate=False,
        num_residue_tokens=22 if args.use_residue_features else None,  # NEW: Conditional
        residue_dim=32 if args.use_residue_features else None,  # NEW: Conditional
    )

    encoder.to(current_device)

    model = torch.nn.ModuleDict()

    # Load weights if pretrained
    if pretrain_path.exists() and pretrain_path.name != "":
        if baseline_model:
            if rank == 0:
                log.logger.info("loading weights for baseline model")
            encoder.load_state_dict(checkpoint["model"])
        else:
            if rank == 0:
                log.logger.info("loading weights for pretrained model")
            # Will load full state_dict later

    # Handle linear probe mode
    if args.linear_probe:
        if rank == 0:
            log.logger.info("Linear probe mode: freezing encoder, training head only")
        for param in encoder.parameters():
            param.requires_grad = False
        # Don't wrap frozen encoder in DDP
        model["encoder"] = encoder.to(current_device)
    else:
        # Wrap encoder in DDP for distributed training
        model["encoder"] = DDP(
            encoder.to(current_device),
            device_ids=[current_device],
            find_unused_parameters=find_unused_parameters,
        )

    # Always wrap trainable head in DDP
    model["finetune"] = DDP(
        Regressor(ckpt_args.dim).to(current_device),
        device_ids=[current_device],
        find_unused_parameters=find_unused_parameters,
    )

    # Load full state dict if pretrained (non-baseline)
    if pretrain_path.exists() and pretrain_path.name != "" and not baseline_model:
        model.load_state_dict(checkpoint["model"])

    model = model.float()

    # After model creation, before training
    def identify_parameters_by_index(model_dict):
        """Map parameter indices to names"""
        all_params = []
        
        for module_name, module in model_dict.items():
            actual_module = module.module if hasattr(module, 'module') else module
            
            for param_name, param in actual_module.named_parameters():
                if param.requires_grad:
                    all_params.append((module_name, param_name, param))
        
        return all_params

    # Run this once
    if rank == 0:
        all_params = identify_parameters_by_index(model)
        
        # Check the problematic indices
        problem_indices = [191, 196, 197, 198, 199]
        
        print(f"\n{'='*60}")
        print("UNUSED PARAMETERS:")
        for idx in problem_indices:
            if idx < len(all_params):
                module_name, param_name, param = all_params[idx]
                print(f"  Index {idx}: {module_name}.{param_name} (shape: {param.shape})")
        print(f"{'='*60}\n")


    if rank == 0:
        summary(model)

    best_model = model
    best_epoch = 0
    criterion = torch.nn.MSELoss()
    best_metric = 1e9

    optimizer = opt.AdamW(
        model.parameters(), lr=lr_per_gpu, weight_decay=args.weight_decay
    )

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
        if rank == 0:
            print(f"resuming training from {resume_path}")
        checkpoint = torch.load(resume_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        resume_epoch = checkpoint["resume_epoch"]
        best_epoch = checkpoint["best_epoch"]
        global_step = checkpoint["global_step"]

        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

        if rank == 0:
            print(
                f"resuming training from {ckpt_root_dir}, epoch {resume_epoch}, global_step {global_step}"
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

                # '''
                batch_ct = 0
                # Only show progress bar on rank 0
                train_iter = (
                    tqdm(train_loader, desc=f"Train Epoch {epoch}")
                    if rank == 0
                    else train_loader
                )
                for batch_idx, batch in enumerate(train_iter):

                    if batch is None:
                        continue
                    else:

                        if batch_idx % 10 == 0 and args.verbose:
                            allocated = torch.cuda.memory_allocated() / 1e9
                            reserved = torch.cuda.memory_reserved() / 1e9
                            print(f"Batch {batch_idx}: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB")
                            print(f"x.shape: {batch[0].shape}, y.shape: {batch[2].shape}")



                        # import pdb; pdb.set_trace()
                        if args.use_residue_features:
                            x, res_feats, pos, y, idx = batch
                        else:
                            x, pos, y, idx = batch
                            res_feats = None


                        optimizer.zero_grad()


                        t0 = time()
                        x, pos, y = (
                            x.long().to(current_device),
                            pos.float().to(current_device),
                            y.float().to(current_device),
                        )

                        t1 = time()
                        mask = x != 0

                        out = (model["encoder"](x, pos, res_feats=res_feats, mask=mask)[1]).to(
                            current_device
                        )
                        out = out.mean(dim=1)

                        pred = model["finetune"](out).to(current_device)

                        loss_batch = criterion(pred, y)



                        loss += loss_batch.item()
                        scaler.scale(loss_batch).backward()
                        scaler.step(optimizer)
                        scaler.update()

                        t2 = time()

                        batch_ct += 1
                        global_step += 1



                        if rank == 0 and batch_idx % 10 == 0 and args.verbose:
                            print(f"Batch {batch_idx}:")
                            print(f"  Data loading: {(t1-t0)*1000:.1f}ms")
                            print(f"  Training step: {(t2-t1)*1000:.1f}ms")

                val_dict = run_eval(
                    model,
                    val_loader,
                    # sampler=val_sampler,
                    args=args,
                    device=current_device,
                )
                pearson, spearman, pdb_pearsonr, pdb_spearmanr, metric, y_pred, y_true = (
                    val_dict["pearson"],
                    val_dict["spearman"],
                    val_dict["pdb_pearson"],
                    val_dict["pdb_spearman"],
                    val_dict["rmse"],
                    val_dict["y_pred"],
                    val_dict["y_true"]
                )

                # Around line 645-680, replace your checkpoint saving code with this:

                if rank == 0:
                    if writer:
                        writer.add_scalar(f"Loss/train-epoch-{rank}", loss, epoch)
                        writer.add_scalar(f"RMSE/val-{rank}", metric, epoch)
                        writer.add_scalar(f"Spearmanr/val-{rank}", spearman, epoch)
                        writer.add_scalar(f"Pearsonr/val-{rank}", pearson, epoch)
                        writer.add_scalar(f"PDB-Pearsonr/val-{rank}", pdb_pearsonr, epoch)
                        writer.add_scalar(f"PDB-Spearmanr/val-{rank}", pdb_spearmanr, epoch)

                    # import pdb; pdb.set_trace() 
                    log.logger.info(
                        f"Val. Epoch: {epoch} | Time: {time() - t1:.1f}s | Train RMSE: {loss/len(train_loader.dataset):.5f} | Val. RMSE: {metric:.5f} | Val. Pearson: {pearson:.5f} | Val. Spearman: {spearman:.5f} | Val. PDB-Pearson: {pdb_pearsonr:.5f} | Val. PDB-Spearman: {pdb_spearmanr:.5f} | Lr*1e-5: {optimizer.param_groups[0]['lr'] * 1e5:.3f} | mean pred. {torch.mean(y_pred).mean().item():.5f} | std pred. {torch.mean(y_pred).std().item():.5f} | mean gt. {torch.mean(y_true).mean().item():.5f} | std gt. {torch.mean(y_true).std().item():.5f}"
                    )
                    
                    # Create config dict to save with checkpoint
                    model_config = {
                        'depth': ckpt_args.depth,
                        'dim': ckpt_args.dim,
                        'num_tokens': ckpt_args.tokens,
                        'num_nearest_neighbors': ckpt_args.num_nearest,
                        'dropout': ckpt_args.dropout,
                        'use_residue_features': args.use_residue_features,
                        'num_residue_tokens': 22 if args.use_residue_features else None,
                        'residue_dim': 32 if args.use_residue_features else None,
                        'model_type': ckpt_args.model,
                    }
                    
                    # Save best model
                    if metric < best_metric:
                        best_metric = metric
                        best_model = copy.deepcopy(model)
                        best_epoch = epoch + 1
                        
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "model": best_model.state_dict(),
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
                                "model_config": model_config,  # NEW: Save model config
                            },
                            model_path.with_name(f"best_model-epoch-{best_epoch}.pt"),
                        )
                    
                    # Save latest checkpoint
                    model_path.parent.mkdir(parents=True, exist_ok=True)
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
                            "model_config": model_config,  # NEW: Save model config
                        },
                        model_path.with_name(f"latest_model-tmp.pt"),
                    )
                    os.replace(
                        model_path.with_name(f"latest_model-tmp.pt"),
                        model_path.with_name(f"latest_model.pt"),
                    )


                
        except Exception as e:
            # All ranks print their error
            print(f"Rank {rank}: Exception occurred!")
            print(f"Error: {e}")
            traceback.print_exc()
            
            # Only rank 0 logs to file
            if rank == 0:
                log.logger.info(f"{e}\tTraining is interrupted.")
            
            # ALL ranks must participate in barrier and cleanup!
            try:
                dist.barrier()  # Moved outside if statement
                dist.destroy_process_group()
            except:
                pass  # If dist is already broken, ignore cleanup errors
            
            raise  # All ranks re-raise
        if rank == 0:
            log.logger.info(
                "{} End Training (Time: {:.2f}h) {}".format(
                    "=" * 20, (time() - t0) / 3600, "=" * 20
                )
            )

    # this needs distributed communication to aggregate results across all ranks
    # best_model_test_dict = run_eval(best_model, test_loader, sampler=test_sampler, args=args, device=current_device)
    # latest_model_test = run_eval(model, test_loader, sampler=test_sampler, args=args, device=current_device )

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("done.")


if __name__ == "__main__":

    main()
