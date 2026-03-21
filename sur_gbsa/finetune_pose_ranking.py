"""
Fine-tune pretrained models on CASF-2016 pose ranking task.
"""

import socket
host = socket.gethostname()
import os

# if "tuo" in host:
    # del os.environ["OMP_PLACES"]
    # del os.environ["OMP_PROC_BIND"]

os.environ["MASTER_ADDR"] = host
os.environ["MASTER_PORT"] = str(23456)
import copy
from time import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as opt
import torch.nn as nn
from scipy import stats
from sur_gbsa.ProtMD.egnn import EGNN_Network, Regressor
from sur_gbsa.ProtMD.utils.utils import Logger, set_seed
from tqdm import tqdm
import pandas as pd
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import numpy as np
import argparse

# Import the CASF dataset
from evaluate_casf_docking_power import CASFActiveDecoyDataset
from sur_gbsa.ProtMD.data.dataset import atom_dict
from sur_gbsa.distributed_utils import distributed_gather

import warnings
warnings.filterwarnings("ignore", message=".*find_unused_parameters=True was specified.*")


def compute_docking_metrics(scores, rmsds, pdb_ids, rmsd_threshold=2.0):
    """
    Compute docking power metrics for a batch of predictions.
    
    Returns dict with per-complex metrics.
    """
    # Convert to numpy
    scores_np = scores.cpu().numpy()
    rmsds_np = rmsds.cpu().numpy()
    
    # Group by PDB
    df = pd.DataFrame({
        'pdb_id': pdb_ids,
        'score': scores_np,
        'rmsd': rmsds_np
    })
    
    metrics = {'success_count': 0, 'total_complexes': 0}
    spearman_list = []
    
    for pdb_id, group in df.groupby('pdb_id'):
        metrics['total_complexes'] += 1
        
        # Top-1 success
        best_idx = group['score'].idxmin()
        best_rmsd = group.loc[best_idx, 'rmsd']
        if best_rmsd <= rmsd_threshold:
            metrics['success_count'] += 1
        
        # Spearman correlation
        if len(group) >= 3:
            corr, _ = spearmanr(group['rmsd'], group['score'])
            if not np.isnan(corr):
                spearman_list.append(corr)
    
    metrics['success_rate'] = metrics['success_count'] / max(metrics['total_complexes'], 1)
    metrics['mean_spearman'] = np.mean(spearman_list) if spearman_list else 0.0
    
    return metrics


def run_eval(model, loader, args, device, rmsd_threshold=2.0):
    """Run evaluation and compute docking power metrics."""
    model.eval()
    
    all_pred_rmsds = []  # Changed: now storing predicted RMSDs
    all_true_rmsds = []
    all_pdb_ids = []
    losses = []

    ranking_criterion = nn.MarginRankingLoss(margin=args.ranking_margin)
    mse_criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            
            if args.use_residue_features:
                x, res_feats, pos, _, idx = batch
            else:
                x, pos, _, idx = batch
                res_feats = None
            
            x = x.long().to(device)
            pos = pos.float().to(device)
            if res_feats is not None:
                res_feats = res_feats.long().to(device)
            
            mask = x != 0
            
            # Forward pass
            out = model["encoder"](x, pos, res_feats=res_feats, mask=mask)[1]
            out = out.mean(dim=1)
            pred_rmsd = model["finetune"](out)
            
            # Get RMSDs and PDB IDs
            for i, sample_idx in enumerate(idx):
                sample_idx = sample_idx.item()
                all_pred_rmsds.append(pred_rmsd[i].cpu())
                all_true_rmsds.append(loader.dataset.__getrmsd__(sample_idx).item())
                all_pdb_ids.append(loader.dataset.__getpdbid__(sample_idx))


                        # Get ground truth RMSDs
            true_rmsd = torch.tensor([loader.dataset.__getrmsd__(i.item()).item() for i in idx], 
                                    dtype=torch.float32, device=device)

            # MSE loss on RMSD prediction
            mse_loss = mse_criterion(pred_rmsd, true_rmsd)

            # # Pairwise ranking loss
            # good_mask = true_rmsd <= 2.0
            # bad_mask = true_rmsd >= 4.0

            # if good_mask.any() and bad_mask.any():
            #     good_preds = pred_rmsd[good_mask]
            #     bad_preds = pred_rmsd[bad_mask]
                
            #     # Create all pairs
            #     n_good = len(good_preds)
            #     n_bad = len(bad_preds)
            #     pairs_good = good_preds.repeat_interleave(n_bad)
            #     pairs_bad = bad_preds.repeat(n_good)
                
            #     # Target: -1 means first input should be ranked lower
            #     targets = -torch.ones(len(pairs_good), device=device)
                
            #     rank_loss = ranking_criterion(pairs_good, pairs_bad, targets)
            # else:
            #     rank_loss = torch.tensor(0.0, device=device)

            rank_loss = 0
            # Combine losses
            loss = mse_loss + args.ranking_weight * rank_loss

            losses.append((loss, rank_loss, mse_loss))
    
    all_pred_rmsds = torch.stack(all_pred_rmsds)
    all_true_rmsds = torch.tensor(all_true_rmsds)
    
    # Compute metrics: now we want to MINIMIZE predicted RMSD
    # So we use predicted RMSD as "score" (lower = better)
    metrics = compute_docking_metrics(all_pred_rmsds, all_true_rmsds, all_pdb_ids, rmsd_threshold)
    
    return {
        'success_rate': metrics['success_rate'],
        'mean_spearman': metrics['mean_spearman'],
        'pred_rmsds': all_pred_rmsds,
        'true_rmsds': all_true_rmsds,
        'pdb_ids': all_pdb_ids,
        'loss': [x[0] for x in losses],
        'mse_loss': [x[1] for x in losses],
        'rank_loss': [x[2] for x in losses]
    }


def main():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 0 
    print(f"torch version: {torch.__version__}")
    print(f"Using device: {device}")


    rank = None
    current_device = 0 # each process sees a single GPU available

    
    gpus_per_node = 4
    



    timeout = timedelta(minutes=60)
    if "tuo" in host:
        world_size = int(os.environ["FLUX_JOB_SIZE"])
        rank = int(os.environ["FLUX_TASK_RANK"])
        # local_rank = int(rank % gpus_per_node)
        local_rank = int(os.environ.get('FLUX_TASK_LOCAL_ID', 0))
        # torch.cuda.set_device(local_rank)  # MUST call this

        # torch.cuda.set_device(current_device)
        print(f"{socket.gethostname()}: FLUX_JOB_SIZE: {os.environ['FLUX_JOB_SIZE']}\tFLUX_TASK_RANK: {os.environ['FLUX_TASK_RANK']}, world_size: {world_size}, rank: {rank}, local_rank:{local_rank}, current_device: {current_device}")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            # timeout=timedelta(minutes=30)
            device_id=current_device,
            # device_id=,
            # timeout=timeout,
        )






    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", help="Path to pretrained model checkpoint", default="")
    parser.add_argument("--train-split", required=True, help="Path to train PDB list")
    parser.add_argument("--val-split", required=True, help="Path to val PDB list")
    parser.add_argument("--test-split", required=True, help="Path to test PDB list")
    parser.add_argument("--casf-root", default="/p/vast1/jones289/pdbbind/CASF-2016")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--use-residue-features", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=1.5)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    # parser.add_argument("--loss-func", choices=["pose-rmse", "pair-ranking"],
                        # default="pose-rmse",
                        #  help="loss function to use, direct regression on rmse values per pose (pose-rmse) or pairwise ranking loss (pair-ranking)")
    
    args = parser.parse_args()
    
    set_seed(args.seed)

    if rank == 0:

        print(f"torch version: {torch.__version__}")
        print(
            f"world size: {world_size}, global rank: {rank}, local rank: {local_rank}, current_device: {current_device}"
        )


    # Create save directory
    pretrain_name = Path(args.pretrain).parent.name
    pretrain_path = Path(args.pretrain)
    pretrain_exists = pretrain_path.exists() and pretrain_path.name != ""
    if pretrain_exists:
        pretrain_name = Path(args.pretrain).parent.name
    else:
        pretrain_name = "scratch"


    if pretrain_exists:
        print(f"Loading pretrained model from {args.pretrain}")
        checkpoint = torch.load(args.pretrain, map_location=f"cuda:{torch.cuda.current_device()}", weights_only=False)
        ckpt_args = checkpoint['args'] if 'args' in checkpoint else args
    else:
        print("Training from scratch (no pretraining)")
        checkpoint = {}
        ckpt_args = args
        # Set default hyperparameters for scratch model
        ckpt_args.tokens = 100
        ckpt_args.dim = 128
        ckpt_args.depth = 6
        ckpt_args.num_nearest = 32
        ckpt_args.dropout = 0.15








    save_dir = Path(args.save_path) / f"casf-decoy-pose-ranking-pretrain={pretrain_name}-lr={args.lr}-seed={args.seed}-linear_probe={args.linear_probe}"
    if args.use_residue_features:
        save_dir = Path(str(save_dir) + "_res")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    log = Logger(save_dir, "main.log", rank=0)
    writer = SummaryWriter(save_dir)






    # Load pretrained model
    print(f"Loading pretrained model from {args.pretrain}")
    
    # checkpoint = torch.load(args.pretrain, map_location=f"cuda:{torch.cuda.current_device()}", weights_only=False)
    # ckpt_args = checkpoint['args'] if 'args' in checkpoint else args
    
    # Build encoder
    if args.use_residue_features:
        encoder = EGNN_Network(
            num_tokens=getattr(ckpt_args, 'tokens', 100),
            dim=getattr(ckpt_args, 'dim', 128),
            depth=getattr(ckpt_args, 'depth', 6),
            num_nearest_neighbors=getattr(ckpt_args, 'num_nearest', 32),
            dropout=getattr(ckpt_args, 'dropout', 0.15),
            global_linear_attn_every=1,
            norm_coors=True,
            coor_weights_clamp_value=2.0,
            aggregate=False,
            num_residue_tokens=22,
            residue_dim=32,
        ).to(device)
    else:
        encoder = EGNN_Network(
            num_tokens=getattr(ckpt_args, 'tokens', 100),
            dim=getattr(ckpt_args, 'dim', 128),
            depth=getattr(ckpt_args, 'depth', 6),
            num_nearest_neighbors=getattr(ckpt_args, 'num_nearest', 32),
            dropout=getattr(ckpt_args, 'dropout', 0.15),
            global_linear_attn_every=1,
            norm_coors=True,
            coor_weights_clamp_value=2.0,
            aggregate=False,
        ).to(device)
    
    # Build model dict
    model = torch.nn.ModuleDict()
    model["encoder"] = encoder
    model["finetune"] = Regressor(getattr(ckpt_args, 'dim', 128)).to(device)
    
    # Load pretrained weights (encoder only)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        # Remove DDP "module." prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "")
            if "encoder" in new_k:
                new_state_dict[new_k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Loaded pretrained encoder weights")
    
    # Freeze encoder if linear probe
    if args.linear_probe:
        print("Linear probe mode: freezing encoder")
        for param in encoder.parameters():
            param.requires_grad = False

    # Load pretrained model




    # Build encoder
    if args.use_residue_features:
        encoder = EGNN_Network(
            num_tokens=getattr(ckpt_args, 'tokens', 100),
            dim=getattr(ckpt_args, 'dim', 128),
            depth=getattr(ckpt_args, 'depth', 6),
            num_nearest_neighbors=getattr(ckpt_args, 'num_nearest', 32),
            dropout=getattr(ckpt_args, 'dropout', 0.15),
            global_linear_attn_every=1,
            norm_coors=True,
            coor_weights_clamp_value=2.0,
            aggregate=False,
            num_residue_tokens=22,
            residue_dim=32,
        ).to(device)
    else:
        encoder = EGNN_Network(
            num_tokens=getattr(ckpt_args, 'tokens', 100),
            dim=getattr(ckpt_args, 'dim', 128),
            depth=getattr(ckpt_args, 'depth', 6),
            num_nearest_neighbors=getattr(ckpt_args, 'num_nearest', 32),
            dropout=getattr(ckpt_args, 'dropout', 0.15),
            global_linear_attn_every=1,
            norm_coors=True,
            coor_weights_clamp_value=2.0,
            aggregate=False,
        ).to(device)

    # Build model dict
    model = torch.nn.ModuleDict()
    model["encoder"] = encoder
    model["finetune"] = Regressor(getattr(ckpt_args, 'dim', 128)).to(device)

    # Load pretrained weights (encoder only) if available
    if pretrain_exists and 'model' in checkpoint:
        state_dict = checkpoint['model']
        # Remove DDP "module." prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "")
            if "encoder" in new_k:
                new_state_dict[new_k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print("Loaded pretrained encoder weights")
    else:
        print("Initialized with random weights")

    # Freeze encoder if linear probe
    if args.linear_probe:
        if not pretrain_exists:
            print("Warning: linear_probe=True but no pretrained model provided. This doesn't make sense!")
        print("Linear probe mode: freezing encoder")
        for param in encoder.parameters():
            param.requires_grad = False
        


    # Load datasets
    print("Loading CASF datasets...")
    
    with open(args.train_split) as f:
        train_pdbs = [line.strip() for line in f if line.strip()]
    with open(args.val_split) as f:
        val_pdbs = [line.strip() for line in f if line.strip()]
    with open(args.test_split) as f:
        test_pdbs = [line.strip() for line in f if line.strip()]
    
    train_dataset = CASFActiveDecoyDataset(
        train_pdbs, args.casf_root, args.use_residue_features, atom_dict, max_len=600
    )
    val_dataset = CASFActiveDecoyDataset(
        val_pdbs, args.casf_root, args.use_residue_features, atom_dict, max_len=600
    )
    test_dataset = CASFActiveDecoyDataset(
        test_pdbs, args.casf_root, args.use_residue_features, atom_dict, max_len=600
    )
    
    log.logger.info(f"Train: {len(train_dataset)} poses from {len(train_pdbs)} complexes")
    log.logger.info(f"Val: {len(val_dataset)} poses from {len(val_pdbs)} complexes")
    log.logger.info(f"Test: {len(test_dataset)} poses from {len(test_pdbs)} complexes")
    
    # Create dataloaders
    batch_size_per_gpu = args.batch_size // world_size

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=1,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=1,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        num_workers=1,
    )


    # MSE loss for RMSD regression
    criterion = nn.MSELoss()
    ranking_criterion = nn.MarginRankingLoss(margin=args.ranking_margin)
    optimizer = opt.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_mode="min"
    # if args.loss_func == "pose-rmse":
    #     loss_mode = "min"
    # elif args.loss_func == "pair-ranking":
    #     loss_mode = "max"
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, loss_mode, factor=0.6, patience=10, min_lr=1e-6)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    

    # Training loop
    best_metric = 0.0
    best_epoch = 0
    best_model = model

    model_path = save_dir / "best_model.pt"
    
    log.logger.info(f"{'='*60}")
    log.logger.info("Starting RMSD regression fine-tuning")
    log.logger.info(f"{'='*60}")

    dist.barrier()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        epoch_loss = 0.0
        batch_ct = 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in train_iter:
            if batch is None:
                continue
            
            if args.use_residue_features:
                x, res_feats, pos, _, idx = batch
            else:
                x, pos, _, idx = batch
                res_feats = None
            
            optimizer.zero_grad()
            
            x = x.long().to(device)
            pos = pos.float().to(device)
            if res_feats is not None:
                res_feats = res_feats.long().to(device)
            
            mask = x != 0
            
            # Forward pass
            out = model["encoder"](x, pos, res_feats=res_feats, mask=mask)[1]
            out = out.mean(dim=1)
            pred_rmsd = model["finetune"](out).squeeze()



            # In training loop, after getting pred_rmsd:

            # Get ground truth RMSDs
            true_rmsd = torch.tensor([train_dataset.__getrmsd__(i.item()).item() for i in idx], 
                                    dtype=torch.float32, device=device)

            # MSE loss on RMSD prediction
            mse_loss = criterion(pred_rmsd, true_rmsd)

            # # Pairwise ranking loss
            # good_mask = true_rmsd <= 2.0
            # bad_mask = true_rmsd >= 4.0

            # if good_mask.any() and bad_mask.any():
            #     good_preds = pred_rmsd[good_mask]
            #     bad_preds = pred_rmsd[bad_mask]
                
            #     # Create all pairs
            #     n_good = len(good_preds)
            #     n_bad = len(bad_preds)
            #     pairs_good = good_preds.repeat_interleave(n_bad)
            #     pairs_bad = bad_preds.repeat(n_good)
                
            #     # Target: -1 means first input should be ranked lower
            #     targets = -torch.ones(len(pairs_good), device=device)
                
            #     rank_loss = ranking_criterion(pairs_good, pairs_bad, targets)
            # else:
            #     rank_loss = torch.tensor(0.0, device=device)

            rank_loss = 0
            # Combine losses
            loss = mse_loss + args.ranking_weight * rank_loss

            # Then backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            batch_ct += 1



        dist.barrier()


        if rank == 0:
            # Validation
            val_dict = run_eval(model, val_loader, args, device)
            
            success_rate = val_dict['success_rate']
            mean_spearman = val_dict['mean_spearman']
            
            # Use success rate as primary metric
            metric = success_rate
        
        else:
            metric = 0.0


        metric_tensor = torch.tensor([metric], dtype=torch.float32, device=device) 
        dist.broadcast(metric_tensor, src=0)
        metric = metric_tensor.item()
        print(f"rank: {rank}, metric: {metric}")
        lr_scheduler.step(metric)
        # wait for rank 0 to compute the validation metric and step the scheduler
        dist.barrier()
        
        avg_loss = epoch_loss / max(batch_ct, 1)
        
        
    
        if rank == 0:
            log.logger.info(
            f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
            f"Val Success Rate: {success_rate:.3f} | "
            f"Val Spearman: {mean_spearman:.3f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Metrics/val_success_rate", success_rate, epoch)
            writer.add_scalar("Metrics/val_spearman", mean_spearman, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # if metric > best_metric:
        #     print(f"New best model! Success rate: {metric:.3f} (prev: {best_metric:.3f})")
        #     best_metric = metric
        #     best_model = copy.deepcopy(model)
        #     best_epoch = epoch
            
        #     torch.save({
        #         'model': best_model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'best_epoch': best_epoch,
        #         'best_metric': best_metric,
        #         'args': args,
        #     }, model_path)

        if rank == 0:
            # Before torch.save, add:
            if metric > best_metric:
                print(f"New best model! Success rate: {metric:.3f} (prev: {best_metric:.3f})")
                best_metric = metric
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                
                # Create parent directory if it doesn't exist
                model_path.parent.mkdir(parents=True, exist_ok=True)  # ADD THIS LINE
                
                torch.save({
                    'model': best_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'best_epoch': best_epoch,
                    'best_metric': best_metric,
                    'args': args,
                }, model_path)

    # Final test evaluation
    print(f"\n{'='*60}")
    print(f"Training complete. Best epoch: {best_epoch}, Best val success rate: {best_metric:.3f}")
    print(f"Evaluating on test set...")
    print(f"{'='*60}\n")

    dist.barrier()
    if rank == 0: 
        test_dict = run_eval(best_model, test_loader, args, device)
        
        log.logger.info(f"Test Success Rate: {test_dict['success_rate']:.3f}")
        log.logger.info(f"Test Spearman: {test_dict['mean_spearman']:.3f}")
        
        # Save test results
        result_path = save_dir / "test_results.pt"
        torch.save({
            'success_rate': test_dict['success_rate'],
            'mean_spearman': test_dict['mean_spearman'],
            'test_dict': test_dict,
            'args': args
        }, result_path)
        
        writer.flush()

    dist.barrier()
    dist.destroy_process_group()    
    print("Done!")


if __name__ == "__main__":
    main()