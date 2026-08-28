################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################
# load and evaluate a model checkpont on a task

import socket

import argparse
host = socket.gethostname()
import os
import csv
from time import time
from pathlib import Path
import torch
import pandas as pd
from scipy import stats
from sur_gbsa.ProtMD.egnn import EGNN_Network
from torchinfo import summary
from sur_gbsa.data_utils import get_train_val_test_datasets
from sur_gbsa.ProtMD.egnn import EGNN_Network, Regressor
from torch.utils.data import DataLoader, ConcatDataset
from sur_gbsa.data_utils import filter_collate_fn
from torch.nn.functional import mse_loss
from tqdm import tqdm
from sur_gbsa import PBSA_DATASET_LIST, GBSA_DATASET_LIST, FINETUNE_DATASET_LIST
from datetime import timedelta
from bisect import bisect_right
DEV_IDS = list(range(torch.cuda.device_count()))
world_size = len(DEV_IDS)
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr




# Add this function to test.py (near the top with other helper functions):

def load_filtered_ground_truth(data_dir, rmsd_thresh=2.0, min_gbsa=-100, max_gbsa=-10):
    """
    Load and compute filtered ground truth from MD data files.
    
    This replicates the filtering logic used during training:
    - RMSD < rmsd_thresh
    - min_gbsa < GBSA < max_gbsa
    
    Returns DataFrame with columns: pdbid, pose, y_true
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    
    data_dir = Path(data_dir)
    records = []
    
    for path in data_dir.glob("*.npy"):
        # Parse filename
        pdbid = path.name.split("_")[0].split("-")[0]
        pose = int(path.name.split("_")[0].split("-")[2][1])
        
        try:
            data = np.load(path, allow_pickle=True).item()
            
            # Get GBSA scores and RMSD
            gbsa_scores = data["DELTA TOTAL"]
            rmsd_values = data["rmsd"]
            
            # Apply filters (same as training)
            mask = (rmsd_values < rmsd_thresh) & \
                   (gbsa_scores > min_gbsa) & \
                   (gbsa_scores <= max_gbsa)
            
            filtered_gbsa = gbsa_scores[mask]
            
            if len(filtered_gbsa) == 0:
                # No frames passed filters, use unfiltered mean as fallback
                # Or skip this entry entirely
                continue
            
            # Compute mean of filtered frames
            mean_gbsa = filtered_gbsa.mean()
            
            records.append({
                'pdbid': pdbid,
                'pose': pose,
                'y_true': mean_gbsa,
                'n_frames_filtered': len(filtered_gbsa),
                'n_frames_total': len(gbsa_scores)
            })
            
        except Exception as e:
            print(f"Warning: Error loading {path}: {e}")
            continue
    
    df = pd.DataFrame(records)
    print(f"Loaded filtered ground truth for {len(df)} (pdbid, pose) pairs")
    print(f"Mean frames used: {df['n_frames_filtered'].mean():.1f} / {df['n_frames_total'].mean():.1f}")
    
    return df


def load_frame_averaged_predictions(data_dir, test_df_raw):
    """
    Load raw MD data and compute frame-averaged predictions.
    
    For each (pdbid, pose) in test_df_raw, average the model predictions
    across all frames in the original MD trajectory.
    
    Parameters:
    -----------
    data_dir : str
        Path to MD data directory
    test_df_raw : pd.DataFrame
        Raw frame-level predictions with columns: pdbid, pose, y_pred, metadata
        
    Returns:
    --------
    pd.DataFrame with averaged predictions per (pdbid, pose)
    """
    from pathlib import Path
    import numpy as np
    
    # Group by (pdbid, pose) and average predictions
    averaged = test_df_raw.groupby(['pdbid', 'pose']).agg({
        'y_pred': 'mean',
        'y_true': 'first',  # Keep first (will be replaced with filtered GT)
        'metadata': 'first'
    }).reset_index()
    
    print(f"Averaged predictions from {len(test_df_raw)} frames to {len(averaged)} (pdbid, pose) combinations")
    
    return averaged

def get_completed_checkpoints(output_path):
    """
    Read the CSV and return set of checkpoint paths that have been completed.
    """
    if not os.path.exists(output_path):
        return set()
    
    completed = set()
    try:
        df = pd.read_csv(output_path)
        # Only consider successful or skipped runs as completed
        valid_statuses = df['status'].isin(['success', 'saved', 'skipped - file exists'])
        completed = set(df.loc[valid_statuses, 'path'].values)
    except Exception as e:
        print(f"Warning: Could not read existing results from {output_path}: {e}")
    
    return completed


def prediction_file_exists(ckpt_path, eval_args_dict):
    """
    Check if prediction file already exists for this checkpoint.
    
    Args:
        ckpt_path: Path to the checkpoint file
        eval_args_dict: Dictionary containing evaluation arguments
    
    Returns:
        tuple: (exists: bool, output_path: str)
    """
    if not eval_args_dict.get("save_predictions", False):
        return False, None
    
    # Extract checkpoint directory and filename components
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_filename = os.path.basename(ckpt_path)
    
    # Extract epoch number from checkpoint name
    try:
        epoch_num = ckpt_filename.split('epoch-')[-1].split('.')[0]
    except:
        epoch_num = "unknown"
    
    # Construct output filename
    predictions_suffix = eval_args_dict.get("predictions_suffix", "test_result")
    dataset_name = eval_args_dict["dataset"]
    output_filename = f"{predictions_suffix}-best_model-epoch-{epoch_num}-rank=0-{dataset_name}.pt"
    output_path = os.path.join(ckpt_dir, output_filename)
    
    return os.path.exists(output_path), output_path

def get_metadata(dataset, idx):
    """
    Extract metadata string from dataset at given index.
    Handles both ConcatDataset and regular datasets.
    """
    if isinstance(dataset, ConcatDataset):
        # Find which sub-dataset this index belongs to
        dataset_idx = bisect_right(dataset.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - dataset.cumulative_sizes[dataset_idx - 1]
        sub_dataset = dataset.datasets[dataset_idx]
        return sub_dataset.__getmetadata__(sample_idx)
    else:
        return dataset.__getmetadata__(idx)

def compare_rankings(df, identifier_col, ground_truth_col, predicted_col, pose_col=None, return_detailed=True):
    """
    Compare rankings between ground truth and predicted values within groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    identifier_col : str
        Name of the column containing string identifiers for grouping
    ground_truth_col : str
        Name of the column containing ground truth values
    predicted_col : str
        Name of the column containing predicted values
    pose_col : str, optional
        Name of the column containing pose identifiers (0 = crystal pose)
    return_detailed : bool, default True
        If True, returns the dataframe with added ranking columns.
        If False, returns only the summary of matches by group.
    
    Returns:
    --------
    If return_detailed=True:
        tuple: (enhanced_df, summary_df)
            - enhanced_df: Original dataframe with added ranking columns
            - summary_df: Summary of ranking matches by group
    If return_detailed=False:
        pandas.Series: Boolean series indicating if top ranks match for each group
    """
    
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Add ranking columns for ground truth and predicted values within each group
    # Lower values get better (lower) ranks (rank 1 = most negative/lowest value)
    gt_rank_col = f'{ground_truth_col}_rank'
    pred_rank_col = f'{predicted_col}_rank'
    
    result_df[gt_rank_col] = result_df.groupby(identifier_col)[ground_truth_col].rank(
        method='min', ascending=True
    )
    result_df[pred_rank_col] = result_df.groupby(identifier_col)[predicted_col].rank(
        method='min', ascending=True
    )
    
    # Add indicator columns for top-ranked items
    gt_top_col = f'is_{ground_truth_col}_top'
    pred_top_col = f'is_{predicted_col}_top'
    both_top_col = 'both_top'
    
    result_df[gt_top_col] = result_df[gt_rank_col] == 1
    result_df[pred_top_col] = result_df[pred_rank_col] == 1
    result_df[both_top_col] = result_df[gt_top_col] & result_df[pred_top_col]
    
    # Create summary dictionary
    summary_dict = {
        both_top_col: 'any',
        gt_top_col: 'sum',
        pred_top_col: 'sum'
    }
    
    # Add crystal pose top-1 if pose column is provided
    if pose_col is not None:
        pred_crystal_top_col = 'pred_top_is_crystal'
        result_df[pred_crystal_top_col] = result_df[pred_top_col] & (result_df[pose_col] == 0)
        summary_dict[pred_crystal_top_col] = 'any'
    
    # Create summary by group
    summary = result_df.groupby(identifier_col).agg(summary_dict)
    
    # Rename columns
    rename_dict = {
        both_top_col: 'has_matching_top_rank',
        gt_top_col: f'num_{ground_truth_col}_top_items',
        pred_top_col: f'num_{predicted_col}_top_items'
    }
    
    if pose_col is not None:
        rename_dict['pred_top_is_crystal'] = 'pred_top_is_crystal'
    
    summary = summary.rename(columns=rename_dict)
    
    if return_detailed:
        return result_df, summary
    else:
        return summary['has_matching_top_rank']

def run_gbsa_eval(model, loader, device="cpu", return_flow_match_loss=False, 
                  data_dir=None, use_filtered_ground_truth=False, 
                  average_over_frames=False):
    """
    Run evaluation on GBSA dataset.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to evaluate
    loader : DataLoader
        Data loader
    device : str
        Device to run on
    return_flow_match_loss : bool
        Whether to return flow matching loss
    data_dir : str, optional
        Path to raw MD data directory (for filtered ground truth)
    use_filtered_ground_truth : bool
        Whether to replace y_true with filtered ground truth
    average_over_frames : bool
        Whether to average predictions over frames per (pdbid, pose)
    """
    model.eval()

    y_pred = []
    y_true = []
    idx_list = []
    latent_list = []
    metadata_list = []
    flow_loss_list = []
    batch_start = 0

    if isinstance(loader.dataset, ConcatDataset):
        use_residue_features = any(
            hasattr(ds, "use_residue_features") and ds.use_residue_features
            for ds in loader.dataset.datasets
        )
    else:
        use_residue_features = loader.dataset.use_residue_features

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            else:
                if use_residue_features:
                    x, res_feats, pos, y, idx = batch
                else:
                    x, pos, y, idx = batch
                    res_feats = None

                batch_size = len(y)
                batch_indices = list(range(batch_start, batch_start + batch_size))
                batch_metadata = [get_metadata(loader.dataset, i) for i in batch_indices]
                metadata_list.extend(batch_metadata)
                idx_list.append(torch.tensor(batch_indices))
                
                x, pos, y = x.long().to(device), pos.float().to(device), y.to(device)
                mask = x != 0
                
                out = model["encoder"](x, pos, res_feats=res_feats, mask=mask)[1]
                out = out.mean(dim=1)
                latent_list.append(out.cpu())
                
                pred = model["finetune"](out)
                y_pred.append(pred.cpu())

                # print(y_true)
                y_true.append(y.cpu())
                
                batch_start += batch_size

    y_pred = torch.cat(y_pred).reshape(-1, 1).cpu() * -10
    y_true = torch.cat(y_true).reshape(-1, 1).cpu() * -10
    idx = torch.cat(idx_list).reshape(-1, 1).cpu()
    latent = torch.cat(latent_list).reshape(-1, out.shape[-1])

    # import pdb; pdb.set_trace()

    # Create initial dataframe with frame-level data
    test_df = pd.DataFrame({
        "y_pred": y_pred.squeeze(),
        "y_true": y_true.squeeze(),
        "idx": idx.squeeze(),
        "metadata": metadata_list,
    })

    # Extract pdbid and pose
    test_df['pdbid'] = test_df['metadata'].apply(lambda x: x.split("-")[0])
    
    def extract_pose(metadata_str):
        try:
            pose_part = metadata_str.split("-")[2]
            pose_num = int(pose_part.split("_")[0][1:])
            return pose_num
        except:
            return -1
    
    test_df['pose'] = test_df['metadata'].apply(extract_pose)
    
    # Average predictions over frames if requested
    if average_over_frames:
        # import pdb; pdb.set_trace()
        print(f"\nAveraging predictions over frames per (pdbid, pose)")
        print(f"Before averaging: {len(test_df)} frame-level predictions")
        
        test_df = test_df.groupby(['pdbid', 'pose']).agg({
                        'y_pred': 'mean', 
                        # 'y_true': 'first', 
                        'y_true': 'mean', 
                        'idx': 'first', 
                        'metadata': 'first'}
                ).reset_index()
        
        print(f"After averaging: {len(test_df)} (pdbid, pose) combinations")
        
        # Update tensors
        y_pred = torch.tensor(test_df['y_pred'].values).reshape(-1, 1)
        y_true = torch.tensor(test_df['y_true'].values).reshape(-1, 1)
    
    # Replace with filtered ground truth if requested
    # if use_filtered_ground_truth and data_dir is not None:
    if (use_filtered_ground_truth and data_dir is not None) and (average_over_frames):
        print(f"\nLoading filtered ground truth from: {data_dir}")
        gt_df = load_filtered_ground_truth(
            data_dir, 
            rmsd_thresh=2.0, 
            min_gbsa=-100, 
            max_gbsa=-10
        )
        
        # Merge
        test_df = test_df.merge(
            gt_df[['pdbid', 'pose', 'y_true', 'n_frames_filtered']], 
            on=['pdbid', 'pose'], 
            how='left',
            suffixes=('_raw', '_filtered')
        )
        
        # Check for missing
        n_missing = test_df['y_true_filtered'].isna().sum()
        if n_missing > 0:
            print(f"WARNING: {n_missing} samples missing filtered ground truth, using raw values")
            test_df['y_true'] = test_df['y_true_filtered'].fillna(test_df['y_true_raw'])
        else:
            test_df['y_true'] = test_df['y_true_filtered']
        
        test_df = test_df.drop(columns=['y_true_raw', 'y_true_filtered'], errors='ignore')
        
        # Update tensor
        y_true = torch.tensor(test_df['y_true'].values).reshape(-1, 1)
        
        print(f"Using filtered ground truth for {len(test_df)} samples")
    
    # Compute global metrics
    pearson = stats.pearsonr(y_pred.numpy().flatten(), y_true.numpy().flatten())[0]
    spearman = stats.spearmanr(y_pred.numpy().flatten(), y_true.numpy().flatten())[0]
    rmse = torch.sqrt(mse_loss(y_true, y_pred, reduction="sum") / len(test_df)).item()
    
    # Compute per-PDB metrics
    pearson_list = []
    spearman_list = []
    for pdbid, pdb_group in test_df.groupby("pdbid"):
        if pdb_group.shape[0] < 2:
            continue
        s_val = spearmanr(pdb_group["y_pred"], pdb_group["y_true"])[0]
        if not np.isnan(s_val):
            spearman_list.append(s_val)
        p_val = pearsonr(pdb_group["y_pred"], pdb_group["y_true"])[0]
        if not np.isnan(p_val):
            pearson_list.append(p_val)

    pdb_spearman = np.mean(spearman_list) if spearman_list else np.nan
    pdb_pearson = np.mean(pearson_list) if pearson_list else np.nan

    # Build result dictionary
    eval_dict = {
        "y_pred": y_pred.squeeze(),
        "y_true": y_true.squeeze(),
        "idx": idx.squeeze() if not average_over_frames else torch.tensor(test_df['idx'].values),
        "metadata": test_df['metadata'].tolist(),
        "latent": latent,
        "df": test_df,
        "dataset_size": len(test_df),
        "n_predictions": len(y_pred),
        "pearson": pearson,
        "spearman": spearman,
        "rmse": rmse,
        "pdb_pearson": pdb_pearson,
        "pdb_spearman": pdb_spearman,
    }

    # Compute ranking metrics
    # has_crystal_pose = (test_df['pose'] == 0).any()
    
    # Compute ranking metrics
    poses_per_pdb = test_df.groupby('pdbid')['pose'].count()
    has_multiple_poses = (poses_per_pdb > 1).any()

    if has_multiple_poses:
        has_crystal_pose = (test_df['pose'] == 0).any()
        
        _, rankings_summary = compare_rankings(
            df=test_df, 
            identifier_col="pdbid", 
            pose_col="pose" if has_crystal_pose else None,
            predicted_col="y_pred", 
            ground_truth_col="y_true",
            return_detailed=True
        )
        
        top1_accuracy = rankings_summary['has_matching_top_rank'].mean()
        
        eval_dict["top_pose_rate"] = top1_accuracy
        eval_dict["top1_accuracy"] = top1_accuracy
        
        # Only compute crystal_top1_rate if crystal exists
        if has_crystal_pose and 'pred_top_is_crystal' in rankings_summary.columns:
            crystal_top1_rate = rankings_summary['pred_top_is_crystal'].mean()
            eval_dict["crystal_top1_rate"] = crystal_top1_rate
        else:
            eval_dict["crystal_top1_rate"] = None
    else:
        eval_dict["top_pose_rate"] = None
        eval_dict["top1_accuracy"] = None
        eval_dict["crystal_top1_rate"] = None
    


    if return_flow_match_loss:
        flow_match_loss = torch.cat(flow_loss_list).sum()
        eval_dict["flow_match_loss"] = flow_match_loss

    return eval_dict

def run_tensor_dataset_eval(model, loader, args, device="cpu"):
    model.eval()
    metric = 0
    y_pred = []
    y_true = []
    pose_list = []
    path_list = []
    frame_list = []
    metadata_list = []
    idx_list = []
    latent_list = []

    with torch.no_grad():
        for batch in tqdm(loader,desc="eval loop."):
            if batch is None:
                continue
            else:

                x, pos, y, idx = batch
                y_true.append(y)
                x, pos, y = x.long().to(device), pos.float().to(device), y.to(device)

                mask = x != 0
                
                out = model["encoder"](x, pos, mask=mask)[1]
                out = out.mean(dim=1)

                pred = model["finetune"](out)
                y_pred.append(pred)
                
                frame_list.append(torch.zeros_like(y))
                pose_list.append(torch.zeros_like(y))
                latent_list.append(out.cpu())

    y_pred = torch.cat(y_pred).reshape(-1, 1).cpu()
    y_true = torch.cat(y_true).reshape(-1, 1).cpu()
    idx = torch.cat(idx_list).reshape(-1, 1).cpu()
    latent = torch.cat(latent_list).reshape(-1, out.shape[-1])
    pearson = stats.pearsonr(y_pred.numpy().flatten(), y_true.numpy().flatten())[0]
    spearman = stats.spearmanr(y_pred.numpy().flatten(), y_true.numpy().flatten())[0]
    rmse = torch.sqrt(
        mse_loss(y_true, y_pred, reduction="sum") / len(loader.dataset)
    ).item()

    test_dict  ={
        "y_pred": y_pred.squeeze(),
        "y_true": y_true.squeeze(),
        "idx": idx.squeeze(),
        "metadata": metadata_list,
        "frame_num": frame_list,
    }

    test_df = pd.DataFrame(test_dict)

    test_df['pdbid'] = test_df['metadata'].apply(lambda x: x.split("-")[0])
    test_df['pose'] = test_df['metadata'].apply(lambda x: x.split("-")[2][-1])

    import numpy as np
    from scipy.stats import spearmanr, pearsonr
    pearson_list = []
    spearman_list = []
    for pdb, pdb_group in test_df.groupby("pdbid"):
        s_val, p_val = np.nan, np.nan
        if pdb_group.shape[0] < 2:
            pass
        else:
            s_val = spearmanr(pdb_group["y_pred"], pdb_group["y_true"])[0]
            spearman_list.append(s_val)
            p_val = pearsonr(pdb_group["y_pred"], pdb_group["y_true"])[0]
            pearson_list.append(p_val)

    spearman_list = [x for x in spearman_list if not np.isnan(x)]
    pearson_list = [x for x in pearson_list if not np.isnan(x)]

    pdb_spearman = np.mean(spearman_list)
    pdb_pearson = np.mean(pearson_list)

    rankings = compare_rankings(df=test_df, identifier_col="pdbid", predicted_col="y_pred", ground_truth_col="y_true")
    matches = rankings[-1]['has_matching_top_rank'].sum()
    top_pose_rate = matches / rankings[-1].shape[0]

    return {"data": test_dict,"df": test_df, "latent": latent, "pearson": pearson, "y_pred": y_pred, "y_true": y_true, 
            "metadata": metadata_list, "idx": idx, "frame_num": frame_list,
            "spearman":spearman, "rmse":rmse ,"pdb_pearson": pdb_pearson, "pdb_spearman": pdb_spearman, "top_pose_rate": top_pose_rate}


def run_eval(model, loader, args, device="cpu"):

    if args.dataset in GBSA_DATASET_LIST or args.dataset in PBSA_DATASET_LIST:
        return run_gbsa_eval(
            model=model, loader=loader, device=device
        )

    elif args.dataset in FINETUNE_DATASET_LIST:
        return run_tensor_dataset_eval(
            model=model, loader=loader, args=args, device=device
        )

    else:
        raise Exception


def filter_to_best_checkpoints_per_run(ckpt_path_list):
    """
    Given a list of checkpoint paths, keep only the checkpoint with the highest epoch
    (most recent/best validation) for each unique hyperparameter configuration.
    
    Assumes checkpoint paths follow pattern:
    .../hparam_dir/checkpoint-epoch-{N}.pt
    
    Returns filtered list keeping only the latest checkpoint per hparam directory.
    """
    from collections import defaultdict
    
    # Group checkpoints by their parent directory (hyperparameter config)
    hparam_to_ckpts = defaultdict(list)
    
    for ckpt_path in ckpt_path_list:
        # Get parent directory (hyperparameter configuration)
        hparam_dir = '/'.join(ckpt_path.split('/')[:-1])
        
        # Extract epoch number from checkpoint filename
        try:
            filename = ckpt_path.split('/')[-1]
            epoch = int(filename.split('epoch-')[-1].split('.')[0])
            hparam_to_ckpts[hparam_dir].append((epoch, ckpt_path))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse epoch from {ckpt_path}, skipping")
            continue
    
    # For each hyperparameter config, keep only the checkpoint with max epoch
    filtered_ckpts = []
    for hparam_dir, ckpt_list in hparam_to_ckpts.items():
        # Sort by epoch and take the last one (highest epoch = best validation)
        ckpt_list.sort(key=lambda x: x[0])
        best_epoch, best_ckpt = ckpt_list[-1]
        filtered_ckpts.append(best_ckpt)
        
        if len(ckpt_list) > 1:
            print(f"Filtered {len(ckpt_list)} checkpoints to epoch {best_epoch}: {hparam_dir}")
    
    return filtered_ckpts


def evaluate_single_checkpoint(args_tuple):
    """
    Evaluate a single checkpoint. This function will be called by multiprocessing.
    
    Args:
        args_tuple: Tuple containing (ckpt_path, eval_args_dict, current_device)
    
    Returns:
        dict: Results including checkpoint path and metrics
    """
    ckpt_path, eval_args_dict, device_id = args_tuple

    # Set device for this process
    torch.cuda.set_device(device_id)
    current_device = device_id
    
    try:
        # Load checkpoint
        checkpoint = torch.load(
            ckpt_path, 
            map_location=f"cuda:{device_id}", 
            weights_only=False,
        )
        
        train_args = checkpoint['args']
        
        # Create dataset
        _, _, dataset = get_train_val_test_datasets(argparse.Namespace(**{
            "dataset": eval_args_dict["dataset"],
            "data_dir": eval_args_dict["data_dir"],
            "test_split": eval_args_dict["split"]
        }))
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=eval_args_dict["batch_size"],
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            pin_memory=True,
            drop_last=False,
        )

        # Build model
        if "use_residue_features" in vars(train_args).keys():
                encoder = EGNN_Network(
                    num_tokens=train_args.tokens,
                    dim=train_args.dim,
                    depth=train_args.depth,
                    num_nearest_neighbors=train_args.num_nearest,
                    dropout=train_args.dropout,
                    global_linear_attn_every=1,
                    norm_coors=True,
                    coor_weights_clamp_value=2.0,
                    aggregate=False,
                    num_residue_tokens=22 if train_args.use_residue_features else None,
                    residue_dim=32 if train_args.use_residue_features else None,
                ).to(current_device)
        else:
            encoder = EGNN_Network(
                num_tokens=train_args.tokens,
                dim=train_args.dim,
                depth=train_args.depth,
                num_nearest_neighbors=train_args.num_nearest,
                dropout=train_args.dropout,
                global_linear_attn_every=1,
                norm_coors=True,
                coor_weights_clamp_value=2.0,
                aggregate=False,
            ).to(current_device)
        
        model = torch.nn.ModuleDict({"encoder": encoder})
        model["finetune"] = Regressor(train_args.dim).to(current_device)
        model = model.float()
        
        # Load state dict
        state_dict = checkpoint["model"]
        
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            if "encoder" in new_key and not new_key.startswith("encoder."):
                new_key = "encoder." + new_key.split("encoder.")[-1]
            if "finetune" in new_key and not new_key.startswith("finetune."):
                new_key = "finetune." + new_key.split("finetune.")[-1]
            new_state_dict[new_key] = value
        
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            print(f"Warning: Strict loading failed for {ckpt_path}, trying non-strict: {e}")
            model.load_state_dict(new_state_dict, strict=False)
        
        model.eval()
        
        # Run evaluation
        # test_dict = run_gbsa_eval(model=model, loader=loader, device=current_device)
        # import pdb; pdb.set_trace()
        test_dict = run_gbsa_eval(
            model=model, 
            loader=loader, 
            device=current_device,
            data_dir=eval_args_dict["data_dir"],
            use_filtered_ground_truth=eval_args_dict.get("use_filtered_ground_truth", False),
            average_over_frames=eval_args_dict.get("average_over_frames", False)
        ) 
        # Save predictions if requested
        output_path = ""
        if eval_args_dict.get("save_predictions", False):
            # Extract checkpoint directory and filename components
            ckpt_dir = os.path.dirname(ckpt_path)
            ckpt_filename = os.path.basename(ckpt_path)
            
            # Extract epoch number from checkpoint name
            try:
                epoch_num = ckpt_filename.split('epoch-')[-1].split('.')[0]
            except:
                epoch_num = "unknown"
            
            # Construct output filename
            predictions_suffix = eval_args_dict.get("predictions_suffix", "test_result")
            dataset_name = eval_args_dict["dataset"]
            output_filename = f"{predictions_suffix}-best_model-epoch-{epoch_num}-rank=0-{dataset_name}.pt"
            output_path = os.path.join(ckpt_dir, output_filename)
            
            # Save the full test_dict
            torch.save(test_dict, output_path)
        
        # # Always return metrics
        # result = {
        #     "path": ckpt_path,
        #     "output_path": output_path,
        #     "dataset": eval_args_dict["dataset"],
        #     "split": eval_args_dict["split"],
        #     "pdb_pearson": test_dict['pdb_pearson'],
        #     "pdb_spearman": test_dict['pdb_spearman'],
        #     "top_pose_rate": test_dict['top_pose_rate'],
        #     "pearson": test_dict['pearson'],
        #     "spearman": test_dict['spearman'],
        #     "status": "saved" if eval_args_dict.get("save_predictions", False) else "success"
        # }
        # And in evaluate_single_checkpoint():
        result = {
            "path": ckpt_path,
            "output_path": output_path,
            "dataset": eval_args_dict["dataset"],
            "split": eval_args_dict["split"],
            "pdb_pearson": test_dict['pdb_pearson'],
            "pdb_spearman": test_dict['pdb_spearman'],
            "top_pose_rate": test_dict['top_pose_rate'],
            "top1_accuracy": test_dict.get('top1_accuracy'),      # NEW
            "crystal_top1_rate": test_dict.get('crystal_top1_rate'),  # NEW
            "pearson": test_dict['pearson'],
            "spearman": test_dict['spearman'],
            "status": "saved" if eval_args_dict.get("save_predictions", False) else "success"
        }
        
        # Clean up GPU memory
        del model, encoder, checkpoint, dataset, loader
        torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        import traceback
        return {
            "path": ckpt_path,
            "output_path": "",
            "dataset": eval_args_dict["dataset"],
            "split": eval_args_dict["split"],
            "pdb_pearson": None,
            "pdb_spearman": None,
            "top_pose_rate": None,
            "pearson": None,
            "spearman": None,
            "status": f"error: {str(e)}",
            "traceback": traceback.format_exc()
        }


def evaluate_serial_dataparallel(eval_args):
    """Serial evaluation using DataParallel for large datasets"""
    
    # Use all available GPUs with DataParallel
    device_ids = list(range(torch.cuda.device_count()))
    primary_device = device_ids[0]
    
    print(f"Running SERIAL evaluation with DataParallel across GPUs: {device_ids}")
    
    # Filter out already-completed checkpoints
    completed = get_completed_checkpoints(eval_args.output_path)
    if completed:
        print(f"\nFound {len(completed)} already-completed checkpoints in {eval_args.output_path}")
        original_count = len(eval_args.ckpt_path_list)
        eval_args.ckpt_path_list = [ckpt for ckpt in eval_args.ckpt_path_list if ckpt not in completed]
        print(f"Filtered out {original_count - len(eval_args.ckpt_path_list)} already-completed checkpoints")
        print(f"Remaining checkpoints to evaluate: {len(eval_args.ckpt_path_list)}\n")
        
        if len(eval_args.ckpt_path_list) == 0:
            print("All checkpoints already evaluated!")
            return
    
    # Determine if we're saving predictions or computing metrics
    save_predictions = eval_args.save_predictions
    
    # Determine file mode and whether to write header
    file_mode = "a" if os.path.exists(eval_args.output_path) else "w"
    write_header = not os.path.exists(eval_args.output_path)
    
    # Open output file
    with open(eval_args.output_path, mode=file_mode, newline='') as handle:
        writer = csv.writer(handle)
        
        # # Write header if creating new file - ALWAYS include metrics columns
        # if write_header:
        #     header = [
        #         "path", "output_path", "dataset", "split", 
        #         "pdb_pearson", "pdb_spearman", "top_pose_rate", 
        #         "pearson", "spearman", "status"
        #     ]
        #     writer.writerow(header)

        # In evaluate_serial_dataparallel():
        if write_header:
            header = [
                "path", "output_path", "dataset", "split", 
                "pdb_pearson", "pdb_spearman", "top_pose_rate", 
                "top1_accuracy", "crystal_top1_rate",  # NEW
                "pearson", "spearman", "status"
            ]
            writer.writerow(header)

        # Evaluate each checkpoint serially
        for ckpt_path in tqdm(eval_args.ckpt_path_list, desc="Evaluating checkpoints"):
            try:
                # Load checkpoint
                checkpoint = torch.load(
                    ckpt_path,
                    map_location=f"cuda:{primary_device}",
                    weights_only=False,
                )
                
                train_args = checkpoint['args']
                
                # Create dataset
                _, _, dataset = get_train_val_test_datasets(argparse.Namespace(**{
                    "dataset": eval_args.dataset,
                    "data_dir": eval_args.data_dir,
                    "test_split": eval_args.split
                }))
                
                # Create dataloader with multiple workers (safe in serial mode)
                loader = DataLoader(
                    dataset,
                    batch_size=eval_args.batch_size,
                    shuffle=False,
                    num_workers=eval_args.num_workers,
                    persistent_workers=eval_args.num_workers > 0,
                    pin_memory=True,
                    drop_last=False,
                )
                
                # Build model WITHOUT DataParallel first
                if "use_residue_features" in vars(train_args).keys():
                    encoder = EGNN_Network(
                        num_tokens=train_args.tokens,
                        dim=train_args.dim,
                        depth=train_args.depth,
                        num_nearest_neighbors=train_args.num_nearest,
                        dropout=train_args.dropout,
                        global_linear_attn_every=1,
                        norm_coors=True,
                        coor_weights_clamp_value=2.0,
                        aggregate=False,
                        num_residue_tokens=22 if train_args.use_residue_features else None,
                        residue_dim=32 if train_args.use_residue_features else None,
                    ).to(primary_device)
                else:
                    encoder = EGNN_Network(
                        num_tokens=train_args.tokens,
                        dim=train_args.dim,
                        depth=train_args.depth,
                        num_nearest_neighbors=train_args.num_nearest,
                        dropout=train_args.dropout,
                        global_linear_attn_every=1,
                        norm_coors=True,
                        coor_weights_clamp_value=2.0,
                        aggregate=False,
                    ).to(primary_device)
                
                finetune_head = Regressor(train_args.dim).to(primary_device)
                
                # Load state dict into unwrapped models
                state_dict = checkpoint["model"]
                
                # Clean up state dict keys
                encoder_state = {}
                finetune_state = {}
                
                for key, value in state_dict.items():
                    # Remove 'module.' prefix if present
                    clean_key = key.replace("module.", "")
                    
                    if "encoder" in clean_key:
                        # Remove 'encoder.' prefix for loading into encoder
                        param_key = clean_key.replace("encoder.", "")
                        encoder_state[param_key] = value
                    elif "finetune" in clean_key:
                        # Remove 'finetune.' prefix for loading into finetune head
                        param_key = clean_key.replace("finetune.", "")
                        finetune_state[param_key] = value
                
                # Load weights
                encoder.load_state_dict(encoder_state, strict=True)
                finetune_head.load_state_dict(finetune_state, strict=True)
                
                # NOW wrap in DataParallel
                encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)
                finetune_head = torch.nn.DataParallel(finetune_head, device_ids=device_ids)
                
                # Create ModuleDict
                model = torch.nn.ModuleDict({
                    "encoder": encoder,
                    "finetune": finetune_head
                }).float()
                
                model.eval()
                
                # Run evaluation
                # test_dict = run_gbsa_eval(model=model, loader=loader, device=primary_device)
                # In evaluate_serial_dataparallel():
                test_dict = run_gbsa_eval(
                    model=model, 
                    loader=loader, 
                    device=primary_device,
                    data_dir=eval_args.data_dir,
                    use_filtered_ground_truth=eval_args.use_filtered_ground_truth,
                    average_over_frames=eval_args.average_over_frames
                )


                # Save predictions if requested
                output_path = ""
                if save_predictions:
                    # Extract checkpoint directory and filename components
                    ckpt_dir = os.path.dirname(ckpt_path)
                    ckpt_filename = os.path.basename(ckpt_path)
                    
                    # Extract epoch number from checkpoint name
                    try:
                        epoch_num = ckpt_filename.split('epoch-')[-1].split('.')[0]
                    except:
                        epoch_num = "unknown"
                    
                    # Construct output filename
                    predictions_suffix = eval_args.predictions_suffix
                    dataset_name = eval_args.dataset
                    output_filename = f"{predictions_suffix}-best_model-epoch-{epoch_num}-rank=0-{dataset_name}.pt"
                    output_path = os.path.join(ckpt_dir, output_filename)
                    
                    # Save the full test_dict
                    torch.save(test_dict, output_path)
                
                # And when writing rows:
                row_data = [
                    ckpt_path,
                    output_path,
                    eval_args.dataset,
                    eval_args.split,
                    test_dict['pdb_pearson'],
                    test_dict['pdb_spearman'],
                    test_dict['top_pose_rate'],
                    test_dict.get('top1_accuracy'),      # NEW
                    test_dict.get('crystal_top1_rate'),  # NEW
                    test_dict['pearson'],
                    test_dict['spearman'],
                    "saved" if save_predictions else "success"
                ]

                # Write result immediately
                writer.writerow(row_data)
                handle.flush()
                
                # Clean up GPU memory
                del model, encoder, finetune_head, checkpoint, dataset, loader
                torch.cuda.empty_cache()
                
            except Exception as e:
                import traceback
                
                row_data = [
                    ckpt_path,
                    "",
                    eval_args.dataset,
                    eval_args.split,
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"error: {str(e)}",
                ]
                
                writer.writerow(row_data)
                handle.flush()
                
                print(f"\nError evaluating {ckpt_path}: {str(e)}")
                print(traceback.format_exc())
    
    print(f"\nEvaluation complete! Results saved to {eval_args.output_path}")

def main():
    import argparse
    import csv
    from multiprocessing import Pool, set_start_method
    
    # Set multiprocessing start method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass  # Already set
    
    dataset_list = GBSA_DATASET_LIST + FINETUNE_DATASET_LIST
    
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints (parallel or serial with DataParallel)"
    )
    parser.add_argument(
        "--dataset", 
        choices=dataset_list, 
        default="gbsa", 
        required=True,
    )
    parser.add_argument("--split", required=True, help="path to file with list of PDB ids to use for evaluation split")
    parser.add_argument(
        "--data-dir",
        type=str,
        help="path to directory containing the MD data with MM/GBSA scores",
        default="/p/vast1/jones289/PDBbind_core_MD/prot_md_input-fixed-03-05-25"
    )
    parser.add_argument(
        '--ckpt-path-list', 
        nargs='+', 
        help="list of checkpoint paths to evaluate"
    )
    parser.add_argument(
        '--ckpt-list-path',
        help="path to a newline separated file with the checkpoints"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="batch size to use for inference"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for dataloading"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--output-path",
        default="debug_test.csv",
        required=True
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        # default=4,
        default=torch.cuda.device_count(),
        help="number of checkpoints to evaluate in parallel (ignored in serial mode)"
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        default=False,
        help="Use serial evaluation with DataParallel instead of parallel multiprocessing"
    )
    parser.add_argument(
        "--filter-to-best-per-run",
        action="store_true",
        default=False,
        help="Only evaluate the best (highest epoch) checkpoint per hyperparameter configuration"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=False,
        help="Save full predictions to checkpoint directory instead of CSV summary"
    )
    parser.add_argument(
        "--predictions-suffix",
        type=str,
        default="test_result",
        help="Prefix for saved prediction files (e.g., 'test_result' -> 'test_result-epoch-100.pt')"
    )
    parser.add_argument(
        "--use-filtered-ground-truth",
        action="store_true",
        default=False,
        help="Use filtered ground truth (RMSD < 2Å, -100 < GBSA < -10)"
    )
    parser.add_argument(
        "--average-over-frames",
        action="store_true",
        default=False,
        help="Average model predictions over MD frames per (pdbid, pose)"
    )
    
    eval_args = parser.parse_args()
    print(eval_args)
    
    # Get available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs")
    
    # Prepare checkpoint list
    if eval_args.ckpt_path_list and eval_args.ckpt_list_path:
        print("can only specify one of ckpt_path_list or ckpt_list_path")
        return
    elif eval_args.ckpt_path_list:
        ckpt_list = eval_args.ckpt_path_list
    elif eval_args.ckpt_list_path:
        ckpt_list = pd.read_csv(eval_args.ckpt_list_path, header=None)[0].values.tolist()
    else:
        print("Must specify either --ckpt-path-list or --ckpt-list-path")
        return
    
    # Filter to best checkpoints if requested
    if eval_args.filter_to_best_per_run:
        print(f"\nFiltering {len(ckpt_list)} checkpoints to best per run...")
        ckpt_list = filter_to_best_checkpoints_per_run(ckpt_list)
        print(f"Filtered to {len(ckpt_list)} checkpoints (best per hyperparameter config)\n")
    
    if eval_args.debug:
        ckpt_list = ckpt_list[:8]
    
    eval_args.ckpt_path_list = ckpt_list


    

    # Choose evaluation mode
    if eval_args.serial or len(ckpt_list) <= 1:
        print(f"Running SERIAL mode with DataParallel across {n_gpus} GPUs")
        
        evaluate_serial_dataparallel(eval_args)
    else:
        print(f"Running PARALLEL mode with {eval_args.n_parallel} processes")
        
        if eval_args.n_parallel > n_gpus:
            print(f"Warning: Requested {eval_args.n_parallel} parallel jobs but only {n_gpus} GPUs available.")
            print(f"Setting n_parallel to {n_gpus}")
            eval_args.n_parallel = n_gpus
        
        
        # Filter out already-completed checkpoints
        completed = get_completed_checkpoints(eval_args.output_path)
        if completed:
            print(f"\nFound {len(completed)} already-completed checkpoints in {eval_args.output_path}")
            original_count = len(ckpt_list)
            ckpt_list = [ckpt for ckpt in ckpt_list if ckpt not in completed]
            print(f"Filtered out {original_count - len(ckpt_list)} already-completed checkpoints")
            print(f"Remaining checkpoints to evaluate: {len(ckpt_list)}\n")
            
            if len(ckpt_list) == 0:
                print("All checkpoints already evaluated!")
                return
        
        
        if eval_args.n_parallel > len(ckpt_list):
            print(f"have fewer jobs than GPUs, setting eval_args.n_parallel to {len(ckpt_list)}")
            eval_args.n_parallel = len(ckpt_list) - 1

        print(f"Evaluating {len(ckpt_list)} checkpoints using {eval_args.n_parallel} parallel processes")
        print(f"Note: DataLoader num_workers forced to 0 in parallel mode")
        
        if eval_args.save_predictions:
            print(f"Saving full predictions to checkpoint directories with suffix: {eval_args.predictions_suffix}")
        
        # eval_args_dict = {
        #     "dataset": eval_args.dataset,
        #     "data_dir": eval_args.data_dir,
        #     "split": eval_args.split,
        #     "batch_size": eval_args.batch_size,
        #     "num_workers": 0,
        #     "save_predictions": eval_args.save_predictions,
        #     "predictions_suffix": eval_args.predictions_suffix,
        # }
        # In eval_args_dict:

        eval_args_dict = {
            "dataset": eval_args.dataset,
            "data_dir": eval_args.data_dir,
            "split": eval_args.split,
            "batch_size": eval_args.batch_size,
            "num_workers": 0,
            "save_predictions": eval_args.save_predictions,
            "predictions_suffix": eval_args.predictions_suffix,
            "use_filtered_ground_truth": eval_args.use_filtered_ground_truth,
            "average_over_frames": eval_args.average_over_frames,
        }
        
        job_args = [
            (ckpt_path, eval_args_dict, i % n_gpus) 
            for i, ckpt_path in enumerate(ckpt_list)
        ]
        
        # Determine file mode and whether to write header
        file_mode = "a" if os.path.exists(eval_args.output_path) else "w"
        write_header = not os.path.exists(eval_args.output_path)
        
        with open(eval_args.output_path, mode=file_mode, newline='') as handle:
            writer = csv.writer(handle)
            
            # # Always write full header with metrics
            # if write_header:
            #     header = [
            #         "path", "output_path", "dataset", "split", 
            #         "pdb_pearson", "pdb_spearman", "top_pose_rate", 
            #         "pearson", "spearman", "status"
            #     ]
            #     writer.writerow(header)
            # And in the parallel mode section:
            if write_header:
                header = [
                    "path", "output_path", "dataset", "split", 
                    "pdb_pearson", "pdb_spearman", "top_pose_rate", 
                    "top1_accuracy", "crystal_top1_rate",  # NEW
                    "pearson", "spearman", "status"
                ]
                writer.writerow(header)
            
            with Pool(processes=eval_args.n_parallel) as pool:
                for result in tqdm(
                    pool.imap(evaluate_single_checkpoint, job_args),
                    total=len(job_args),
                    desc="Evaluating checkpoints"
                ):
                    # Always write all columns
                    # row_data = [
                    #     result["path"],
                    #     result.get("output_path", ""),
                    #     result["dataset"],
                    #     result["split"],
                    #     result["pdb_pearson"],
                    #     result["pdb_spearman"],
                    #     result["top_pose_rate"],
                    #     result["pearson"],
                    #     result["spearman"],
                    #     result["status"]
                    # ]
                    # Always write all columns
                    row_data = [
                        result["path"],
                        result.get("output_path", ""),
                        result["dataset"],
                        result["split"],
                        result["pdb_pearson"],
                        result["pdb_spearman"],
                        result["top_pose_rate"],
                        result.get("top1_accuracy"),       # ADD THIS
                        result.get("crystal_top1_rate"),   # ADD THIS
                        result["pearson"],
                        result["spearman"],
                        result["status"]
                    ]
                    writer.writerow(row_data)
                    handle.flush()
                    
                    if result["status"] not in ["success", "saved"]:
                        print(f"\nError evaluating {result['path']}: {result['status']}")
                        if "traceback" in result:
                            print(result["traceback"])
        
        print(f"\nEvaluation complete! Results saved to {eval_args.output_path}") 


if __name__ == "__main__":
    import socket
    host = socket.gethostname()

    # import pdb; pdb.set_trace()
    main()