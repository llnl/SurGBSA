################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import pandas as pd

from sur_gbsa import GBSA_DATASET_LIST, FINETUNE_DATASET_LIST


def load_gbsa_result_path(data):
    y_pred = data["y_pred"].squeeze()
    y_true = data["y_true"].squeeze()
    ids = [x for sublist in data["id"] for x in sublist]

    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    rmse = mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)

    return {
        "train_size": data["train_size"],
        "val_size": data["val_size"],
        "test_size": data["test_size"],
        "y_pred": y_pred,
        "y_true": y_true,
        "id": [0] * len(y_pred),
        "pdbid": ["foo"] * len(y_pred),
        "pose": [0] * len(y_pred),
        "pearsonr": pearson,
        "spearmanr": spearman,
        "rmse": rmse,
    }


def load_mpro_result_path(data):
    y_pred = data["y_pred"].squeeze()
    y_true = data["y_true"].squeeze()
    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    rmse = mean_squared_error(y_pred=y_pred, y_true=y_true, squared=False)

    return {
        "train_size": data["train_size"],
        "val_size": data["val_size"],
        "test_size": data["test_size"],
        "y_pred": y_pred,
        "y_true": y_true,
        "id": [data["args"].dataset] * len(y_true),
        "pdbid": [data["args"].dataset] * len(y_true),
        "pose": [0] * len(y_true),
        "pearsonr": pearson,
        "spearmanr": spearman,
        "rmse": rmse,
    }


def load_result_path(path, device="cpu"):
    data = torch.load(path, map_location=device)
    dataset = data["args"].dataset
    if dataset in GBSA_DATASET_LIST:
        return load_gbsa_result_path(data)
    elif dataset in FINETUNE_DATASET_LIST:
        return load_mpro_result_path(data)
    else:
        raise NotImplementedError


def load_result_path_as_df(path, device="cpu"):
    data = load_result_path(path, device=device)

    df_len = len(data["y_pred"])

    df = pd.DataFrame()

    for key, value in data.items():
        print(key, len(value)) if isinstance(value, list) else print(key, value)
        # list of scalar values that are constant across the time dimension
        if key in [
            "model",
            "train_size",
            "test_size",
            "val_size",
            "split",
            "pearsonr",
            "spearmanr",
            "rmse",
            "train_pdbid_list",
            "test_pdbid_list",
            "train_pose_list",
            "test_pose_list",
        ]:
            df[key] = [value] * df_len

        else:
            if len(value) > 0:
                try:
                    df[key] = value
                except Exception as e:
                    print(e)
                    return

    return df

# from pathlib import Path
# from sur_gbsa.data_utils import filter_collate_fn, get_train_val_test_datasets
# from torch.utils.data import DataLoader, DistributedSampler
# import torch.distributed as dist
# from sur_gbsa.test import run_eval

# def run_full_test_eval(
#     args,
#     model_path,
#     best_epoch,
#     best_model,
#     ckpt_root_dir,
#     current_device,
#     rank,
#     writer,
#     epoch,
#     batch_size_per_gpu,
#     train_loader,
#     val_loader,
#     test_loader,
#     log,
# ):
#     for dataset in GBSA_DATASET_LIST:
#         # save the predictions and ground truth labels
#         result_path = ckpt_root_dir / Path(
#             f"test_result-{model_path.with_name(f'best_model-epoch-{best_epoch}').name}-{dataset}.pt"
#         )
#         if result_path.exists():
#             if rank == 0:
#                 log.logger.info(f"{result_path} already exists, work is finished.")
#             # pass
#         else:
#             # if True:
#             config = args
#             config.dataset = dataset

#             _, _, eval_dataset = get_train_val_test_datasets(config)
#             eval_sampler = DistributedSampler(eval_dataset)
#             if isinstance(eval_sampler, DistributedSampler):
#                 eval_sampler.set_epoch(0)
#             eval_loader = DataLoader(
#                 eval_dataset,
#                 batch_size=batch_size_per_gpu,
#                 shuffle=False,
#                 num_workers=1,
#                 collate_fn=filter_collate_fn,
#                 sampler=eval_sampler,
#                 pin_memory=False,
#                 persistent_workers=False,
#                 drop_last=False,
#             )

#             eval_dict = run_eval(
#                 best_model,
#                 eval_loader,
#                 sampler=eval_sampler,
#                 args=config,
#                 device=current_device,
#             )
#             pearson, spearman, metric, pred, gt, idx = (
#                 eval_dict["pearson"],
#                 eval_dict["spearman"],
#                 eval_dict["rmse"],
#                 eval_dict["y_pred"],
#                 eval_dict["y_true"],
#                 eval_dict["idx"],
#             )

#             dist.barrier()

#             pred = distributed_gather(pred.float())
#             gt = distributed_gather(gt.float())
#             idx = distributed_gather(idx)
#             latent = distributed_gather(eval_dict["latent"])
#             # metadata = [None] * world_size
            
#             metadata = gather_to_rank0(eval_dict["metadata"])
#             frame_num = gather_to_rank0(
#                 eval_dict["frame_num"]
#             )  # todo: can use torch tensors instead
#             dist.barrier()

#             if rank == 0:

#                 # stack the data on the rank dimension
#                 pred = pred.view(-1)
#                 gt = gt.view(-1)
#                 idx = idx.view(-1)
#                 latent = latent.view(-1, args.dim)
#                 metadata = [item for sublist in metadata for item in sublist]
#                 # frame_num = torch.cat([item for sublist in frame_num for item in sublist]).view(-1)
#                 frame_num = torch.cat(
#                     [item.view(-1) for sublist in frame_num for item in sublist]
#                 ).view(-1)

#                 metric = mse_loss(pred, gt)
#                 pearson = stats.pearsonr(pred, gt)[0]
#                 spearman = stats.spearmanr(pred, gt)[0]

#                 # compute pdb pearson and spearman

#                 test_dict  ={
#                     "y_pred": pred,
#                     "y_true": gt,
#                     "idx": idx,
#                     # "latent": latent,
#                     "metadata": metadata,
#                     # "frame_num": frame_list,
#                 }

#                 test_df = pd.DataFrame(test_dict)

#                 test_df['pdbid'] = test_df['metadata'].apply(lambda x: x.split("-")[0])
#                 test_df['pose'] = test_df['metadata'].apply(lambda x: x.split("-")[2][-1])


#                 import numpy as np
#                 from scipy.stats import spearmanr, pearsonr
#                 pearson_list = []
#                 spearman_list = []
#                 for pdb, pdb_group in test_df.groupby("pdbid"):
#                     s_val, p_val = np.nan, np.nan
#                     if pdb_group.shape[0] < 2:
#                         pass
#                     else:
#                         s_val = spearmanr(pdb_group["y_pred"], pdb_group["y_true"])[0]
#                         spearman_list.append(s_val)
#                         p_val = pearsonr(pdb_group["y_pred"], pdb_group["y_true"])[0]
#                         pearson_list.append(p_val)


#                 spearman_list = [x for x in spearman_list if not np.isnan(x)]

#                 pearson_list = [x for x in pearson_list if not np.isnan(x)]

#                 pdb_spearmanr = np.mean(spearman_list)
#                 pdb_pearsonr = np.mean(pearson_list)


#                 writer.add_scalar(f"RMSE/{dataset}/test-{rank}", metric, epoch)
#                 writer.add_scalar(f"Pearsonr/{dataset}/test-{rank}", pearson, epoch)
#                 writer.add_scalar(f"Spearmanr/{dataset}/test-{rank}", spearman, epoch)
#                 writer.add_scalar(f"PDB-Pearsonr/{dataset}/test-{rank}", pdb_pearsonr, epoch)
#                 writer.add_scalar(f"PDB-Spearmanr/{dataset}/test-{rank}", pdb_spearmanr, epoch)
#                 writer.flush()

#                 torch.save(
#                     {
#                         "train_size": len(train_loader.dataset),
#                         "val_size": len(val_loader.dataset),
#                         "test_size": len(test_loader.dataset),
#                         "y_pred": pred,
#                         "y_true": gt,
#                         "id": idx,
#                         "latent": latent,
#                         "metadata": metadata,
#                         "frame_num": frame_num,
#                         # "pdbid": pdbid_list,
#                         # "pose": pose_list,
#                         # "frame": frame_list,
#                         # "id": path_list,
#                         # "frame_num": frame_num,
#                         "args": args,
#                     },
#                     result_path,
#                 )

#                 log.logger.info(
#                     f"rank: {rank} | dataset: {dataset} | RMSE: {metric:.5f} | Test Pearson: {pearson:.5f} | Test Spearman: {spearman:.5f} | Test PDB-Pearson: {pdb_pearsonr:.5f} | Test PDB-Spearman: {pdb_spearmanr:.5f}"
#                 )
#                 log.logger.info(
#                     f"Save the best model as {str(model_path)}\nBest result as {result_path}\nBest Epoch: {best_epoch}"
#                 )

#                 log.logger.info(f"{result_path.parent}")

#                 plot_path = result_path.with_name(
#                     f"{result_path.stem}_scatterplot-{dataset}.png"
#                 )

#                 log.logger.info(plot_path)
#                 if plot_path.exists():
#                     pass
#                 else:

#                     f, ax = plt.subplots(1, 2, figsize=(18, 8))
#                     ax = ax.flatten()

#                     sns.scatterplot(x=gt.numpy(), y=pred.numpy(), ax=ax[0])

#                     sns.kdeplot(
#                         x=pred,
#                         color="red",
#                         ax=ax[1],
#                         shade=True,
#                         label="Predicted MM/GBSA",
#                     )
#                     sns.kdeplot(
#                         x=gt,
#                         color="blue",
#                         ax=ax[1],
#                         shade=True,
#                         label="Calculated MM/GBSA",
#                     )

#                     ax[1].legend()

#                     pearson_score = pearsonr(pred.numpy(), gt.numpy())[0]
#                     hparam_str = plot_path.parent.stem
#                     f.suptitle(f"{hparam_str}\n(r={pearson_score:.2f})")
#                     plt.tight_layout()
#                     f.savefig(plot_path, dpi=450)
#                     plt.close(f)

