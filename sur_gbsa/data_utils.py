################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

# from sur_gbsa.ProtMD.data.dataset import GBSAMDDataset
# from sur_gbsa.ProtMD.data.dataset import PLAS20kDataset
from sur_gbsa.datasets import GBSAMDDataset, MisatoDataset, PDBBindDataset
# from sur_gbsa.misato.dataloader_misato import MisatoDataset



def filter_collate_fn(batch):
    # Filter out None values from the batch
    batch = [item for item in batch if item is not None]

    # Check if the batch is empty after filtering
    if len(batch) == 0:
        return None

    # Use the default collate function to handle the rest of the batch
    return torch.utils.data.default_collate(batch)


def get_train_val_test_datasets(config):
    """Main entry point for loading datasets based on config.dataset"""
    
    # Map dataset names to their configurations
    dataset_configs = {
        # MD datasets
        "md-crystal": {"target": "GBSA", "max_frames": 1000, "pose_list": [0]},
        "md-dock_top_1": {"target": "GBSA", "max_frames": 1000, "pose_list": [1]},
        "md-dock_top_3": {"target": "GBSA", "max_frames": 1000, "pose_list": [1, 2, 3]},
        "md-dock_top_5": {"target": "GBSA", "max_frames": 1000, "pose_list": [1, 2, 3, 4, 5]},
        "md-dock_top_5+crystal": {"target": "GBSA", "max_frames": 1000, "pose_list": [0, 1, 2, 3, 4, 5]},
        
        # Single-point datasets
        "sp-crystal": {"target": "GBSA", "max_frames": 1, "pose_list": [0]},
        "sp-dock_top_1": {"target": "GBSA", "max_frames": 1, "pose_list": [1]},
        "sp-dock_top_3": {"target": "GBSA", "max_frames": 1, "pose_list": [1, 2, 3]},
        "sp-dock_top_5": {"target": "GBSA", "max_frames": 1, "pose_list": [1, 2, 3, 4, 5]},
        "sp-dock_top_5+crystal": {"target": "GBSA", "max_frames": 1, "pose_list": [0, 1, 2, 3, 4, 5]},
        
        # Mean datasets
        "mean-crystal": {"target": "GBSA_mean", "max_frames": 1, "pose_list": [0]},
        "mean-dock_top_1": {"target": "GBSA_mean", "max_frames": 1, "pose_list": [1]},
        "mean-dock_top_3": {"target": "GBSA_mean", "max_frames": 1, "pose_list": [1, 2, 3]},
        "mean-dock_top_5": {"target": "GBSA_mean", "max_frames": 1, "pose_list": [1, 2, 3, 4, 5]},
        "mean-dock_top_5+crystal": {"target": "GBSA_mean", "max_frames": 1, "pose_list": [0, 1, 2, 3, 4, 5]},
    }
    
    # Handle standard GBSA datasets
    if config.dataset in dataset_configs:
        cfg = dataset_configs[config.dataset]
        config_dict = vars(config)
        config_dict.update({
            "target": cfg["target"],
            "max_frames": cfg["max_frames"],
            "train_pose_list": cfg["pose_list"],
            "val_pose_list": cfg["pose_list"],
            "test_pose_list": cfg["pose_list"],
        })
        train_list, val_list, test_list = load_disjoint_gbsa_dataset(**config_dict)
        
        # Wrap lists in ConcatDataset
        train_dataset = ConcatDataset(train_list) if train_list else None
        val_dataset = ConcatDataset(val_list) if val_list else None
        test_dataset = ConcatDataset(test_list) if test_list else None
        
        return train_dataset, val_dataset, test_dataset
    
    # Handle special datasets
    elif config.dataset == "mpro":
        return load_mpro_dataset()
    
    # elif config.dataset == "pdbbind-30":
        # return load_pdbbind30_dataset()
    
    # elif config.dataset == "pdbbind-60":
        # return load_pdbbind60_dataset()

    elif config.dataset == "pdbbind-30":
        return load_pdbbind30_dataset(
            use_residue_features=getattr(config, 'use_residue_features', False)
        )
    
    elif config.dataset == "pdbbind-60":
        return load_pdbbind60_dataset(
            use_residue_features=getattr(config, 'use_residue_features', False)
        )



    elif config.dataset == "PLAS-20k":
        return load_plas20k_dataset(config)
    
    elif config.dataset == "misato":
        print("loading misato")
        train_list, val_list, test_list = load_misato(config)
        
        # Wrap lists in ConcatDataset
        train_dataset = ConcatDataset(train_list) if train_list else None
        val_dataset = ConcatDataset(val_list) if val_list else None
        test_dataset = ConcatDataset(test_list) if test_list else None
        
        return train_dataset, val_dataset, test_dataset
    
    elif config.dataset == "misato-coremd-combo":
        # Load both datasets as lists
        misato_train_list, misato_val_list, misato_test_list = load_misato(config)
        
        # Load CoreMD with same config
        config_dict = vars(config)
        config_dict.update({
            "target": "GBSA",
            "max_frames": 1000,
            "train_pose_list": [0, 1, 2, 3, 4, 5],
            "val_pose_list": [0, 1, 2, 3, 4, 5],
            "test_pose_list": [0, 1, 2, 3, 4, 5],
        })
        coremd_train_list, coremd_val_list, coremd_test_list = load_disjoint_gbsa_dataset(**config_dict)
        
        # Combine lists and create single-level ConcatDataset
        combo_train = ConcatDataset(misato_train_list + coremd_train_list)
        combo_val = ConcatDataset(misato_val_list + coremd_val_list)
        combo_test = ConcatDataset(misato_test_list + coremd_test_list)
        
        return combo_train, combo_val, combo_test
    
    else:
        raise NotImplementedError(f"Dataset {config.dataset} not implemented")
def load_mpro_dataset(**kwargs):
    x_train, pose_num_train, pos_train, y_train = torch.load(
        f"/p/vast1/jones289/dtra_mrpo/train.pt"
    )
    x_val, pose_num_val, pos_val, y_val = torch.load(
        f"/p/vast1/jones289/dtra_mrpo/val.pt"
    )
    x_test, pose_num_test, pos_test, y_test = torch.load(
        f"/p/vast1/jones289/dtra_mrpo/test.pt"
    )

    train_dataset = TensorDataset(x_train, pos_train, y_train, torch.arange(len(y_train)))
    val_dataset = TensorDataset(x_val, pos_val, y_val, torch.arange(len(y_val)))
    test_dataset = TensorDataset(x_test, pos_test, y_test, torch.arange(len(y_test)))

    return train_dataset, val_dataset, test_dataset





def load_pdbbind30_dataset(use_residue_features=False, **kwargs):
    # Try to load with residue names (new format)
    try:
        x_train, idx_train, pos_train, y_train, resnames_train = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_train_30.pt"
        )
        x_val, idx_val, pos_val, y_val, resnames_val = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_val_30.pt"
        )
        x_test, idx_test, pos_test, y_test, resnames_test = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_test_30.pt"
        )
    except ValueError:
        # Fall back to old format (no residue names)
        print("WARNING: Loading old PDBbind files without residue information")
        x_train, idx_train, pos_train, y_train = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_train_30.pt"
        )
        x_val, idx_val, pos_val, y_val = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_val_30.pt"
        )
        x_test, idx_test, pos_test, y_test = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_test_30.pt"
        )
        resnames_train = resnames_val = resnames_test = None
        use_residue_features = False

    train_dataset = PDBBindDataset(x_train, pos_train, y_train, 
                                   resnames=resnames_train,
                                   use_residue_features=use_residue_features)
    val_dataset = PDBBindDataset(x_val, pos_val, y_val, 
                                 resnames=resnames_val,
                                 use_residue_features=use_residue_features)
    test_dataset = PDBBindDataset(x_test, pos_test, y_test, 
                                  resnames=resnames_test,
                                  use_residue_features=use_residue_features)
    
    return train_dataset, val_dataset, test_dataset


def load_pdbbind60_dataset(use_residue_features=False, **kwargs):
    # Same pattern as above
    try:
        x_train, idx_train, pos_train, y_train, resnames_train = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_train_60.pt"
        )
        x_val, idx_val, pos_val, y_val, resnames_val = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_val_60.pt"
        )
        x_test, idx_test, pos_test, y_test, resnames_test = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_test_60.pt"
        )
    except ValueError:
        print("WARNING: Loading old PDBbind files without residue information")
        x_train, idx_train, pos_train, y_train = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_train_60.pt"
        )
        x_val, idx_val, pos_val, y_val = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_val_60.pt"
        )
        x_test, idx_test, pos_test, y_test = torch.load(
            "/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/pdb/pdb_test_60.pt"
        )
        resnames_train = resnames_val = resnames_test = None
        use_residue_features = False

    train_dataset = PDBBindDataset(x_train, pos_train, y_train, 
                                   resnames=resnames_train,
                                   use_residue_features=use_residue_features)
    val_dataset = PDBBindDataset(x_val, pos_val, y_val, 
                                 resnames=resnames_val,
                                 use_residue_features=use_residue_features)
    test_dataset = PDBBindDataset(x_test, pos_test, y_test, 
                                  resnames=resnames_test,
                                  use_residue_features=use_residue_features)
    
    return train_dataset, val_dataset, test_dataset


def load_plas20k_dataset(config, **kwargs):
    
    data_dir = Path(config.data_dir)
    train_pdb_list = pd.read_csv(config.train_split, header=None)[0].values.tolist()
    val_pdb_list = pd.read_csv(config.val_split, header=None)[0].values.tolist()
    test_pdb_list = pd.read_csv(config.test_split, header=None)[0].values.tolist()

    train_dataset = PLAS20kDataset(data_dir=data_dir, pdb_list=train_pdb_list)
    val_dataset = PLAS20kDataset(data_dir=data_dir, pdb_list=val_pdb_list)
    test_dataset = PLAS20kDataset(data_dir=data_dir, pdb_list=test_pdb_list)

    return train_dataset, val_dataset, test_dataset


def load_disjoint_gbsa_dataset(
    target="GBSA",
    max_frames=1000,
    rmsd_thresh=2.0,
    train_frac=0.8,
    max_len=600,
    data_dir=None,
    train_pose_list=[1, 2, 3, 4, 5],
    val_pose_list=[1, 2, 3, 4, 5],
    test_pose_list=[1, 2, 3, 4, 5],
    noise=False,
    noise_scale=0,
    prompt=None,
    label_process="sign-flip",
    use_residue_features=False,
    verbose=False,
    **kwargs,
):
    """
    Load GBSA dataset with disjoint train/val/test splits.
    
    Returns lists of datasets (not ConcatDataset) for each split.
    """
    
    if verbose:
        print(
            f"loading gbsa dock dataset from: {data_dir}, using target {target} for training."
        )
    
    data_dir = Path(data_dir)
    
    # Define split configurations
    split_configs = {
        "train": ("train_split", train_pose_list),
        "val": ("val_split", val_pose_list),
        "test": ("test_split", test_pose_list),
    }
    
    # Check at least one split is provided
    requested_splits = [
        split for split, (key, _) in split_configs.items() if key in kwargs
    ]
    
    if not requested_splits:
        raise ValueError(
            "At least one of train_split, val_split, or test_split must be provided in kwargs"
        )
    
    # Process splits
    results = {}
    for split, (split_key, pose_list) in split_configs.items():
        if split_key not in kwargs:
            results[split] = []  # Return empty list instead of None
            continue
        
        # Load PDB list for this split
        pdb_list = pd.read_csv(kwargs[split_key], header=None)[0].values.tolist()
        file_list = [
            x.name
            for x in data_dir.glob("*.npy")
            if x.name.split("-")[0] in pdb_list
            and int(x.name.split("-")[2].split("_")[0].replace("p", "")) in pose_list
        ]
        
        if verbose:
            print(f"{split} split: found {len(file_list)} files")
        
        # Create datasets for this split
        datasets = []
        for pdbid in file_list:
            dataset = GBSAMDDataset(
                target=target,
                max_frames=max_frames,
                backprop=None,
                rmsd_thresh=rmsd_thresh,
                max_len=max_len,
                pdb_list=[pdbid],
                data_dir=data_dir,
                prompt=prompt,
                noise=noise,
                noise_scale=noise_scale,
                label_process=label_process,
                use_residue_features=use_residue_features,
                verbose=verbose,
            )
            if len(dataset) > 0:
                datasets.append(dataset)
        
        results[split] = datasets  # Return list of datasets
        
        if verbose:
            print(f"{split} split: created {len(datasets)} datasets")
    
    return results["train"], results["val"], results["test"]


def load_union_gbsa_dataset(
    target="GBSA",
    max_frames=1000,
    rmsd_thresh=2.0,
    train_frac=0.8,
    max_len=600,
    data_dir=None,
    train_pdb_list=None,
    train_pose_list=None,
    test_pdb_list=None,
    test_pose_list=None,
    noise=False,
    noise_scale=0,
    prompt=None,
    split_strat="temporal",
    label_process="sign-flip",
    use_residue_features=False,
    **kwargs,
):
    """
    Load GBSA dataset where same PDB IDs can appear in train/val/test.
    
    Uses temporal or random split to divide frames from each PDB.
    """
    
    data_dir = Path(data_dir)

    train_pdb_list = pd.read_csv(kwargs["train_split"], header=None)[0].values.tolist()
    train_pdb_list = [
        x.name
        for x in data_dir.glob("*.npy")
        if x.name.split("-")[0] in train_pdb_list
        and int(x.name.split("-")[2].split("_")[0].replace("p", "")) in train_pose_list
    ]

    val_pdb_list = pd.read_csv(kwargs["val_split"], header=None)[0].values.tolist()
    val_pdb_list = [
        x.name
        for x in data_dir.glob("*.npy")
        if x.name.split("-")[0] in val_pdb_list
        and int(x.name.split("-")[2].split("_")[0].replace("p", "")) in val_pose_list
    ]

    test_pdb_list = pd.read_csv(kwargs["test_split"], header=None)[0].values.tolist()
    test_pdb_list = [
        x.name
        for x in data_dir.glob("*.npy")
        if x.name.split("-")[0] in test_pdb_list
        and int(x.name.split("-")[2].split("_")[0].replace("p", "")) in test_pose_list
    ]

    train_dataset_list, val_dataset_list, test_dataset_list = [], [], []
    
    for pdbid in train_pdb_list:
        if split_strat == "random":
            train_dataset = GBSAMDDataset(
                target=target,
                max_frames=max_frames,
                backprop=None,
                rmsd_thresh=rmsd_thresh,
                max_len=max_len,
                pdb_list=[pdbid],
                data_dir=data_dir,
                prompt=prompt,
                noise=noise,
                noise_scale=noise_scale,
                label_process=label_process,
                use_residue_features=use_residue_features,
            )

            train_dataset, test_dataset = random_split(
                train_dataset, [train_frac, 1 - train_frac]
            )
            train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])

            train_dataset_list.append(train_dataset)
            val_dataset_list.append(val_dataset)
            test_dataset_list.append(test_dataset)

        elif split_strat == "temporal":
            train_dataset = GBSAMDDataset(
                target=target,
                max_frames=max_frames,
                max_time_step_prop=train_frac,
                backprop=True,
                rmsd_thresh=rmsd_thresh,
                max_len=max_len,
                pdb_list=[pdbid],
                data_dir=data_dir,
                prompt=prompt,
                noise=noise,
                noise_scale=noise_scale,
                label_process=label_process,
                use_residue_features=use_residue_features,
            )

            train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])

            test_dataset = GBSAMDDataset(
                target=target,
                max_frames=1000,
                max_time_step_prop=train_frac,
                backprop=False,
                rmsd_thresh=rmsd_thresh,
                max_len=max_len,
                pdb_list=[pdbid],
                data_dir=data_dir,
                prompt=prompt,
                noise=noise,
                noise_scale=noise_scale,
                label_process=label_process,
                use_residue_features=use_residue_features,
            )

            train_dataset_list.append(train_dataset)
            val_dataset_list.append(val_dataset)
            test_dataset_list.append(test_dataset)

    train_dataset = [x for x in train_dataset_list if len(x) > 0]
    val_dataset = [x for x in val_dataset_list if len(x) > 0]
    test_dataset = [x for x in test_dataset_list if len(x) > 0]

    return train_dataset, val_dataset, test_dataset


def load_misato(config):
    """Load MISATO dataset - returns lists of datasets"""
    from sur_gbsa.misato.dataloader_misato import MisatoDataset
    
    train_dataset = MisatoDataset(
        data_path="/g/g13/jones289/workspace/pretrain_md/sur_gbsa/misato/data/misato/misato_train.pt",
        pdb_path=config.train_split,
        max_len=config.gbsa_max_len,
        use_residue_features=getattr(config, 'use_residue_features', False),
        verbose=getattr(config, 'verbose', False),
    )
    val_dataset = MisatoDataset(
        data_path="/g/g13/jones289/workspace/pretrain_md/sur_gbsa/misato/data/misato/misato_val.pt",
        pdb_path=config.val_split,
        max_len=config.gbsa_max_len,
        use_residue_features=getattr(config, 'use_residue_features', False),
        verbose=getattr(config, 'verbose', False),
    )
    test_dataset = MisatoDataset(
        data_path="/g/g13/jones289/workspace/pretrain_md/sur_gbsa/misato/data/misato/misato_test.pt",
        pdb_path=config.test_split,
        max_len=config.gbsa_max_len,
        use_residue_features=getattr(config, 'use_residue_features', False),
        verbose=getattr(config, 'verbose', False),
    )

    # Return as lists (even though each is a single dataset)
    return [train_dataset], [val_dataset], [test_dataset]


if __name__ == "__main__":
    pass