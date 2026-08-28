import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from sur_gbsa.datasets import GBSAMDDataset, MisatoDataset, PDBBindDataset, FEPPlusDataset


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


def objective_flags(config):
    train_objectives = _as_list(getattr(config, "train_objectives", []))
    val_objectives = _as_list(getattr(config, "val_objectives", []))
    objectives = set(train_objectives) | set(val_objectives)

    return {
        "use_coord_pairs": "coord_pred" in objectives,
        "use_rmsf": "rmsf" in objectives,
        "use_ordering": "ordering" in objectives,
    }


def filter_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)


def make_pose_list(mode):
    if mode == "crystal":
        return [0]
    if mode == "dock_top_1":
        return [1]
    if mode == "dock_top_3":
        return [1, 2, 3]
    if mode == "dock_top_5":
        return [1, 2, 3, 4, 5]
    if mode == "dock_top_5+crystal":
        return [0, 1, 2, 3, 4, 5]
    raise ValueError(f"Unknown pose mode: {mode}")


def build_dataset_config(name):
    mapping = {
        "md-crystal": ("GBSA", "crystal", 1000),
        "md-dock_top_1": ("GBSA", "dock_top_1", 1000),
        "md-dock_top_3": ("GBSA", "dock_top_3", 1000),
        "md-dock_top_5": ("GBSA", "dock_top_5", 1000),
        "md-dock_top_5+crystal": ("GBSA", "dock_top_5+crystal", 1000),
        "sp-crystal": ("GBSA", "crystal", 1),
        "sp-dock_top_1": ("GBSA", "dock_top_1", 1),
        "sp-dock_top_3": ("GBSA", "dock_top_3", 1),
        "sp-dock_top_5": ("GBSA", "dock_top_5", 1),
        "sp-dock_top_5+crystal": ("GBSA", "dock_top_5+crystal", 1),
        "mean-crystal": ("GBSA_mean", "crystal", 1),
        "mean-dock_top_1": ("GBSA_mean", "dock_top_1", 1),
        "mean-dock_top_3": ("GBSA_mean", "dock_top_3", 1),
        "mean-dock_top_5": ("GBSA_mean", "dock_top_5", 1),
        "mean-dock_top_5+crystal": ("GBSA_mean", "dock_top_5+crystal", 1),
    }

    if name not in mapping:
        raise ValueError(f"Unknown dataset: {name}")

    target, pose_mode, max_frames = mapping[name]
    pose_list = make_pose_list(pose_mode)

    return {
        "target": target,
        "max_frames": max_frames,
        "pose_list": pose_list,
    }


def get_train_val_test_datasets(config):
    import copy

    dataset_configs = {
        "md-crystal": {"target": "GBSA", "max_frames": 1000, "pose_list": [0]},
        "md-dock_top_1": {"target": "GBSA", "max_frames": 1000, "pose_list": [1]},
        "md-dock_top_3": {"target": "GBSA", "max_frames": 1000, "pose_list": [1, 2, 3]},
        "md-dock_top_5": {"target": "GBSA", "max_frames": 1000, "pose_list": [1, 2, 3, 4, 5]},
        "md-dock_top_5+crystal": {"target": "GBSA", "max_frames": 1000, "pose_list": [0, 1, 2, 3, 4, 5]},
        "sp-crystal": {"target": "GBSA", "max_frames": 1, "pose_list": [0]},
        "sp-dock_top_1": {"target": "GBSA", "max_frames": 1, "pose_list": [1]},
        "sp-dock_top_3": {"target": "GBSA", "max_frames": 1, "pose_list": [1, 2, 3]},
        "sp-dock_top_5": {"target": "GBSA", "max_frames": 1, "pose_list": [1, 2, 3, 4, 5]},
        "sp-dock_top_5+crystal": {"target": "GBSA", "max_frames": 1, "pose_list": [0, 1, 2, 3, 4, 5]},
        "mean-crystal": {"target": "GBSA_mean", "max_frames": 1, "pose_list": [0]},
        "mean-dock_top_1": {"target": "GBSA_mean", "max_frames": 1, "pose_list": [1]},
        "mean-dock_top_3": {"target": "GBSA_mean", "max_frames": 1, "pose_list": [1, 2, 3]},
        "mean-dock_top_5": {"target": "GBSA_mean", "max_frames": 1, "pose_list": [1, 2, 3, 4, 5]},
        "mean-dock_top_5+crystal": {"target": "GBSA_mean", "max_frames": 1, "pose_list": [0, 1, 2, 3, 4, 5]},
    }

    if config.dataset in dataset_configs:
        cfg = dataset_configs[config.dataset]
        config_dict = copy.deepcopy(vars(config))
        config_dict.update({
            "target": cfg["target"],
            "max_frames": cfg["max_frames"],
            "train_pose_list": cfg["pose_list"],
            "val_pose_list": cfg["pose_list"],
            "test_pose_list": cfg["pose_list"],
        })

        flags = objective_flags(config)
        config_dict.update(flags)

        dataset_dict = load_disjoint_gbsa_dataset(**config_dict)

        for key in dataset_dict.keys():
            dataset_dict[key] = ConcatDataset(dataset_dict[key])

        return dataset_dict

    elif config.dataset == "mpro":
        return load_mpro_dataset()

    elif config.dataset == "pdbbind-30":
        return load_pdbbind30_dataset(
            use_residue_features=getattr(config, "use_residue_features", False)
        )

    elif config.dataset == "pdbbind-60":
        return load_pdbbind60_dataset(
            use_residue_features=getattr(config, "use_residue_features", False)
        )

    elif config.dataset == "PLAS-20k":
        return load_plas20k_dataset(config)

    elif config.dataset == "misato":
        print("loading misato")
        return load_misato(config)

    elif config.dataset == "misato-coremd-combo":
        misato_dict = load_misato(config)

        config_dict = copy.deepcopy(vars(config))
        config_dict.update({
            "target": "GBSA",
            "max_frames": 1000,
            "train_pose_list": [0, 1, 2, 3, 4, 5],
            "val_pose_list": [0, 1, 2, 3, 4, 5],
            "test_pose_list": [0, 1, 2, 3, 4, 5],
        })
        config_dict.update(objective_flags(config))

        coremd_dict = load_disjoint_gbsa_dataset(**config_dict)

        combo_dict = {}
        for split in ["train", "val", "test"]:
            if split in misato_dict and split in coremd_dict:
                combo_dict[split] = misato_dict[split] + coremd_dict[split]

        return combo_dict

    else:
        raise NotImplementedError(f"Dataset {config.dataset} not implemented")


def load_mpro_dataset(**kwargs):
    x_train, pose_num_train, pos_train, y_train = torch.load(f"/p/vast1/jones289/dtra_mrpo/train.pt")
    x_val, pose_num_val, pos_val, y_val = torch.load(f"/p/vast1/jones289/dtra_mrpo/val.pt")
    x_test, pose_num_test, pos_test, y_test = torch.load(f"/p/vast1/jones289/dtra_mrpo/test.pt")

    train_dataset = TensorDataset(x_train, pos_train, y_train, torch.arange(len(y_train)))
    val_dataset = TensorDataset(x_val, pos_val, y_val, torch.arange(len(y_val)))
    test_dataset = TensorDataset(x_test, pos_test, y_test, torch.arange(len(y_test)))

    return train_dataset, val_dataset, test_dataset


def load_pdbbind30_dataset(use_residue_features=False, **kwargs):
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

    train_dataset = PDBBindDataset(x_train, pos_train, y_train, resnames=resnames_train, use_residue_features=use_residue_features)
    val_dataset = PDBBindDataset(x_val, pos_val, y_val, resnames=resnames_val, use_residue_features=use_residue_features)
    test_dataset = PDBBindDataset(x_test, pos_test, y_test, resnames=resnames_test, use_residue_features=use_residue_features)

    return train_dataset, val_dataset, test_dataset


def load_pdbbind60_dataset(use_residue_features=False, **kwargs):
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

    train_dataset = PDBBindDataset(x_train, pos_train, y_train, resnames=resnames_train, use_residue_features=use_residue_features)
    val_dataset = PDBBindDataset(x_val, pos_val, y_val, resnames=resnames_val, use_residue_features=use_residue_features)
    test_dataset = PDBBindDataset(x_test, pos_test, y_test, resnames=resnames_test, use_residue_features=use_residue_features)

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
    use_coord_pairs=False,
    use_rmsf=False,
    use_ordering=False,
    verbose=False,
    **kwargs,
):
    if verbose:
        print(f"loading gbsa dock dataset from: {data_dir}, using target {target} for training.")

    data_dir = Path(data_dir)

    split_configs = {
        "train": ("train_split", train_pose_list),
        "val": ("val_split", val_pose_list),
        "test": ("test_split", test_pose_list),
    }

    requested_splits = [split for split, (key, _) in split_configs.items() if key in kwargs]
    if not requested_splits:
        raise ValueError("At least one of train_split, val_split, or test_split must be provided in kwargs")

    results = {}
    for split, (split_key, pose_list) in split_configs.items():
        if split_key in kwargs:
            pdb_list = pd.read_csv(kwargs[split_key], header=None)[0].values.tolist()
            file_list = [
                x.name
                for x in data_dir.glob("*.npy")
                if x.name.split("-")[0] in pdb_list
                and int(x.name.split("-")[2].split("_")[0].replace("p", "")) in pose_list
            ]

            if verbose:
                print(f"{split} split: found {len(file_list)} files")

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
                    use_coord_pairs=use_coord_pairs,
                    use_rmsf=use_rmsf,
                    use_ordering=use_ordering,
                )

                if len(dataset) > 0:
                    datasets.append(dataset)

            results[split] = datasets

            if verbose:
                print(f"{split} split: created {len(datasets)} datasets")

    return results


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

            train_dataset, test_dataset = random_split(train_dataset, [train_frac, 1 - train_frac])
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
    data_dict = {}
    flags = objective_flags(config)

    if hasattr(config, "train_split"):
        data_dict["train"] = MisatoDataset(
            data_path="/g/g13/jones289/workspace/pretrain_md/sur_gbsa/misato/data/misato/misato_train.pt",
            pdb_path=config.train_split,
            max_len=config.gbsa_max_len,
            use_residue_features=getattr(config, "use_residue_features", False),
            verbose=getattr(config, "verbose", False),
            use_coord_pairs=flags["use_coord_pairs"],
            use_rmsf=flags["use_rmsf"],
            use_ordering=flags["use_ordering"],
        )

    if hasattr(config, "val_split"):
        data_dict["val"] = MisatoDataset(
            data_path="/g/g13/jones289/workspace/pretrain_md/sur_gbsa/misato/data/misato/misato_val.pt",
            pdb_path=config.val_split,
            max_len=config.gbsa_max_len,
            use_residue_features=getattr(config, "use_residue_features", False),
            verbose=getattr(config, "verbose", False),
            use_coord_pairs=flags["use_coord_pairs"],
            use_rmsf=flags["use_rmsf"],
            use_ordering=flags["use_ordering"],
        )

    if hasattr(config, "test_split"):
        data_dict["test"] = MisatoDataset(
            data_path="/g/g13/jones289/workspace/pretrain_md/sur_gbsa/misato/data/misato/misato_test.pt",
            pdb_path=config.test_split,
            max_len=config.gbsa_max_len,
            use_residue_features=getattr(config, "use_residue_features", False),
            verbose=getattr(config, "verbose", False),
            use_coord_pairs=flags["use_coord_pairs"],
            use_rmsf=flags["use_rmsf"],
            use_ordering=flags["use_ordering"],
        )

    return data_dict



def load_fepplus_dataset(config, **kwargs):
    data_root = Path(config.data_dir)

    data_kwargs = {
        "root_dir": data_root,
        "max_len": getattr(config, "gbsa_max_len", 600),
        "use_residue_features": getattr(config, "use_residue_features", False),
        "verbose": getattr(config, "verbose", False),
        "residue_mode": getattr(config, "residue_mode", "full"),
        "pocket_cutoff": getattr(config, "pocket_cutoff", 6.0),
    }

    train_dataset = FEPPlusDataset(**data_kwargs)
    val_dataset = FEPPlusDataset(**data_kwargs)
    test_dataset = FEPPlusDataset(**data_kwargs)

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }

if __name__ == "__main__":
    pass