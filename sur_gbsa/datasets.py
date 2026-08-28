import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import MDAnalysis as mda
from MDAnalysis.core.universe import Merge
from tqdm import tqdm
import h5py
import pickle
from typing_extensions import deprecated
from rdkit import Chem

atom_dict = {
    "C": 6,
    "H": 1,
    "O": 8,
    "N": 7,
    "S": 9,
    "P": 10,
    "F": 11,
    "I": 12,
    "B": 13,
    "CL": 14,  # Chlorine (not in training data)
    "BR": 15,  # Bromine (not in training data)
}

# todo (derek): need to rely upon a commonly used set of 3 letter keys from an existing python package if available
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HIE", "HID", "HIP", "CYX", "ACE", "NME",
}

RESIDUE_VOCAB = {
    "PAD": 0,
    "PROT": 1,
    "LIG": 2,
    "ALA": 3,
    "ARG": 4,
    "ASN": 5,
    "ASP": 6,
    "CYS": 7,
    "GLN": 8,
    "GLU": 9,
    "GLY": 10,
    "HIS": 11,
    "ILE": 12,
    "LEU": 13,
    "LYS": 14,
    "MET": 15,
    "PHE": 16,
    "PRO": 17,
    "SER": 18,
    "THR": 19,
    "TRP": 20,
    "TYR": 21,
    "VAL": 22,
    "HIE": 23,
    "HID": 24,
    "HIP": 25,
    "CYX": 26,
    "ACE": 27,
    "NME": 28,
}


def normalize_residue_name(resname, residue_mode="full"):
    if resname is None or resname == "":
        return "PAD"

    resname = str(resname).upper()

    if resname == "LIG":
        return "LIG"

    if resname == "PROT":
        return "PROT"

    if resname in STANDARD_RESIDUES:
        return resname if residue_mode == "full" else "PROT"

    return "LIG"



def compute_rmsf_from_positions(pos_seq, mask=None, per_residue=False, residue_ids=None):
    """
    pos_seq: [T, N, 3]
    mask:    [N] or [T, N], optional
    residue_ids: [N], optional integer grouping if per_residue=True
    """
    pos_seq = pos_seq.float()

    if mask is not None:
        if mask.dim() == 1:
            pos_seq = pos_seq[:, mask]
            if residue_ids is not None:
                residue_ids = residue_ids[mask]
        elif mask.dim() == 2:
            valid = mask.any(dim=0)
            pos_seq = pos_seq[:, valid]
            if residue_ids is not None:
                residue_ids = residue_ids[valid]

    mean_pos = pos_seq.mean(dim=0, keepdim=True)
    sq_disp = (pos_seq - mean_pos).pow(2).sum(dim=-1)
    rmsf = torch.sqrt(sq_disp.mean(dim=0) + 1e-8)

    if not per_residue:
        return rmsf

    if residue_ids is None:
        raise ValueError("residue_ids is required when per_residue=True")

    unique_res = torch.unique(residue_ids)
    out = []
    for r in unique_res:
        idx = residue_ids == r
        out.append(rmsf[idx].mean())
    return torch.stack(out)


def load_misato_split(
    path="/p/vast1/mldrug/misato/misato_MD.hdf5",
    pdb_list=None,
    output_file=None,
    dataset_name="misato",
    max_len=600,
    verbose=False
):
    """Load MISATO dataset for given PDB IDs with max length enforcement"""

    if pdb_list is None:
        pdb_list = []

    peptide_list = pd.read_csv(
        "/g/g13/jones289/workspace/pretrain_md/sur_gbsa/misato/peptides.txt",
        header=None
    )[0].values.tolist()

    with open(
        "/g/g13/jones289/workspace/pretrain_md/sur_gbsa/misato/atoms_residue_map.pickle",
        "rb"
    ) as handle:
        res_map = pickle.load(handle)

    f = h5py.File(path, "r")
    file_pdbids = list(f)

    x_all, pos_all, y_all, resnames_all = [], [], [], []
    pdb_ids_processed = []
    skipped_counts = {"peptide": 0, "missing": 0, "no_valid_atoms": 0, "truncated": 0}

    pdb_iter = tqdm(pdb_list, desc="Processing PDB IDs") if verbose else pdb_list

    for pdbid in pdb_iter:
        if pdbid in peptide_list:
            skipped_counts["peptide"] += 1
            continue
        elif pdbid.upper() not in file_pdbids:
            skipped_counts["missing"] += 1
            continue

        try:
            from sur_gbsa.ProtMD.data.extract_md import process_gbsa_frames
            dataset = f[pdbid.upper()]
            pbsa = torch.tensor(
                process_gbsa_frames(
                    f"/p/vast1/jones289/misato/mmgbsa_alignment/{pdbid.upper()}/{pdbid.upper()}_FINAL_RESULTS_MMPBSA_perframe.dat"
                )["DELTA TOTAL"].values
            )
        except FileNotFoundError as e:
            if verbose:
                print(e)
            continue

        atom_element = dataset["atoms_element"][:]
        atom_pos = dataset["trajectory_coordinates"][:]
        atom_residue = dataset["atoms_residue"][:]

        lig_mask = np.array([res_map[x] == "MOL" for x in atom_residue])

        if not lig_mask.any():
            skipped_counts["no_valid_atoms"] += 1
            continue

        for frame_idx in range(atom_pos.shape[0]):
            ligand_coords = atom_pos[frame_idx, lig_mask]
            protein_coords = atom_pos[frame_idx, ~lig_mask]

            from scipy.spatial.distance import cdist
            distances = cdist(protein_coords, ligand_coords).min(axis=1)

            protein_pocket_mask = distances < 6.0
            protein_pocket_indices = np.where(~lig_mask)[0][protein_pocket_mask]
            ligand_indices = np.where(lig_mask)[0]

            num_ligand = len(ligand_indices)
            num_pocket_protein = len(protein_pocket_indices)
            total_atoms = num_ligand + num_pocket_protein

            if total_atoms > max_len:
                num_protein_to_keep = max_len - num_ligand
                if num_protein_to_keep < 0:
                    skipped_counts["truncated"] += 1
                    continue
                protein_pocket_indices = protein_pocket_indices[:num_protein_to_keep]
                skipped_counts["truncated"] += 1

            combined_indices = np.concatenate([ligand_indices, protein_pocket_indices])

            filtered_pos = atom_pos[frame_idx, combined_indices]
            filtered_atoms = atom_element[combined_indices]
            filtered_residues = atom_residue[combined_indices]

            residue_names = [res_map[res_id] for res_id in filtered_residues]

            x_tmp = torch.tensor(filtered_atoms, dtype=torch.long)
            pos_tmp = torch.tensor(filtered_pos, dtype=torch.float32)

            if len(x_tmp) < max_len:
                pad_len = max_len - len(x_tmp)
                x_tmp = torch.cat([x_tmp, torch.zeros(pad_len, dtype=torch.long)])
                pos_tmp = torch.cat([pos_tmp, torch.zeros(pad_len, 3, dtype=torch.float32)])
                residue_names.extend([""] * pad_len)

            x_all.append(x_tmp)
            pos_all.append(pos_tmp)
            resnames_all.append(residue_names)
            pdb_ids_processed.append(f"{pdbid}_frame{frame_idx}")
            y_all.append(pbsa[frame_idx])

    f.close()

    if verbose:
        print("\nProcessing summary:")
        print(f"  Input PDB IDs: {len(pdb_list)}")
        print(f"  Skipped (peptides): {skipped_counts['peptide']}")
        print(f"  Skipped (missing): {skipped_counts['missing']}")
        print(f"  Skipped (no valid atoms/frames): {skipped_counts['no_valid_atoms']}")
        print(f"  Truncated frames: {skipped_counts['truncated']}")
        print(f"  Successfully processed: {len(x_all)} frames")

    if len(x_all) == 0:
        print("ERROR: No valid data processed!")
        return None

    x = torch.stack(x_all)
    pos = torch.stack(pos_all)
    y = torch.stack(y_all) / -10

    if output_file:
        out_path = Path(output_file)
    else:
        out_dir = Path(f"data/{dataset_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_name}_processed.pt"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "x": x,
        "pos": pos,
        "y": y,
        "pdb_ids": pdb_ids_processed,
        "resnames": resnames_all,
        "num_samples": len(x),
    }

def parse_mol2_names(mol2_file):
    """Extract molecule names from multi-structure MOL2 file."""
    names = []
    with open(mol2_file, 'r') as f:
        in_molecule_block = False
        for line in f:
            if line.startswith('@<TRIPOS>MOLECULE'):
                in_molecule_block = True
                continue
            if in_molecule_block:
                names.append(line.strip())
                in_molecule_block = False
    return names


class BaseMDDataset(Dataset, ABC):
    """Base class for MD datasets with common functionality"""
    
    def __init__(
        self,
        max_len=600,
        use_residue_features=False,
        rmsd_thresh=2.0,
        verbose=False,
        residue_mode="full",
    ):
        self.max_len = max_len
        self.use_residue_features = use_residue_features
        self.rmsd_thresh = rmsd_thresh
        self.verbose = verbose
        self.residue_mode = residue_mode
        
        self.x_0 = None
        self.pos = None
        self.y = None
        self.resname_indices = None
        self.rmsd = None
        self.frame_num = None
        self.file_list = None
        self.metadata_list = None

    @abstractmethod
    def _load_data(self):
        pass

    def _normalize_residue_names(self, resname_list):
        return [
            normalize_residue_name(res, residue_mode=self.residue_mode)
            for res in resname_list
        ]

    def _residue_to_indices(self, resname_list):
        normalized = self._normalize_residue_names(resname_list)
        return torch.tensor(
            [RESIDUE_VOCAB.get(res, RESIDUE_VOCAB["LIG"]) for res in normalized],
            dtype=torch.long
        )

    def __getitem__(self, idx):
        item = {
            "x": self.x_0[idx],
            "pos": self.pos[idx],
            "y": self.y[idx],
            "idx": torch.tensor(idx),
        }


        if self.use_residue_features and self.resname_indices is not None:
            item["res_feats"] = self.resname_indices[idx] 

        if self.rmsd is not None:
            item["rmsd"] = self.rmsd[idx]

        if self.frame_num is not None:
            item["frame_num"] = self.frame_num[idx]

        if self.file_list is not None:
            item["file_list"] = self.file_list[idx]

        if self.metadata_list is not None:
            item["metadata"] = self.metadata_list[idx]

        return item

    def __len__(self):
        return len(self.x_0)



class MisatoDataset(BaseMDDataset):
    def __init__(
        self,
        data_path,
        pdb_path,
        max_len=600,
        use_residue_features=False,
        rmsd_thresh=2.0,
        verbose=False,
        use_coord_pairs=False,
        use_rmsf=False,
        use_ordering=False,
        residue_mode="full",
    ):
        super().__init__(
            max_len=max_len,
            use_residue_features=use_residue_features,
            rmsd_thresh=rmsd_thresh,
            verbose=verbose,
            residue_mode=residue_mode,
        )

        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.pdb_path = Path(pdb_path) if isinstance(pdb_path, str) else pdb_path

        self.use_coord_pairs = use_coord_pairs
        self.use_rmsf = use_rmsf
        self.use_ordering = use_ordering

        self.pos_t1 = None
        self.rmsf_target = None
        self.order_target = None

        self._load_data()
    



    def _load_data(self):
        pdb_list = pd.read_csv(self.pdb_path, header=None)[0].values.tolist()
        data_dict = load_misato_split(pdb_list=pdb_list, max_len=self.max_len)

        self.x_0 = data_dict["x"]
        self.pos = data_dict["pos"]
        self.y = data_dict["y"]
        self.metadata_list = data_dict["pdb_ids"]
        self.file_list = data_dict["pdb_ids"]

        self.frame_num = torch.tensor(
            [int(x.split("frame")[-1]) for x in self.metadata_list],
            dtype=torch.long,
        )

        if self.use_residue_features and "resnames" in data_dict:
            self.resname_indices = torch.stack([
                self._residue_to_indices(resnames)
                for resnames in data_dict["resnames"]
            ])
        else:
            self.resname_indices = None
            self.use_residue_features = False

        if self.use_coord_pairs:
            self.pos_t1 = torch.zeros_like(self.pos)
            if len(self.pos) > 1:
                self.pos_t1[:-1] = self.pos[1:]

        if self.use_ordering:
            self.order_target = torch.zeros(len(self.pos), dtype=torch.float32)
            if len(self.pos) > 1:
                self.order_target[:-1] = 1.0

        if self.use_rmsf:
            self.rmsf_target = self._compute_framewise_rmsf_targets(self.pos)

        if self.verbose:
            print(f"Loaded MISATO dataset: {len(self)} samples")

    def _compute_framewise_rmsf_targets(self, pos):
        targets = []
        for i in range(len(pos)):
            sample_pos = pos[i]
            valid = (sample_pos.abs().sum(dim=-1) > 0)
            if valid.any():
                rmsf_vec = compute_rmsf_from_positions(
                    sample_pos.unsqueeze(0),
                    mask=valid
                )
                targets.append(rmsf_vec.mean())
            else:
                targets.append(torch.tensor(0.0))
        return torch.stack(targets)


    def __getitem__(self, i):
        item = {
            "x": self.x_0[i],
            "pos": self.pos[i],
            "y": self.y[i],
            "pose_list": 0,
            "idx": torch.tensor(i),
            "frame_num": self.frame_num[i],
            "metadata": self.metadata_list[i],
        }

        if self.use_residue_features and self.resname_indices is not None:
            item["res_feats"] = self.resname_indices[i]

        if self.pos_t1 is not None:
            item["pos_t1"] = self.pos_t1[i]

        if self.rmsf_target is not None:
            item["rmsf_target"] = self.rmsf_target[i]

        if self.order_target is not None:
            item["order_target"] = self.order_target[i]

        return item


    def __getmetadata__(self, i):
        return self.metadata_list[i]

class GBSAMDDataset(BaseMDDataset):
    """GBSA MD dataset - loads from individual .npy files"""

    def __init__(
        self,
        backprop,
        pdb_list,
        data_dir,
        target="GBSA",
        max_frames=1000,
        max_time_step_prop=0.9,
        max_len=600,
        use_residue_features=False,
        rmsd_thresh=2.0,
        label_process="sign-flip",
        verbose=False,
        use_coord_pairs=False,
        use_rmsf=False,
        use_ordering=False,
        residue_mode="full",
        **kwargs,
    ):

        # super().__init__(max_len, use_residue_features, rmsd_thresh, verbose)
        super().__init__(
            max_len=max_len,
            use_residue_features=use_residue_features,
            rmsd_thresh=rmsd_thresh,
            verbose=verbose,
            residue_mode=residue_mode,
        )

        self.backprop = backprop
        self.pdb_list = pdb_list
        self.data_dir = Path(data_dir)
        self.target = target
        self.max_frames = max_frames
        self.max_time_step_prop = max_time_step_prop
        self.label_process = label_process
        self.use_coord_pairs = use_coord_pairs
        self.use_rmsf = use_rmsf
        self.use_ordering = use_ordering

        self.pose_list = []
        self.pdb_to_idx = {}

        self._load_data()


    def _load_data(self):
        pos_list = []
        x_0_list = []
        resname_indices_list = []
        y_list = []
        rmsd_list = []
        frame_num_list = []
        file_list = []

        pos_t1_list = []
        rmsf_target_list = []
        order_target_list = []

        pdbid_count = 1

        for file in self.pdb_list:
            if not file.endswith(".npy"):
                if self.verbose:
                    print(f"Warning: {file} is not a numpy file")
                continue

            pdbid = file.split("-")[0]
            if pdbid not in self.pdb_to_idx:
                self.pdb_to_idx[pdbid] = pdbid_count
                pdbid_count += 1

            d = np.load(self.data_dir / file, allow_pickle=True).item()

            file_pos = torch.tensor(d["R"])
            pos_seq = file_pos.float()  # [T, N, 3]
            z = torch.tensor([atom_dict[x] for x in d["z"]])
            z = z.repeat(file_pos.shape[0], 1)

            pose_num = int(file.split("_")[0].split("-")[-1][1])
            self.pose_list.append([pose_num] * len(file_pos))

            if self.use_residue_features:
                res_indices = self._residue_to_indices(d["res-name"])
                res_indices = res_indices.repeat(file_pos.shape[0], 1)
                resname_indices_list.append(res_indices)

            if self.target == "GBSA":
                y = d["DELTA TOTAL"]
            elif self.target == "GBSA_mean":
                y = d["GBSA_mean"] * np.ones(len(file_pos))
                y = y.astype(np.float32)
            elif self.target == "PLAS_20k":
                if len(d["PLAS_20k_data"]["DELTA_TOTAL"]) > 0:
                    y = d["PLAS_20k_data"]["DELTA_TOTAL"].values * np.ones(len(file_pos))
                    y = y.astype(np.float32)
                else:
                    self.x_0 = []
                    return

            y_list.append(torch.from_numpy(y))

            rmsd = d["rmsd"]
            rmsd_list.append(torch.from_numpy(rmsd))

            frame_num = torch.tensor(list(range(len(y))), dtype=torch.int32)
            frame_num_list.append(frame_num)

            file_list.append([file] * len(y))

            if self.use_coord_pairs and len(pos_seq) > 1:
                pos_t1_list.append(pos_seq[1:])
                order_target_list.append(torch.ones(len(pos_seq) - 1, dtype=torch.float32))

            if self.use_rmsf:
                traj_rmsf = compute_rmsf_from_positions(pos_seq)
                rmsf_target_list.append(traj_rmsf.repeat(pos_seq.shape[0], 1))

            pos_list.append(file_pos)
            x_0_list.append(z)

        self.x_0 = pad_sequence(
            [x.transpose(1, 0) for x in x_0_list],
            batch_first=True,
            padding_value=0,
        ).transpose(1, 2)

        self.pos = pad_sequence(
            [x.transpose(1, 0) for x in pos_list],
            batch_first=True,
            padding_value=0,
        ).transpose(1, 2)

        if self.use_residue_features:
            self.resname_indices = pad_sequence(
                [x.transpose(1, 0) for x in resname_indices_list],
                batch_first=True,
                padding_value=0,
            ).transpose(1, 2)

        self.x_0 = self.x_0[:, :self.max_frames, :]
        self.pos = self.pos[:, :self.max_frames, :, :]
        if self.use_residue_features:
            self.resname_indices = self.resname_indices[:, :self.max_frames, :]

        self.pose_list = [p[:self.max_frames] for p in self.pose_list]
        y_list = [y[:self.max_frames] for y in y_list]
        rmsd_list = [r[:self.max_frames] for r in rmsd_list]
        frame_num_list = [f[:self.max_frames] for f in frame_num_list]
        file_list = [f[:self.max_frames] for f in file_list]

        if self.use_coord_pairs:
            self.pos_t1 = torch.cat(pos_t1_list, dim=0) if len(pos_t1_list) > 0 else None
        else:
            self.pos_t1 = None

        if self.use_ordering:
            self.order_target = torch.cat(order_target_list, dim=0) if len(order_target_list) > 0 else None
        else:
            self.order_target = None

        if self.use_rmsf:
            rmsf_target_list = [r[:self.max_frames] for r in rmsf_target_list]

        n_frames = self.x_0.shape[1]
        self._apply_temporal_split(n_frames, y_list, rmsd_list, frame_num_list, file_list)

        self._reshape_to_flat()
        self._pad_to_max_len()

        self.y = self.y / -10

        if self.use_coord_pairs:
            self.pos_t1 = torch.cat(pos_t1_list).reshape(-1, self.pos.shape[1], 3) if len(pos_t1_list) > 0 else None
            if self.pos_t1 is not None:
                self.pos_t1 = self.pos_t1.reshape(-1, self.pos_t1.shape[-2], 3)
        else:
            self.pos_t1 = None

        if self.use_rmsf:
            self.rmsf_target = torch.cat(rmsf_target_list) if len(rmsf_target_list) > 0 else None
        else:
            self.rmsf_target = None

        if self.use_ordering:
            self.order_target = torch.cat(order_target_list) if len(order_target_list) > 0 else None
        else:
            self.order_target = None

        if self.verbose:
            print(f"Loaded GBSA dataset: {len(self)} samples")

    def _apply_temporal_split(self, n_frames, y_list, rmsd_list, frame_num_list, file_list):
        if self.backprop is True:
            cutoff = int(n_frames * self.max_time_step_prop)
            self.x_0 = self.x_0[:, :cutoff, -self.max_len:]
            self.pos = self.pos[:, :cutoff, -self.max_len:, :]
            

            self.y = torch.cat([y[:cutoff] for y in y_list])
            self.rmsd = torch.cat([r[:cutoff] for r in rmsd_list])
            self.frame_num = torch.cat([f[:cutoff] for f in frame_num_list])
            self.file_list = [item for sublist in file_list for item in sublist[:cutoff]]
            self.pose_list = [item for sublist in self.pose_list for item in sublist[:cutoff]]

        elif self.backprop is False:
            cutoff = int(n_frames * self.max_time_step_prop)
            self.x_0 = self.x_0[:, cutoff:, -self.max_len:]
            self.pos = self.pos[:, cutoff:, -self.max_len:, :]
            

            self.y = torch.cat([y[cutoff:] for y in y_list])
            self.rmsd = torch.cat([r[cutoff:] for r in rmsd_list])
            self.frame_num = torch.cat([f[cutoff:] for f in frame_num_list])
            self.file_list = [item for sublist in file_list for item in sublist[cutoff:]]
            self.pose_list = [item for sublist in self.pose_list for item in sublist[cutoff:]]
        else:
            self.x_0 = self.x_0[:, :, -self.max_len:]
            self.pos = self.pos[:, :, -self.max_len:, :]
            

            self.y = torch.cat(y_list)
            self.rmsd = torch.cat(rmsd_list)
            self.frame_num = torch.cat(frame_num_list)
            self.file_list = [item for sublist in file_list for item in sublist]
            self.pose_list = [item for sublist in self.pose_list for item in sublist]
        
        if self.use_residue_features:
            self.resname_indices = self.resname_indices[:, cutoff:, -self.max_len:]


    def _reshape_to_flat(self):
        self.x_0 = self.x_0.reshape(-1, self.x_0.shape[2])
        self.pos = self.pos.reshape(-1, self.pos.shape[2], 3)

        if self.use_residue_features and self.resname_indices is not None:
            self.resname_indices = self.resname_indices.reshape(
                -1, self.resname_indices.shape[2]
            )    

    def _pad_to_max_len(self):
        seq_len = self.x_0.size(1)
        if seq_len < self.max_len:
            pad_size = self.max_len - seq_len
            self.x_0 = torch.cat([
                self.x_0,
                torch.zeros((self.x_0.size(0), pad_size), dtype=self.x_0.dtype)
            ], dim=1)

            self.pos = torch.cat([
                self.pos,
                torch.zeros((self.pos.size(0), pad_size, 3), dtype=self.pos.dtype)
            ], dim=1)

            
    def __getitem__(self, i):
        out = {
            "x": self.x_0[i],
            "pos": self.pos[i],
            "y": self.y[i],
            "idx": torch.tensor(i),
            "pose_list": self.pose_list[i],
            "rmsd": self.rmsd[i],
            "frame_num": self.frame_num[i],
            "file_list": self.file_list[i],
        }

        if self.use_residue_features and self.resname_indices is not None:
            out["res_feats"] = self.resname_indices[i]

        if self.pos_t1 is not None:
            out["pos_t1"] = self.pos_t1[i]

        if self.rmsf_target is not None:
            out["rmsf_target"] = self.rmsf_target[i]

        if self.order_target is not None:
            out["order_target"] = self.order_target[i]

        return out


    def __len__(self):
        return len(self.x_0)

    def __getposenum__(self, i):
        return self.pose_list[i]

    def __getrmsd__(self, i):
        return self.rmsd[i]

    def __getframenum__(self, i):
        return self.frame_num[i]

    def __getfilename__(self, i):
        return self.file_list[i]

    def __getmetadata__(self, i):
        return self.__getfilename__(i)

class PDBBindDataset(BaseMDDataset):
    """Wrapper for PDBbind data to support residue features"""
    
    def __init__(
        self,
        x,
        pos,
        y,
        resnames=None,
        use_residue_features=False,
        residue_mode="full",
    ):
        super().__init__(
            max_len=pos.shape[1] if hasattr(pos, "shape") and len(pos.shape) > 1 else 600,
            use_residue_features=use_residue_features,
            residue_mode=residue_mode,
        )

        self.x = x
        self.pos = pos
        self.y = y
        self.use_residue_features = use_residue_features and resnames is not None

        if self.use_residue_features:
            self.resname_indices = torch.stack([
                self._residue_to_indices(sample_resnames)
                for sample_resnames in resnames
            ])


    def _load_data(self):
        pass

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        item = {
            "x": self.x[idx],
            "pos": self.pos[idx],
            "y": self.y[idx],
            "idx": torch.tensor(idx),
        }

        if self.use_residue_features and self.resname_indices is not None:
            item["res_feats"] = self.resname_indices[idx]

        return item





# FEP+ Dataset (ross et. al 2023)

class FEPPlusDataset(BaseMDDataset):
    """
    Static FEP+ benchmark dataset.

    Layout:
      root_dir/fep_benchmark_inputs/structure_inputs/<group>/
        *_protein.pdb
        *_ligands.sdf

    One sample per ligand record in the SDF, paired with the matching protein.
    """

    def __init__(
        self,
        root_dir,
        max_len=600,
        use_residue_features=False,
        verbose=False,
        residue_mode="full",
        pocket_cutoff=6.0,
    ):
        super().__init__(
            max_len=max_len,
            use_residue_features=use_residue_features,
            verbose=verbose,
            residue_mode=residue_mode,
        )
        self.root_dir = Path(root_dir)
        self.pocket_cutoff = pocket_cutoff

        self._load_data()


    def _find_group_files(self):
        groups_root = self.root_dir / "fep_benchmark_inputs" / "structure_inputs"

        # groups_root = self.root_dir
        for group_dir in sorted(groups_root.iterdir()):
            if not group_dir.is_dir():
                continue

            protein_files = sorted(group_dir.glob("*_protein.pdb"))
            for protein_file in protein_files:
                prefix = protein_file.name.replace("_protein.pdb", "")
                ligand_file = group_dir / f"{prefix}_ligands.sdf"

                if not ligand_file.exists():
                    continue

                # import pdb; pdb.set_trace()
                metadata = f"{group_dir.name}-{ligand_file.name.replace('_ligands.sdf', '')}"
                yield metadata, protein_file, ligand_file

    def _load_protein(self, pdb_file):
        u = mda.Universe(str(pdb_file))
        protein = u.select_atoms("protein")

        if len(protein) == 0:
            raise ValueError(f"No protein atoms found in {pdb_file}")

        atom_symbols = np.array([
            (a.element if a.element else a.name[0]) for a in protein
        ], dtype=object)
        coords = protein.positions.astype(np.float32)
        resnames = [a.resname for a in protein.atoms]
        return atom_symbols, coords, resnames

    def _load_ligands(self, sdf_file):
        suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        ligands = []

        for mol in suppl:
            if mol is None:
                continue
            if not mol.HasProp("r_exp_dg"):
                continue
            if mol.GetNumConformers() == 0:
                continue
            ligands.append(mol)

        return ligands


    def _ligand_features(self, mol):
        conf = mol.GetConformer()
        coords = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        atom_symbols = np.array([a.GetSymbol() for a in mol.GetAtoms()], dtype=object)
        return atom_symbols, coords



    def _atom_to_z(self, atom_symbols):
        return torch.tensor(
            [atom_dict.get(str(sym).upper(), 0) for sym in atom_symbols],
            dtype=torch.long,
        )

    def _trim_pocket(self, protein_symbols, protein_coords, protein_resnames, lig_coords):
        from scipy.spatial.distance import cdist

        dist = cdist(protein_coords, lig_coords).min(axis=1)
        keep = dist < self.pocket_cutoff

        protein_symbols = protein_symbols[keep]
        protein_coords = protein_coords[keep]
        protein_resnames = np.array(protein_resnames, dtype=object)[keep].tolist()

        return protein_symbols, protein_coords, protein_resnames

    def _pad_or_trim(self, x, pos, resnames):
        if len(x) > self.max_len:
            x = x[: self.max_len]
            pos = pos[: self.max_len]
            resnames = resnames[: self.max_len]
        elif len(x) < self.max_len:
            pad_len = self.max_len - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            pos = torch.cat([pos, torch.zeros(pad_len, 3, dtype=torch.float32)])
            resnames = resnames + [""] * pad_len

        return x, pos, resnames

    def _load_data(self):
        x_list = []
        pos_list = []
        y_list = []
        resname_list = []
        metadata_list = []

        for group_name, protein_file, ligand_file in self._find_group_files():
            protein_syms, protein_coords, protein_resnames = self._load_protein(protein_file)
            ligands = self._load_ligands(ligand_file)

            for lig_idx, mol in enumerate(ligands):
                lig_syms, lig_coords = self._ligand_features(mol)

                pocket_syms, pocket_coords, pocket_resnames = self._trim_pocket(
                    protein_syms,
                    protein_coords,
                    protein_resnames,
                    lig_coords,
                )

                all_syms = np.concatenate([pocket_syms, lig_syms])
                all_coords = np.concatenate([pocket_coords, lig_coords], axis=0)
                all_resnames = pocket_resnames + ["LIG"] * len(lig_syms)

                x = self._atom_to_z(all_syms)
                pos = torch.tensor(all_coords, dtype=torch.float32)
                y = torch.tensor(float(mol.GetProp("r_exp_dg")), dtype=torch.float32)

                x, pos, all_resnames = self._pad_or_trim(x, pos, all_resnames)

                x_list.append(x)
                pos_list.append(pos)
                y_list.append(y)
                resname_list.append(all_resnames)
                metadata_list.append(
                    {
                        "group": group_name,
                        "protein_file": str(protein_file),
                        "ligand_file": str(ligand_file),
                        "ligand_index": lig_idx,
                        "mol_name": mol.GetProp("_Name") if mol.HasProp("_Name") else "",
                    }
                )

        self.x_0 = torch.stack(x_list)
        self.pos = torch.stack(pos_list)
        self.y = torch.stack(y_list)
        self.metadata_list = metadata_list

        if self.use_residue_features:
            self.resname_indices = torch.stack([
                self._residue_to_indices(rnames) for rnames in resname_list
            ])
        else:
            self.resname_indices = None
            self.use_residue_features = False

        if self.verbose:
            print(f"Loaded FEPPlusDataset: {len(self)} samples")

    def __getitem__(self, idx):
        item = {
            "x": self.x_0[idx],
            "pos": self.pos[idx],
            "y": self.y[idx],
            "idx": torch.tensor(idx),
            "metadata": self.metadata_list[idx],
        }

        if self.use_residue_features and self.resname_indices is not None:
            item["res_feats"] = self.resname_indices[idx]

        return item

    def __len__(self):
        return len(self.x_0)

    def __getmetadata__(self, idx):
        return self.metadata_list[idx]





























# OpenFE Dataset (gowers et.al 2023)

from pathlib import Path
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
# import MDAnalysis as mda

# from sur_gbsa.datasets.base import BaseMDDataset
# from sur_gbsa.data import atom_dict


class IndustryBenchmarksDataset(BaseMDDataset):
    def __init__(
        self,
        root_dir,
        max_len=600,
        use_residue_features=False,
        verbose=False,
        residue_mode="full",
        pocket_cutoff=6.0,
        canon_csv_path=None,
    ):
        super().__init__(
            max_len=max_len,
            use_residue_features=use_residue_features,
            verbose=verbose,
            residue_mode=residue_mode,
        )

        self.root_dir = Path(root_dir)
        self.original_structures_root = (
            self.root_dir
            / "industry_benchmarks"
            / "input_structures"
            / "original_structures"
        )
        self.pocket_cutoff = pocket_cutoff

        self.canon_csv_path = Path(canon_csv_path) if canon_csv_path else None
        if self.canon_csv_path is None:
            raise ValueError("canon_csv_path is required for IndustryBenchmarksDataset")
        self.canon_df = pd.read_csv(self.canon_csv_path)

        self.group_map = {
            "charge_annihilation_set": "charge_annhil",
            "fragments": "fragments",
            "water_set": "waterset",
            "merck": "merck",
            "janssen_bace": "janssen_bace",
            "jacs_set": "jacs_set",
            "mcs_docking_set": "mcs_docking",
            "miscellaneous_set": "misc",
            "scaffold_hopping_set": "scaffold_hopping",
        }

        self.system_map = {
            ("charge_annihilation_set", "cdk2"): "cdk2_ligands.sdf",
            ("charge_annihilation_set", "dlk"): "dlk_ligands.sdf",
            ("charge_annihilation_set", "btk"): "btk_ligands.sdf",
            ("charge_annihilation_set", "egfr"): "egfr_ligands.sdf",
            ("charge_annihilation_set", "ephx2"): "ephx2_ligands.sdf",
            ("charge_annihilation_set", "irak4_s2"): "irak4_s2_ligands.sdf",
            ("charge_annihilation_set", "irak4_s3"): "irak4_s3_ligands.sdf",
            ("charge_annihilation_set", "itk"): "itk_ligands.sdf",
            ("charge_annihilation_set", "jak1"): "jak1_ligands.sdf",
            ("charge_annihilation_set", "jnk1"): "jnk1_ligands.sdf",
            ("charge_annihilation_set", "ptp1b"): "ptp1b_ligands.sdf",
            ("charge_annihilation_set", "thrombin"): "thrombin_whole_map_ligands.sdf",
            ("charge_annihilation_set", "tyk2"): "tyk2_ligands.sdf",
            ("janssen_bace", "bace_ciordia_prospective"): "bace_ciordia_prospective_ligands.sdf",
            ("janssen_bace", "ciordia_retro"): "bace_ciordia_retro_ligands.sdf",
            ("janssen_bace", "keranen_p2"): "bace_keranen_p2_ligands.sdf",
            ("janssen_bace", "bace_p3_arg368_in"): "bace_p3_arg368_in_ligands.sdf",
            ("jacs_set", "bace"): "bace/bace_ligands.sdf",
            ("jacs_set", "cdk2"): "cdk2/cdk2_ligands.sdf",
            ("jacs_set", "jnk1"): "jnk1/jnk1_manual_flips_ligands.sdf",
            ("jacs_set", "mcl1"): "mcl1/mcl1_extra_flips_ligands.sdf",
            ("jacs_set", "p38"): "p38/p38_ligands.sdf",
            ("jacs_set", "ptp1b"): "ptp1b/ptp1b_ligands.sdf",
            ("jacs_set", "thrombin"): "thrombin/thrombin_core_ligands.sdf",
            ("jacs_set", "tyk2"): "tyk2/tyk2_ligands.sdf",
            ("fragments", "hsp90_2rings"): "hsp90_2rings/hsp90_frag_2rings_ligands.sdf",
            ("fragments", "hsp90_single_ring"): "hsp90_single_ring/hsp90_frag_single_ring_ligands.sdf",
            ("fragments", "jak2_set1"): "jak2_set1/jak2_set1_ligands.sdf",
            ("fragments", "jak2_set2"): "jak2_set2/jak2_set2_extra_ligands.sdf",
            ("fragments", "liga"): "liga/frag_liga_auto_ligands.sdf",
            ("fragments", "mcl1"): "mcl1/frag_mcl1_noweak_ligands.sdf",
            ("fragments", "mup1"): "mup1/frag_mup1_ligands.sdf",
            ("fragments", "p38"): "p38/frag_p38_ligands.sdf",
            ("fragments", "t4_lysozyme"): "t4_lysozyme/t4lysozyme_uvt_ligands.sdf",
            ("merck", "cdk8"): "cdk8/cdk8_5cei_new_helix_loop_extra_ligands.sdf",
            ("merck", "cmet"): "cmet/cmet_ligands.sdf",
            ("merck", "eg5"): "eg5/eg5_extraprotomers_ligands.sdf",
            ("merck", "hif2a"): "hif2a/hif2a_automap_ligands.sdf",
            ("merck", "pfkfb3"): "pfkfb3/pfkfb3_automap_ligands.sdf",
            ("merck", "shp2"): "shp2/shp2_ligands.sdf",
            ("merck", "syk"): "syk/syk_4puz_fullmap_ligands.sdf",
            ("merck", "tnks2"): "tnks2/tnks2_fullmap_ligands.sdf",
            ("miscellaneous_set", "btk"): "btk_extra_flip_ligands.sdf",
            ("miscellaneous_set", "cdk8"): "cdk8_koehler_ligands.sdf",
            ("miscellaneous_set", "galectin"): "galectin3_extra_ligands.sdf",
            ("miscellaneous_set", "faah"): "hfaah_ligands.sdf",
            ("miscellaneous_set", "hiv1_protease"): "hiv_prot_ekegren_ligands.sdf",
            ("mcs_docking_set", "hne"): "hne_ligands.sdf",
            ("mcs_docking_set", "renin"): "renin_customcore_ligands.sdf",
            ("scaffold_hopping_set", "bace1"): "Bace1_4zsp_ligands.sdf",
            ("scaffold_hopping_set", "factor_xa"): "Fxa_2ei8_ligands.sdf",
            ("water_set", "brd4"): "brd41_ASH106_ligands.sdf",
            ("water_set", "thrombin"): "throm_nozob_hip75_ligands.sdf",
            ("water_set", "chk1"): "chk1_ligands.sdf",
            ("water_set", "hsp90_kung"): "hsp90_kung_ligands.sdf",
            ("water_set", "hsp90_woodhead"): "hsp90_woodhead_ligands.sdf",
            ("water_set", "scyt_dehyd"): "scyt_dehyd_ligands.sdf",
            ("water_set", "taf12"): "taf12_ligands.sdf",
            ("water_set", "urokinase"): "urokinase_ligands.sdf",
        }

        self.aliases = {
            "28o": "28 out",
            "36o": "36 out",
            "37o": "37 out",
            "38o": "38 out",
            "39o": "39 out",
            "28i": "28 in",
            "29i": "29 in",
            "36i": "36 in",
            "37i": "37 in",
            "38i": "38 in",
            "39i": "39 in",
        }

        self.hard_aliases = {
            "janssen_bace_keranen_p2_28o": "28 out",
            "janssen_bace_keranen_p2_36o": "36 out",
            "janssen_bace_keranen_p2_37o": "37 out",
            "janssen_bace_keranen_p2_38o": "38 out",
            "janssen_bace_keranen_p2_39o": "39 out",
        }

        self._load_data()

    def _safe_str(self, x):
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        return str(x).strip()

    def _norm_key(self, x):
        return self._safe_str(x).lower().replace(" ", "_")

    def _norm_csv(self, s):
        if pd.isna(s):
            return ""
        s = str(s).lower().strip()
        s = re.sub(r"\([^)]*\)", "", s)
        s = s.replace("-", " ").replace("_", " ").replace("/", " ")
        s = re.sub(r"[^a-z0-9\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _score(self, a, b):
        return SequenceMatcher(None, self._norm_csv(a), self._norm_csv(b)).ratio()

    def _best_match(self, query, candidates):
        if not candidates:
            return None, 0.0
        scored = sorted(((self._score(query, c), c) for c in candidates), reverse=True)
        return scored[0][1], scored[0][0]

    def _apply_aliases(self, name):
        raw = str(name)
        return self.aliases.get(raw, raw)

    def _ligand_queries(self, raw_name):
        raw_name = self._apply_aliases(raw_name)
        parts = raw_name.split("_")

        queries = [raw_name]
        for n in range(min(4, len(parts)), 0, -1):
            queries.append("_".join(parts[-n:]))
        if parts:
            queries.append(parts[-1])

        out = []
        seen = set()
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out

    def _find_system_pairs(self):
        return []

    def _resolve_metadata_from_path(self, protein_file):
        rel = protein_file.relative_to(self.original_structures_root)
        parts = rel.parts

        group_folder = parts[0] if len(parts) > 0 else ""
        system_folder = parts[1] if len(parts) > 1 else group_folder

        return {
            "group": self._safe_str(group_folder),
            "group_abbreviation": self._safe_str(group_folder),
            "system_name": self._safe_str(system_folder),
            "protein": "",
            "reference_pdb": "",
            "subset_metadata_file": "",
        }

    def _load_protein(self, pdb_file):
        u = mda.Universe(str(pdb_file))
        protein = u.select_atoms("protein")

        if len(protein) == 0:
            raise ValueError(f"No protein atoms found in {pdb_file}")

        atom_symbols = np.array(
            [(a.element if a.element else a.name[0]) for a in protein],
            dtype=object,
        )
        coords = protein.positions.astype(np.float32)
        resnames = [a.resname for a in protein.atoms]
        return atom_symbols, coords, resnames

    def _load_ligands(self, sdf_file):
        suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
        ligands = []

        for mol in suppl:
            if mol is None:
                continue
            if mol.GetNumConformers() == 0:
                continue
            ligands.append(mol)

        return ligands

    def _ligand_features(self, mol):
        conf = mol.GetConformer()
        coords = np.array(
            [
                [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ],
            dtype=np.float32,
        )
        atom_symbols = np.array([a.GetSymbol() for a in mol.GetAtoms()], dtype=object)
        return atom_symbols, coords

    def _atom_to_z(self, atom_symbols):
        return torch.tensor(
            [atom_dict.get(str(sym).upper(), 0) for sym in atom_symbols],
            dtype=torch.long,
        )

    def _trim_pocket(self, protein_symbols, protein_coords, protein_resnames, lig_coords):
        from scipy.spatial.distance import cdist

        dist = cdist(protein_coords, lig_coords).min(axis=1)
        keep = dist < self.pocket_cutoff

        protein_symbols = protein_symbols[keep]
        protein_coords = protein_coords[keep]
        protein_resnames = np.array(protein_resnames, dtype=object)[keep].tolist()

        return protein_symbols, protein_coords, protein_resnames

    def _pad_or_trim(self, x, pos, resnames):
        if len(x) > self.max_len:
            x = x[: self.max_len]
            pos = pos[: self.max_len]
            resnames = resnames[: self.max_len]
        elif len(x) < self.max_len:
            pad_len = self.max_len - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            pos = torch.cat([pos, torch.zeros(pad_len, 3, dtype=torch.float32)])
            resnames = resnames + [""] * pad_len

        return x, pos, resnames

    def _resolve_csv_row(self, row):
        canon_group = self._safe_str(row.get("system_group", row.get("group", "")))
        canon_system = self._safe_str(row.get("system_name", row.get("system", "")))
        canon_ligand = self._safe_str(row.get("ligand_name", row.get("ligand", "")))
        canon_row_index = int(row.name) if row.name is not None else -1

        mapped_group = self.group_map.get(canon_group, canon_group)
        mapped_system = self.system_map.get((canon_group, canon_system))

        if mapped_system is None:
            return {
                "canon_row_index": canon_row_index,
                "canon_system_group": canon_group,
                "canon_system_name": canon_system,
                "canon_ligand_name": canon_ligand,
                "mapped_group": mapped_group,
                "mapped_system_file": "",
                "resolved_path": "",
                "matched_ligand": "",
                "match_score": 0.0,
                "resolved": False,
                "resolve_reason": "missing SYSTEM_MAP entry",
            }

        resolved_path = self.original_structures_root / mapped_group / mapped_system
        if not resolved_path.exists():
            return {
                "canon_row_index": canon_row_index,
                "canon_system_group": canon_group,
                "canon_system_name": canon_system,
                "canon_ligand_name": canon_ligand,
                "mapped_group": mapped_group,
                "mapped_system_file": mapped_system,
                "resolved_path": str(resolved_path),
                "matched_ligand": "",
                "match_score": 0.0,
                "resolved": False,
                "resolve_reason": "ligand file not found",
            }

        ligand_names = []
        for mol in Chem.SDMolSupplier(str(resolved_path), removeHs=False):
            if mol is None:
                continue
            try:
                ligand_names.append(mol.GetProp("_Name"))
            except Exception:
                pass

        raw_name = canon_ligand

        if raw_name in self.hard_aliases:
            target = self.hard_aliases[raw_name]
            if target in ligand_names:
                return {
                    "canon_row_index": canon_row_index,
                    "canon_system_group": canon_group,
                    "canon_system_name": canon_system,
                    "canon_ligand_name": canon_ligand,
                    "mapped_group": mapped_group,
                    "mapped_system_file": mapped_system,
                    "resolved_path": str(resolved_path),
                    "matched_ligand": target,
                    "match_score": 1.0,
                    "resolved": True,
                    "resolve_reason": "hard alias match",
                }

        for query in self._ligand_queries(raw_name):
            if query in ligand_names:
                return {
                    "canon_row_index": canon_row_index,
                    "canon_system_group": canon_group,
                    "canon_system_name": canon_system,
                    "canon_ligand_name": canon_ligand,
                    "mapped_group": mapped_group,
                    "mapped_system_file": mapped_system,
                    "resolved_path": str(resolved_path),
                    "matched_ligand": query,
                    "match_score": 1.0,
                    "resolved": True,
                    "resolve_reason": f"exact match via {query}",
                }

        best = None
        best_score = -1.0
        best_query = None

        for query in self._ligand_queries(raw_name):
            matched, s = self._best_match(query, ligand_names)
            if s > best_score:
                best = matched
                best_score = s
                best_query = query

        return {
            "canon_row_index": canon_row_index,
            "canon_system_group": canon_group,
            "canon_system_name": canon_system,
            "canon_ligand_name": canon_ligand,
            "mapped_group": mapped_group,
            "mapped_system_file": mapped_system,
            "resolved_path": str(resolved_path),
            "matched_ligand": best or "",
            "match_score": float(best_score if best_score >= 0 else 0.0),
            "resolved": bool(best_score >= 0.80),
            "resolve_reason": f"fuzzy match via {best_query}" if best_score >= 0.80 else "no good match",
        }

    def _load_data(self):
        x_list = []
        pos_list = []
        y_list = []
        resname_list = []
        self.metadata_list = []

        for _, row in self.canon_df.iterrows():
            resolved = self._resolve_csv_row(row)
            if not resolved["resolved"]:
                continue

            ligand_file = Path(resolved["resolved_path"])
            if not ligand_file.exists():
                continue

            stem = ligand_file.name.replace("_ligands.sdf", "")
            protein_file = ligand_file.with_name(f"{stem}_protein.pdb")
            if not protein_file.exists():
                continue

            protein_syms, protein_coords, protein_resnames = self._load_protein(protein_file)
            ligands = self._load_ligands(ligand_file)

            target_ligand = resolved["matched_ligand"]
            chosen = None
            chosen_idx = -1

            for lig_idx, mol in enumerate(ligands):
                lig_name = ""
                if mol.HasProp("_Name"):
                    lig_name = self._safe_str(mol.GetProp("_Name"))

                if lig_name == target_ligand or target_ligand == "":
                    chosen = mol
                    chosen_idx = lig_idx
                    break

            if chosen is None and ligands:
                chosen = ligands[0]
                chosen_idx = 0

            if chosen is None:
                continue

            lig_name = ""
            if chosen.HasProp("_Name"):
                lig_name = self._safe_str(chosen.GetProp("_Name"))

            lig_syms, lig_coords = self._ligand_features(chosen)
            pocket_syms, pocket_coords, pocket_resnames = self._trim_pocket(
                protein_syms,
                protein_coords,
                protein_resnames,
                lig_coords,
            )

            all_syms = np.concatenate([pocket_syms, lig_syms])
            all_coords = np.concatenate([pocket_coords, lig_coords], axis=0)
            all_resnames = list(pocket_resnames) + ["LIG"] * len(lig_syms)

            x = self._atom_to_z(all_syms)
            pos = torch.tensor(all_coords, dtype=torch.float32)

            y_val = float("nan")
            if chosen.HasProp("r_exp_dg"):
                try:
                    y_val = float(chosen.GetProp("r_exp_dg"))
                except Exception:
                    pass
            y = torch.tensor(y_val, dtype=torch.float32)

            x, pos, all_resnames = self._pad_or_trim(x, pos, all_resnames)

            x_list.append(x)
            pos_list.append(pos)
            y_list.append(y)
            resname_list.append(all_resnames)

            self.metadata_list.append(
                {
                    **resolved,
                    "ligand_name": lig_name,
                    "ligand_index": int(chosen_idx),
                    "protein_file": str(protein_file),
                    "ligand_file": str(ligand_file),
                }
            )

        if not x_list:
            raise ValueError("No valid structures were loaded from IndustryBenchmarksDataset.")

        self.x_0 = torch.stack(x_list)
        self.pos = torch.stack(pos_list)
        self.y = torch.stack(y_list)

        if self.use_residue_features:
            self.resname_indices = torch.stack(
                [self._residue_to_indices(r) for r in resname_list]
            )
        else:
            self.resname_indices = None
            self.use_residue_features = False

    def __getitem__(self, idx):
        item = {
            "x": self.x_0[idx],
            "pos": self.pos[idx],
            "y": self.y[idx],
            "idx": torch.tensor(idx),
            "metadata": self.metadata_list[idx],
        }
        if self.use_residue_features and self.resname_indices is not None:
            item["res_feats"] = self.resname_indices[idx]
        return item

    def __len__(self):
        return len(self.x_0)

    def __getmetadata__(self, idx):
        return self.metadata_list[idx]

# "a Focused 4-target dataset" (hahn et.al 2022)



@deprecated("PLAS20KDataset is currently deprecated and will be refactored to use the authors MD directly")
class PLAS20kDataset(BaseMDDataset):
    def __init__(
        self,
        pdb_list,
        data_dir=Path("data/md/"),
        max_len=600,
        rmsd_thresh=2.0,
        echo=True,
    ):
        self.rmsd_thresh = rmsd_thresh
        self.pdb_to_idx = {}
        self.max_len = max_len
        pos = []
        x_0_ = []

        self.file_list = []
        # if prompt is None or len(prompt) <= 1: prompt = [1]

        # Traverse all pdbs
        self.pose_list = []
        self.y = []
        self.pdbid_list = []
        self.resname = []
        self.rmsd = []
        self.frame_num = []

        path_list = [
            x for x in data_dir.glob("*-p0_*.npy") if x.stem.split("-")[0] in pdb_list
        ]

        for file in path_list:
            # print(file)

            d = np.load(Path(data_dir) / Path(file), allow_pickle=True).item()

            d["R"] = np.expand_dims(d["R"][0], axis=0)

            file_pos = torch.tensor(d["R"])

            # self.pose_list.append([int(file.split("_")[0].split("-")[-1][1])]*len(file_pos))
            self.pose_list.append([0])
            z = torch.tensor([atom_dict[x] for x in d["z"]])
            z = z.repeat(file_pos.shape[0], 1)

            if len(d["PLAS_20k_data"]["DELTA_TOTAL"]) > 0:
                # print(d['PLAS_20k_data']['DELTA_TOTAL'].values,np.ones(total_frames) )
                y = d["PLAS_20k_data"]["DELTA_TOTAL"].values.astype(np.float32)
            else:
                # move onto next pdbid
                continue

            pos.append(file_pos)
            x_0_.append(z)

            rmsd = np.expand_dims(d["rmsd"][0], axis=0)
            # resname = d['res-name']

            frame_num = torch.tensor(list(range(y.shape[0])), dtype=torch.int32)

            assert y.shape == rmsd.shape
            # print(y.shape)
            self.y.append(torch.from_numpy(y))
            self.rmsd.append(torch.from_numpy(rmsd))
            self.file_list.append([file.name.split("-")[0]] * len(y))
            # self.resname.append(resname)
            self.frame_num.append(frame_num)

        self.x_0 = pad_sequence(
            [x.transpose(1, 0) for x in x_0_], batch_first=True, padding_value=0
        ).transpose(1, 2)
        self.pos = pad_sequence(
            [x.transpose(1, 0) for x in pos], batch_first=True, padding_value=0
        ).transpose(1, 2)

        self.x_0 = self.x_0[
            :, :, -self.max_len :
        ]  # the ligand is at the end of the array dimension and we want to keep all of it

        self.x_0 = self.x_0.reshape(-1, self.x_0.shape[2])

        # Current sequence length
        seq_len = self.x_0.size(1)

        if seq_len < max_len:
            # Calculate how many zeros to add
            pad_size = max_len - seq_len

            # Create padding tensor of zeros (batch_size, pad_size)
            padding = torch.zeros((self.x_0.size(0), pad_size), dtype=self.x_0.dtype)

            # Concatenate along the sequence dimension
            self.x_0 = torch.cat((self.x_0, padding), dim=1)

        self.pos = self.pos[
            :, :, -self.max_len :, :
        ]  # the ligand is inclued at the end
        self.pos = self.pos.reshape(-1, self.pos.shape[2], 3)

        seq_len = self.pos.size(1)

        if seq_len < max_len:
            # Calculate how many zeros to add
            pad_size = max_len - seq_len

            # Create padding tensor of zeros (batch_size, pad_size)
            padding = torch.zeros((self.pos.size(0), pad_size, 3), dtype=self.pos.dtype)

            # Concatenate along the sequence dimension
            self.pos = torch.cat((self.pos, padding), dim=1)

        self.y = torch.cat(self.y)

        self.frame_num = torch.cat(self.frame_num)

        self.rmsd = torch.cat(self.rmsd)

        backprop_file_list = []
        for id_list in self.file_list:
            backprop_file_list.extend(id_list)

        self.file_list = backprop_file_list

        backprop_pose_list = []
        for pose_list in self.pose_list:
            backprop_pose_list.extend(pose_list)

        self.pose_list = backprop_pose_list

        if echo:
            print("Got {:d} protein-ligand simulations as input!".format(len(x_0_)))

        self.y = self.y / -10

        # print(self.file_list)

    def __getitem__(self, i):
        return (
            self.x_0[i],
            self.pose_list[i],
            self.pos[i],
            self.y[i],
            self.file_list[i],
            self.rmsd[i],
            self.frame_num[i],
        )

    def __len__(self):
        return len(self.x_0)
    



class OpenBindDataset(BaseMDDataset):
    """
    OpenBind EV-A71 2A protease dataset.
    
    Layout:
      root_dir/
        EV-A71_2A_metadata.csv
        structures/<compound_group>/<complex_name>/
          <complex_name>_prepared.pdb
          <complex_name>_ligand_prepared.sdf
          <complex_name>_complex_ref.pdb (optional)
          <complex_name>_ligand_ref.sdf (optional)
    
    Loads structures and pairs them with experimental pKD values from metadata.
    """
    
    def __init__(
        self,
        root_dir,
        max_len=600,
        use_residue_features=False,
        verbose=False,
        residue_mode="full",
        pocket_cutoff=6.0,
    ):
        super().__init__(
            max_len=max_len,
            use_residue_features=use_residue_features,
            verbose=verbose,
            residue_mode=residue_mode,
        )
        self.root_dir = Path(root_dir)
        self.pocket_cutoff = pocket_cutoff
        
        # Load metadata
        metadata_path = self.root_dir / "EV-A71_2A_metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        self.metadata_df = pd.read_csv(metadata_path)
        if verbose:
            print(f"Loaded metadata with {len(self.metadata_df)} complexes")
            print(f"  Complexes with pKD: {self.metadata_df['experimental_pKD'].notna().sum()}")
        
        self._load_data()
    
    def _find_structures(self):
        """Find all structure directories."""
        structures_root = self.root_dir / "structures"
        
        for compound_group in sorted(structures_root.iterdir()):
            if not compound_group.is_dir():
                continue
            
            for complex_dir in sorted(compound_group.iterdir()):
                if not complex_dir.is_dir():
                    continue
                
                complex_name = complex_dir.name

                # Always use complex_ref.pdb which contains protein + ligand + waters
                pdb_file = complex_dir / f"{complex_name}_complex_ref.pdb"

                if not pdb_file.exists():
                    if self.verbose:
                        print(f"Warning: No PDB file found for {complex_name}")
                    continue
                
                # Get metadata for this complex
                meta_row = self.metadata_df[self.metadata_df["complex_name"] == complex_name]
                if len(meta_row) == 0:
                    if self.verbose:
                        print(f"Warning: No metadata found for {complex_name}")
                    continue
                
                meta_row = meta_row.iloc[0]
                
                yield {
                    "complex_name": complex_name,
                    "compound_group": compound_group.name,
                    "pdb_path": pdb_file,
                    "smiles": meta_row.get("smiles"),
                    "pKD": meta_row.get("experimental_pKD") if pd.notna(meta_row.get("experimental_pKD")) else None,
                    "covalent": meta_row.get("covalent", False),
                    "fragment_screen": meta_row.get("fragment_screen", False),
                }
    
    def _load_data(self):
        """Load all structures and metadata."""
        self.samples = []
        
        for structure_info in tqdm(list(self._find_structures()), desc="Loading structures", disable=not self.verbose):
            try:
                # Load the complex
                u = mda.Universe(str(structure_info["pdb_path"]))

                # Filter out waters and metals (same as training data preprocessing)
                # This matches the filtering in extract_md.py:
                # "not resname WAT and not resname Na+ and not element Zn and not element Mg"
                filtered_u = u.select_atoms(
                    "not resname WAT and not resname NA and not element Zn and not element Mg"
                )

                # Separate protein and ligand from filtered atoms
                # OpenBind uses "LIG" for ligand residue name
                protein = filtered_u.select_atoms("protein")
                ligand = filtered_u.select_atoms("resname LIG")

                if len(ligand) == 0:
                    if self.verbose:
                        print(f"Warning: No ligand atoms found in {structure_info['complex_name']}")
                    continue

                # Extract pocket around ligand
                # Select protein atoms within cutoff distance of ligand atoms
                # Use filtered universe to ensure pocket doesn't include waters/metals
                pocket = filtered_u.select_atoms(f"protein and around {self.pocket_cutoff} resname LIG")
                
                if len(pocket) == 0:
                    if self.verbose:
                        print(f"Warning: No pocket atoms found for {structure_info['complex_name']}")
                    continue
                
                # Combine pocket + ligand
                complex_atoms = Merge(pocket.atoms, ligand.atoms)
                
                # Extract features
                atom_types = []
                positions = []
                residue_names = []
                
                for atom in complex_atoms.atoms:
                    element = atom.element.upper()
                    if element not in atom_dict:
                        if self.verbose:
                            print(f"Warning: Unknown element {element} in {structure_info['complex_name']}, skipping atom")
                        continue
                    
                    atom_types.append(atom_dict[element])
                    positions.append(atom.position)
                    
                    # Track residue type
                    res_name = atom.residue.resname
                    residue_names.append(res_name)
                
                if len(atom_types) == 0:
                    if self.verbose:
                        print(f"Warning: No valid atoms for {structure_info['complex_name']}")
                    continue
                
                # Truncate if too long
                if len(atom_types) > self.max_len:
                    atom_types = atom_types[:self.max_len]
                    positions = positions[:self.max_len]
                    residue_names = residue_names[:self.max_len]

                # Pad to max_len
                current_len = len(atom_types)
                if current_len < self.max_len:
                    pad_len = self.max_len - current_len
                    atom_types = np.concatenate([atom_types, np.zeros(pad_len, dtype=np.int64)])
                    positions = np.concatenate([positions, np.zeros((pad_len, 3), dtype=np.float32)])
                    residue_names = residue_names + ["PAD"] * pad_len

                # Convert to residue features if needed
                res_feats = None
                if self.use_residue_features:
                    res_feats = []
                    for res_name in residue_names:
                        if res_name == "PAD":
                            res_feats.append(RESIDUE_VOCAB["PAD"])
                        elif self.residue_mode == "collapsed":
                            # Collapsed mode: PROT/LIG/PAD only
                            if res_name in STANDARD_RESIDUES:
                                res_feats.append(RESIDUE_VOCAB["PROT"])
                            else:
                                res_feats.append(RESIDUE_VOCAB["LIG"])
                        else:
                            # Full mode: use specific residue vocabulary
                            if res_name in RESIDUE_VOCAB:
                                res_feats.append(RESIDUE_VOCAB[res_name])
                            elif res_name in STANDARD_RESIDUES:
                                res_feats.append(RESIDUE_VOCAB["PROT"])
                            else:
                                res_feats.append(RESIDUE_VOCAB["LIG"])
                
                # Store sample
                sample = {
                    "x": torch.tensor(atom_types, dtype=torch.long),
                    "pos": torch.tensor(positions, dtype=torch.float32),
                    "complex_name": structure_info["complex_name"],
                    "compound_group": structure_info["compound_group"],
                    "smiles": structure_info["smiles"],
                    "pKD": structure_info["pKD"],
                    "covalent": structure_info["covalent"],
                    "fragment_screen": structure_info["fragment_screen"],
                }
                
                if res_feats is not None:
                    sample["res_feats"] = torch.tensor(res_feats, dtype=torch.long)
                
                self.samples.append(sample)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error loading {structure_info['complex_name']}: {e}")
                continue
        
        if len(self.samples) == 0:
            raise RuntimeError("No valid samples loaded from OpenBind dataset")
        
        if self.verbose:
            print(f"Successfully loaded {len(self.samples)} samples")
            samples_with_pkd = sum(1 for s in self.samples if s["pKD"] is not None)
            print(f"  Samples with pKD: {samples_with_pkd}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        result = {
            "x": sample["x"],
            "pos": sample["pos"],
            "idx": torch.tensor(idx, dtype=torch.long),
        }

        if sample["pKD"] is not None:
            # Convert pKD to binding affinity (kcal/mol)
            # ΔG = -RT ln(10) * pKD
            # R = 1.987e-3 kcal/(mol·K), T = 297 K
            dG = -1.987e-3 * 297 * 2.302585 * sample["pKD"]
            result["y"] = torch.tensor(dG, dtype=torch.float32)
        else:
            # Use NaN for samples without affinity labels
            # This ensures all samples have 'y' key for batching
            result["y"] = torch.tensor(float('nan'), dtype=torch.float32)

        if "res_feats" in sample:
            result["res_feats"] = sample["res_feats"]

        return result
    
    def __getmetadata__(self, idx):
        sample = self.samples[idx]
        return {
            "complex_name": sample["complex_name"],
            "compound_group": sample["compound_group"],
            "smiles": sample["smiles"],
            "pKD": sample["pKD"],
            "covalent": sample["covalent"],
            "fragment_screen": sample["fragment_screen"],
        }


class BindingDBDataset(BaseMDDataset):
    """
    BindingDB benchmark dataset from AEV-PLIG.

    Layout:
      root_dir/
        bindingdb_processed.csv  (metadata with pK values)
        surflex/<folder>/
          <pdb_file>       (protein structure)
          <mol2_file>      (ligand structure from docking)

    Uses mol2 files for ligands (docked poses) and PDB files for receptors.
    Follows the same directory structure as AEV-PLIG for fair comparison.
    """

    def __init__(
        self,
        root_dir,
        metadata_csv=None,
        max_len=600,
        use_residue_features=False,
        verbose=False,
        residue_mode="full",
        pocket_cutoff=6.0,
    ):
        super().__init__(
            max_len=max_len,
            use_residue_features=use_residue_features,
            verbose=verbose,
            residue_mode=residue_mode,
        )
        self.root_dir = Path(root_dir)
        self.pocket_cutoff = pocket_cutoff

        # Load metadata
        if metadata_csv is None:
            metadata_csv = Path(root_dir) / "bindingdb_processed.csv"
        else:
            metadata_csv = Path(metadata_csv)

        if not metadata_csv.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_csv}")

        self.metadata_df = pd.read_csv(metadata_csv)
        if verbose:
            print(f"Loaded BindingDB metadata with {len(self.metadata_df)} complexes")
            print(f"  Complexes with pK values: {self.metadata_df['pK'].notna().sum()}")

        self._load_data()

    def _load_data(self):
        """Load all structures and metadata."""
        self.samples = []

        for idx, row in tqdm(self.metadata_df.iterrows(), total=len(self.metadata_df),
                             desc="Loading BindingDB structures", disable=not self.verbose):
            try:
                folder = row["folder"]
                pdb_file = self.root_dir / "surflex" / folder / row["pdb_file"]
                mol2_file = self.root_dir / "surflex" / folder / row["mol2_file"]

                if not pdb_file.exists():
                    if self.verbose:
                        print(f"Warning: PDB file not found: {pdb_file}")
                    continue

                if not mol2_file.exists():
                    if self.verbose:
                        print(f"Warning: MOL2 file not found: {mol2_file}")
                    continue

                # Load protein from PDB
                protein_u = mda.Universe(str(pdb_file))

                # Load ligand from MOL2 using RDKit
                ligand_mol = Chem.MolFromMol2File(str(mol2_file), removeHs=False)
                if ligand_mol is None:
                    if self.verbose:
                        print(f"Warning: Could not load ligand from {mol2_file}")
                    continue

                # Get ligand positions and elements from RDKit conformer
                conformer = ligand_mol.GetConformer()
                ligand_positions = []
                ligand_elements = []

                for atom in ligand_mol.GetAtoms():
                    # Skip hydrogens to match training data preprocessing
                    if atom.GetSymbol() == "H":
                        continue

                    element = atom.GetSymbol().upper()
                    if element not in atom_dict:
                        if self.verbose:
                            print(f"Warning: Unknown element {element} in {mol2_file}, skipping")
                        continue

                    pos = conformer.GetAtomPosition(atom.GetIdx())
                    ligand_positions.append([pos.x, pos.y, pos.z])
                    ligand_elements.append(element)

                if len(ligand_positions) == 0:
                    if self.verbose:
                        print(f"Warning: No valid ligand atoms in {mol2_file}")
                    continue

                ligand_positions = np.array(ligand_positions, dtype=np.float32)

                # Filter protein: remove waters and metals (match training preprocessing)
                filtered_protein = protein_u.select_atoms(
                    "protein and not name H* and not resname WAT and not resname NA "
                    "and not element Zn and not element Mg"
                )

                if len(filtered_protein) == 0:
                    if self.verbose:
                        print(f"Warning: No protein atoms found in {pdb_file}")
                    continue

                # Find pocket: protein atoms within cutoff of any ligand atom
                # Calculate distances between all protein atoms and ligand center
                ligand_center = ligand_positions.mean(axis=0)
                protein_positions = filtered_protein.positions
                distances = np.linalg.norm(protein_positions - ligand_center, axis=1)

                # Select pocket atoms within cutoff (with some buffer for the whole pocket)
                pocket_cutoff_extended = self.pocket_cutoff + 3.0  # Add buffer for pocket selection
                pocket_mask = distances < pocket_cutoff_extended

                if pocket_mask.sum() == 0:
                    if self.verbose:
                        print(f"Warning: No pocket atoms found for {row['unique_id']}")
                    continue

                pocket_atoms = filtered_protein[pocket_mask]

                # Extract pocket features
                pocket_positions = pocket_atoms.positions
                pocket_elements = [atom.element.upper() for atom in pocket_atoms.atoms]
                pocket_residues = [atom.residue.resname for atom in pocket_atoms.atoms]

                # Filter out unknown elements from pocket
                valid_pocket_mask = np.array([elem in atom_dict for elem in pocket_elements])
                pocket_positions = pocket_positions[valid_pocket_mask]
                pocket_elements = [elem for i, elem in enumerate(pocket_elements) if valid_pocket_mask[i]]
                pocket_residues = [res for i, res in enumerate(pocket_residues) if valid_pocket_mask[i]]

                if len(pocket_positions) == 0:
                    if self.verbose:
                        print(f"Warning: No valid pocket atoms for {row['unique_id']}")
                    continue

                # Combine pocket + ligand
                all_positions = np.concatenate([pocket_positions, ligand_positions], axis=0)
                all_elements = pocket_elements + ligand_elements
                all_residues = pocket_residues + ["LIG"] * len(ligand_elements)

                # Convert to atom types
                atom_types = np.array([atom_dict[elem] for elem in all_elements], dtype=np.int64)

                # Truncate if too long
                if len(atom_types) > self.max_len:
                    atom_types = atom_types[:self.max_len]
                    all_positions = all_positions[:self.max_len]
                    all_residues = all_residues[:self.max_len]

                # Pad to max_len
                current_len = len(atom_types)
                if current_len < self.max_len:
                    pad_len = self.max_len - current_len
                    atom_types = np.concatenate([atom_types, np.zeros(pad_len, dtype=np.int64)])
                    all_positions = np.concatenate([all_positions, np.zeros((pad_len, 3), dtype=np.float32)])
                    all_residues = all_residues + ["PAD"] * pad_len

                # Convert to residue features if needed
                res_feats = None
                if self.use_residue_features:
                    res_feats = []
                    for res_name in all_residues:
                        if res_name == "PAD":
                            res_feats.append(RESIDUE_VOCAB["PAD"])
                        elif self.residue_mode == "collapsed":
                            if res_name in STANDARD_RESIDUES:
                                res_feats.append(RESIDUE_VOCAB["PROT"])
                            else:
                                res_feats.append(RESIDUE_VOCAB["LIG"])
                        else:
                            if res_name in RESIDUE_VOCAB:
                                res_feats.append(RESIDUE_VOCAB[res_name])
                            elif res_name in STANDARD_RESIDUES:
                                res_feats.append(RESIDUE_VOCAB["PROT"])
                            else:
                                res_feats.append(RESIDUE_VOCAB["LIG"])

                # Parse pK value (can be pKi, pIC50, or pKd)
                pk_value = row.get("pK")
                if pd.notna(pk_value):
                    pk_value = float(pk_value)
                else:
                    pk_value = None

                # Store sample
                sample = {
                    "x": torch.tensor(atom_types, dtype=torch.long),
                    "pos": torch.tensor(all_positions, dtype=torch.float32),
                    "unique_id": row["unique_id"],
                    "folder": folder,
                    "pdb_file": row["pdb_file"],
                    "mol2_file": row["mol2_file"],
                    "pK": pk_value,
                    "surflex_score": row.get("surflex_score"),
                }

                if res_feats is not None:
                    sample["res_feats"] = torch.tensor(res_feats, dtype=torch.long)

                self.samples.append(sample)

            except Exception as e:
                if self.verbose:
                    print(f"Error loading {row.get('unique_id', idx)}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

        if len(self.samples) == 0:
            raise RuntimeError("No valid samples loaded from BindingDB dataset")

        if self.verbose:
            print(f"Successfully loaded {len(self.samples)} BindingDB complexes")
            samples_with_pk = sum(1 for s in self.samples if s["pK"] is not None)
            print(f"  Samples with pK values: {samples_with_pk}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        result = {
            "x": sample["x"],
            "pos": sample["pos"],
            "idx": torch.tensor(idx, dtype=torch.long),
        }

        if sample["pK"] is not None:
            # Convert pK to binding affinity (kcal/mol)
            # ΔG = -RT ln(10) * pK
            # R = 1.987e-3 kcal/(mol·K), T = 297 K
            dG = -1.987e-3 * 297 * 2.302585 * sample["pK"]
            result["y"] = torch.tensor(dG, dtype=torch.float32)
        else:
            # Use NaN for samples without affinity labels
            result["y"] = torch.tensor(float('nan'), dtype=torch.float32)

        if "res_feats" in sample:
            result["res_feats"] = sample["res_feats"]

        return result

    def __getmetadata__(self, idx):
        sample = self.samples[idx]
        return {
            "unique_id": sample["unique_id"],
            "folder": sample["folder"],
            "pdb_file": sample["pdb_file"],
            "mol2_file": sample["mol2_file"],
            "pK": sample["pK"],
            "surflex_score": sample["surflex_score"],
        }
