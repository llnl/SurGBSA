################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
# adapted from ProtMD (https://github.com/smiles724/ProtMD)
# All rights reserved.
################################################################################


import numpy as np
import json
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from atom3d.datasets import LMDBDataset
import atom3d.util.formats as fo
from tqdm import tqdm
from sur_gbsa.ProtMD.data.dataset import atom_dict


def pdb_loader(
    mode="train",
    split="30",
    path="/p/vast1/jones289/atom3d/split-by-sequence-identity-*/data/",
    dataset_name="pdb",
):
    """Load PDB data with consistent pocket extraction matching MD preprocessing"""
    path = path.replace("*", split)
    dataset = LMDBDataset(path + mode)

    x, idx, pos, resnames = [], [], [], []  # NEW: added resnames
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        if i % 1000 == 0:
            print(f"Currently processed at {i}")
        struct = dataset[i]
        atoms_pocket = struct["atoms_pocket"]
        atoms_ligand = struct["atoms_ligand"]

        atoms_pocket = atoms_pocket[
            atoms_pocket["element"].apply(lambda x: x in atom_dict.keys())
        ]
        atoms_ligand = atoms_ligand[
            atoms_ligand["element"].apply(lambda x: x in atom_dict.keys())
        ]

        x_tmp = []
        for m in np.append(atoms_pocket["element"], atoms_ligand["element"]):
            if m in atom_dict.keys():
                x_tmp.append(atom_dict[m])
            else:
                print(f"{m} not in atom_types. moving to next atom")
                continue

        pos_1, pos_2 = fo.get_coordinates_from_df(
            atoms_pocket
        ), fo.get_coordinates_from_df(atoms_ligand)

        # FIXED: Use exclusive cutoff to match MD preprocessing
        dists = torch.cdist(torch.from_numpy(pos_1), torch.from_numpy(pos_2), p=2)
        mask = dists < 6  # Changed from <= 6 to < 6

        # Pocket atoms: keep only those within 6A of ANY ligand atom
        a_indices = torch.where(mask.sum(dim=1) > 0)[0]
        
        # FIXED: Keep ALL ligand atoms (don't filter based on distance)
        pos_1 = pos_1[a_indices]
        pos_2 = pos_2  # Keep all ligand atoms
        
        # Build combined indices: filtered pocket atoms + ALL ligand atoms
        ligand_indices = torch.arange(len(atoms_pocket), len(atoms_pocket) + len(atoms_ligand))
        combined_indices = torch.cat([a_indices, ligand_indices])

        x_tmp = torch.index_select(torch.tensor(x_tmp), dim=0, index=combined_indices)
        x.append(x_tmp)
        pos.append(torch.tensor(np.append(pos_1, pos_2, axis=0)))
        
        # NEW: Create residue labels (protein vs ligand)
        # Extract residue names from pocket atoms
        if 'resname' in atoms_pocket.columns:
            pocket_resnames = atoms_pocket['resname'].iloc[a_indices.numpy()].tolist()
        else:
            # If no resname column, use generic protein residue labels
            pocket_resnames = ['PROT'] * len(a_indices)
        
        # Ligand atoms get 'LIG' label
        ligand_resnames = ['LIG'] * len(atoms_ligand)
        
        # Combine
        combined_resnames = pocket_resnames + ligand_resnames
        resnames.append(combined_resnames)

    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    
    # NEW: Pad residue names with empty strings
    max_len = max(len(r) for r in resnames)
    resnames_padded = []
    for r in resnames:
        padded = r + [''] * (max_len - len(r))
        resnames_padded.append(padded)
    
    y = torch.tensor([item["scores"]["neglog_aff"] for item in dataset])

    out_dir = Path(f"/usr/WS1/jones289/pretrain_md/sur_gbsa/ProtMD/data/{dataset_name}")
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    out_f = f"{out_dir}/{dataset_name}_{mode}_{split}.pt"
    print(f"Saving to {out_f}")
    print(f"  x shape: {x.shape}")
    print(f"  pos shape: {pos.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  resnames: {len(resnames_padded)} samples")
    
    # NEW: Save with residue names
    torch.save([x, idx, pos, y, resnames_padded], out_f)


def load_final_30():
    pdb_loader(mode="train")
    pdb_loader(mode="val")
    pdb_loader(mode="test")


def load_final_60():
    pdb_loader(mode="train", split="60")
    pdb_loader(mode="val", split="60") 
    pdb_loader(mode="test", split="60")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pdbbind-30", "pdbbind-60"])
    args = parser.parse_args()
    if args.dataset == "pdbbind-30":
        load_final_30()
    elif args.dataset == "pdbbind-60":
        load_final_60()
    else:
        raise ValueError("Dataset not recognized.")