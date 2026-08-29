################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################
import netCDF4
from MDAnalysis.analysis import rms, align
from pathlib import Path
import MDAnalysis as mda
import MDAnalysis.transformations
import matplotlib.pyplot as plt
import torch
import imageio
import pandas as pd
import numpy as np
from io import StringIO
import os
from tqdm import tqdm
import h5py
import multiprocessing as mp

# Periodic table dictionary for organic chemistry
# Maps atomic numbers to element symbols for atoms commonly found in organic molecules

organic_periodic_table = {
    1: "H",  # Hydrogen - most common in organic molecules
    2: "He",  # Helium - rarely in organic compounds but included for completeness
    3: "Li",  # Lithium - used in organolithium reagents
    5: "B",  # Boron - organoborane compounds
    6: "C",  # Carbon - backbone of organic molecules
    7: "N",  # Nitrogen - amines, amides, nitro compounds
    8: "O",  # Oxygen - alcohols, ethers, carbonyls, carboxyls
    9: "F",  # Fluorine - fluorinated organic compounds
    11: "Na",  # Sodium - organosodium compounds, salts
    12: "Mg",  # Magnesium - Grignard reagents
    13: "Al",  # Aluminum - organoaluminum compounds
    14: "Si",  # Silicon - organosilicon compounds
    15: "P",  # Phosphorus - phosphates, phosphonates
    16: "S",  # Sulfur - thiols, sulfides, sulfoxides
    17: "Cl",  # Chlorine - chlorinated organic compounds
    19: "K",  # Potassium - organopotassium compounds, salts
    20: "Ca",  # Calcium - organic salts, coordination compounds
    25: "Mn",  # Manganese - organometallic catalysts
    26: "Fe",  # Iron - organometallic compounds, heme
    27: "Co",  # Cobalt - vitamin B12, organometallic compounds
    28: "Ni",  # Nickel - organometallic catalysts
    29: "Cu",  # Copper - organocopper reagents, catalysts
    30: "Zn",  # Zinc - organozinc reagents, enzyme cofactor
    35: "Br",  # Bromine - brominated organic compounds
    47: "Ag",  # Silver - organometallic compounds
    50: "Sn",  # Tin - organotin compounds
    53: "I",  # Iodine - iodinated organic compounds
    78: "Pt",  # Platinum - organometallic catalysts
    79: "Au",  # Gold - organogold compounds
    82: "Pb",  # Lead - organolead compounds (historical)
}


def process_misato_file(path):
    with h5py.File(path, "r") as f:
        for pdb in f.keys():
            coors = f[pdb]["trajectory_coordinates"][:]
            atom_num = f[pdb]["atoms_number"][:]
            atom_name = [organic_periodic_table[x] for x in atom_num]
            pbsa = f[pdb]["frames_interaction_energy"][:]

            data_dict = {
                "z": atom_name,
                "R": coors,
                "DELTA TOTAL": pbsa,
                "PBSA_mean": pbsa.mean(),
                "PBSA_std": pbsa.std(),
            }

            # todo: link gbsa information for the crysal structure

            out_file = (
                args.out_dir / f"{pdb}-0-p0.npy"
            )  # for misato dataset we only have one trajectory per pdb using the crystal structure pose
            np.save(out_file, data_dict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", default="/p/vast1/jones289/PDBbind_core_MD/misato-protmd-format/"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)

    args = parser.parse_args()

    file_dir = Path("/p/vast1/mldrug/misato/misato_chunks")

    file_list = list(file_dir.glob("*.h5"))

    # todo: process the keys in parallel and just loop over the chunks
    with mp.Pool(16) as pool:
        # file_list = list(tqdm(pool.imap(lambda x: x.with_name("com.nc"), file_list), total=len(file_list)))
        process_misato_file(file_list)
    # for path in file_dir.glob("*.h5"):

    # import pdb
    # pdb.set_trace()
    # path_df = pd.read_csv("/g/g13/jones289/workspace/fast_md/fast_md/feature_pipeline/revised_casf_2016_md.txt",
    # header=None)
    # file_list = path_df[0].values.tolist()

    # file_list = [x for x in file_list if x == "/p/vast1/jones289/PDBbind_core_MD/md/2zb1/1613/p1/com.nc"]
    # file_list = [x for x in file_list if x == "/p/vast1/jones289/PDBbind_core_MD/md/1a30/9/p0/com.nc"]
    # file_list = [x for x in file_list if "1ps3" in x]
    # file_list = [x for x in file_list if "3oe4" in x]

    # PLAS_5k_data = pd.read_csv("/g/g13/jones289/workspace/pretrain_md/PLAS/5000_final.csv")
    # PLAS_20k_data = pd.read_csv("/g/g13/jones289/workspace/pretrain_md/PLAS/data.csv", skipfooter=3)

    # main()
