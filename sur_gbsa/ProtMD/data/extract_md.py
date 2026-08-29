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


def process_gbsa_frames(path):
    complex_data = []
    delta_flag = False
    with open(path, "r") as handle:
        for line in handle.readlines():
            if "DELTA Energy Terms" in line:
                delta_flag = True

            elif delta_flag:
                complex_data.append(line)
            else:
                pass
        complex_data = "".join(complex_data)

    df = pd.read_csv(StringIO(complex_data))
    return df


def process_nc_file_to_numpy(path, out_path):
    # get corresponding path for crystal structure
    # import pdb
    # pdb.set_trace()

    pdbid = path.parent.parent.parent.name

    # PLAS_data = pd.concat([PLAS_5k_data, PLAS_20k_data]).reset_index(drop=True)

    # PLAS_data = PLAS_data.loc[PLAS_data['PDB_ID'] == pdbid]

    crystal_path = path.parent.parent / Path("p0/com.nc")
    crystal_u = mda.Universe(
        crystal_path.with_name("com.prmtop"), crystal_path.with_name("com.nc")
    )

    n_atoms_complex = len(crystal_u.atoms)
    # nc_obj=Dataset(path)
    # u = mda.Universe(path.with_name("com.pdb"))
    u = mda.Universe(path.with_name("com.prmtop"), path.with_name("com.nc"))

    cluster_df = pd.read_csv(
        path.parent / Path("clust/cnumvtime.dat"), delim_whitespace=True
    )
    # cluster_df = pd.read_csv(path.parent / Path('clust/cnumvtime.dat'), sep='\s+', engine="python")

    cluster_v_time_arr = cluster_df.loc[:, cluster_df.columns[1]].values

    # crystal_traj = crystal_u.trajectory.timeseries()
    # traj = u.trajectory.timeseries()
    # traj = traj.transpose(1,0, 2)

    if len(u.trajectory) != 1000:
        # if traj.shape[0] != 1000:
        # if traj.shape[1] != 1000: # after applying align it seems that we use index 1 to refer to the time dimension
        print(f"found corrupt file {path}, only {len(u.trajectory)} frames / 1000.")
        return

    # import ipdb as pdb
    # pdb.set_trace()
    # crystal_u.atoms.unwrap()
    # transform = mda.transformations.unwrap(crystal_u.atoms)
    # crystal_u.trajectory.add_transformations(transform)

    # unwrap both sets of atoms in each trajectory and make sure they are on the starting frame
    crystal_u.atoms.unwrap()
    crystal_u.trajectory[0]
    u.atoms.unwrap()
    u.trajectory[0]

    # import pdb
    # pdb.set_trace()
    # align the current trajectory to the crystal
    align.AlignTraj(u, crystal_u, select="all", in_memory=True).run()

    traj = u.trajectory.timeseries()
    traj = traj.transpose(1, 0, 2)

    # no_wat_metal_u = u.select_atoms("not resname WAT and not resname Na+ and not element Zn and not element Mg and not element H")
    no_wat_metal_crystal_u = crystal_u.select_atoms(
        "not resname WAT and not resname Na+ and not element Zn and not element Mg"
    )
    no_wat_metal_u = u.select_atoms(
        "not resname WAT and not resname Na+ and not element Zn and not element Mg"
    )

    u_ligand = no_wat_metal_u.select_atoms("resname LIG")
    u_protein = no_wat_metal_u.select_atoms("not resname LIG")

    dist = torch.cdist(
        torch.tensor(u_protein.atoms.positions), torch.tensor(u_ligand.atoms.positions)
    )
    mask = torch.sum(dist < 6, dim=-1) > 0
    mask = torch.cat((mask, torch.ones(len(u_ligand)))).bool().numpy()

    selected_atoms = no_wat_metal_u.atoms[mask]

    tr = {"name": out_path}

    # print(cluster_v_time_arr.shape)
    assert cluster_v_time_arr.shape[0] == 1000
    tr["cluster_id"] = cluster_v_time_arr
    tr["z"] = [x[0] for x in selected_atoms.atoms.names]
    # tr['R'] = nc_obj.variables['coordinates'][:,  mask]

    traj_select_mask = np.zeros(u.atoms.n_atoms, dtype=bool)
    traj_select_mask[selected_atoms.indices] = True

    # list(set(list(range(no_wat_metal_u.atoms.indices[-1]+1))).difference(list(no_wat_metal_u.atoms.indices)))
    tr["R"] = traj[:, traj_select_mask, :]

    # import pdb
    # pdb.set_trace()
    # ref = no_wat_metal_u.copy()
    # rmsd_analysis = rms.RMSD(no_wat_metal_u, select="all")
    rmsd_analysis = rms.RMSD(selected_atoms, select="all")
    rmsd_analysis.run()
    rmsd_values = rmsd_analysis.results["rmsd"][:, 2]

    # rmsd_analysis.rmsd[:,1] contains the simulation time (in pico seconds). complete simulations should be 15000 - 5000 ps (10ns)

    tr["rmsd"] = rmsd_values
    tr["rmsd-std"] = np.std(rmsd_values)
    tr["rmsd-mean"] = np.mean(rmsd_values)

    lig_rmsd = rms.RMSD(u_ligand, select="all")
    lig_rmsd.run()
    tr["lig-rmsd"] = lig_rmsd.results["rmsd"][:, 2]

    # import pdb
    # pdb.set_trace()
    # tr['res-name'] = [res.resname for res in selected_atoms.residues]
    tr["res-name"] = [atom.resname for atom in selected_atoms.atoms]

    # crystal-structure rmsd
    # import pdb
    # pdb.set_trace()
    # crystal_rmsd = rms.RMSD(no_wat_metal_u, no_wat_metal_crystal_u)
    ref_crystal_frame = no_wat_metal_crystal_u.select_atoms("resname LIG")
    crystal_u.trajectory[0]
    # crystal_rmsd = rms.RMSD(u_ligand, ref_crystal_frame)
    crystal_rmsd = rms.RMSD(u_ligand, ref_crystal_frame)
    crystal_rmsd.run()

    tr["crystal-rmsd"] = crystal_rmsd.results["rmsd"][:, 2]
    tr["n_atoms_complex"] = n_atoms_complex
    tr["n_residue_pocket"] = len(selected_atoms.residues)
    tr["n_residue_complex"] = len(u.residues)
    # import pdb
    # pdb.set_trace()
    tr["PLAS_5k_data"] = PLAS_5k_data.loc[PLAS_5k_data["pdbid"] == pdbid]
    tr["PLAS_20k_data"] = PLAS_20k_data.loc[PLAS_20k_data["PDB_ID"] == pdbid]
    # load MM/GBSA data

    mmgbsa_path = path.with_name("frames.dat")

    mmgbsa_df = process_gbsa_frames(path=mmgbsa_path)

    for col_name, col_val in mmgbsa_df.items():
        if col_name == "Frame #":
            pass
        else:
            tr[col_name] = col_val.values

    tr["GBSA_mean"] = mmgbsa_df["DELTA TOTAL"].mean()
    tr["GBSA_std"] = mmgbsa_df["DELTA TOTAL"].std()

    # tr['universe'] = u
    # tr['atom_selection'] = no_wat_metal_u

    if args.dry_run:
        return
    else:
        with open(out_path, "wb") as f:
            np.save(f, tr)
            tqdm.write(f"processed {out_path}")


def process_nc_file_job(path):
    apath = Path(str(path).replace("/p/lustre1/ahashare", "/p/vast1/jones289"))
    npy_file = apath.with_name("prot-md-input.npy")
    if npy_file.exists():
        tqdm.write(f"{npy_file} exists. moving to next file.")
    elif Path(apath).exists():
        pose = apath.parent.name
        ligid = apath.parent.parent.name
        pdbid = apath.parent.parent.parent.name
        out_path = Path(f"{args.out_dir}/{pdbid}-{ligid}-{pose}_prot-md-input.npy")
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
        process_nc_file_to_numpy(apath, out_path=out_path)

    else:
        tqdm.write(f"{apath} does not exist")


def main():
    if args.n_jobs == 1:
        for path in tqdm(file_list):
            process_nc_file_job(path=path)
    else:
        import multiprocessing as mp

        with mp.Pool(args.n_jobs) as pool:
            list(tqdm(pool.imap(process_nc_file_job, file_list), total=len(file_list)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--md-dir",
        default="/p/vast1/jones289/PDBbind_core_MD/md",
        help="top level to amber MD simulation directory as described in docs.",
    )
    parser.add_argument(
        "--out-dir", default="/p/vast1/jones289/PDBbind_core_MD/prot_md_input-fixed/"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)

    args = parser.parse_args()

    md_dir = Path(args.md_dir)
    file_list = list(md_dir.glob("**/com.nc"))

    PLAS_5k_data = pd.read_csv("./PLAS/5000_final.csv")
    PLAS_20k_data = pd.read_csv("./PLAS/data.csv", skipfooter=3)

    main()
