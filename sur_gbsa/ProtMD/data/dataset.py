################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from pathlib import Path
import pandas as pd
from tqdm import tqdm 

atom_dict = {"C": 6, "H": 1, "O": 8, "N": 7, "S": 9, "P": 10, "F": 11, "I": 12, "B": 13}


class PDBMDDataset:
    def __init__(
        self,
        backprop,
        pdb_list,
        prompt=None,
        data_dir=None,
        max_len=600,
        noise=True,
        noise_scale=10,
        echo=False,
        rmsd_thresh=4.0,
    ):

        pdbid_df = pd.read_csv(f"{Path(data_dir)/ Path('pdbid_list.csv')}", header=None)
        pose_df = pd.read_csv(f"{Path(data_dir) / Path('pose_list.csv')}", header=None)
        self.pdbid_map = {
            element: index for index, element in enumerate(pdbid_df[0].unique())
        }
        self.pose_map = {
            element: index for index, element in enumerate(pose_df[0].unique())
        }

        self.rmsd_thresh = rmsd_thresh

        self.pdbid_to_idx = {}
        max_len = int(max_len)
        self.backprop = backprop
        self.noise = noise
        self.noise_scale = noise_scale
        x_0_, x_t_ = [], []
        feats_, prompt_ = [], []
        self.file_list = []
        self.y = []
        if prompt is None or len(prompt) <= 1:
            prompt = [1]

        self.pdbid_list = []
        self.pose_list = []
        self.rmsd = []
        self.cluster = []
        pdbid_count = 1
        # Traverse all pdbs

        for file in pdb_list:
            if not file.endswith(".npy"):
                if echo:
                    print(f"Warning: data {file} is not a numpy file.")
                continue

            pdbid = file.split("-")[0]
            complex_id = file.split("_")[0]

            d = np.load(data_dir / file, allow_pickle=True).item()

            # x.shape: (T,N,3)
            x = torch.tensor(d["R"])
            z = torch.tensor([atom_dict[x] for x in d["z"][: x.shape[1]]])
            n_times = len(x)


            if backprop == True:
                x = x[
                    : int(0.9 * n_times)
                ]  
            elif backprop == False:
                x = x[int(0.9 * n_times) :]
            else:
                pass



            # Traverse all prompts
            for pt in prompt:
                x_0, x_t = x[:-pt], x[pt:]
                x_0_.append(x_0)
                x_t_.append(x_t)
                feats_ += [z] * len(x_0)
                prompt_ += [pt] * len(x_0)

            self.pose_list.extend([self.pose_map[complex_id]] * len(x_0))
            self.pdbid_list.extend([self.pdbid_map[pdbid]] * len(x_0))
            self.file_list.extend([file] * len(x_0))

            y = d["DELTA TOTAL"][: len(x_0)]
            rmsd = d["rmsd"][: len(x_0)]  # store the other rmsd as well?
            cluster = d["cluster_id"][: len(x_0)]

            self.y.extend(torch.from_numpy(y))
            self.rmsd.append(rmsd)
            self.cluster.append(cluster)

        self.mole_idx = pad_sequence(feats_, batch_first=True, padding_value=0)[
            :, -max_len:
        ] #keep the ligand and include amino acids by counting backwards
         
        self.prompt = torch.tensor(prompt_).unsqueeze(-1)
        self.pose = torch.tensor(self.pose_list)
        # first collect the initial/end frame time step coordinates
        for i in range(len(x_0_)):
            x_0_[i] = F.pad(
                x_0_[i], (0, 0, 0, self.mole_idx.shape[-1] - x_0_[i].shape[1])
            )
            x_t_[i] = F.pad(
                x_t_[i], (0, 0, 0, self.mole_idx.shape[-1] - x_t_[i].shape[1])
            )

        # (derek): then the padded frame data is concatenated and then a filter on number of atoms is applied
        self.x_0, self.x_t = (torch.cat(x_0_)[:, -max_len:]).float(), (
            torch.cat(x_t_)[:, -max_len:]
        ).float()
        self.rmsd = torch.cat([torch.from_numpy(x) for x in self.rmsd]).float()
        self.cluster = torch.tensor(np.concatenate(self.cluster))
        if echo:
            print("Got {:d} protein-ligand simulations as input!".format(len(x_0_)))

    def __getitem__(self, i):

        rmsd = self.rmsd[i]

        if rmsd > self.rmsd_thresh:
            return None

        pdbid = self.file_list[i].split("-")[0]

        # Return the coordinates at the initial time, the atom type, and the coordinates at the end time

        return (
            self.x_0[i],
            self.mole_idx[i],
            self.x_t[i],
            self.prompt[i],
            self.y[i] / -10,
            self.pose[i],
            self.file_list[i],
            torch.tensor(self.pdbid_map[pdbid]),
            self.rmsd[i],
            self.cluster[i],
        )

    def __len__(self):
        return len(self.x_0)


