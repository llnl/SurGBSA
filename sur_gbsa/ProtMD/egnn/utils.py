################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
# adapted from ProtMD (https://github.com/smiles724/ProtMD)
# All rights reserved.
################################################################################
import torch
import torch.nn as nn
from torch import sin, cos


def rot_z(gamma):
    return torch.tensor(
        [[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]],
        dtype=gamma.dtype,
    )


def rot_y(beta):
    return torch.tensor(
        [[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]],
        dtype=beta.dtype,
    )


def rot(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


# this is a classifier for snapshot ordering
class Classifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid()
        )

    def forward(self, x1, x2):
        return self.out(torch.cat((x1, x2), dim=-1)).squeeze(-1)


class Regressor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(dim, int(dim / 2)), nn.ReLU(), nn.Linear(int(dim / 2), 1)
        )

    def forward(self, x):
        return self.out(x).squeeze(-1)
