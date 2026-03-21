################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################
GBSA_DATASET_LIST = [
    "sp-crystal",
    "sp-dock_top_1",
    "sp-dock_top_3",
    "sp-dock_top_5",
    "sp-dock_top_5+crystal",
    "mean-crystal",
    "mean-dock_top_1",
    "mean-dock_top_3",
    "mean-dock_top_5",
    "mean-dock_top_5+crystal",
    "md-crystal",
    "md-dock_top_1",
    "md-dock_top_3",
    "md-dock_top_5",
    "md-dock_top_5+crystal",
]

# misato is GBSA so rename this or combine into the other list 
PBSA_DATASET_LIST = ["misato", "misato-coremd-combo"] #todo: misato-coremd and misato (full)

FINETUNE_DATASET_LIST = ["mpro", "pdbbind-30", "pdbbind-60", "PLAS-20k"]
