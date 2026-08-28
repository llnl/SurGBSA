# SurGBSA

Official implementation of **"SurGBSA: Learning Representations From Molecular Dynamics Simulations"**

[![arXiv](https://img.shields.io/badge/arXiv-2509.03084-b31b1b.svg)](https://arxiv.org/abs/2509.03084)
[![DOI](https://zenodo.org/badge/1180359891.svg)](https://doi.org/10.5281/zenodo.21961791)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

SurGBSA is a deep learning framework for predicting protein-ligand binding affinities and MM-GBSA scores using E(n)-equivariant graph neural networks (EGNN) trained on molecular dynamics (MD) simulations.

## Table of Contents

- [Installation](#installation)
- [Dataset and Model Checkpoints](#dataset-and-model-checkpoints)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Supported Datasets](#supported-datasets)
- [Citation](#citation)
- [License](#license)

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA/ROCm support (GPU recommended)
- CUDA 11.8+ or ROCm 6.0+ (for GPU acceleration)

### Step 1: Clone the Repository

```bash
git clone https://github.com/llnl/SurGBSA.git
cd SurGBSA
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv surgbsa-env
source surgbsa-env/bin/activate  # On Windows: surgbsa-env\Scripts\activate
```

### Step 3: Install PyTorch

Install PyTorch with GPU support. Choose based on your hardware:

**For NVIDIA GPUs (CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For AMD GPUs (ROCm):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

See [PyTorch Get Started](https://pytorch.org/get-started/locally/) for other configurations.

### Step 4: Install SurGBSA and Dependencies

```bash
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0.0` - PyTorch deep learning framework
- `MDAnalysis>=2.8.0` - MD trajectory analysis
- `rdkit>=2023.9.1` - Cheminformatics toolkit
- `h5py>=3.8.0` - HDF5 file handling
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scipy>=1.10.0` - Scientific computing
- `tensorboard>=2.12.0` - Training visualization
- `einops>=0.6.0` - Tensor operations

### Step 5: Set Python Path

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Dataset and Model Checkpoints

Our datasets (MD trajectories, MM-GBSA scores, and processed ML-ready files), pre-trained model checkpoints, and train/val/test splits are available on HuggingFace:

**🤗 [https://huggingface.co/datasets/llnl/SurGBSA](https://huggingface.co/datasets/llnl/SurGBSA)**

### Available Data

The dataset is organized into the following directories:

```
SurGBSA-data/
├── md/                          # Raw MD simulation data
│   └── {pdb_id}/                # One directory per PDB structure
│       └── p{N}/                # Pose directory (p0=crystal, p1-p5=docked)
│           ├── com.prmtop       # AMBER topology file
│           ├── com.nc           # NetCDF trajectory file
│           └── frames.dat       # MM-GBSA scores per frame
├── ml_data/                     # Pre-processed ML-ready numpy arrays
│   └── {pdb_id}-{rep}-p{N}_prot-md-input.npy
├── splits/                      # Train/val/test split definitions
│   ├── coreMD-fold-0-train.csv
│   ├── coreMD-fold-0-val.csv
│   ├── coreMD-fold-0-test.csv
│   ├── pdbid_list.csv          # Metadata for all structures
│   └── pose_list.csv            # Pose information
└── weights/                     # Pre-trained model checkpoints
    └── best_model-epoch-{N}.pt
```

#### Data Contents

- **MD Trajectories**: Protein-ligand MD simulations from PDBBind CoreSet with topology (`.prmtop`) and trajectory (`.nc`) files
- **MM-GBSA Scores**: Frame-by-frame energy calculations in `frames.dat` files
- **ML-Ready Files**: Preprocessed numpy arrays (`.npy`) for direct model training
- **Splits**: Train/validation/test splits for reproducibility and cross-validation
- **Pre-trained Checkpoints**: PyTorch model weights pre-trained on MD data with self-supervised tasks

#### Data Format Details

- **Topology files** (`.prmtop`): AMBER parameter/topology files
- **Trajectory files** (`.nc`): NetCDF format trajectories readable with MDAnalysis
- **MM-GBSA files** (`.frames.dat`): Text files with one score per MD frame
- **ML arrays** (`.npy`): NumPy arrays with shape `(n_frames, n_atoms, features)`
- **Model checkpoints** (`.pt`): PyTorch state dicts with model weights and training args

## Directory Structure

```
SurGBSA/
├── sur_gbsa/                      # Main package directory
│   ├── __init__.py               # Package initialization with dataset lists
│   ├── datasets.py               # Dataset classes (GBSAMDDataset, MisatoDataset, PDBBindDataset)
│   ├── data_utils.py             # Data loading utilities and collate functions
│   ├── utils.py                  # General utilities (metrics, result loading)
│   ├── distributed_utils.py      # Distributed training utilities
│   │
│   ├── pretrain.py               # Single-GPU pretraining script (not used in paper)
│   ├── pretrain_distributed.py   # Multi-GPU distributed pretraining (main pretraining)
│   ├── finetune_distributed.py   # Distributed MM-GBSA training for EGMN models
│   ├── finetune_distributed_egnn.py  # Distributed MM-GBSA training for GNN/EGNN models
│   ├── finetune_affinity.py      # Finetune for binding affinity prediction
│   ├── finetune_decoy_pose_ranking.py  # Finetune for CASF-2016 decoy pose ranking
│   ├── test.py                   # Model evaluation script
│   ├── extract.py                # Extract embeddings from checkpoints
│   ├── sample.py                 # Sample trajectories from model
│   ├── score_mdanalysis.py       # Run inference on MDAnalysis-compatible structures
│   ├── rank_poses.py             # Rank protein-ligand poses
│   ├── optimize_poses.py         # Optimize poses with trained models
│   ├── evaluate_casf_docking_power.py  # CASF benchmark evaluation
│   │
│   ├── ProtMD/                   # Core neural network implementations
│   │   ├── egnn/                 # E(n) Equivariant GNN modules
│   │   │   ├── egnn_pytorch.py            # Standard EGNN implementation
│   │   │   ├── egnn_pytorch_gbsa.py       # Coordinate-sensitive EGNN
│   │   │   ├── egnn_pytorch_geometric.py  # PyTorch Geometric version
│   │   │   └── utils.py                   # Model utilities (Regressor, Classifier)
│   │   └── utils/                # Training utilities
│   │
│   ├── egnn/                     # Additional GNN implementations
│   │   ├── gcl.py               # Graph Contrastive Learning
│   │   └── gnn.py               # Standard GNN layers
│   │
│   ├── splits/                   # Dataset split definitions
│   │   ├── generate_coremd_kfold.py      # Generate CoreMD k-fold splits
│   │   ├── generate_plas20k_kfold.py     # Generate PLAS-20k splits
│   │   └── match_atom3d_splits.py        # Match ATOM3D split definitions
│   │
│   └── notebooks/                # Jupyter notebooks for analysis
│
├── README.md                     # This file
├── LICENSE                       # MIT License
├── CONTRIBUTING.md               # Contribution guidelines
└── requirements.txt              # Python dependencies
```

## Quick Start

### 1. Download Data

Download the dataset from HuggingFace:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download the complete dataset (~90 GB)
huggingface-cli download llnl/SurGBSA --repo-type dataset --local-dir ./data

# Extract the MD trajectories
cd ./data
tar -xzf md.tar.gz

# After extraction, your data directory will contain:
# ./data/
#   ├── md/          - Raw MD simulation trajectories for all systems
#   ├── ml_data/     - Pre-processed ML-ready numpy arrays
#   ├── splits/      - Train/val/test splits
#   └── weights/     - Pre-trained model checkpoints
```

**Directory structure:**
```
data/
├── md/                          # Raw MD simulation data
│   └── {pdb_id}/                # One directory per PDB structure (e.g., 1a30)
│       └── p{N}/                # Pose directory (p0=crystal, p1-p5=docked)
│           ├── com.prmtop       # AMBER topology file
│           ├── com.nc           # NetCDF trajectory file
│           └── frames.dat       # MM-GBSA scores per frame
├── ml_data/                     # Pre-processed numpy arrays
│   └── {pdb_id}-{rep}-p{N}_prot-md-input.npy
├── splits/                      # Train/val/test split definitions
│   ├── coreMD-fold-0-train.csv
│   ├── coreMD-fold-0-val.csv
│   ├── coreMD-fold-0-test.csv
│   ├── pdbid_list.csv
│   └── pose_list.csv
└── weights/                     # Pre-trained model checkpoints
    └── best_model-epoch-{N}.pt
```

### 2. Working with the MD Data

The downloaded `md/` directory contains organized MD trajectories:

```bash
# Example: Explore the data structure
ls ./data/md/                    # Lists all PDB IDs (e.g., 1a30, 1a42, ...)
ls ./data/md/1a30/              # Lists pose directories (p0, p1, p2, ...)
ls ./data/md/1a30/p0/           # Shows com.prmtop, com.nc, frames.dat

# Load trajectory data with MDAnalysis (Python)
import MDAnalysis as mda
u = mda.Universe("./data/md/1a30/p0/com.prmtop", "./data/md/1a30/p0/com.nc")
print(f"Number of atoms: {u.atoms.n_atoms}")
print(f"Number of frames: {len(u.trajectory)}")

# Load MM-GBSA scores
import numpy as np
gbsa_scores = np.loadtxt("./data/md/1a30/p0/frames.dat")
```

**Pose numbering:**
- `p0` = Crystal structure pose
- `p1-p5` = Docked poses (ranked by docking score)

### 3. Pretraining

Pre-train a model on MD trajectories with self-supervised tasks using distributed training:

```bash
# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Run distributed pretraining (used in paper)
python -m torch.distributed.launch --nproc_per_node=4 \
  sur_gbsa/pretrain_distributed.py \
  --dataset md-dock_top_5+crystal \
  --data_path ./data/md \
  --split_path ./data/splits \
  --batch_size 256 \
  --lr 1e-4 \
  --epochs 100 \
  --pretrain_tasks order,rmsd,pose
```

**Note**: `pretrain.py` (single-GPU version) is provided for convenience but was not used in the paper. Use `pretrain_distributed.py` for reproducing paper results.

### 4. MM-GBSA Training

Train models for MM-GBSA score prediction (main task in the paper):

**For EGMN models:**
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  sur_gbsa/finetune_distributed.py \
  --dataset md-dock_top_5+crystal \
  --data_path ./data/md \
  --split_path ./data/splits \
  --pretrain ./data/weights/best_model-epoch-574.pt \
  --save_path ./results/mmgbsa_egmn \
  --batch_size 256 \
  --lr 1e-4 \
  --epochs 100
```

**For GNN/EGNN models:**
```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  sur_gbsa/finetune_distributed_egnn.py \
  --dataset md-dock_top_5+crystal \
  --data_path ./data/md \
  --split_path ./data/splits \
  --pretrain ./data/weights/best_model-epoch-574.pt \
  --save_path ./results/mmgbsa_egnn \
  --batch_size 256 \
  --lr 1e-4 \
  --epochs 100
```

### 5. Fine-tuning for Binding Affinity and Pose Ranking

Fine-tune on binding affinity or pose ranking tasks:

**Binding Affinity Prediction:**
```bash
python sur_gbsa/finetune_affinity.py \
  --pretrain ./data/weights/best_model-epoch-574.pt \
  --dataset md-crystal \
  --data-dir ./data/md \
  --train-split ./data/splits/coreMD-fold-0-train.csv \
  --val-split ./data/splits/coreMD-fold-0-val.csv \
  --test-split ./data/splits/coreMD-fold-0-test.csv \
  --save_path ./results/affinity_model \
  --lr 1e-4 \
  --batch_size 64 \
  --epochs 500
```

**Pose Ranking (CASF-2016):**
```bash
python sur_gbsa/finetune_decoy_pose_ranking.py \
  --pretrain ./data/weights/best_model-epoch-574.pt \
  --train-split ./casf_splits/train_pdbs.txt \
  --val-split ./casf_splits/val_pdbs.txt \
  --test-split ./casf_splits/test_pdbs.txt \
  --casf-root /path/to/CASF-2016 \
  --save-path ./results/pose_ranking \
  --batch-size 32 \
  --lr 1e-4 \
  --epochs 50
```

### 6. Inference on Structures

Run inference using the pre-trained model on MD trajectories or custom structures:

```bash
# Score an MD trajectory from the downloaded data
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./data/weights/best_model-epoch-574.pt \
  --topology-path ./data/md/1a30/p0/com.prmtop \
  --trajectory-path ./data/md/1a30/p0/com.nc \
  --output predictions.csv \
  --frame-stride 10

# Score your own custom PDB file
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./data/weights/best_model-epoch-574.pt \
  --input-paths protein_ligand.pdb \
  --output predictions.csv

# Score multiple custom structures
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./data/weights/best_model-epoch-574.pt \
  --input-paths structure1.pdb structure2.cif structure3.pdb \
  --output batch_predictions.csv
```

### 7. Evaluation

Evaluate a trained model on test set:

```bash
# Evaluate using downloaded splits
python sur_gbsa/test.py \
  --checkpoint ./data/weights/best_model-epoch-574.pt \
  --dataset md-crystal \
  --data_path ./data/md \
  --split_path ./data/splits/coreMD-fold-0-test.csv
```

## Usage Examples

### MM-GBSA Prediction (Main Paper Results)

Train models for MM-GBSA score prediction using distributed training:

**EGMN Model (used in paper):**
```bash
# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Train EGMN for MM-GBSA prediction
python -m torch.distributed.launch --nproc_per_node=4 \
  sur_gbsa/finetune_distributed.py \
  --dataset md-dock_top_5+crystal \
  --data_path ./data/md \
  --split_path ./data/splits \
  --pretrain ./data/weights/best_model-epoch-574.pt \
  --save_path ./results/mmgbsa_egmn \
  --batch_size 256 \
  --lr 1e-4 \
  --epochs 100 \
  --seed 0
```

**GNN/EGNN Models:**
```bash
# Train GNN/EGNN for MM-GBSA prediction
python -m torch.distributed.launch --nproc_per_node=4 \
  sur_gbsa/finetune_distributed_egnn.py \
  --dataset md-dock_top_5+crystal \
  --data_path ./data/md \
  --split_path ./data/splits \
  --pretrain ./data/weights/best_model-epoch-574.pt \
  --save_path ./results/mmgbsa_egnn \
  --batch_size 256 \
  --lr 1e-4 \
  --epochs 100 \
  --seed 0
```

**Key Parameters:**
- `--dataset`: Dataset variant (e.g., `md-crystal`, `md-dock_top_5+crystal`)
- `--pretrain`: Path to pretrained checkpoint from `pretrain_distributed.py`
- `--data_path`: Directory containing MD trajectories
- `--split_path`: Directory containing train/val/test split CSV files

### Fine-tuning for Binding Affinity Prediction

Fine-tune a model for binding affinity prediction using `finetune_affinity.py`:

```bash
# Fine-tune from pretrained checkpoint
python sur_gbsa/finetune_affinity.py \
  --pretrain ./data/weights/best_model-epoch-574.pt \
  --dataset md-crystal \
  --data-dir ./data/md \
  --train-split ./data/splits/coreMD-fold-0-train.csv \
  --val-split ./data/splits/coreMD-fold-0-val.csv \
  --test-split ./data/splits/coreMD-fold-0-test.csv \
  --save_path ./results/affinity_model \
  --lr 1e-4 \
  --batch_size 64 \
  --epochs 500 \
  --seed 0

# Train from scratch (no pretraining)
python sur_gbsa/finetune_affinity.py \
  --dataset md-crystal \
  --data-dir ./data/md \
  --train-split ./data/splits/coreMD-fold-0-train.csv \
  --val-split ./data/splits/coreMD-fold-0-val.csv \
  --test-split ./data/splits/coreMD-fold-0-test.csv \
  --save_path ./results/from_scratch \
  --lr 1e-4 \
  --batch_size 64 \
  --epochs 500 \
  --seed 0
```

**Key Parameters:**
- `--pretrain`: Path to pretrained checkpoint (optional, trains from scratch if not provided)
- `--dataset`: Dataset type (e.g., `md-crystal`, `md-dock_top_5`, `pdbbind-30`)
- `--data-dir`: Path to directory containing MD trajectories
- `--train-split/--val-split/--test-split`: Paths to CSV files defining splits
- `--use_residue_features`: Enable residue-level features (advanced)
- `--linear_probe`: Freeze encoder and only train the regression head

### Fine-tuning for Pose Ranking (CASF-2016)

Fine-tune a model for the CASF-2016 decoy pose ranking task using `finetune_decoy_pose_ranking.py`:

```bash
python sur_gbsa/finetune_decoy_pose_ranking.py \
  --pretrain ./data/weights/best_model-epoch-574.pt \
  --train-split ./data/casf_splits/train_pdbs.txt \
  --val-split ./data/casf_splits/val_pdbs.txt \
  --test-split ./data/casf_splits/test_pdbs.txt \
  --casf-root /path/to/CASF-2016 \
  --save-path ./results/pose_ranking \
  --batch-size 32 \
  --lr 1e-4 \
  --epochs 50 \
  --ranking-margin 1.5 \
  --ranking-weight 0.5 \
  --seed 42
```

**Key Parameters:**
- `--pretrain`: Path to pretrained checkpoint
- `--casf-root`: Path to CASF-2016 dataset directory
- `--train-split/--val-split/--test-split`: Text files with PDB IDs (one per line)
- `--ranking-margin`: Margin for pairwise ranking loss (default: 1.5)
- `--ranking-weight`: Weight balancing ranking loss vs MSE loss (default: 0.5)
- `--linear-probe`: Only train regression head, freeze encoder
- `--use-residue-features`: Enable residue-level features

**Metrics computed:**
- Success rate (Top-1): % of complexes where best-scored pose has RMSD ≤ 2.0 Å
- Mean Spearman correlation: Average correlation between predicted scores and true RMSDs

### Distributed Training Notes

All main training scripts (`pretrain_distributed.py`, `finetune_distributed.py`, `finetune_distributed_egnn.py`) use PyTorch DDP for multi-GPU training:

```bash
# Set required environment variables
export MASTER_ADDR=localhost  # or hostname of rank 0 node
export MASTER_PORT=29500      # any available port

# Launch with desired number of GPUs
python -m torch.distributed.launch --nproc_per_node=N_GPUS script.py [args]
```

**Important**: The paper results use distributed training scripts. Single-GPU versions are provided for convenience but were not used in the published work.

### Extract Learned Representations

Extract embeddings from a trained model:

```bash
python sur_gbsa/extract.py \
  --checkpoint ./data/weights/best_model-epoch-574.pt \
  --dataset md-crystal \
  --data_path ./data/md \
  --split_path ./data/splits/coreMD-fold-0-test.csv \
  --output ./embeddings/embeddings.pt
```

### Pose Ranking

Rank multiple protein-ligand binding poses:

```bash
python sur_gbsa/rank_poses.py \
  --checkpoint ./results/finetuned_model/best_model.pt \
  --input_poses ./poses/*.pdb \
  --output ./ranked_poses.csv
```

### Running Inference on Custom Structures

The `score_mdanalysis.py` script allows you to run inference on any MDAnalysis-compatible structure or trajectory format (PDB, CIF, mmCIF, XTC, DCD, NC, etc.).

#### Basic Usage - Single Structure

Score a single protein-ligand complex:

```bash
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./data/weights/best_model-epoch-574.pt \
  --input-paths complex.pdb \
  --output predictions.csv
```

#### Multiple Structures

Score multiple structures in batch:

```bash
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./data/weights/best_model-epoch-574.pt \
  --input-paths structure1.pdb structure2.cif structure3.mmcif \
  --output batch_predictions.csv \
  --batch-size 16
```

#### MD Trajectory Analysis

Score all frames in an MD trajectory from the downloaded data:

```bash
# Example: Score the 1a30 crystal pose (p0) trajectory
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./data/weights/best_model-epoch-574.pt \
  --topology-path ./data/md/1a30/p0/com.prmtop \
  --trajectory-path ./data/md/1a30/p0/com.nc \
  --output 1a30_trajectory_scores.csv \
  --frame-stride 10 \
  --max-frames 100
```

#### Advanced Options

Customize ligand detection and pocket definition:

```bash
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./data/weights/best_model-epoch-574.pt \
  --input-paths complex.pdb \
  --output predictions.csv \
  --ligand-selection "resname LIG" \
  --protein-selection "protein and not resname LIG" \
  --pocket-cutoff 8.0 \
  --max-len 800
```

#### Output Formats

Save as CSV (human-readable):
```bash
--output predictions.csv
```

Save as PyTorch tensor (for further analysis):
```bash
--output predictions.pt
```

#### Key Parameters

- `--ckpt`: Path to trained model checkpoint (required)
- `--input-paths`: List of structure files for batch inference
- `--topology-path`: Topology file for trajectory mode (e.g., .prmtop, .psf)
- `--trajectory-path`: Trajectory file for trajectory mode (e.g., .nc, .xtc, .dcd)
- `--output`: Output file path (.csv or .pt)
- `--ligand-selection`: MDAnalysis selection string for ligand atoms
- `--protein-selection`: MDAnalysis selection string for protein atoms (default: "protein")
- `--pocket-cutoff`: Distance cutoff (Å) for pocket residue selection (default: 6.0)
- `--max-len`: Maximum number of atoms to include (default: 600)
- `--frame-stride`: Process every Nth frame in trajectory mode (default: 1)
- `--max-frames`: Maximum number of frames to process from trajectory
- `--batch-size`: Number of structures to process in parallel (default: 8)
- `--device`: Device to use ("cuda" or "cpu")

#### Example Output (CSV)

```csv
sample_id,structure_id,pose,frame,source_file,y_pred
0,complex,None,None,complex.pdb,-7.234
1,traj_frame_0,None,0,trajectory.nc,-6.891
2,traj_frame_10,None,10,trajectory.nc,-7.102
```

The `y_pred` column contains the predicted binding affinity (or MM-GBSA score depending on the model).

## Supported Datasets

SurGBSA supports multiple datasets for different tasks:

### MM-GBSA Prediction Datasets

- `md-crystal` - MD trajectories from crystal structures
- `md-dock_top_1` - MD trajectories from top docked pose
- `md-dock_top_5` - MD trajectories from top 5 docked poses
- `sp-*` variants - Single-point (single frame) versions

### Binding Affinity Prediction Datasets

- `pdbbind-30` - PDBBind with 30% sequence identity split (ATOM3D)
- `pdbbind-60` - PDBBind with 60% sequence identity split (ATOM3D)
- `PLAS-20k` - PLAS-20k protein-ligand affinity dataset
- `misato` - MISATO MD dataset

### Dataset Configuration

Datasets are automatically configured in `sur_gbsa/data_utils.py`. Key parameters:

- `max_frames` - Number of MD frames to use (1 for single-point, 1000 for full trajectory)
- `pose_list` - Which poses to include (0=crystal, 1-5=docked poses)
- `target` - Prediction target (`GBSA` or `affinity`)

## Model Architecture

SurGBSA uses **E(n) Equivariant Graph Neural Networks (EGNN)** which are:

- **Rotation and translation invariant** - Predictions unchanged by rigid transformations
- **Coordinate-aware** - Directly processes 3D atomic coordinates
- **Scalable** - Handles variable-size protein-ligand complexes

### Key Components

1. **Encoder**: EGNN layers that process atom features and 3D coordinates
2. **Regressor Head**: Predicts continuous values (binding affinity, GBSA scores)
3. **Self-Supervised Tasks** (pretraining):
   - `order` - Predict temporal order of MD frames
   - `rmsd` - Predict RMSD between frames
   - `pose` - Contrastive learning across different poses
   - `pdbid` - Contrastive learning across protein systems

## Citation

If you use SurGBSA in your research, please cite:

```bibtex
@ARTICLE{Jones2025-dj,
  title         = "{SurGBSA}: Learning representations from molecular dynamics
                   simulations",
  author        = "Jones, Derek and Yang, Yue and Lightstone, Felice C and
                   Moshiri, Niema and Allen, Jonathan E and Rosing, Tajana S",
  journal       = "arXiv [q-bio.BM]",
  month         =  sep,
  year          =  2025,
  archivePrefix = "arXiv",
  primaryClass  = "q-bio.BM",
  eprint        = "2509.03084"
}
```

**Paper**: Jones, D., Yang, Y., Lightstone, F. C., Moshiri, N., Allen, J. E., & Rosing, T. S. (2025). SurGBSA: Learning representations from molecular dynamics simulations. *arXiv preprint arXiv:2509.03084*. [https://arxiv.org/abs/2509.03084](https://arxiv.org/abs/2509.03084)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Release**: LLNL-CODE-2016774

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## Acknowledgments

This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.

The EGNN implementation is adapted from [ProtMD](https://github.com/smiles724/ProtMD) and the original [EGNN](https://github.com/vgsatorras/egnn) repository.
