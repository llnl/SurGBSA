# SurGBSA

Official implementation of **"SurGBSA: Learning Representations From Molecular Dynamics Simulations"**

[![arXiv](https://img.shields.io/badge/arXiv-2509.03084-b31b1b.svg)](https://arxiv.org/abs/2509.03084)
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

- **MD Trajectories**: Protein-ligand MD simulations from PDBBind CoreSet
- **MM-GBSA Scores**: Energy calculations for binding affinity prediction
- **ML-Ready Files**: Preprocessed HDF5 files for direct model training
- **Splits**: Train/validation/test splits for reproducibility
- **Pre-trained Checkpoints**: Models pre-trained on MD data with self-supervised tasks

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
│   ├── pretrain.py               # Single-GPU pretraining script
│   ├── pretrain_distributed.py   # Multi-GPU distributed pretraining
│   ├── finetune_affinity.py      # Finetune for binding affinity prediction
│   ├── finetune_distributed-egnn.py  # Distributed finetuning for EGNN
│   ├── finetune_pose_ranking.py  # Finetune for pose ranking
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

Download the preprocessed datasets from HuggingFace:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download the dataset (adjust path as needed)
huggingface-cli download llnl/SurGBSA --repo-type dataset --local-dir ./data
```

### 2. Pretraining (Optional)

Pre-train a model on MD trajectories with self-supervised tasks:

```bash
python sur_gbsa/pretrain.py \
  --dataset md-dock_top_5+crystal \
  --batch_size 128 \
  --lr 1e-4 \
  --epochs 100 \
  --pretrain_tasks order,rmsd,pose
```

### 3. Fine-tuning

Fine-tune on binding affinity prediction (e.g., PDBBind with ATOM3D splits):

```bash
python sur_gbsa/finetune_affinity.py \
  --pretrain ./checkpoints/pretrained_model.pt \
  --dataset pdbbind-30 \
  --save_path ./results/finetuned_model \
  --lr 1e-4 \
  --batch_size 64 \
  --epochs 500 \
  --seed 0
```

### 4. Inference on Custom Structures

Run inference on your own protein-ligand structures using MDAnalysis-compatible formats:

```bash
# Score a single PDB file
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./checkpoints/finetuned_model.pt \
  --input-paths protein_ligand.pdb \
  --output predictions.csv

# Score multiple structures
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./checkpoints/finetuned_model.pt \
  --input-paths structure1.pdb structure2.cif structure3.pdb \
  --output predictions.csv

# Score an MD trajectory
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./checkpoints/finetuned_model.pt \
  --topology-path protein.prmtop \
  --trajectory-path trajectory.nc \
  --output trajectory_scores.csv
```

### 5. Evaluation

Evaluate a trained model on test set:

```bash
python sur_gbsa/test.py \
  --checkpoint ./results/finetuned_model/best_model.pt \
  --dataset pdbbind-30 \
  --split test
```

## Usage Examples

### Training from Scratch

Train a model without pretraining:

```bash
python sur_gbsa/finetune_affinity.py \
  --dataset pdbbind-30 \
  --save_path ./results/from_scratch \
  --lr 1e-4 \
  --batch_size 64 \
  --epochs 500 \
  --seed 0
```

### Distributed Training (Multi-GPU)

For multi-GPU training using PyTorch DDP:

```bash
# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Launch distributed training (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 \
  sur_gbsa/pretrain_distributed.py \
  --dataset md-dock_top_5+crystal \
  --batch_size 256 \
  --lr 1e-4 \
  --epochs 100
```

### Extract Learned Representations

Extract embeddings from a trained model:

```bash
python sur_gbsa/extract.py \
  --checkpoint ./results/finetuned_model/best_model.pt \
  --dataset pdbbind-30 \
  --output ./embeddings/pdbbind_embeddings.pt
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
  --ckpt ./checkpoints/model.pt \
  --input-paths complex.pdb \
  --output predictions.csv
```

#### Multiple Structures

Score multiple structures in batch:

```bash
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./checkpoints/model.pt \
  --input-paths structure1.pdb structure2.cif structure3.mmcif \
  --output batch_predictions.csv \
  --batch-size 16
```

#### MD Trajectory Analysis

Score all frames in an MD trajectory:

```bash
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./checkpoints/model.pt \
  --topology-path system.prmtop \
  --trajectory-path trajectory.nc \
  --output trajectory_scores.csv \
  --frame-stride 10 \
  --max-frames 100
```

#### Advanced Options

Customize ligand detection and pocket definition:

```bash
python sur_gbsa/score_mdanalysis.py \
  --ckpt ./checkpoints/model.pt \
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
