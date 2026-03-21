# SurGBSA 

Official implementation of [SurGBSA: Learning Representations From Molecular Dynamics Simulations](https://arxiv.org/abs/2509.03084)


## Organization

* `data_utils.py`: utilities for dataloading 
* `distributed_utils.py`: utilities for distributed communication
* `extract.py`: script that loads a model checkpoint and extracts the embeddings for a user-specified dataset
* `finetune_distributed-egnn.py`: distributed training script for the GNN and EGNN models
* `finetune_distributed.py`: distributed training script for the EGMN models
* `finetune_distributed-plus.py`: upated training script with bug fixes
* `pretrain.py`: pretraining script
* `reorganize_results.py`: scans result directory files and computes metrics then saves to a smaller summary file
* `sample.py`: loads a model checkpoint and samples trajectories
* `test.py`: loads a model checkpoint and perform inference while capturing timing information
* `utils.py`: general utilities
* `distributed_example/`: contains a simple distributed training example to help configure environment
* `egnn/`: contains the implementation of the GNN and EGNN networks
* `lightning/`: pytorch lightning models and utilities
* `notebooks/`: jupyter notebooks for data analysis
* `scripts/`: scripts for training
* `splits/`: scripts for generating the training splits and the dataset information