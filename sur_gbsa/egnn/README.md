# E(n) Equivariant Graph Neural Networks

<!--Official implementation (Pytorch 1.7.1) of:  -->

\*adapted from <https://github.com/vgsatorras/egnn>\*

**E(n) Equivariant Graph Neural Networks**  
Victor Garcia Satorras, Emiel Hogeboom, Max Welling  
https://arxiv.org/abs/2102.09844


### Example code
For a simple example of a EGNN implementation [click here](https://github.com/vgsatorras/egnn/blob/3c079e7267dad0aa6443813ac1a12425c3717558/models/egnn_clean/egnn_clean.py#L106). Or copy the file `models/egnn_clean/egnn_clean.py` into your working directory and run:

```python
import egnn_clean as eg
import torch

# Dummy parameters
batch_size = 8
n_nodes = 4
n_feat = 1
x_dim = 3

# Dummy variables h, x and fully connected edges
h = torch.ones(batch_size * n_nodes, n_feat)
x = torch.ones(batch_size * n_nodes, x_dim)
edges, edge_attr = eg.get_edges_batch(n_nodes, batch_size)

# Initialize EGNN
egnn = eg.EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

# Run EGNN
h, x = egnn(h, x, edges, edge_attr)
```

If you are using the EGNN in a new application we recommend checking the EGNN [attributes description](https://github.com/vgsatorras/egnn/blob/3c079e7267dad0aa6443813ac1a12425c3717558/models/egnn_clean/egnn_clean.py#L119) that contains some upgrades not included in the paper.

#### Acknowledgements
The Robert Bosch GmbH is acknowledged for financial support.

