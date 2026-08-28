################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################

# adapted from https://github.com/vgsatorras/egnn

import torch
from torch import nn
from sur_gbsa.egnn.gcl import GCL


class GNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_nf,
        out_node_nf=3,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        attention=0,
        recurrent=False,
    ):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        # self.add_module("gcl_0", GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=recurrent))
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_nf=1,
                    act_fn=act_fn,
                    attention=attention,
                    recurrent=recurrent,
                ),
            )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, out_node_nf)
        )
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.to(self.device)

    def forward(self, nodes, edges, edge_attr=None):
        # return nodes
        # import pdb
        # pdb.set_trace()
        h = self.embedding(nodes)

        # print(f"embedding shape: {h.shape}")
        # torch.save(h, "h.pt")
        # torch.save(edges, "edges.pt")
        # torch.save(edge_attr, "edge_attr.pt")
        # torch.save(self._modules["gcl_0"], "gnn_layer_0.pt")
        # h, _ = self._modules["gcl_0"](h, edges, edge_attr=edge_attr)

        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
            # print(f"embedding shape: {h.shape}")

        # '''
        # return h
        return self.decoder(h)
        # '''

        # return h
