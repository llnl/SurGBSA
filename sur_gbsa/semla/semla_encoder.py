import torch
import torch.nn as nn

class SemlaEncoderWrapper(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        dynamics,
        use_residue_features=False,
        num_residue_tokens=22,
        residue_dim=32,
    ):
        super().__init__()
        self.dim = dim
        self.use_residue_features = use_residue_features

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.prompt_embed = None

        if self.use_residue_features:
            self.residue_emb = nn.Embedding(num_residue_tokens, residue_dim)
            self.residue_projection = nn.Linear(dim + residue_dim, dim)

        self.dynamics = dynamics

    def build_adj_matrix(self, mask):
        adj = mask[:, :, None] & mask[:, None, :]
        eye = torch.eye(mask.shape[-1], device=mask.device, dtype=torch.bool).unsqueeze(0)
        adj = adj & ~eye
        return adj.float()

    def forward(
        self,
        x,
        pos,
        res_feats=None,
        prompt=None,
        adj_mat=None,
        mask=None,
        edge_feats=None,
        return_coor_changes=False,
        coors_only=False,
    ):
        b, n = x.shape[:2]
        device = x.device

        if mask is None:
            mask = x != 0
        mask = mask.bool()

        feats = self.token_emb(x.long())

        if self.use_residue_features and res_feats is not None:
            res_embedded = self.residue_emb(res_feats.long())
            feats = torch.cat([feats, res_embedded], dim=-1)
            feats = self.residue_projection(feats)

        if adj_mat is None:
            adj_mat = self.build_adj_matrix(mask)
        else:
            if adj_mat.dim() == 2:
                adj_mat = adj_mat.unsqueeze(0).expand(b, -1, -1)
            adj_mat = adj_mat.bool()

        out = self.dynamics(
            coords=pos.float(),
            inv_feats=feats,
            adj_matrix=adj_mat,
            atom_mask=mask,
            edge_feats=edge_feats,
        )

        if len(out) == 2:
            out_coords, out_feats = out
            out_edges = None
        elif len(out) == 3:
            out_coords, out_feats, out_edges = out
        else:
            raise RuntimeError(f"Unexpected dynamics output length: {len(out)}")

        if coors_only:
            return out_coords

        if return_coor_changes:
            return out_coords, out_feats, out_edges if out_edges is not None else None

        return out_coords, out_feats