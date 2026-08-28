################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################
#Script to run SurGBSA models on MDAnalysis readable files (.nc,.xtc,.pdb,.cif,etc)


import argparse
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader

from sur_gbsa.ProtMD.egnn import EGNN_Network, Regressor
from mda_inference_dataset import MDAnalysisInferenceDataset, SampleMetadata


def load_model_from_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = checkpoint["args"]

    if "use_residue_features" in vars(train_args):
        use_residue_features = train_args.use_residue_features
    else:
        use_residue_features = False

    encoder = EGNN_Network(
        num_tokens=train_args.tokens,
        dim=train_args.dim,
        depth=train_args.depth,
        num_nearest_neighbors=train_args.num_nearest,
        dropout=train_args.dropout,
        global_linear_attn_every=1,
        norm_coors=True,
        coor_weights_clamp_value=2.0,
        aggregate=False,
        num_residue_tokens=22 if use_residue_features else None,
        residue_dim=32 if use_residue_features else None,
    ).to(device)

    finetune = Regressor(train_args.dim).to(device)
    model = torch.nn.ModuleDict({
        "encoder": encoder,
        "finetune": finetune,
    }).float()

    state_dict = checkpoint["model"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        if "encoder" in new_key and not new_key.startswith("encoder."):
            new_key = "encoder." + new_key.split("encoder.")[-1]
        if "finetune" in new_key and not new_key.startswith("finetune."):
            new_key = "finetune." + new_key.split("finetune.")[-1]
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    return model, train_args, use_residue_features


def metadata_to_row(meta):
    if isinstance(meta, SampleMetadata):
        return {
            "sample_id": meta.sample_id,
            "structure_id": meta.structure_id,
            "pose": meta.pose,
            "frame": meta.frame,
            "source_file": meta.source_file,
            "topology_file": meta.topology_file,
            "trajectory_file": meta.trajectory_file,
        }

    return {
        "sample_id": str(meta),
        "structure_id": str(meta),
        "pose": None,
        "frame": None,
        "source_file": str(meta),
        "topology_file": None,
        "trajectory_file": None,
    }


def run_inference(model, dataset, batch_size, device, use_residue_features):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    y_pred = []
    latent_list = []
    idx_list = []

    with torch.no_grad():
        for batch in loader:
            if use_residue_features:
                x, res_feats, pos, y, idx = batch
                res_feats = res_feats.to(device)
            else:
                x, pos, y, idx = batch
                res_feats = None

            x = x.long().to(device)
            pos = pos.float().to(device)

            mask = x != 0
            out = model["encoder"](x, pos, res_feats=res_feats, mask=mask)[1]
            out = out.mean(dim=1)
            pred = model["finetune"](out)

            y_pred.append(pred.detach().cpu())
            latent_list.append(out.detach().cpu())
            idx_list.append(idx.detach().cpu())

    y_pred = torch.cat(y_pred).reshape(-1, 1).cpu() * -10
    latent = torch.cat(latent_list)
    idx = torch.cat(idx_list).reshape(-1)

    rows = []
    for i in idx.tolist():
        meta = dataset.__getmetadata__(i)
        row = metadata_to_row(meta)
        row["idx"] = i
        row["y_pred"] = float(y_pred[i].item())
        rows.append(row)

    df = pd.DataFrame(rows)

    return {
        "y_pred": y_pred.squeeze(),
        "idx": idx,
        "latent": latent,
        "metadata": [dataset.__getmetadata__(i) for i in idx.tolist()],
        "df": df,
        "dataset_size": len(dataset),
        "n_predictions": len(y_pred),
    }


def main():
    parser = argparse.ArgumentParser(description="Run on-the-fly MDAnalysis inference on raw structures")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint")
    parser.add_argument("--input-paths", nargs="*", default=None, help="List of pdb/cif/mmcif files")
    parser.add_argument("--topology-path", type=str, default=None, help="Topology file for trajectory mode")
    parser.add_argument("--trajectory-path", type=str, default=None, help="Trajectory file for trajectory mode")
    parser.add_argument("--output", required=True, type=str, help="Output .pt or .csv path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=600)
    parser.add_argument("--pocket-cutoff", type=float, default=6.0)
    parser.add_argument("--protein-selection", type=str, default="protein")
    parser.add_argument("--ligand-selection", type=str, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--legacy-metadata-string", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    device = torch.device(args.device)
    model, train_args, use_residue_features = load_model_from_checkpoint(args.ckpt, device)

    dataset = MDAnalysisInferenceDataset(
        input_paths=args.input_paths,
        topology_path=args.topology_path,
        trajectory_path=args.trajectory_path,
        max_len=args.max_len,
        use_residue_features=use_residue_features,
        ligand_selection=args.ligand_selection,
        protein_selection=args.protein_selection,
        pocket_cutoff=args.pocket_cutoff,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        return_legacy_metadata_string=args.legacy_metadata_string,
        verbose=args.verbose,
    )

    result = run_inference(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        use_residue_features=use_residue_features,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        result["df"].to_csv(output_path, index=False)
    else:
        torch.save(result, output_path)

    print(f"Saved predictions to {output_path}")
    print(f"Num samples: {result['n_predictions']}")


if __name__ == "__main__":
    main()