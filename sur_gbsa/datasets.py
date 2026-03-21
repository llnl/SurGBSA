import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import MDAnalysis as mda
from MDAnalysis.core.universe import Merge
from tqdm import tqdm
import h5py
import pickle

atom_dict = {"C": 6, "H": 1, "O": 8, "N": 7, "S": 9, "P": 10, "F": 11, "I": 12, "B": 13}


def load_misato_split(
    path="/p/vast1/mldrug/misato/misato_MD.hdf5",
    pdb_list=[],
    output_file=None,
    dataset_name="misato",
    max_len=600,
    debug=False
):
    """Load MISATO dataset for given PDB IDs with max length enforcement"""
    
    peptide_list = pd.read_csv("/g/g13/jones289/workspace/pretrain_md/sur_gbsa/misato/peptides.txt", header=None)[0].values.tolist()

    with open("/g/g13/jones289/workspace/pretrain_md/sur_gbsa/misato/atoms_residue_map.pickle", 'rb') as handle:
        res_map = pickle.load(handle)

    f = h5py.File(path, 'r')
    file_pdbids = list(f)

    x_all, pos_all, y_all, resnames_all = [], [], [], []  # NEW: added resnames_all
    pdb_ids_processed = []
    skipped_counts = {'peptide': 0, 'missing': 0, 'no_valid_atoms': 0, 'truncated': 0}
    
    for pdbid in tqdm(pdb_list, desc="Processing PDB IDs"):
        
        if pdbid in peptide_list:
            skipped_counts['peptide'] += 1
            continue
        elif pdbid.upper() not in file_pdbids:
            skipped_counts['missing'] += 1
            continue
        
        dataset = f[pdbid.upper()]
        pbsa = torch.tensor(dataset['frames_interaction_energy'][:])
        atom_element = dataset['atoms_element'][:]
        atom_pos = dataset['trajectory_coordinates'][:]
        atom_residue = dataset['atoms_residue'][:]  # NEW: get residue IDs
        
        lig_mask = np.array([res_map[x] == "MOL" for x in atom_residue])
        
        if not lig_mask.any():
            skipped_counts['no_valid_atoms'] += 1
            continue
        
        # Process each frame
        for frame_idx in range(atom_pos.shape[0]):
            ligand_coords = atom_pos[frame_idx, lig_mask]
            protein_coords = atom_pos[frame_idx, ~lig_mask]
            
            # Distance to any ligand atom
            from scipy.spatial.distance import cdist
            distances = cdist(protein_coords, ligand_coords).min(axis=1)
            
            # Get pocket protein atoms
            protein_pocket_mask = distances <= 6.0
            protein_pocket_indices = np.where(~lig_mask)[0][protein_pocket_mask]
            
            # Get ligand indices
            ligand_indices = np.where(lig_mask)[0]
            
            # Combine: all ligand atoms + pocket protein atoms
            num_ligand = len(ligand_indices)
            num_pocket_protein = len(protein_pocket_indices)
            total_atoms = num_ligand + num_pocket_protein
            
            # Handle max_len constraint
            if total_atoms > max_len:
                # Keep all ligand atoms, truncate protein atoms
                num_protein_to_keep = max_len - num_ligand
                if num_protein_to_keep < 0:
                    # Even ligand alone exceeds max_len - skip this frame
                    skipped_counts['truncated'] += 1
                    continue
                protein_pocket_indices = protein_pocket_indices[:num_protein_to_keep]
                skipped_counts['truncated'] += 1
            
            # Combine indices: ligand first, then pocket protein
            combined_indices = np.concatenate([ligand_indices, protein_pocket_indices])
            
            filtered_pos = atom_pos[frame_idx, combined_indices]
            filtered_atoms = atom_element[combined_indices]
            filtered_residues = atom_residue[combined_indices]  # NEW: filter residues
            
            # NEW: Convert residue IDs to residue names using res_map
            residue_names = [res_map[res_id] for res_id in filtered_residues]
            
            # Convert to tensors
            x_tmp = torch.tensor(filtered_atoms, dtype=torch.long)
            pos_tmp = torch.tensor(filtered_pos, dtype=torch.float32)
            
            # Pad if needed
            if len(x_tmp) < max_len:
                # Pad with zeros
                pad_len = max_len - len(x_tmp)
                x_tmp = torch.cat([x_tmp, torch.zeros(pad_len, dtype=torch.long)])
                pos_tmp = torch.cat([pos_tmp, torch.zeros(pad_len, 3, dtype=torch.float32)])
                residue_names.extend([''] * pad_len)  # NEW: pad residue names with empty strings
            
            x_all.append(x_tmp)
            pos_all.append(pos_tmp)
            resnames_all.append(residue_names)  # NEW: store residue names as list
            pdb_ids_processed.append(f"{pdbid}_frame{frame_idx}")
            y_all.append(pbsa[frame_idx])

    f.close()
    
    print(f"\nProcessing summary:")
    print(f"  Input PDB IDs: {len(pdb_list)}")
    print(f"  Skipped (peptides): {skipped_counts['peptide']}")
    print(f"  Skipped (missing): {skipped_counts['missing']}")
    print(f"  Skipped (no valid atoms/frames): {skipped_counts['no_valid_atoms']}")
    print(f"  Truncated frames: {skipped_counts['truncated']}")
    print(f"  Successfully processed: {len(x_all)} frames")
    
    if len(x_all) == 0:
        print("ERROR: No valid data processed!")
        return None
    
    # Stack (already padded to same size)
    x = torch.stack(x_all)
    pos = torch.stack(pos_all)
    y = torch.stack(y_all) / -10
    
    if output_file:
        out_path = Path(output_file)
    else:
        out_dir = Path(f"data/{dataset_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{dataset_name}_processed.pt"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    return {
        'x': x,
        'pos': pos,
        'y': y,
        'pdb_ids': pdb_ids_processed,
        'resnames': resnames_all,  # NEW: include residue names (list of lists of strings)
        'num_samples': len(x)
    }


def parse_mol2_names(mol2_file):
    """Extract molecule names from multi-structure MOL2 file."""
    names = []
    with open(mol2_file, 'r') as f:
        in_molecule_block = False
        for line in f:
            if line.startswith('@<TRIPOS>MOLECULE'):
                in_molecule_block = True
                continue
            if in_molecule_block:
                names.append(line.strip())
                in_molecule_block = False
    return names

class BaseMDDataset(Dataset, ABC):
    """Base class for MD datasets with common functionality"""
    
    def __init__(self,
                 max_len=600,
                 use_residue_features=False,
                 rmsd_thresh=2.0,
                 verbose=False):
        
        self.max_len = max_len
        self.use_residue_features = use_residue_features
        self.rmsd_thresh = rmsd_thresh
        self.verbose = verbose
        
        # Will be populated by subclasses
        self.x_0 = None
        self.pos = None
        self.y = None
        self.resname_indices = None
        self.rmsd = None
        self.frame_num = None
        self.file_list = None
        self.metadata_list = None
    
    @abstractmethod
    def _load_data(self):
        """Load and process data - implemented by subclasses"""
        pass


    def _residue_to_indices(self, resname_list):
        """
        Convert residue names to binary indices.
        
        Indices:
        0: padding token (empty string or None)
        1: protein atom (any standard amino acid)
        2: ligand atom (MOL or anything else)
        """
        # Standard amino acids + common variants in MISATO
        STANDARD_RESIDUES = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            # MISATO variants
            'HIE', 'HID', 'HIP',  # Histidine protonation states
            'CYX',  # Disulfide-bonded cysteine
            'ACE', 'NME',  # Capping groups
        ]
        
        indices = []
        for res in resname_list:
            if res == '' or res is None:
                indices.append(0)  # Padding
            elif res in STANDARD_RESIDUES:
                indices.append(1)  # Protein atom
            else:
                # MOL or any other non-standard residue = ligand
                indices.append(2)

        # indices = []
        # for res in resname_list:
        #     if res in self.STANDARD_RESIDUES:
        #         # Add 1 to reserve 0 for padding
        #         idx = self.STANDARD_RESIDUES.index(res) + 1
        #         indices.append(idx)
        #     else:
        #         # Ligand atoms or non-standard residues
        #         indices.append(21)  # "other" category
         
        return torch.tensor(indices, dtype=torch.long)

    def _apply_rmsd_filter(self, mask_initial=None):
        """Apply RMSD and label filtering"""
        mask_list = []
        
        for i in range(len(self.y)):
            # Check initial mask if provided
            if mask_initial is not None and not mask_initial[i]:
                mask_list.append(False)
                continue
            
            # RMSD filter
            if self.rmsd is not None and self.rmsd[i] > self.rmsd_thresh:
                mask_list.append(False)
            # Label range filters (adjust as needed per dataset)
            elif self.y[i] >= -10:
                mask_list.append(False)
            elif self.y[i] < -100:
                mask_list.append(False)
            else:
                mask_list.append(True)
        
        mask = torch.tensor(mask_list).bool()
        
        if self.verbose:
            print(f"Filtered {(~mask).sum()}/{len(mask)} samples")
        
        return mask
    
    def _apply_mask(self, mask):
        """Apply boolean mask to all data tensors"""
        self.x_0 = self.x_0[mask]
        self.pos = self.pos[mask]
        self.y = self.y[mask]
        
        if self.resname_indices is not None:
            self.resname_indices = self.resname_indices[mask]
        
        if self.rmsd is not None:
            self.rmsd = self.rmsd[mask]
        
        if self.frame_num is not None:
            self.frame_num = self.frame_num[mask]
        
        if self.file_list is not None:
            self.file_list = [x for x, keep in zip(self.file_list, mask) if keep]
        
        if self.metadata_list is not None:
            self.metadata_list = [x for x, keep in zip(self.metadata_list, mask) if keep]
    
    def __getitem__(self, i):
        """Standard getitem - works for both datasets"""
        if self.use_residue_features and self.resname_indices is not None:
            return (self.x_0[i], self.resname_indices[i], self.pos[i], 
                    self.y[i], torch.tensor(i))
        else:
            return self.x_0[i], self.pos[i], self.y[i], torch.tensor(i)
    
    def __len__(self):
        return len(self.x_0)
    
    # Common accessor methods
    def __getframenum__(self, i):
        return self.frame_num[i] if self.frame_num is not None else torch.tensor(0)
    
    def __getfilename__(self, i):
        return self.file_list[i] if self.file_list is not None else ""
    
    def __getposenum__(self, i):
        return torch.tensor(0)  # Override in subclass if needed
    
    def __getrmsd__(self, i):
        return self.rmsd[i] if self.rmsd is not None else torch.tensor(0.0)
    
    def __getmetadata__(self, i):
        return self.metadata_list[i] if self.metadata_list is not None else ""
    
    @staticmethod
    def get_num_residue_tokens():
        """Number of residue tokens for embedding layer"""
        return 3  # 0=padding, 1=protein, 2=ligand

class MisatoDataset(BaseMDDataset):
    """MISATO dataset - loads from preprocessed data"""
    
    def __init__(self, 
                 data_path,
                 pdb_path,
                 max_len=600,
                 use_residue_features=False,
                 rmsd_thresh=2.0,
                 verbose=False):
        
        super().__init__(max_len, use_residue_features, rmsd_thresh, verbose)
        
        self.data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.pdb_path = Path(pdb_path) if isinstance(pdb_path, str) else pdb_path
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load MISATO data"""
        # Read PDB list
        pdb_list = pd.read_csv(self.pdb_path, header=None)[0].values.tolist()
        
        # Load or generate data
        data_dict = load_misato_split(pdb_list=pdb_list, max_len=self.max_len)
        
        # Set core attributes
        self.x_0 = data_dict['x']
        self.pos = data_dict['pos']
        self.y = data_dict['y'] / -10  # Flip sign and scale
        self.metadata_list = data_dict['pdb_ids']
        
        # Frame numbers from metadata
        self.frame_num = torch.tensor([
            int(x.split("-")[-1].split("frame")[-1]) 
            for x in self.metadata_list
        ])
        
        # Load residue info if needed
        if self.use_residue_features:
            if 'resnames' in data_dict:
                # If load_misato_split provides residue names
                self.resname_indices = torch.stack([
                    self._residue_to_indices(resnames) 
                    for resnames in data_dict['resnames']
                ])
            else:
                print("Warning: residue features requested but not available in MISATO data")
                self.use_residue_features = False
        
        # RMSD if available
        self.rmsd = data_dict.get('rmsd', None)
        
        # Apply filters if RMSD available
        if self.rmsd is not None:
            mask = self._apply_rmsd_filter()
            self._apply_mask(mask)
        
        if self.verbose:
            print(f"Loaded MISATO dataset: {len(self)} samples")
    
    def __getmetadata__(self, i):
        """MISATO-specific metadata format"""
        pdbid = self.metadata_list[i].split('_')[0]
        pose = 0
        lig_num = 0
        return f"{pdbid}-{pose}-{lig_num}"

class GBSAMDDataset(BaseMDDataset):
    """GBSA MD dataset - loads from individual .npy files"""
    
    def __init__(self,
                 backprop,
                 pdb_list,
                 data_dir,
                 target="GBSA",
                 max_frames=1000,
                 max_time_step_prop=0.9,
                 max_len=600,
                 use_residue_features=False,
                 rmsd_thresh=2.0,
                 label_process="sign-flip",
                 verbose=False,
                 **kwargs):  # Absorb other args
        
        super().__init__(max_len, use_residue_features, rmsd_thresh, verbose)
        
        self.backprop = backprop
        self.pdb_list = pdb_list
        self.data_dir = Path(data_dir)
        self.target = target
        self.max_frames = max_frames
        self.max_time_step_prop = max_time_step_prop
        self.label_process = label_process
        
        # Additional tracking
        self.pose_list = []
        self.pdb_to_idx = {}
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load GBSA MD simulation data"""
        pos_list = []
        x_0_list = []
        resname_indices_list = []
        y_list = []
        rmsd_list = []
        frame_num_list = []
        file_list = []
        
        pdbid_count = 1
        
        for file in self.pdb_list:
            if not file.endswith(".npy"):
                if self.verbose:
                    print(f"Warning: {file} is not a numpy file")
                continue
            
            # Track PDB IDs
            pdbid = file.split("-")[0]
            if pdbid not in self.pdb_to_idx:
                self.pdb_to_idx[pdbid] = pdbid_count
                pdbid_count += 1
            
            # Load file
            d = np.load(self.data_dir / file, allow_pickle=True).item()

            # import pdb; pdb.set_trace() 
            # Extract coordinates and atom types
            file_pos = torch.tensor(d["R"])
            z = torch.tensor([atom_dict[x] for x in d["z"]])
            z = z.repeat(file_pos.shape[0], 1)
            
            pos_list.append(file_pos)
            x_0_list.append(z)
            
            # Pose information
            pose_num = int(file.split("_")[0].split("-")[-1][1])
            self.pose_list.append([pose_num] * len(file_pos))
            
            # Residue features
            if self.use_residue_features:
                resname = d["res-name"]
                res_indices = self._residue_to_indices(resname)
                res_indices = res_indices.repeat(file_pos.shape[0], 1)
                resname_indices_list.append(res_indices)
            
            # Labels
            if self.target == "GBSA":
                y = d["DELTA TOTAL"]
            elif self.target == "GBSA_mean":
                y = d["GBSA_mean"] * np.ones(len(file_pos))
                y = y.astype(np.float32)
            elif self.target == "PLAS_20k":
                if len(d["PLAS_20k_data"]["DELTA_TOTAL"]) > 0:
                    y = d["PLAS_20k_data"]["DELTA_TOTAL"].values * np.ones(len(file_pos))
                    y = y.astype(np.float32)
                else:
                    self.x_0 = []
                    return
            
            y_list.append(torch.from_numpy(y))
            
            # RMSD and frame numbers
            rmsd = d["rmsd"]
            rmsd_list.append(torch.from_numpy(rmsd))
            
            frame_num = torch.tensor(list(range(len(y))), dtype=torch.int32)
            frame_num_list.append(frame_num)
            
            file_list.append([file] * len(y))
        
        # Pad sequences
        self.x_0 = pad_sequence(
            [x.transpose(1, 0) for x in x_0_list],
            batch_first=True,
            padding_value=0
        ).transpose(1, 2)
        
        self.pos = pad_sequence(
            [x.transpose(1, 0) for x in pos_list],
            batch_first=True,
            padding_value=0
        ).transpose(1, 2)
        
        if self.use_residue_features:
            self.resname_indices = pad_sequence(
                [x.transpose(1, 0) for x in resname_indices_list],
                batch_first=True,
                padding_value=0
            ).transpose(1, 2)
        
        # Apply max_frames limit
        self.x_0 = self.x_0[:, :self.max_frames, :]
        self.pos = self.pos[:, :self.max_frames, :, :]
        if self.use_residue_features:
            self.resname_indices = self.resname_indices[:, :self.max_frames, :]
        
        # Truncate other lists
        self.pose_list = [p[:self.max_frames] for p in self.pose_list]
        y_list = [y[:self.max_frames] for y in y_list]
        rmsd_list = [r[:self.max_frames] for r in rmsd_list]
        frame_num_list = [f[:self.max_frames] for f in frame_num_list]
        file_list = [f[:self.max_frames] for f in file_list]
        
        # Apply temporal split (backprop filtering)
        n_frames = self.x_0.shape[1]
        self._apply_temporal_split(n_frames, y_list, rmsd_list, frame_num_list, file_list)
        
        # Reshape to (batch*frames, seq_len, ...)
        self._reshape_to_flat()
        
        # Pad to max_len
        self._pad_to_max_len()
        


        # import pdb; pdb.set_trace() 
        # Apply quality filters
        mask = self._apply_rmsd_filter()
        self._apply_mask(mask)

         # Process labels
        self.y = self.y / -10  # Scale labels       
        if self.verbose:
            print(f"Loaded GBSA dataset: {len(self)} samples")
    
    def _apply_temporal_split(self, n_frames, y_list, rmsd_list, frame_num_list, file_list):
        """Apply backprop/temporal filtering"""
        if self.backprop == True:
            cutoff = int(n_frames * self.max_time_step_prop)
            self.x_0 = self.x_0[:, :cutoff, -self.max_len:]
            self.pos = self.pos[:, :cutoff, -self.max_len:, :]
            if self.use_residue_features:
                self.resname_indices = self.resname_indices[:, :cutoff, -self.max_len:]  # 3 indices
            
            self.y = torch.cat([y[:cutoff] for y in y_list])
            self.rmsd = torch.cat([r[:cutoff] for r in rmsd_list])
            self.frame_num = torch.cat([f[:cutoff] for f in frame_num_list])
            self.file_list = [item for sublist in file_list for item in sublist[:cutoff]]
            self.pose_list = [item for sublist in self.pose_list for item in sublist[:cutoff]]
            
        elif self.backprop == False:
            cutoff = int(n_frames * self.max_time_step_prop)
            self.x_0 = self.x_0[:, cutoff:, -self.max_len:]
            self.pos = self.pos[:, cutoff:, -self.max_len:, :]
            if self.use_residue_features:
                self.resname_indices = self.resname_indices[:, cutoff:, -self.max_len:]  # 3 indices
            
            self.y = torch.cat([y[cutoff:] for y in y_list])
            self.rmsd = torch.cat([r[cutoff:] for r in rmsd_list])
            self.frame_num = torch.cat([f[cutoff:] for f in frame_num_list])
            self.file_list = [item for sublist in file_list for item in sublist[cutoff:]]
            self.pose_list = [item for sublist in self.pose_list for item in sublist[cutoff:]]
        else:
            # Use all frames
            self.x_0 = self.x_0[:, :, -self.max_len:]
            self.pos = self.pos[:, :, -self.max_len:, :]
            if self.use_residue_features:
                self.resname_indices = self.resname_indices[:, :, -self.max_len:]  # 3 indices
            
            self.y = torch.cat(y_list)
            self.rmsd = torch.cat(rmsd_list)
            self.frame_num = torch.cat(frame_num_list)
            self.file_list = [item for sublist in file_list for item in sublist]
            self.pose_list = [item for sublist in self.pose_list for item in sublist]

    
    def _reshape_to_flat(self):
        """Reshape from (n_pdbs, n_frames, seq_len, ...) to (n_pdbs*n_frames, seq_len, ...)"""
        self.x_0 = self.x_0.reshape(-1, self.x_0.shape[2])
        self.pos = self.pos.reshape(-1, self.pos.shape[2], 3)
        if self.use_residue_features:
            self.resname_indices = self.resname_indices.reshape(-1, self.resname_indices.shape[2])
    
    def _pad_to_max_len(self):
        """Pad sequences to max_len"""
        seq_len = self.x_0.size(1)
        if seq_len < self.max_len:
            pad_size = self.max_len - seq_len
            
            # Pad atom types
            self.x_0 = torch.cat([
                self.x_0,
                torch.zeros((self.x_0.size(0), pad_size), dtype=self.x_0.dtype)
            ], dim=1)
            
            # Pad positions
            self.pos = torch.cat([
                self.pos,
                torch.zeros((self.pos.size(0), pad_size, 3), dtype=self.pos.dtype)
            ], dim=1)
            
            # Pad residue indices
            if self.use_residue_features:
                self.resname_indices = torch.cat([
                    self.resname_indices,
                    torch.zeros((self.resname_indices.size(0), pad_size), 
                               dtype=self.resname_indices.dtype)
                ], dim=1)
    
    def __getposenum__(self, i):
        return self.pose_list[i]

    def __getrmsd__(self, i):
        return self.rmsd[i]

    def __getframenum__(self, i):
        return self.frame_num[i]

    def __getfilename__(self, i):
        return self.file_list[i]

    def __len__(self):
        return len(self.x_0)
    
    def __getmetadata__(self, i):
        metadata_str = self.__getfilename__(i)
        return metadata_str

class PDBBindDataset(BaseMDDataset):
    """Wrapper for PDBbind data to support residue features"""
    
    def __init__(self, x, pos, y, resnames=None, use_residue_features=False):
        self.x = x
        self.pos = pos
        self.y = y
        self.use_residue_features = use_residue_features and resnames is not None
        
        if self.use_residue_features:
            # Convert residue names to indices
            self.resname_indices = self._process_residues(resnames)
        else:
            self.resname_indices = None
    
    def _process_residues(self, resnames):
        """Convert residue names to indices"""
        # Standard amino acids
        STANDARD_RESIDUES = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'HIE', 'HID', 'HIP', 'CYX', 'ACE', 'NME', 'PROT'  # Include generic PROT
        ]
        
        indices_list = []
        for sample_resnames in resnames:
            indices = []
            for res in sample_resnames:
                if res == '' or res is None:
                    indices.append(0)  # Padding
                elif res in STANDARD_RESIDUES:
                    indices.append(1)  # Protein
                else:  # 'LIG' or anything else
                    indices.append(2)  # Ligand
            indices_list.append(torch.tensor(indices, dtype=torch.long))
        
        return torch.stack(indices_list)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.use_residue_features:
            return self.x[idx], self.resname_indices[idx], self.pos[idx], self.y[idx], torch.tensor(idx)
        else:
            return self.x[idx], self.pos[idx], self.y[idx], torch.tensor(idx)

class PLAS20kDataset(BaseMDDataset):
    def __init__(
        self,
        pdb_list,
        data_dir=Path("data/md/"),
        max_len=600,
        rmsd_thresh=2.0,
        echo=True,
    ):
        self.rmsd_thresh = rmsd_thresh
        self.pdb_to_idx = {}
        self.max_len = max_len
        pos = []
        x_0_ = []

        self.file_list = []
        # if prompt is None or len(prompt) <= 1: prompt = [1]

        # Traverse all pdbs
        self.pose_list = []
        self.y = []
        self.pdbid_list = []
        self.resname = []
        self.rmsd = []
        self.frame_num = []

        path_list = [
            x for x in data_dir.glob("*-p0_*.npy") if x.stem.split("-")[0] in pdb_list
        ]

        for file in path_list:
            # print(file)

            d = np.load(Path(data_dir) / Path(file), allow_pickle=True).item()

            d["R"] = np.expand_dims(d["R"][0], axis=0)

            file_pos = torch.tensor(d["R"])

            # self.pose_list.append([int(file.split("_")[0].split("-")[-1][1])]*len(file_pos))
            self.pose_list.append([0])
            z = torch.tensor([atom_dict[x] for x in d["z"]])
            z = z.repeat(file_pos.shape[0], 1)

            if len(d["PLAS_20k_data"]["DELTA_TOTAL"]) > 0:
                # print(d['PLAS_20k_data']['DELTA_TOTAL'].values,np.ones(total_frames) )
                y = d["PLAS_20k_data"]["DELTA_TOTAL"].values.astype(np.float32)
            else:
                # move onto next pdbid
                continue

            pos.append(file_pos)
            x_0_.append(z)

            rmsd = np.expand_dims(d["rmsd"][0], axis=0)
            # resname = d['res-name']

            frame_num = torch.tensor(list(range(y.shape[0])), dtype=torch.int32)

            assert y.shape == rmsd.shape
            # print(y.shape)
            self.y.append(torch.from_numpy(y))
            self.rmsd.append(torch.from_numpy(rmsd))
            self.file_list.append([file.name.split("-")[0]] * len(y))
            # self.resname.append(resname)
            self.frame_num.append(frame_num)

        self.x_0 = pad_sequence(
            [x.transpose(1, 0) for x in x_0_], batch_first=True, padding_value=0
        ).transpose(1, 2)
        self.pos = pad_sequence(
            [x.transpose(1, 0) for x in pos], batch_first=True, padding_value=0
        ).transpose(1, 2)

        self.x_0 = self.x_0[
            :, :, -self.max_len :
        ]  # the ligand is at the end of the array dimension and we want to keep all of it

        self.x_0 = self.x_0.reshape(-1, self.x_0.shape[2])

        # Current sequence length
        seq_len = self.x_0.size(1)

        if seq_len < max_len:
            # Calculate how many zeros to add
            pad_size = max_len - seq_len

            # Create padding tensor of zeros (batch_size, pad_size)
            padding = torch.zeros((self.x_0.size(0), pad_size), dtype=self.x_0.dtype)

            # Concatenate along the sequence dimension
            self.x_0 = torch.cat((self.x_0, padding), dim=1)

        self.pos = self.pos[
            :, :, -self.max_len :, :
        ]  # the ligand is inclued at the end
        self.pos = self.pos.reshape(-1, self.pos.shape[2], 3)

        seq_len = self.pos.size(1)

        if seq_len < max_len:
            # Calculate how many zeros to add
            pad_size = max_len - seq_len

            # Create padding tensor of zeros (batch_size, pad_size)
            padding = torch.zeros((self.pos.size(0), pad_size, 3), dtype=self.pos.dtype)

            # Concatenate along the sequence dimension
            self.pos = torch.cat((self.pos, padding), dim=1)

        self.y = torch.cat(self.y)

        self.frame_num = torch.cat(self.frame_num)

        self.rmsd = torch.cat(self.rmsd)

        backprop_file_list = []
        for id_list in self.file_list:
            backprop_file_list.extend(id_list)

        self.file_list = backprop_file_list

        backprop_pose_list = []
        for pose_list in self.pose_list:
            backprop_pose_list.extend(pose_list)

        self.pose_list = backprop_pose_list

        if echo:
            print("Got {:d} protein-ligand simulations as input!".format(len(x_0_)))

        self.y = self.y / -10

        # print(self.file_list)

    def __getitem__(self, i):
        return (
            self.x_0[i],
            self.pose_list[i],
            self.pos[i],
            self.y[i],
            self.file_list[i],
            self.rmsd[i],
            self.frame_num[i],
        )

    def __len__(self):
        return len(self.x_0)