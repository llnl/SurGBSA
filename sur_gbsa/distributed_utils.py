################################################################################
# Copyright (c) 2021-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIBUTING.md
#
# All rights reserved.
################################################################################
try:
    import torch
    import torch.distributed as dist
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyTorch is not installed. Please install it using 'pip install torch'."
    )


def distributed_gather(tensor, device="cuda"):
    """
    Gathers tensors from all ranks in the process group to rank 0 and returns as a floating point tensor.

    Parameters:
        tensor (torch.Tensor or None): The tensor to gather from each process. If None, a zero tensor is used.
        device (str): The device to place the tensor on (default: "cuda").

    Returns:
        torch.Tensor or None: The gathered tensors as a floating point tensor on rank 0 (moved to CPU), None on other ranks.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "Distributed process group is not initialized. Call dist.init_process_group() first."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Ensure tensor is not None by replacing it with a zero tensor of appropriate shape
    if tensor is None:
        tensor = torch.zeros(1, dtype=torch.float32).to(device)
    else:
        tensor = tensor.to(device)  # Ensure tensor is on CUDA

    # print(f"Rank {rank}: Tensor is on device {tensor.device}")

    if not tensor.is_cuda:
        raise ValueError(
            f"Tensor must be on CUDA for NCCL backend. Current device: {tensor.device}"
        )

    tensor = tensor.contiguous()  # Ensures memory alignment for gather operation

    if rank == 0:
        gather_list = [
            torch.zeros_like(tensor, dtype=tensor.dtype, device=device)
            for _ in range(world_size)
        ]
        dist.gather(tensor, gather_list, dst=0)
        # Stack all tensors from all ranks into a single tensor along the rank (0) dimension and move to CPU
        return torch.stack(
            gather_list, dim=0
        ).cpu()  # Stack along the 0th dimension (rank dimension)
    else:
        dist.gather(tensor, dst=0)
        return None


try:
    import torch
    import torch.distributed as dist
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "PyTorch is not installed. Please install it using 'pip install torch'."
    )


def distributed_gather_strings(string_list, max_length=256, device="cuda"):
    """
    Gathers lists of strings from all ranks in the process group to rank 0.

    Parameters:
        string_list (list of str or None): The list of strings to gather from each process.
        max_length (int): Maximum length of each string (strings are truncated or padded to this size).
        device (str): The device to place the tensor on (default: "cuda").

    Returns:
        list of list of str or None: The gathered strings on rank 0, None on other ranks.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "Distributed process group is not initialized. Call dist.init_process_group() first."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if string_list is None:
        string_list = []

    # Convert each string to a fixed-size byte tensor
    encoded_tensors = []
    for string in string_list:
        print(string)
        string = string[:max_length]  # Truncate if too long
        encoded = torch.tensor(list(map(ord, string)), dtype=torch.uint8)
        # encoded = torch.tensor(list(string.encode("utf-8")), dtype=torch.uint8)

        padded = torch.zeros(max_length, dtype=torch.uint8)
        padded[: len(encoded)] = encoded  # Pad with zeros
        encoded_tensors.append(padded)

    # Stack into a single tensor
    if encoded_tensors:
        tensor = torch.stack(encoded_tensors).to(device)
    else:
        tensor = torch.zeros((1, max_length), dtype=torch.uint8).to(
            device
        )  # Empty placeholder

    tensor = tensor.contiguous()  # Ensure memory alignment

    if rank == 0:
        gather_list = [
            torch.zeros_like(tensor, device=device) for _ in range(world_size)
        ]
        dist.gather(tensor, gather_list, dst=0)

        # Decode gathered tensors back into string lists
        gathered_strings = []
        for rank_tensor in gather_list:
            rank_strings = []
            for row in rank_tensor:
                decoded_str = "".join(map(chr, row.tolist())).rstrip(
                    "\x00"
                )  # Remove padding
                rank_strings.append(decoded_str)
            gathered_strings.append(rank_strings)

        return gathered_strings  # List of lists (one per rank)

    else:
        dist.gather(tensor, dst=0)
        return None



import subprocess

def get_gpu_partition_mode():
    """Detect if running in SPX, TPX, or CPX mode"""
    try:
        result = subprocess.run(
            ['amd-smi', 'static', '--partition'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if 'COMPUTE_PARTITION: SPX' in result.stdout:
            return 'SPX', 4  # 4 GPUs per node
        elif 'COMPUTE_PARTITION: TPX' in result.stdout:
            return 'TPX', 12  # 12 GPUs per node
        elif 'COMPUTE_PARTITION: CPX' in result.stdout:
            return 'CPX', 24  # 24 GPUs per node
    except:
        pass
    
    # Default fallback
    return 'SPX', 4

