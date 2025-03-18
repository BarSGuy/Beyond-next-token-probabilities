import os
import torch
from pathlib import Path


def save_raw_data(
    LLM: str,
    dataset_name: str,
    idx: int,
    label: torch.Tensor,
    probs_input: torch.Tensor = None,
    probs_output: torch.Tensor = None,
    base_dir: str = "./data"
):
    """
    Save model results (PyTorch tensors) and labels to a specified directory structure.

    Args:
        LLM (str): Name of the language model.
        dataset_name (str): Name of the dataset.
        probs_input (torch.Tensor): Input probabilities tensor to save.
        probs_output (torch.Tensor): Output probabilities tensor to save.
        label (torch.Tensor): Label data to save.
        idx (int): Index of the sample, used for file naming.
        base_dir (str, optional): Base directory for saving results (default: './data').
    """

    # Construct save directory path
    save_dir = Path(base_dir) / LLM / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Function to save tensor safely
    def save_tensor(tensor, filename):
        if tensor is not None:
            try:
                torch.save(tensor, save_dir / filename)
                # print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")

    # Save label
    save_tensor(label, f'label_{idx}.pt')

    # Save probability tensors
    save_tensor(probs_input, f'probs_input_{idx}.pt')
    save_tensor(probs_output, f'probs_output_{idx}.pt')

