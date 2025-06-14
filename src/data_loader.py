# src/data_loader.py
# Handles loading, preprocessing, and tokenizing the dataset.

import os
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset, DatasetDict, Dataset
from src.linearize import get_linearizer
from omegaconf import DictConfig

def load_and_prepare_dataset(cfg: DictConfig, tokenizer) -> tuple[Dataset, Dataset]:
    """
    Loads the WebNLG dataset from the Hugging Face Hub and prepares it for training.
    
    Returns:
        A tuple containing:
        1. The raw training dataset.
        2. The raw validation dataset.
    """
    print(f"Loading dataset '{cfg.dataset.name}' from the Hugging Face Hub...")
    
    # Load both train and dev (validation) splits
    raw_datasets = load_dataset(cfg.dataset.name, cfg.dataset.subset, split=['train', 'dev'])
    
    # We will return the raw datasets. The formatting will now be handled by a
    # `formatting_func` directly inside the SFTTrainer for maximum robustness.
    # This ensures the chat template is applied correctly during training.
    
    train_dataset = raw_datasets[0]
    validation_dataset = raw_datasets[1]

    print("Raw datasets loaded successfully.")
    
    return train_dataset, validation_dataset
