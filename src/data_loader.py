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

# Alpaca prompt template
PROMPT_TEMPLATE = """Below is a graph describing a set of entities and their relationships. Write a coherent and fluent paragraph that accurately describes this information.

### Graph:
{}

### Description:
{}"""

def load_and_prepare_dataset(cfg: DictConfig, tokenizer) -> tuple[DatasetDict, Dataset]:
    """
    Loads the WebNLG dataset from the Hugging Face Hub, applies linearization and formatting.
    
    Returns:
        A tuple containing:
        1. The processed dataset ready for the SFTTrainer.
        2. The raw, unprocessed validation dataset for use in callbacks.
    """
    print(f"Loading dataset '{cfg.dataset.name}' from the Hugging Face Hub...")
    
    raw_datasets = load_dataset(cfg.dataset.name, cfg.dataset.subset, split=['train', 'dev'])
    dataset_dict = DatasetDict({
        'train': raw_datasets[0],
        'validation': raw_datasets[1]
    })
    
    linearizer = get_linearizer(cfg.dataset.linearization_strategy)
    
    def format_prompt(entry):
        linearized_graph = linearizer(entry)
        target_text = entry['lex']['text'][0]
        
        # Explicitly add the EOS token to the end of the text.
        # This makes the separation between prompt and completion unambiguous for the SFTTrainer.
        full_prompt = PROMPT_TEMPLATE.format(linearized_graph, target_text) + tokenizer.eos_token
        
        return {cfg.dataset.text_col: full_prompt}

    # Format the datasets
    processed_datasets = dataset_dict.map(
        format_prompt,
        remove_columns=list(dataset_dict['train'].features)
    )

    print("Dataset prepared.")
    print(f"Sample prompt:\n{processed_datasets['train'][0][cfg.dataset.text_col]}")
    
    # Return both the processed dataset for training and the raw validation set for the callback
    return processed_datasets, dataset_dict['validation']
