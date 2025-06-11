# src/data_loader.py
# Handles loading, preprocessing, and tokenizing the dataset.

import os
import sys

# FIX: Add project root to the Python path
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

def load_and_prepare_dataset(cfg: DictConfig, tokenizer) -> DatasetDict:
    """
    Loads the WebNLG dataset from the Hugging Face Hub, applies linearization and formatting.
    This replaces the need for the download_data.sh script.
    """
    print(f"Loading dataset '{cfg.dataset.name}' from the Hugging Face Hub...")
    
    # Load the specific subset of the dataset from the hub
    raw_datasets = load_dataset(cfg.dataset.name, cfg.dataset.subset, split=['train', 'dev'])
    dataset_dict = DatasetDict({
        'train': raw_datasets[0],
        'validation': raw_datasets[1]
    })
    
    linearizer = get_linearizer(cfg.dataset.linearization_strategy)
    
    def format_prompt(entry):
        # The structure from the hub is slightly different
        triples = entry['modified_triple_sets']['mtriple_set'][0]
        # We need to manually re-package the entry for the existing linearizers
        repackaged_entry = {'modified_triple_sets': {'mtriple_set': [triples]}}
        linearized_graph = linearizer(repackaged_entry)
        
        # Get one of the reference texts as the target
        target_text = entry['lex']['text'][0]
        
        # Create the full prompt using the Alpaca template
        full_prompt = PROMPT_TEMPLATE.format(linearized_graph, target_text)
        
        return {cfg.dataset.text_col: full_prompt}

    # Format the datasets
    processed_datasets = dataset_dict.map(
        format_prompt,
        remove_columns=list(dataset_dict['train'].features)
    )

    print("Dataset prepared.")
    print(f"Sample prompt:\n{processed_datasets['train'][0][cfg.dataset.text_col]}")
    
    return processed_datasets
