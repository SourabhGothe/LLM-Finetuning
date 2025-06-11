# src/data_loader.py
# Handles loading, preprocessing, and tokenizing the dataset.

from datasets import load_dataset, DatasetDict, Dataset
from src.linearize import get_linearizer
from typing import Dict, Any

# Alpaca prompt template
PROMPT_TEMPLATE = """Below is a graph describing a set of entities and their relationships. Write a coherent and fluent paragraph that accurately describes this information.

### Graph:
{}

### Description:
{}"""

def load_and_prepare_dataset(cfg: Dict[str, Any], tokenizer) -> DatasetDict:
    """
    Loads the WebNLG dataset, applies linearization and formatting.
    """
    print("Loading and preparing dataset...")
    
    # Load the dataset from JSON files
    data_files = {
        'train': f"{cfg.data_dir}/{cfg.dataset.train_file}",
        'validation': f"{cfg.data_dir}/{cfg.dataset.val_file}"
    }
    raw_datasets = load_dataset('json', data_files=data_files)
    
    linearizer = get_linearizer(cfg.dataset.linearization_strategy)
    
    def format_prompt(entry):
        # Linearize the graph data
        linearized_graph = linearizer(entry)
        
        # Get one of the reference texts as the target
        # In a real scenario, you might want to handle multiple references
        target_text = entry['lexicalisations'][0]['lex']
        
        # Create the full prompt using the Alpaca template
        full_prompt = PROMPT_TEMPLATE.format(linearized_graph, target_text)
        
        return {cfg.dataset.text_col: full_prompt}

    # Format the datasets
    processed_datasets = raw_datasets.map(
        format_prompt,
        remove_columns=list(raw_datasets['train'].features)
    )

    print("Dataset prepared.")
    print(f"Sample prompt:\n{processed_datasets['train'][0][cfg.dataset.text_col]}")
    
    return processed_datasets
