# src/data_loader.py
# Handles loading, preprocessing, and tokenizing the dataset.

import os
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset, Dataset
from src.linearize import get_linearizer
from omegaconf import DictConfig

# The prompt template for the user's instruction
PROMPT_TEMPLATE = """Below is a graph describing a set of entities and their relationships. Write a coherent and fluent paragraph that accurately describes this information.

### Graph:
{}

### Description:
"""

def load_and_prepare_dataset(cfg: DictConfig, tokenizer) -> tuple[Dataset, Dataset, Dataset]:
    """
    Loads the WebNLG dataset, "flattens" it to handle multiple references,
    and formats it using the model's chat template.
    
    Returns:
        A tuple containing:
        1. The processed training dataset.
        2. The processed validation dataset.
        3. The raw validation dataset for callback generation.
    """
    print(f"Loading raw dataset '{cfg.dataset.name}' from the Hugging Face Hub...")
    raw_train_dataset, raw_validation_dataset = load_dataset(
        cfg.dataset.name, cfg.dataset.subset, split=['train', 'dev']
    )
    
    linearizer = get_linearizer(cfg.dataset.linearization_strategy)

    def process_and_flatten_dataset(dataset: Dataset) -> Dataset:
        """
        Takes a raw dataset and transforms it. Each graph with N text descriptions
        will become N separate examples in the new dataset.
        """
        processed_data = []
        for entry in dataset:
            linearized_graph = linearizer(entry)
            # Create a new example for each lexicalization
            for text in entry['lex']['text']:
                messages = [
                    {"role": "user", "content": PROMPT_TEMPLATE.format(linearized_graph, "").strip()},
                    {"role": "assistant", "content": text},
                ]
                # Each example is a single formatted string
                processed_data.append({
                    "text": tokenizer.apply_chat_template(messages, tokenize=False)
                })
        return Dataset.from_list(processed_data)

    print("Processing and formatting datasets...")
    train_dataset = process_and_flatten_dataset(raw_train_dataset)
    validation_dataset = process_and_flatten_dataset(raw_validation_dataset)

    print("Datasets prepared successfully.")
    print(f"Original training examples: {len(raw_train_dataset)}. Flattened training examples: {len(train_dataset)}")
    print(f"Sample formatted example:\n{train_dataset[0]['text']}")
    
    # Return processed sets for the trainer, and the raw validation set for the callback
    return train_dataset, validation_dataset, raw_validation_dataset
