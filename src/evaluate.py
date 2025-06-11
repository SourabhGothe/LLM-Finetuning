# src/evaluate.py
# Script for evaluating the fine-tuned model on the test set.

import os
import sys
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
import evaluate as hf_evaluate
from datasets import load_dataset
from unsloth import FastLanguageModel

# Add project root to path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.linearize import get_linearizer

# We need a slightly different prompt template for inference, without the answer part.
INFERENCE_PROMPT_TEMPLATE = """Below is a graph describing a set of entities and their relationships. Write a coherent and fluent paragraph that accurately describes this information.

### Graph:
{}

### Description:
"""

def load_model_for_inference(model_path: str, cfg: DictConfig):
    """
    Loads the fine-tuned model from a checkpoint for inference.
    """
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}. Please ensure you've trained the model and provided the correct path.")
        
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg.model.max_seq_length,
        dtype=getattr(torch, cfg.model.dtype) if cfg.model.dtype else None,
        load_in_4bit=False, # We are loading a merged model adapter
    )
    model.eval()
    print("Model loaded successfully for evaluation.")
    return model, tokenizer

def load_test_dataset(cfg: DictConfig):
    """
    Loads the test portion of the WebNLG dataset.
    """
    test_file = cfg.dataset.test_file
    data_files = {
        'test': os.path.join(cfg.data_dir, test_file)
    }
    try:
        raw_dataset = load_dataset('json', data_files=data_files)
    except FileNotFoundError:
        print(f"Error: Test file not found at {data_files['test']}")
        print("Please ensure the dataset is downloaded correctly and the path in your config is correct.")
        return None
    return raw_dataset['test']
    
@hydra.main(version_base=None, config_path="../configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation function.
    Expects an additional command-line argument: `model_checkpoint_path`
    Example: python src/evaluate.py +experiment=llama3_qlora_webnlg model_checkpoint_path=outputs/.../final_checkpoint
    """
    print("Starting evaluation process...")
    
    if "model_checkpoint_path" not in cfg:
        print("\nERROR: `model_checkpoint_path` not provided.")
        print("Please specify the path to the trained model checkpoint on the command line.")
        print("Example: python src/evaluate.py +experiment=llama3_qlora_webnlg model_checkpoint_path=outputs/MODEL_DIR/final_checkpoint\n")
        return

    model_path = cfg.model_checkpoint_path
    
    # --- 1. Load Model and Tokenizer ---
    model, tokenizer = load_model_for_inference(model_path, cfg)
    
    # --- 2. Load Test Dataset ---
    test_dataset = load_test_dataset(cfg)
    if test_dataset is None:
        return
        
    linearizer = get_linearizer(cfg.dataset.linearization_strategy)
    
    # --- 3. Generate Predictions ---
    predictions = []
    references = []
    
    print(f"Generating predictions for {len(test_dataset)} test samples...")
    for entry in tqdm(test_dataset):
        linearized_graph = linearizer(entry)
        prompt = INFERENCE_PROMPT_TEMPLATE.format(linearized_graph)
        
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and clean the output
        prediction_text = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:])[0]
        prediction_text = prediction_text.replace(tokenizer.eos_token, "").strip()
        
        # Collect prediction and references
        predictions.append(prediction_text)
        references.append([lex['lex'] for lex in entry['lexicalisations']])
        
    print("Generation complete.")
    
    # --- 4. Compute Metrics ---
    print("Computing metrics (BLEU and ROUGE)...")
    
    bleu_metric = hf_evaluate.load("sacrebleu")
    rouge_metric = hf_evaluate.load("rouge")
    
    # Sacrebleu expects a list of lists of references
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    
    # ROUGE also supports a list of lists of references for a more robust score
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    
    results = {
        "bleu": bleu_results['score'],
        "rouge1": rouge_results['rouge1'],
        "rouge2": rouge_results['rouge2'],
        "rougeL": rouge_results['rougeL'],
    }
    
    print("\n--- Evaluation Results ---")
    print(json.dumps(results, indent=2))
    
    # --- 5. Save Results ---
    output_path = os.path.join(model_path, "evaluation_results.json")
    
    # Save metrics and a few examples
    full_results_data = {
        "metrics": results,
        "samples": [
            {
                "linearized_graph": linearizer(test_dataset[i]),
                "prediction": predictions[i],
                "references": references[i],
            }
            for i in range(min(10, len(predictions))) # Save first 10 samples
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results_data, f, indent=4, ensure_ascii=False)
        
    print(f"\nEvaluation results saved to {output_path}")

if __name__ == "__main__":
    main()
