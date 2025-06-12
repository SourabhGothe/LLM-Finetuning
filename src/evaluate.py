# src/evaluate.py
# Script for evaluating the fine-tuned model on the test set.
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import sys
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
import evaluate as hf_evaluate
from datasets import load_dataset
from unsloth import FastLanguageModel

# FIX: Add project root to the Python path
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
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}. Please ensure you've trained the model and provided the correct path.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg.model.max_seq_length,
        dtype=getattr(torch, cfg.model.dtype) if cfg.model.dtype else None,
        load_in_4bit=False,
    )
    model.eval()
    print("Model loaded successfully for evaluation.")
    return model, tokenizer

def load_test_dataset(cfg: DictConfig):
    print(f"Loading test split from '{cfg.dataset.name}/{cfg.dataset.subset}'...")
    try:
        # The test set is called 'test' in this dataset version
        test_dataset = load_dataset(cfg.dataset.name, cfg.dataset.subset, split='test')
    except Exception as e:
        print(f"Error loading dataset from Hugging Face Hub: {e}")
        return None
    return test_dataset

@hydra.main(version_base=None, config_path="../configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    print("Starting evaluation process...")
    if "model_checkpoint_path" not in cfg:
        print("\nERROR: `model_checkpoint_path` not provided.")
        print("Example: python src/evaluate.py +experiment=llama3_qlora_webnlg model_checkpoint_path=outputs/MODEL_DIR/final_checkpoint\n")
        return

    model_path = cfg.model_checkpoint_path
    model, tokenizer = load_model_for_inference(model_path, cfg)
    test_dataset = load_test_dataset(cfg)
    if test_dataset is None: return

    linearizer = get_linearizer(cfg.dataset.linearization_strategy)
    predictions, references = [], []

    print(f"Generating predictions for {len(test_dataset)} test samples...")
    for entry in tqdm(test_dataset):
        triples = entry['modified_triple_sets']['mtriple_set'][0]
        repackaged_entry = {'modified_triple_sets': {'mtriple_set': [triples]}}
        linearized_graph = linearizer(repackaged_entry)
        prompt = INFERENCE_PROMPT_TEMPLATE.format(linearized_graph)
        
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, use_cache=True, pad_token_id=tokenizer.eos_token_id
            )
        
        prediction_text = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:])[0]
        prediction_text = prediction_text.replace(tokenizer.eos_token, "").strip()
        
        predictions.append(prediction_text)
        references.append(entry['lex']['text'])

    print("Generation complete.")
    print("Computing metrics (BLEU and ROUGE)...")
    
    bleu_metric = hf_evaluate.load("sacrebleu")
    rouge_metric = hf_evaluate.load("rouge")
    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    
    results = { "bleu": bleu_results['score'], **rouge_results }
    print("\n--- Evaluation Results ---")
    print(json.dumps(results, indent=2))
    
    output_path = os.path.join(model_path, "evaluation_results.json")
    full_results_data = { "metrics": results, "samples": [] }
    for i in range(min(10, len(predictions))):
        triples = test_dataset[i]['modified_triple_sets']['mtriple_set'][0]
        repackaged_entry = {'modified_triple_sets': {'mtriple_set': [triples]}}
        full_results_data["samples"].append({
            "linearized_graph": linearizer(repackaged_entry),
            "prediction": predictions[i],
            "references": references[i],
        })
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results_data, f, indent=4, ensure_ascii=False)
    print(f"\nEvaluation results saved to {output_path}")

if __name__ == "__main__":
    main()
