# src/train.py
# Main script to orchestrate the finetuning process. (Offline Edition)
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# src/train.py
# Main script to orchestrate the finetuning process. (Final Version)

import os
import sys
import torch
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import load_model_for_training
from src.data_loader import load_and_prepare_dataset
from src.linearize import get_linearizer

# Define the inference prompt template used by the callback
INFERENCE_PROMPT_TEMPLATE = """Below is a graph describing a set of entities and their relationships. Write a coherent and fluent paragraph that accurately describes this information.

### Graph:
{}

### Description:
"""

# FIX: Added a custom callback to generate sample outputs after each epoch.
class SampleGenerationCallback(TrainerCallback):
    "A callback that generates sample predictions from the validation set at the end of each epoch."

    def __init__(self, raw_eval_dataset, tokenizer, cfg, num_samples=3):
        super().__init__()
        # Take a small, fixed sample from the raw validation set
        self.sample_dataset = raw_eval_dataset.select(range(min(num_samples, len(raw_eval_dataset))))
        self.tokenizer = tokenizer
        self.linearizer = get_linearizer(cfg.dataset.linearization_strategy)
        self.cfg = cfg

    def on_epoch_end(self, args, state, control, model, **kwargs):
        # Ensure this runs only on the main process
        if not state.is_world_process_zero:
            return

        print(f"\n--- Generating samples for epoch {int(state.epoch)} ---")
        output_dir = self.cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"sample_outputs_epoch_{int(state.epoch)}.txt")

        # Set model to eval mode for generation
        model.eval()

        with open(output_file_path, "w", encoding="utf-8") as f:
            for i, entry in enumerate(self.sample_dataset):
                linearized_graph = self.linearizer(entry)
                prompt = INFERENCE_PROMPT_TEMPLATE.format(linearized_graph)
                
                inputs = self.tokenizer([prompt], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                decoded_output = self.tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                
                f.write(f"===== SAMPLE {i+1} =====\n")
                f.write(f"INPUT GRAPH:\n{linearized_graph}\n\n")
                f.write(f"MODEL OUTPUT:\n{decoded_output.strip()}\n\n")
                f.write(f"REFERENCE:\n{entry['lex']['text'][0]}\n")
                f.write("="*20 + "\n\n")

        print(f"Sample outputs saved to {output_file_path}")
        # Set model back to train mode
        model.train()


@hydra.main(version_base=None, config_path="../configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    """
    Main training function.
    """
    print("--- Starting training process ---")
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    model, tokenizer = load_model_for_training(cfg)
    
    # Get both the processed dataset for training and the raw validation set for the callback
    processed_dataset, raw_validation_dataset = load_and_prepare_dataset(cfg, tokenizer)

    # Instantiate the custom callback
    sample_generation_callback = SampleGenerationCallback(
        raw_eval_dataset=raw_validation_dataset,
        tokenizer=tokenizer,
        cfg=cfg
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        dataset_text_field=cfg.dataset.text_col,
        max_seq_length=cfg.model.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        # Add the callback to the trainer
        callbacks=[sample_generation_callback],
        args=TrainingArguments(
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            warmup_steps=cfg.training.warmup_steps,
            num_train_epochs=cfg.training.num_train_epochs,
            learning_rate=cfg.training.learning_rate,
            fp16=not cfg.training.bf16,
            bf16=cfg.training.bf16,
            logging_steps=cfg.training.logging_steps,
            optim=cfg.training.optim,
            weight_decay=cfg.training.weight_decay,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            seed=cfg.training.seed,
            output_dir=cfg.output_dir,
            report_to="tensorboard",
        ),
    )

    print("\n--- Starting model training ---")
    trainer_stats = trainer.train()

    print("\n--- Training finished ---")
    print(trainer_stats)
    
    final_model_path = os.path.join(cfg.output_dir, "final_checkpoint")
    trainer.save_model(final_model_path)
    print(f"\nFinal model adapter saved to {final_model_path}")


if __name__ == "__main__":
    main()
