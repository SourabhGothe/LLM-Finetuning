# src/train.py
# Main script to orchestrate the finetuning process. (Offline Edition)
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import unsloth
import os
import sys
# src/train.py
# Main script to orchestrate the finetuning process. (Fixed Configuration)

import os
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig, OmegaConf
from trl import SFTTrainer
from transformers import TrainingArguments

from src.model import load_model_for_training
from src.data_loader import load_and_prepare_dataset

# FIX: The config_path now points directly to the 'experiment' directory.
# The config_name will be the specific experiment file to run (e.g., 'deepseek_coder_qlora_webnlg').
# Hydra will load this file, which in turn loads the base_config.
@hydra.main(version_base=None, config_path="../configs/experiment", config_name="llama3_qlora_webnlg")
def main(cfg: DictConfig) -> None:
    """
    Main training function.
    """
    print("--- Starting training process ---")
    print("Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # The rest of the script remains the same
    model, tokenizer = load_model_for_training(cfg)
    dataset = load_and_prepare_dataset(cfg, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field=cfg.dataset.text_col,
        max_seq_length=cfg.model.max_seq_length,
        dataset_num_proc=2,
        packing=False,
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
    print(f"Final model adapter saved to {final_model_path}")


if __name__ == "__main__":
    main()
