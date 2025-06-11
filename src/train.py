# src/train.py
# Main script to orchestrate the finetuning process. (Offline Edition)

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from trl import SFTTrainer
from transformers import TrainingArguments

from src.model import load_model_for_training
from src.data_loader import load_and_prepare_dataset

@hydra.main(version_base=None, config_path="../configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    """
    Main training function.
    """
    print("Starting training process...")
    print(OmegaConf.to_yaml(cfg))

    # --- 1. Load Model and Tokenizer ---
    model, tokenizer = load_model_for_training(cfg)

    # --- 2. Load and Prepare Dataset ---
    dataset = load_and_prepare_dataset(cfg, tokenizer)

    # --- 3. Configure Training ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field=cfg.dataset.text_col,
        max_seq_length=cfg.model.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training faster, but requires more memory
        args=TrainingArguments(
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            warmup_steps=cfg.training.warmup_steps,
            num_train_epochs=cfg.training.num_train_epochs,
            learning_rate=cfg.training.learning_rate,
            fp16=not cfg.training.bf16, # Unsloth manages this
            bf16=cfg.training.bf16,     # Unsloth manages this
            logging_steps=cfg.training.logging_steps,
            optim=cfg.training.optim,
            weight_decay=cfg.training.weight_decay,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            seed=cfg.training.seed,
            output_dir=cfg.output_dir,
            # Report logs to TensorBoard. The Trainer will handle creating the log files.
            report_to="tensorboard",
        ),
    )

    # --- 4. Start Training ---
    print("Starting model training...")
    trainer_stats = trainer.train()

    print("Training finished.")
    print(trainer_stats)
    
    # --- 5. Save Final Model ---
    final_model_path = os.path.join(cfg.output_dir, "final_checkpoint")
    trainer.save_model(final_model_path)
    print(f"Final model adapter saved to {final_model_path}")


if __name__ == "__main__":
    main()
