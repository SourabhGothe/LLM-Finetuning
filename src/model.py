# src/model.py
# Handles loading the LLM and applying PEFT configuration.
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverfied_context
from unsloth import FastLanguageModel
from peft import LoraConfig
from typing import Dict, Any
import torch

def load_model_for_training(cfg: Dict[str, Any]):
    """
    Loads the model and tokenizer using Unsloth's FastLanguageModel,
    configured for QLoRA training.
    """
    print(f"Loading model: {cfg.model.name}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        dtype=getattr(torch, cfg.model.dtype) if cfg.model.dtype else None,
        load_in_4bit=cfg.model.load_in_4bit,
    )

    print("Applying PEFT (LoRA) configuration...")
    # It's important to do this step to enable gradient checkpointing and prepare the model for LoRA.
    # This is an Unsloth-specific optimization.
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.peft.r,
        target_modules=cfg.peft.target_modules,
        lora_alpha=cfg.peft.lora_alpha,
        lora_dropout=cfg.peft.lora_dropout,
        bias=cfg.peft.bias,
        use_gradient_checkpointing=cfg.peft.use_gradient_checkpointing,
        random_state=cfg.training.seed,
        task_type=cfg.peft.task_type,
    )

    print("Model and tokenizer are ready for training.")
    return model, tokenizer
