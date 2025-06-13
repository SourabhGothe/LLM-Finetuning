# src/model.py
# Handles loading the LLM and applying PEFT configuration.
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# src/model.py
# Handles loading the LLM and applying PEFT configuration.

from unsloth import FastLanguageModel
from omegaconf import OmegaConf
import torch

def load_model_for_training(cfg):
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
    
    # FIX: target_modules is now optional.
    # We use OmegaConf.get to safely access the key. If it's not found in any
    # config file, it defaults to None. When target_modules is None,
    # Unsloth automatically finds all linear layers to apply LoRA to,
    # which is the most robust and recommended approach.
    target_modules = OmegaConf.get(cfg, "peft.target_modules", None)
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.peft.r,
        target_modules=target_modules,
        lora_alpha=cfg.peft.lora_alpha,
        lora_dropout=cfg.peft.lora_dropout,
        bias=cfg.peft.bias,
        use_gradient_checkpointing=cfg.peft.use_gradient_checkpointing,
        random_state=cfg.training.seed,
    )

    print("Model and tokenizer are ready for training.")
    return model, tokenizer
