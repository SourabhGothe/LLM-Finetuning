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

    # FIX: Explicitly set the padding token for the tokenizer.
    # Llama-3 models do not have a default padding token. This is a critical step.
    # We use the end-of-sequence (EOS) token as the padding token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer.pad_token to tokenizer.eos_token")


    print("Applying PEFT (LoRA) configuration...")
    
    # We default to None for target_modules to let Unsloth auto-detect them,
    # which is the most robust approach.
    try:
        target_modules = cfg.peft.target_modules
        print(f"Found manually specified target_modules: {target_modules}")
    except AttributeError:
        target_modules = None
        print("`target_modules` not specified. Unsloth will auto-detect all linear layers.")
    
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
