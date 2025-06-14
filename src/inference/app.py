import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# src/inference/app.py
# Gradio app for interactive inference with the fine-tuned model. (Final Version)

import os
import sys
import gradio as gr
from unsloth import FastLanguageModel
import torch

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration ---
# =================================================================================
#  !!! CRITICAL: YOU MUST UPDATE THIS PATH !!!
# =================================================================================
MODEL_DIR = "outputs/llama-3-2-1b-instruct/llama_3_2_1b_qlora_webnlg_3_epochs_.../final_checkpoint"
MAX_SEQ_LENGTH = 2048

# --- Load Model ---
print(f"Loading fine-tuned model from: {MODEL_DIR}")

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found at '{MODEL_DIR}'. Please update the path.")

if torch.cuda.is_available():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR, max_seq_length=MAX_SEQ_LENGTH, dtype=torch.bfloat16, load_in_4bit=False,
    )
else:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR, max_seq_length=MAX_SEQ_LENGTH, dtype=torch.float32, load_in_4bit=False,
    )

model.eval()
print("Model loaded successfully.")

# --- Inference Logic ---
def generate_text(linearized_graph: str) -> str:
    if not linearized_graph.strip():
        return "Please enter a linearized graph."
        
    # FIX: Use the official chat template for consistency with the new training format.
    messages = [
        {
            "role": "user",
            "content": f"""Below is a graph describing a set of entities and their relationships. Write a coherent and fluent paragraph that accurately describes this information.

### Graph:
{linearized_graph}

### Description:"""
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, use_cache=True,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id
        )
    
    decoded_output = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    
    return decoded_output.strip()

# --- Gradio Interface ---
demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(
        lines=10, label="Linearized Graph Input", placeholder="Enter graph data here..."
    ),
    outputs=gr.Textbox(lines=10, label="Generated Description"),
    title="Graph-to-Text Generation with Fine-tuned LLM",
    description="This demo showcases a fine-tuned LLM that generates fluent text from structured graph data.",
    examples=[
        ["14th_New_York_State_Legislature | session | 13th_New_York_State_Legislature."],
        ["Ag शिरोमणि | leader | Harchand_Singh_Longowal. Harchand_Singh_Longowal | deathPlace | Punjab."]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Launching Gradio app...")
    demo.launch(server_name="107.99.236.79")
