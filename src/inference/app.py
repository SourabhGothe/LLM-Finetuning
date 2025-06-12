import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# src/inference/app.py
# Gradio app for interactive inference with the fine-tuned model. (Offline Edition)

import os
import sys
import gradio as gr
from unsloth import FastLanguageModel
import torch
from langchain.prompts import PromptTemplate

# FIX: Add project root to the Python path
# This allows the script to find the 'src' module and its submodules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Go up two levels from src/inference
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration ---
# =================================================================================
#
#  !!! CRITICAL: YOU MUST UPDATE THIS PATH !!!
#
#  Replace the path below with the path to YOUR fine-tuned model checkpoint.
#  This is the 'final_checkpoint' directory created by the training script.
#
#  Example: "outputs/llama-3-8b-instruct-bnb-4bit/llama3_8b_qlora_webnlg_YYYY-MM-DD_HH-MM-SS/final_checkpoint"
#
# =================================================================================
MODEL_DIR = "outputs/llama-3-8b-instruct-bnb-4bit/llama3_8b_qlora_webnlg_2024-06-11_16-08-34/final_checkpoint"


MAX_SEQ_LENGTH = 2048

# --- Load Model ---
# Load the base model and merge the adapter.
print(f"Loading fine-tuned model from: {MODEL_DIR}")

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(
        f"Model directory not found at '{MODEL_DIR}'.\n"
        "Please update the MODEL_DIR variable in this script to point to your 'final_checkpoint' directory."
    )

if torch.cuda.is_available():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
else: # For CPU inference
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float32,
        load_in_4bit=False,
    )

# FIX: Explicitly set the padding token for the tokenizer.
# Llama-3 models do not have a default padding token. Setting it to the
# end-of-sequence (EOS) token is a standard and required practice.
tokenizer.pad_token = tokenizer.eos_token

# Set to evaluation mode
model.eval()
print("Model loaded successfully.")


# --- Langchain Prompt Template ---
# This ensures the input to the model is in the exact same format as during training.
prompt_template_str = """Below is a graph describing a set of entities and their relationships. Write a coherent and fluent paragraph that accurately describes this information.

### Graph:
{graph_input}

### Description:
"""

prompt_template = PromptTemplate(
    input_variables=["graph_input"],
    template=prompt_template_str,
)


def generate_text(linearized_graph: str) -> str:
    """
    The main inference function for the Gradio app.
    """
    if not linearized_graph.strip():
        return "Please enter a linearized graph."
        
    formatted_prompt = prompt_template.format(graph_input=linearized_graph)
    
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id # Use the newly set pad token ID
        )
    
    # Decode the output, skipping the prompt part
    decoded_output = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    
    return decoded_output.strip()

# --- Gradio Interface ---
css = """
body { font-family: 'Helvetica Neue', 'Arial', sans-serif; }
.gr-button { background-color: #4CAF50; color: white; border-radius: 8px; }
footer { display: none !important; }
"""
demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(
        lines=10, 
        label="Linearized Graph Input", 
        placeholder="Enter graph data here, e.g., 'Entity1 | relation | Entity2. Entity2 | relation2 | Entity3.'"
    ),
    outputs=gr.Textbox(
        lines=10, 
        label="Generated Description"
    ),
    title="Graph-to-Text Generation with Fine-tuned LLM (Offline)",
    description="""
    This demo showcases an LLM fine-tuned on the WebNLG dataset to generate fluent text from structured graph data.
    Enter a linearized graph in the input box and click 'Submit' to see the model's description.
    The model is powered by Unsloth for efficient training and inference.
    """,
    examples=[
        ["14th_New_York_State_Legislature | session | 13th_New_York_State_Legislature."],
        ["Ag शिरोमणि | leader | Harchand_Singh_Longowal. Harchand_Singh_Longowal | deathPlace | Punjab."]
    ],
    css=css,
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Launching Gradio app...")
    demo.launch()
