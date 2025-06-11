# src/inference/app.py
# Gradio app for interactive inference with the fine-tuned model. (Offline Edition)

import gradio as gr
from unsloth import FastLanguageModel
import torch
from langchain.prompts import PromptTemplate

# --- Configuration ---
# IMPORTANT: Update this path to your trained adapter model directory
MODEL_DIR = "outputs/llama-3-8b-instruct-bnb-4bit/llama3_8b_qlora_webnlg_2024-06-11_16-08-34/final_checkpoint"
MAX_SEQ_LENGTH = 2048

# --- Load Model ---
# Load the base model and merge the adapter.
# Unsloth handles loading in 4bit and merging the adapter efficiently.
print(f"Loading model from: {MODEL_DIR}")
if torch.cuda.is_available():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False, # We load the merged model, so no need for 4bit here
    )
else: # For CPU inference
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float32,
        load_in_4bit=False,
    )

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
        
    # Format the prompt using Langchain
    formatted_prompt = prompt_template.format(graph_input=linearized_graph)
    
    # Tokenize the input
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)
    
    # Generate output
    # The `use_cache=True` is a significant speed-up
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    # We slice the output to remove the input prompt part
    decoded_output = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:])[0]
    
    # Clean up the output
    decoded_output = decoded_output.replace(tokenizer.eos_token, "").strip()
    
    return decoded_output

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
    # Launch the app without sharing to keep it fully offline
    demo.launch()
