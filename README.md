Graph-to-Text LLM Finetuning Framework (Offline Edition)
This repository provides a complete, end-to-end framework for fine-tuning open-source Large Language Models (LLMs) on Graph-to-Text generation tasks. It is designed to be modular, extensible, and easy to use for researchers and developers to conduct novel experiments in an offline-first environment.

The system leverages cutting-edge, open-source tools like Unsloth for memory-efficient training, Hugging Face for models and training infrastructure, Hydra for configuration management, TensorBoard for local experiment tracking, and Gradio for interactive demos.

Features
Efficient Finetuning: Integrates Unsloth for significantly faster training and lower memory usage.

PEFT Strategies: Supports various Parameter-Efficient Finetuning techniques, including LoRA and QLoRA.

Model Agnostic: Easily switch between different open-source LLMs like Llama 3, DeepSeek, Mistral, and more.

Config-Driven Experiments: Uses Hydra to manage all experiment configurations.

Robust Data Loading: Loads datasets directly from the Hugging Face Hub, eliminating unreliable download scripts.

Offline Experiment Tracking: Uses TensorBoard to log and visualize training metrics locally.

Comprehensive Evaluation: Includes a dedicated script (evaluate.py) to measure model performance using standard NLG metrics like BLEU and ROUGE.

Rich Visualization: Includes utility scripts to visualize input graphs using pyvis and networkx.

Interactive Inference: Comes with a Gradio application for real-time inference.

Project Structure
The download_data.sh script is no longer needed and has been removed.

graph2text_finetuning/
├── configs/
│   ├── base_config.yaml
│   └── experiments/
│       └── llama3_qlora_webnlg.yaml
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── linearize.py
│   ├── model.py
│   ├── train.py
│   ├── visualize_graph.py
│   └── inference/
│       └── app.py
├── requirements.txt
└── README.md

Setup and Installation
Clone the Repository:

git clone <repository_url>
cd graph2text_finetuning

Install Dependencies:
It is highly recommended to use a virtual environment. This step requires an internet connection.

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Download Assets (Automatic):
The framework now automatically downloads the required dataset and LLM from the Hugging Face Hub the first time you run a script. Subsequent runs will use the cached versions for a fully offline experience.

How to Run
1. Configure Your Experiment
Experiments are defined in .yaml files inside the configs/experiments/ directory. The configuration now points to the Hugging Face dataset name (web_nlg) instead of local files.

2. Run Training
Use the run_experiment.sh script to launch a training run.

bash scripts/run_experiment.sh llama3_qlora_webnlg

3. Monitor with TensorBoard
To view your training progress, open a new terminal and start the TensorBoard server.

tensorboard --logdir=outputs

Then, open your web browser and go to http://localhost:6006/.

4. Run Evaluation
After training is complete, run the evaluation script. You must provide the path to the saved model checkpoint.

# Replace YOUR_MODEL_PATH with the actual path from the training output
python src/evaluate.py +experiment=llama3_qlora_webnlg model_checkpoint_path=YOUR_MODEL_PATH

5. Visualize a Graph
To understand the input data, you can visualize a graph from the dataset.

python src/visualize_graph.py

This will create an interactive_graph.html file.

6. Launch the Inference Demo
To interact with your fine-tuned model, run the Gradio application. Make sure to update the MODEL_DIR in src/inference/app.py to point to your trained adapter checkpoint.

python src/inference/app.py
