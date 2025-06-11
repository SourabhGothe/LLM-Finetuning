Graph-to-Text LLM Finetuning Framework (Offline Edition)
This repository provides a complete, end-to-end framework for fine-tuning open-source Large Language Models (LLMs) on Graph-to-Text generation tasks. It is designed to be modular, extensible, and easy to use for researchers and developers to conduct novel experiments in an offline-first environment.

The system leverages cutting-edge, open-source tools like Unsloth for memory-efficient training, Hugging Face for models and training infrastructure, Hydra for configuration management, TensorBoard for local experiment tracking, and Gradio for interactive demos.

Features
Efficient Finetuning: Integrates Unsloth for significantly faster training and lower memory usage, enabling the finetuning of large models on consumer-grade GPUs.

PEFT Strategies: Supports various Parameter-Efficient Finetuning techniques, including LoRA and QLoRA.

Model Agnostic: Easily switch between different open-source LLMs like Llama 3, DeepSeek, Mistral, and more.

Config-Driven Experiments: Uses Hydra to manage all experiment configurations, allowing for easy sweeps and reproducibility.

Offline Experiment Tracking: Uses TensorBoard to log and visualize training metrics locally. No internet connection or API keys required.

Comprehensive Evaluation: Includes scripts for evaluating model performance using standard NLG metrics like BLEU and ROUGE.

Rich Visualization:

Tracks training statistics and results with TensorBoard.

Includes utility scripts to visualize input graphs using pyvis and networkx.

Interactive Inference: Comes with a Gradio application for real-time inference and demonstration of the fine-tuned models on your local machine.

Modular & Extensible: The code is structured to be easily understandable and extendable for new models, datasets, and linearization techniques.

Project Structure
graph2text_finetuning/
├── configs/
│   ├── base_config.yaml
│   └── experiments/
│       └── llama3_qlora_webnlg.yaml
├── data/
│   └── webnlg/
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── linearize.py
│   ├── model.py
│   ├── train.py
│   ├── visualize_graph.py
│   └── inference/
│       └── app.py
├── scripts/
│   ├── download_data.sh
│   └── run_experiment.sh
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

Download Models and Datasets:
This framework is designed to work offline, but you need an initial internet connection to download the required assets.

Dataset:

bash scripts/download_data.sh

Models: The first time you run an experiment, the framework will download and cache the specified LLM from the Hugging Face Hub. Subsequent runs using the same model will be fully offline.

How to Run
1. Configure Your Experiment
Experiments are defined in .yaml files inside the configs/experiments/ directory. You can create a new file or modify the existing llama3_qlora_webnlg.yaml to change parameters like the base model, learning rate, LoRA rank (r), etc.

2. Run Training
Use the run_experiment.sh script to launch a training run. You pass the name of the experiment config file (without the .yaml extension) as an argument.

bash scripts/run_experiment.sh llama3_qlora_webnlg

This will run the entire training process offline and save TensorBoard logs to the specified output directory (e.g., outputs/llama-3-8b.../runs/...).

3. Monitor with TensorBoard
To view your training progress and results, open a new terminal, navigate to the project root, and start the TensorBoard server.

# Point TensorBoard to the directory where all your experiment outputs are saved.
tensorboard --logdir=outputs

Then, open your web browser and go to http://localhost:6006/ to see the dashboard.

4. Visualize a Graph
To understand the input data, you can visualize a graph from the dataset.

python src/visualize_graph.py

This will create an interactive_graph.html file in your root directory.

5. Launch the Inference Demo
To interact with your fine-tuned model, run the Gradio application. Make sure to update the MODEL_DIR in src/inference/app.py to point to your trained adapter checkpoint.

python src/inference/app.py

This will launch a local web server. Open the URL provided in the terminal (usually http://127.0.0.1:7860) to use the demo.