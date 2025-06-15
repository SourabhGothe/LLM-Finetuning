#!/bin/bash

# This script runs a training experiment using Hydra.
# It takes the name of the model config file (without .yaml) as an argument.

# Ensure the script exits if any command fails
set -e

# Check if a model name is provided
if [ -z "$1" ];
then
  echo "Usage: $0 <model_config_name>"
  echo "Example: $0 deepseek_qwen_14b"
  exit 1
fi

MODEL_NAME=$1

# Get the absolute directory where this script is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Change the current directory to the project root.
# This ensures all relative paths in the python script are correct.
cd "$PROJECT_ROOT"

echo "---"
echo "Project root set to: $(pwd)"
echo "Starting experiment with model: $MODEL_NAME"
echo "---"

# Set PYTHONPATH to include the project's root directory.
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Execute the python script using the most robust Hydra syntax.
# 'model=$MODEL_NAME' overrides the default model specified in main_config.yaml.
python src/train.py model=$MODEL_NAME
