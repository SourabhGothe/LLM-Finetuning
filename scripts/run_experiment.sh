#!/bin/bash

#bash scripts/run_experiment.sh llama_3_2_1b_instruct strict
# This script runs a training experiment using Hydra.
# It takes the model config name as the first argument, and an optional
# training config name as the second argument.

# Ensure the script exits if any command fails
set -e

# Check if a model name is provided
if [ -z "$1" ];
then
  echo "Usage: $0 <model_config_name> [training_config_name]"
  echo "Example 1 (default training): $0 llama3_qlora_webnlg"
  echo "Example 2 (strict training):  $0 llama_3_2_1b_instruct strict"
  exit 1
fi

MODEL_NAME=$1
TRAINING_PROFILE=$2

# Get the absolute directory where this script is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Change the current directory to the project root.
cd "$PROJECT_ROOT"

echo "---"
echo "Project root set to: $(pwd)"
echo "Starting experiment with model: $MODEL_NAME"

# Build the python command
PYTHON_CMD="python src/train.py model=$MODEL_NAME"

# FIX: If a second argument (training profile) is provided, add it to the command.
# This is the correct way to override a config group in Hydra.
if [ ! -z "$TRAINING_PROFILE" ]; then
  echo "Using training profile: $TRAINING_PROFILE"
  PYTHON_CMD="$PYTHON_CMD training=$TRAINING_PROFILE"
fi
echo "---"


# Set PYTHONPATH to include the project's root directory.
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Execute the final command
eval $PYTHON_CMD
