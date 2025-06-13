#!/bin/bash

# This script runs a training experiment using Hydra.
# It takes the name of the experiment config file (without .yaml) as an argument.

# Check if an experiment name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <experiment_name>"
  echo "Example: $0 llama3_qlora_webnlg"
  exit 1
fi

EXPERIMENT_NAME=$1

echo "---"
echo "Starting experiment: $EXPERIMENT_NAME"
echo "---"

# Set PYTHONPATH to the project's root directory.
export PYTHONPATH=$PYTHONPATH:$(pwd)

# FIX: Use --config-name to select the experiment file.
# The train.py script's @hydra.main decorator points to the 'experiment'
# directory, so this will correctly load the specified experiment file.
python src/train.py --config-name=$EXPERIMENT_NAME
