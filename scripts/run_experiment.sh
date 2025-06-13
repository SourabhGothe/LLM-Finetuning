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

# FIX: This is the most robust way to run Hydra.
# We explicitly pass the base config and the experiment config.
# Hydra will merge them, with the experiment config taking priority.
python src/train.py \
  --config-name base_config.yaml \
  --config-dir configs \
  +experiment=$EXPERIMENT_NAME
