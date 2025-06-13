#!/bin/bash

# This script runs a training experiment using Hydra.
# It takes the name of the experiment config file as an argument.

# Check if an experiment name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <experiment_name>"
  echo "Example: $0 llama3_qlora_webnlg"
  exit 1
fi

EXPERIMENT_NAME=$1

echo "Running experiment: $EXPERIMENT_NAME"

# Set PYTHONPATH to the project's root directory.
export PYTHONPATH=$PYTHONPATH:$(pwd)

# FIX: Removed the '+' from '+experiment'.
# The correct Hydra syntax to override a value from a config group
# in the defaults list is just 'experiment=<value>'.
python src/train.py experiment=$EXPERIMENT_NAME
