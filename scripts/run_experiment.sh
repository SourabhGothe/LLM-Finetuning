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
# This ensures that Python can find the 'src' module.
export PYTHONPATH=$PYTHONPATH:$(pwd)

# FIX: The correct Hydra syntax for overriding a default from a config group
# is 'config_group_name=value'. The '+' is not needed and was causing issues.
# Our main script now points to the 'experiment' directory, so we just need
# to pass the experiment name to override the default.
python src/train.py experiments=$EXPERIMENT_NAME
