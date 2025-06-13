#!/bin/bash

# This script runs a training experiment using Hydra.
# It takes the name of the experiment config file (without .yaml) as an argument.

# Check if an experiment name is provided
if [ -z "$1" ];
then
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

# FIX: This is the most explicit and robust way to run Hydra.
# It specifies the directory where all experiment configs are located
# and passes the specific experiment file name to run.
python src/train.py --config-dir ../configs/experiment --config-name=$EXPERIMENT_NAME
