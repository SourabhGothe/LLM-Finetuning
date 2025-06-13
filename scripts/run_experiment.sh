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

# --- FIX: Make the script robust to where it is called from ---

# 1. Get the absolute directory where this script is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 2. Get the project root, which is one level up from the scripts directory.
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# 3. Change the current directory to the project root.
#    This ensures all relative paths in the python script are correct.
cd "$PROJECT_ROOT"

echo "---"
echo "Project root set to: $(pwd)"
echo "Starting experiment: $EXPERIMENT_NAME"
echo "---"

# 4. Set PYTHONPATH to the project's root directory.
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# 5. Execute the python script.
#    The path to the config directory is now simple and correct because we are
#    in the project root.
python src/train.py --config-dir configs/experiment --config-name=$EXPERIMENT_NAME
