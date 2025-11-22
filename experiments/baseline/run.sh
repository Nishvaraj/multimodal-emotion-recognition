#!/bin/bash

# Activate the virtual environment
source ../venv/bin/activate

# Run the training script
python ../src/training/train.py --config ../configs/config.yaml

# Deactivate the virtual environment
deactivate