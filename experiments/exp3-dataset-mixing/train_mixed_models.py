#!/usr/bin/env python3
"""
Training script for mixed datasets in Experiment 3
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.exp1_pure_synthetic.model_configs import create_model_config
from experiments.exp1_pure_synthetic.train_pure_synthetic import main as train_main

def main():
    parser = argparse.ArgumentParser(description="Train models on mixed datasets")
    parser.add_argument("--strategy", type=str, required=True, help="Mixing strategy name")
    parser.add_argument("--model_size", type=str, default="500M", help="Model size")
    parser.add_argument("--dataset_dir", type=str, default="./data/mixed_datasets", help="Mixed datasets directory")
    
    args = parser.parse_args()
    
    # Set dataset path
    dataset_path = f"{args.dataset_dir}/{args.strategy}"
    
    # Run training with mixed dataset
    train_args = [
        "--dataset_path", dataset_path,
        "--model_size", args.model_size,
        "--output_dir", f"./models/exp3-{args.strategy}",
        "--run_name", f"exp3-{args.strategy}-{args.model_size}",
        "--fp16",
        "--gradient_checkpointing"
    ]
    
    # Call training function
    sys.argv = ["train_mixed_models.py"] + train_args
    train_main()

if __name__ == "__main__":
    main()