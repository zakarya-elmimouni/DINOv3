import yaml
from src.train import Trainer
import torch
import os

if __name__ == '__main__':
    # Ensure the dinov3 library is in the Python path
    # This is necessary for the direct imports in `src/model.py` to work
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # Load configuration from YAML file
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at '{config_path}'")
        sys.exit(1)

    print("--- Configuration Loaded ---")
    for section, settings in config.items():
        if isinstance(settings, dict):
            print(f"  {section.capitalize()}:")
            for key, value in settings.items():
                print(f"    {key}: {value}")
        else:
            print(f"  {section}: {settings}")
    print("--------------------------\n")

    # Set CUDA device visibility if you have multiple GPUs
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Start the training process
    trainer = Trainer(config)
    trainer.train()