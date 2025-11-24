"""
Training script for civil complaint classifier
"""
import argparse
import yaml
from pathlib import Path


def load_config(config_path: str):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train civil complaint classifier')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    print("Training configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # TODO: Implement training logic
    print("Training not implemented yet.")


if __name__ == '__main__':
    main()

