import argparse
from .config import TrainingConfig
from .birdclef_trainer import train_model

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='BirdCLEF Model Training')
    parser.add_argument('--stage', choices=['pretrain_bce', 'train_ce'], required=True,
                        help='Training stage to execute')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Output directory for models and logs')
    return parser.parse_args()

def main():

    args = parse_args()
    
    # Initialize configuration
    config = TrainingConfig()
    config.train_data_path = args.data_dir
    config.output_dir = args.output_dir
    
    # Execute training stage
    train_model(config, args.stage)

if __name__ == '__main__':
    main()
    