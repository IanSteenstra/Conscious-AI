"""
Main Training Script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import torch

from conscious_agent.models.agent import ConsciousAgent
from conscious_agent.training.trainer import ConsciousAgentTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Conscious Agent')
    parser.add_argument(
        '--config',
        type=str,
        default='config/agent_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if specified
    if args.device:
        config['model']['device'] = args.device
    
    # Check CUDA availability
    if config['model']['device'] == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config['model']['device'] = 'cpu'
    
    print(f"\n{'='*60}")
    print("Conscious Agent Training")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Device: {config['model']['device']}")
    print(f"WandB: {'disabled' if args.no_wandb else 'enabled'}")
    print(f"{'='*60}\n")
    
    # Initialize agent
    print("Initializing agent...")
    agent = ConsciousAgent(config)
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = ConsciousAgentTrainer(
        agent=agent,
        config=config,
        use_wandb=not args.no_wandb
    )
    
    # Train
    print("\nStarting training...\n")
    trainer.train()
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()