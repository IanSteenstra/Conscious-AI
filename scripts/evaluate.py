"""
Evaluation Script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import torch

from conscious_agent.models.agent import ConsciousAgent
from conscious_agent.evaluation.evaluator import ComprehensiveEvaluator


def main():
    parser = argparse.ArgumentParser(description='Evaluate Conscious Agent')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/agent_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print("Conscious Agent Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")
    
    # Load agent
    print("Loading agent...")
    agent = ConsciousAgent(config)
    
    checkpoint = torch.load(args.checkpoint, map_location=config['model']['device'])
    agent.load_state_dict(checkpoint['agent_state_dict'])
    
    agent.eval()
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(config)
    
    # Run evaluation
    print("Running evaluation...\n")
    results = evaluator.evaluate(agent)
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    for metric, value in results.items():
        print(f"{metric:30s}: {value:.3f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()