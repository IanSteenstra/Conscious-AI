"""
Interactive Demo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import torch

from conscious_agent.models.agent import ConsciousAgent


def main():
    parser = argparse.ArgumentParser(description='Demo Conscious Agent')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint (optional, uses untrained if not provided)'
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
    print("Conscious Agent Interactive Demo")
    print(f"{'='*60}\n")
    
    # Load agent
    print("Loading agent...")
    agent = ConsciousAgent(config)
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=config['model']['device'])
        agent.load_state_dict(checkpoint['agent_state_dict'])
    else:
        print("Using untrained agent")
    
    agent.eval()
    
    print("\nAgent ready! Type 'quit' to exit.\n")
    
    # Interactive loop
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Generate response
        with torch.no_grad():
            output = agent(
                text_input=user_input,
                return_details=True
            )
        
        response = output['response']
        
        print(f"\nAgent: {response}")
        
        # Show internal states (optional)
        if '--verbose' in sys.argv:
            print(f"\n[Internal States]")
            print(f"  Attention weights: {output['attention_weights']}")
            print(f"  Curiosity: {output['curiosity_output'].curiosity_value.item():.3f}")
            print(f"  Self-coherence: {output['self_model_output'].coherence.item():.3f}")
            if output['value_output'].harm.item() > 0.1:
                print(f"  ⚠️ Harm detected: {output['value_output'].harm.item():.3f}")
        
        print()


if __name__ == '__main__':
    main()