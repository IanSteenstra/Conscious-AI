"""
Cognitive Pre-training Script
Teaches the agent "who it is" and "how to think" before interaction.
Uses Supervised Learning with Multi-Task Loss (Text + Cognitive Targets).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import yaml
import argparse
import sys
import os
from tqdm import tqdm
from typing import Dict, Iterator

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conscious_agent.models.agent import ConsciousAgent

class CognitiveDataset(IterableDataset):
    """
    Streaming Dataset for Cognitive Pre-training.
    Designed to handle millions of examples without loading all into RAM.
    """
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # In production, this would be a file path to a JSONL file
        # self.data_path = "data/cognitive_pretrain_1M.jsonl"
        
    def __iter__(self) -> Iterator[Dict]:
        """
        Yields batches of data.
        In production, read from file line by line here.
        """
        # SIMULATED DATA GENERATOR
        # Generates infinite stream of synthetic examples for demonstration
        while True:
            yield self._generate_synthetic_example()
            
    def _generate_synthetic_example(self):
        """Generate a synthetic example with cognitive targets"""
        
        # Example 1: Curiosity Trigger
        prompt = "I found a strange artifact."
        target = "That is fascinating. What does it look like?"
        
        # Tokenize
        full_text = prompt + " " + target
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        
        # Create labels (mask out prompt, only train on response)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[:prompt_len] = -100  # Ignore prompt in loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'cognitive_targets': {
                'curiosity_level': 0.9,  # High curiosity
                'emotional_valence': 0.5
            }
        }

def train(config_path: str):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['model']['device'] = device
    
    # 2. Initialize Agent
    print("Initializing Conscious Agent...")
    agent = ConsciousAgent(config)
    agent.to(device)
    agent.train() # Set to training mode
    
    # 3. Setup Data
    dataset = CognitiveDataset(agent.tokenizer)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'])
    
    # 4. Optimizer (Only trains the new cognitive layers, LLM is frozen)
    optimizer = optim.AdamW(
        agent.get_trainable_parameters(),
        lr=config['training']['learning_rate']
    )
    
    # 5. Training Loop
    print("Starting Cognitive Pre-training...")
    num_steps = 1000 # Demo steps
    pbar = tqdm(total=num_steps)
    
    step = 0
    for batch in dataloader:
        if step >= num_steps:
            break
            
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward Pass (using the new forward_train method)
        outputs = agent.forward_train(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # === LOSS COMPUTATION ===
        
        # A. Text Loss (Next Token Prediction)
        # Shift logits and labels for causal LM loss
        # Logits: [batch, seq_len + 1 (thought), vocab] -> we ignore thought token for text loss mapping
        # Actually, the LLM output includes the thought token position.
        # The output logits shape is [batch, seq_len + 1, vocab]
        # We need to align them.
        
        logits = outputs['logits']
        # Remove the first token (thought embedding position) from logits to match text labels
        text_logits = logits[:, 1:, :] 
        
        # Standard Causal LM Loss
        loss_fct = nn.CrossEntropyLoss()
        text_loss = loss_fct(
            text_logits.reshape(-1, text_logits.size(-1)), 
            labels.reshape(-1)
        )
        
        # B. Cognitive Loss (Supervised Thoughts)
        cog_loss = 0.0
        targets = batch['cognitive_targets']
        
        # Curiosity Loss
        pred_curiosity = outputs['curiosity_output'].novelty_score
        target_curiosity = torch.tensor(targets['curiosity_level'], device=device)
        cog_loss += nn.MSELoss()(pred_curiosity, target_curiosity)
        
        # Total Loss
        total_loss = text_loss + (0.5 * cog_loss)
        
        # Backward Pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.get_trainable_parameters(), 1.0)
        optimizer.step()
        
        # Logging
        pbar.update(1)
        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'text_loss': f"{text_loss.item():.4f}",
            'cog_loss': f"{cog_loss.item():.4f}"
        })
        
        step += 1
        
    # Save Pre-trained Model
    print("\nSaving pre-trained cognitive weights...")
    torch.save(agent.state_dict(), "data/checkpoints/cognitive_pretrained.pt")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/agent_config.yaml")
    args = parser.parse_args()
    
    train(args.config)
