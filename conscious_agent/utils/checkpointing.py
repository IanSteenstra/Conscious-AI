"""
Checkpoint Management
"""

import torch
import os
from datetime import datetime
from typing import Dict


class CheckpointManager:
    """
    Manages model checkpoints
    - Saves checkpoints at intervals
    - Keeps best checkpoints
    - Enables rollback
    """
    
    def __init__(self, save_dir: str = './checkpoints'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.checkpoints = []
    
    def save(self, checkpoint: Dict, step: int) -> str:
        """
        Save checkpoint
        
        Args:
            checkpoint: Dict with model state, optimizer state, etc.
            step: Training step
        
        Returns:
            Path to saved checkpoint
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step{step}_{timestamp}.pt"
        path = os.path.join(self.save_dir, filename)
        
        torch.save(checkpoint, path)
        
        self.checkpoints.append({
            'path': path,
            'step': step,
            'timestamp': timestamp
        })
        
        return path
    
    def save_final(self, checkpoint: Dict) -> str:
        """Save final checkpoint"""
        
        filename = "checkpoint_final.pt"
        path = os.path.join(self.save_dir, filename)
        
        torch.save(checkpoint, path)
        
        return path
    
    def load(self, step: int) -> Dict:
        """Load checkpoint at specific step"""
        
        for ckpt in self.checkpoints:
            if ckpt['step'] == step:
                return torch.load(ckpt['path'])
        
        raise ValueError(f"No checkpoint found at step {step}")
    
    def load_latest(self) -> Dict:
        """Load most recent checkpoint"""
        
        if not self.checkpoints:
            raise ValueError("No checkpoints available")
        
        latest = max(self.checkpoints, key=lambda x: x['step'])
        return torch.load(latest['path'])