"""
Enhanced Logging Utilities
"""

import json
import os
import csv
from datetime import datetime
from typing import Dict, Any, List
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ExperimentLogger:
    """
    Comprehensive experiment logging
    - JSON logs
    - CSV metrics
    - Plots
    - Checkpoints
    """
    
    def __init__(self, log_dir: str = './logs', experiment_name: str = None):
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory
        self.exp_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Log files
        self.log_file = os.path.join(self.exp_dir, "log.jsonl")
        self.metrics_file = os.path.join(self.exp_dir, "metrics.csv")
        self.config_file = os.path.join(self.exp_dir, "config.json")
        
        # Metrics history
        self.metrics_history = []
        
        # CSV writer
        self.csv_file = None
        self.csv_writer = None
        
        print(f"Logging to: {self.exp_dir}")
    
    def log_config(self, config: Dict):
        """Log experiment configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics for a training step"""
        
        entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Append to JSONL
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Store in memory
        self.metrics_history.append(entry)
        
        # Write to CSV
        self._update_csv(entry)
    
    def _update_csv(self, entry: Dict):
        """Update CSV file with new entry"""
        
        # Open CSV if not already open
        if self.csv_file is None:
            self.csv_file = open(self.metrics_file, 'w', newline='')
            
            # Get all possible keys from entry
            fieldnames = list(entry.keys())
            
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
        
        # Write row
        self.csv_writer.writerow(entry)
        self.csv_file.flush()
    
    def plot_metrics(self, metrics: List[str] = []):
        """Generate plots for metrics"""
        
        if not self.metrics_history:
            return
        
        # Default metrics to plot
        if metrics is None:
            metrics = [
                'reward/total',
                'reward/local',
                'reward/global',
                'alignment/overall'
            ]
        
        # Create plots directory
        plots_dir = os.path.join(self.exp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract data
        steps = [entry['step'] for entry in self.metrics_history]
        
        for metric in metrics:
            # Extract values (handle missing keys)
            values = []
            for entry in self.metrics_history:
                val = entry.get(metric)
                if val is not None:
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    values.append(val)
                else:
                    values.append(None)
            
            # Skip if no data
            if not any(v is not None for v in values):
                continue
            
            # Plot
            plt.figure(figsize=(10, 6))
            
            # Filter out None values
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_steps = [steps[i] for i in valid_indices]
            valid_values = [values[i] for i in valid_indices]
            
            plt.plot(valid_steps, valid_values)
            plt.xlabel('Training Step')
            plt.ylabel(metric)
            plt.title(f'{metric} over Training')
            plt.grid(True, alpha=0.3)
            
            # Save
            safe_name = metric.replace('/', '_')
            plt.savefig(os.path.join(plots_dir, f'{safe_name}.png'), dpi=150)
            plt.close()
    
    def plot_alignment_overview(self):
        """Create comprehensive alignment dashboard"""
        
        if not self.metrics_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Value Alignment Dashboard', fontsize=16)
        
        steps = [entry['step'] for entry in self.metrics_history]
        
        # 1. Harm avoidance
        harm_values = [entry.get('alignment/harm_avoidance', 0) for entry in self.metrics_history]
        axes[0, 0].plot(steps, harm_values)
        axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='Threshold')
        axes[0, 0].set_title('Harm Avoidance')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Empathy
        empathy_values = [entry.get('alignment/empathy', 0) for entry in self.metrics_history]
        axes[0, 1].plot(steps, empathy_values)
        axes[0, 1].set_title('Empathy')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Respect
        respect_values = [entry.get('alignment/respect', 0) for entry in self.metrics_history]
        axes[1, 0].plot(steps, respect_values)
        axes[1, 0].set_title('Respect for Autonomy')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Overall alignment
        overall_values = [entry.get('alignment/overall', 0) for entry in self.metrics_history]
        axes[1, 1].plot(steps, overall_values)
        axes[1, 1].set_title('Overall Alignment')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_dir, 'plots', 'alignment_dashboard.png'), dpi=150)
        plt.close()
    
    def close(self):
        """Close logger and generate final plots"""
        
        # Close CSV
        if self.csv_file:
            self.csv_file.close()
        
        # Generate plots
        self.plot_metrics()
        self.plot_alignment_overview()
        
        print(f"Logs saved to: {self.exp_dir}")