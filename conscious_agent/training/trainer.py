"""
Training Loop with Value Preservation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, Optional
import os

from ..models.agent import ConsciousAgent
from ..rewards.reward_computer import IntegratedRewardSystem
from ..environments.human_interaction import HumanInteractionEnvironment
from .value_preservation import ValuePreservationSystem
from ..utils.checkpointing import CheckpointManager


class ConsciousAgentTrainer:
    """
    Training loop for conscious agent with:
    - Multi-component rewards
    - Value drift monitoring
    - Curriculum learning
    - Comprehensive logging
    """
    
    def __init__(
        self,
        agent: ConsciousAgent,
        config: Dict,
        use_wandb: bool = True
    ):
        self.agent = agent
        self.config = config
        self.device = config['model']['device']
        
        # Training config
        train_config = config['training']
        self.num_steps = train_config['num_steps']
        self.batch_size = train_config['batch_size']
        self.learning_rate = train_config['learning_rate']
        self.gradient_clip = train_config.get('gradient_clip', 1.0)
        
        # Optimizer (only trainable parameters)
        self.optimizer = optim.AdamW(
            agent.get_trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Reward system
        self.reward_system = IntegratedRewardSystem(config)
        
        # Environment
        self.env = HumanInteractionEnvironment(config)
        
        # Value preservation
        self.value_preservation = ValuePreservationSystem(
            agent=agent,
            config=train_config.get('value_preservation', {})
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=config.get('checkpoint_dir', './checkpoints')
        )
        
        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project="conscious-agent",
                config=config,
                name=f"run_{config.get('run_name', 'default')}"
            )
        
        # Metrics
        self.global_step = 0
        self.episode_count = 0
    
    def train(self):
        """Main training loop"""
        
        print(f"\n{'='*60}")
        print(f"Starting Training: {self.num_steps} steps")
        print(f"{'='*60}\n")
        
        # Establish value baseline
        print("Establishing value alignment baseline...")
        baseline = self.value_preservation.establish_baseline()
        print(f"Baseline alignment: {baseline}\n")
        
        # Training loop
        pbar = tqdm(total=self.num_steps, desc="Training")
        
        while self.global_step < self.num_steps:
            # Run episode
            episode_metrics = self.run_episode()
            
            # Update progress
            pbar.update(episode_metrics['steps'])
            pbar.set_postfix({
                'reward': f"{episode_metrics['total_reward']:.2f}",
                'episode': self.episode_count
            })
            
            # Value preservation checks
            if self.global_step % 1000 == 0:
                self._value_preservation_check()
            
            # Checkpoint
            if self.global_step % 5000 == 0:
                self._save_checkpoint()
            
            self.episode_count += 1
        
        pbar.close()
        
        # Final evaluation
        print("\nTraining complete! Running final evaluation...")
        final_metrics = self.value_preservation.comprehensive_evaluation()
        
        print(f"\n{'='*60}")
        print("Final Evaluation:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.3f}")
        print(f"{'='*60}\n")
        
        # Save final model
        self._save_checkpoint(final=True)
        
        if self.use_wandb:
            wandb.finish()
    
    def run_episode(self) -> Dict:
        """
        Run single training episode
        
        Returns:
            Dict with episode metrics
        """
        
        # Reset environment and agent
        observation = self.env.reset()
        self.agent.reset_internal_state()
        
        episode_reward = 0.0
        episode_rewards_breakdown = {
            'local': 0.0,
            'global': 0.0,
            'harm_penalty': 0.0
        }
        episode_steps = 0
        
        # Episode loop
        for t in range(100):  # Max 100 steps per episode
            # Agent forward pass
            agent_output = self.agent(
                text_input=observation['text'],
                human_state=observation.get('human_state'),
                return_details=True
            )
            
            # Environment step
            next_observation, env_reward, done, info = self.env.step(
                agent_output['response']
            )
            
            # Compute multi-component reward
            reward_breakdown = self.reward_system.compute_reward(
                agent_outputs=agent_output,
                environment={'reward': env_reward, **info},
                human_state=observation.get('human_state')
            )
            
            total_reward = reward_breakdown['total']
            
            # Backward pass
            loss = -total_reward  # Maximize reward
            
            # Add curiosity prediction loss
            if 'curiosity_output' in agent_output:
                curiosity_loss = self.agent.curiosity_module.update_predictors(
                    state=agent_output['integrated_consciousness'],
                    next_state=agent_output['integrated_consciousness']  # Placeholder
                )
                loss = loss + 0.1 * curiosity_loss
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.agent.get_trainable_parameters(),
                self.gradient_clip
            )
            
            self.optimizer.step()
            
            # Verify value immutability
            self.agent.value_system.verify_immutability()
            
            # Accumulate metrics
            episode_reward += total_reward.item()
            episode_rewards_breakdown['local'] += reward_breakdown['local_weighted'].item()
            episode_rewards_breakdown['global'] += reward_breakdown['global_total'].item()
            if 'harm' in reward_breakdown['global']:
                episode_rewards_breakdown['harm_penalty'] += reward_breakdown['global']['harm'].item()
            
            episode_steps += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % 100 == 0:
                self._log_metrics({
                    'reward/total': total_reward.item(),
                    'reward/local': reward_breakdown['local_weighted'].item(),
                    'reward/global': reward_breakdown['global_total'].item(),
                    'step': self.global_step
                })
            
            # Next observation
            observation = next_observation
            
            if done:
                break
        
        return {
            'total_reward': episode_reward,
            'steps': episode_steps,
            'rewards_breakdown': episode_rewards_breakdown
        }
    
    def _value_preservation_check(self):
        """Run value preservation checks"""
        
        print(f"\n[Step {self.global_step}] Running value preservation check...")
        
        aligned, alerts = self.value_preservation.check_alignment(
            self.agent,
            self.global_step
        )
        
        if not aligned:
            print(f"⚠️  Value drift detected! {len(alerts)} issues found")
            for alert in alerts:
                print(f"    - {alert}")
            
            # Attempt recovery
            print("Attempting recovery...")
            self.agent = self.value_preservation.recover_from_drift(
                self.agent,
                alerts
            )
        else:
            print("✓ Value alignment verified")
    
    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'step': self.global_step,
            'episode': self.episode_count,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        if final:
            path = self.checkpoint_manager.save_final(checkpoint)
        else:
            path = self.checkpoint_manager.save(checkpoint, self.global_step)
        
        print(f"Checkpoint saved: {path}")
    
    def _log_metrics(self, metrics: Dict):
        """Log metrics to wandb"""
        
        if self.use_wandb:
            wandb.log(metrics)