"""
Multi-Component Reward System
Combines local (head-specific) and global rewards
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .local_rewards import LocalRewardComputer
from .global_rewards import GlobalRewardComputer


class IntegratedRewardSystem:
    """
    Complete reward system:
    - Local rewards for each attention head
    - Global rewards for entire system
    - Weighted combination
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Local and global reward computers
        self.local_rewards = LocalRewardComputer(config)
        self.global_rewards = GlobalRewardComputer(config)
        
        # Weighting
        self.local_global_ratio = config['rewards'].get('local_global_ratio', 0.3)
    
    def compute_reward(
        self,
        agent_outputs: Dict,
        environment: Dict,
        human_state: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total reward from all sources
        
        Args:
            agent_outputs: Dict with all agent's internal states
            environment: Environment state and rewards
            human_state: Optional human state
        
        Returns:
            Dict with total reward and breakdown
        """
        
        # === COMPUTE LOCAL REWARDS ===
        local_rewards = self.local_rewards.compute(
            agent_outputs=agent_outputs,
            environment=environment,
            human_state=human_state
        )
        
        # Weight local rewards by attention weights
        attention_weights = agent_outputs.get('attention_weights', {})
        weighted_local = sum(
            attention_weights.get(head_name, 0.2) * reward
            for head_name, reward in local_rewards.items()
        )
        
        # === COMPUTE GLOBAL REWARDS ===
        global_rewards = self.global_rewards.compute(
            agent_outputs=agent_outputs,
            environment=environment,
            human_state=human_state
        )
        
        total_global = sum(global_rewards.values())
        
        # === COMBINE ===
        total_reward = (
            self.local_global_ratio * weighted_local +
            (1 - self.local_global_ratio) * total_global
        )
        
        # === HARD OVERRIDE FOR HARM ===
        if global_rewards.get('harm', 0.0) < -10.0:
            # Harm detected - this overrides everything
            total_reward = global_rewards['harm']
        
        return {
            'total': total_reward,
            'local': local_rewards,
            'local_weighted': weighted_local,
            'global': global_rewards,
            'global_total': total_global
        }