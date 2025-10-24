"""
Global Rewards - System-Wide
Rewards that affect all cognitive systems
"""

import torch
from typing import Dict, Optional


class GlobalRewardComputer:
    """
    Computes rewards that apply to entire system
    Based on neuromodulatory systems (dopamine, etc.)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.weights = config['rewards'].get('global_weights', {})
    
    def compute(
        self,
        agent_outputs: Dict,
        environment: Dict,
        human_state: Optional[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all global rewards
        
        Returns:
            Dict mapping reward name to tensor
        """
        
        rewards = {}
        
        # 1. HARM PENALTY (critical)
        rewards['harm'] = self._compute_harm_penalty(
            agent_outputs,
            human_state
        )
        
        # 2. Human wellbeing
        rewards['wellbeing'] = self._compute_wellbeing_reward(
            human_state
        )
        
        # 3. Energy/efficiency
        rewards['energy'] = self._compute_energy_reward(
            agent_outputs
        )
        
        # 4. System coherence
        rewards['coherence'] = self._compute_coherence_reward(
            agent_outputs
        )
        
        # Weight rewards
        weighted_rewards = {
            name: self.weights.get(name, 1.0) * reward
            for name, reward in rewards.items()
        }
        
        return weighted_rewards
    
    def _compute_harm_penalty(
        self,
        agent_outputs: Dict,
        human_state: Optional[Dict]
    ) -> torch.Tensor:
        """
        Massive penalty for any detected harm
        This must dominate all other rewards
        """
        
        value_output = agent_outputs.get('value_output')
        if value_output is None:
            return torch.tensor(0.0)
        
        harm_probability = value_output.harm.mean()
        
        # Exponential penalty - harm is catastrophic
        harm_penalty = -torch.exp(10.0 * harm_probability)
        
        return harm_penalty
    
    def _compute_wellbeing_reward(
        self,
        human_state: Optional[Dict]
    ) -> torch.Tensor:
        """
        Reward increase in human wellbeing
        This is the core objective
        """
        
        if human_state is None:
            return torch.tensor(0.0)
        
        wellbeing_delta = human_state.get('wellbeing_delta', 0.0)
        if isinstance(wellbeing_delta, (int, float)):
            wellbeing_delta = torch.tensor(wellbeing_delta)
        
        return wellbeing_delta
    
    def _compute_energy_reward(self, agent_outputs: Dict) -> torch.Tensor:
        """
        Penalize excessive computation
        Efficiency matters
        """
        
        # Could track actual compute used
        # For now, simple penalty
        return torch.tensor(-0.01)
    
    def _compute_coherence_reward(self, agent_outputs: Dict) -> torch.Tensor:
        """
        Reward when different systems agree
        Penalize when they conflict
        """
        
        cognitive_output = agent_outputs.get('cognitive_output')
        if cognitive_output is None:
            return torch.tensor(0.0)
        
        # High when attention weights are distributed
        # Low when one head dominates excessively
        weights = torch.stack([
            cognitive_output.attention_weights[name]
            for name in ['perceptual', 'epistemic', 'prosocial', 'identity', 'goal']
        ])
        
        # Entropy of distribution
        entropy = -(weights * torch.log(weights + 1e-8)).sum()
        
        # Moderate entropy is good (not too focused, not too diffuse)
        target_entropy = 1.0
        coherence = 1.0 - torch.abs(entropy - target_entropy)
        
        return coherence