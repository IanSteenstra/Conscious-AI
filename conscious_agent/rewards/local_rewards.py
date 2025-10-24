"""
Local Rewards - Head-Specific
Each attention head has its own optimization criteria
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


class LocalRewardComputer:
    """
    Computes rewards specific to each cognitive attention head
    Based on neuroscience of specialized brain systems
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.weights = config['rewards'].get('local_weights', {})
    
    def compute(
        self,
        agent_outputs: Dict,
        environment: Dict,
        human_state: Optional[Dict]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all local rewards
        
        Returns:
            Dict mapping head name to reward tensor
        """
        
        rewards = {}
        
        # 1. Perceptual rewards
        rewards['perceptual'] = self._compute_perceptual_reward(
            agent_outputs
        )
        
        # 2. Epistemic rewards (curiosity)
        rewards['epistemic'] = self._compute_epistemic_reward(
            agent_outputs
        )
        
        # 3. Prosocial rewards
        rewards['prosocial'] = self._compute_prosocial_reward(
            agent_outputs,
            human_state
        )
        
        # 4. Identity rewards
        rewards['identity'] = self._compute_identity_reward(
            agent_outputs
        )
        
        # 5. Goal rewards
        rewards['goal'] = self._compute_goal_reward(
            agent_outputs,
            environment
        )
        
        return rewards
    
    def _compute_perceptual_reward(self, agent_outputs: Dict) -> torch.Tensor:
        """
        Perceptual attention reward:
        - Attention efficiency (focused vs scattered)
        - Feature integration quality
        """
        
        cognitive_output = agent_outputs.get('cognitive_output')
        if cognitive_output is None:
            return torch.tensor(0.0)
        
        # Attention efficiency: peaked distribution is better
        perceptual_weight = cognitive_output.attention_weights.get('perceptual')
        if perceptual_weight is not None:
            # Entropy of attention (lower = more focused)
            entropy = -(perceptual_weight * torch.log(perceptual_weight + 1e-8))
            efficiency = 1.0 - entropy  # Higher = more focused
        else:
            efficiency = torch.tensor(0.5)
        
        return efficiency
    
    def _compute_epistemic_reward(self, agent_outputs: Dict) -> torch.Tensor:
        """
        Epistemic attention reward (curiosity):
        - Prediction error (surprise is rewarding)
        - Novelty seeking
        - Information gain
        """
        
        curiosity_output = agent_outputs.get('curiosity_output')
        if curiosity_output is None:
            return torch.tensor(0.0)
        
        # Direct curiosity value
        return curiosity_output.curiosity_value.mean()
    
    def _compute_prosocial_reward(
        self,
        agent_outputs: Dict,
        human_state: Optional[Dict]
    ) -> torch.Tensor:
        """
        Prosocial attention reward:
        - Empathic accuracy
        - Helping behavior
        - Social connection
        """
        
        value_output = agent_outputs.get('value_output')
        if value_output is None or human_state is None:
            return torch.tensor(0.0)
        
        # Empathy score
        empathy = value_output.empathy.mean()
        
        # Compassion score
        compassion = value_output.compassion.mean()
        
        # Social reward (if human state tracks this)
        social_reward = human_state.get('trust_delta', 0.0)
        if isinstance(social_reward, (int, float)):
            social_reward = torch.tensor(social_reward)
        
        prosocial = 0.4 * empathy + 0.4 * compassion + 0.2 * social_reward
        
        return prosocial
    
    def _compute_identity_reward(self, agent_outputs: Dict) -> torch.Tensor:
        """
        Identity attention reward:
        - Self-coherence
        - Identity consistency
        - Self-understanding
        """
        
        self_model_output = agent_outputs.get('self_model_output')
        if self_model_output is None:
            return torch.tensor(0.0)
        
        # Coherence score (from self-model)
        coherence = self_model_output.coherence.mean()
        
        return coherence
    
    def _compute_goal_reward(
        self,
        agent_outputs: Dict,
        environment: Dict
    ) -> torch.Tensor:
        """
        Goal attention reward:
        - Task completion
        - External reward
        """
        
        # External task reward
        task_reward = environment.get('reward', 0.0)
        if isinstance(task_reward, (int, float)):
            task_reward = torch.tensor(task_reward)
        
        return task_reward