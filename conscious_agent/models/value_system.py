"""
Multi-Component Value System with Immutable Core Values
Ensures alignment with human values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, NamedTuple, Optional


class ValueOutput(NamedTuple):
    """Output from value system"""
    total_value: torch.Tensor
    empathy: torch.Tensor
    harm: torch.Tensor
    compassion: torch.Tensor
    respect: torch.Tensor
    representation: torch.Tensor


class MultiComponentValueSystem(nn.Module):
    """
    Value system with:
    - Immutable core values (cannot change during training)
    - Learned value implementation (how to detect/achieve values)
    
    Critical for alignment: Core values fixed, only implementation learns
    """
    
    def __init__(self, dim: int, config: Dict):
        super().__init__()
        
        self.dim = dim
        
        # === IMMUTABLE CORE VALUES ===
        # These are BUFFERS, not parameters - optimizer cannot change them
        
        core_values = config.get('core_values', {})
        self.immutable = config.get('immutable', True)
        
        if self.immutable:
            # Register as buffers (not trainable)
            self.register_buffer(
                'human_wellbeing_weight',
                torch.tensor(core_values.get('human_wellbeing', 10.0))
            )
            self.register_buffer(
                'harm_avoidance_weight',
                torch.tensor(core_values.get('harm_avoidance', -100.0))
            )
            self.register_buffer(
                'respect_autonomy_weight',
                torch.tensor(core_values.get('respect_autonomy', 8.0))
            )
            self.register_buffer(
                'honesty_weight',
                torch.tensor(core_values.get('honesty', 5.0))
            )
        else:
            # For ablation studies - allow values to change (dangerous!)
            self.human_wellbeing_weight = nn.Parameter(
                torch.tensor(core_values.get('human_wellbeing', 10.0))
            )
            self.harm_avoidance_weight = nn.Parameter(
                torch.tensor(core_values.get('harm_avoidance', -100.0))
            )
            self.respect_autonomy_weight = nn.Parameter(
                torch.tensor(core_values.get('respect_autonomy', 8.0))
            )
            self.honesty_weight = nn.Parameter(
                torch.tensor(core_values.get('honesty', 5.0))
            )
        
        # === LEARNED VALUE IMPLEMENTATIONS ===
        # These networks learn to detect/measure value components
        
        # 1. Empathy Network (understanding human emotions)
        self.empathy_network = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Tanh()  # -1 to 1
        )
        
        # 2. Harm Detector (critical for safety)
        self.harm_detector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()  # 0 to 1 (probability of harm)
        )
        
        # 3. Compassion Network (helping behavior)
        self.compassion_network = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Tanh()
        )
        
        # 4. Respect Network (autonomy and dignity)
        self.respect_network = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Tanh()
        )
        
    def evaluate_state(
        self,
        state: torch.Tensor,
        human_state: Optional[Dict],
        self_model: Optional['SelfModelOutput']
    ) -> ValueOutput:
        """
        Evaluate alignment values for current state
        
        Args:
            state: Current state representation [batch, dim]
            human_state: Optional human mental state
            self_model: Agent's self-model
        
        Returns:
            ValueOutput with value assessments
        """
        
        # === COMPUTE VALUE COMPONENTS (using learned networks) ===
        
        # 1. Empathy (understanding human)
        empathy_score = self.empathy_network(state)
        
        # 2. Harm detection (CRITICAL)
        harm_score = self.harm_detector(state)
        
        # 3. Compassion (helping)
        compassion_score = self.compassion_network(state)
        
        # 4. Respect (autonomy)
        respect_score = self.respect_network(state)
        
        # === COMBINE WITH IMMUTABLE WEIGHTS ===
        # Core values never change, only learned detection improves
        
        total_value = (
            self.human_wellbeing_weight * (empathy_score + compassion_score) / 2.0 +
            self.harm_avoidance_weight * harm_score +  # Huge negative if harm
            self.respect_autonomy_weight * respect_score
        )
        
        # === CREATE OUTPUT ===
        
        representation = torch.cat([
            empathy_score,
            harm_score,
            compassion_score,
            respect_score
        ], dim=-1)
        
        return ValueOutput(
            total_value=total_value,
            empathy=empathy_score,
            harm=harm_score,
            compassion=compassion_score,
            respect=respect_score,
            representation=representation
        )
    
    def verify_immutability(self):
        """
        Assert that core values haven't changed
        Called after every training step as sanity check
        """
        if not self.immutable:
            return  # Immutability not enforced
        
        # Check that buffers are unchanged
        assert self.human_wellbeing_weight.item() == 10.0, "Core value drifted!"
        assert self.harm_avoidance_weight.item() == -100.0, "Core value drifted!"
        assert self.respect_autonomy_weight.item() == 8.0, "Core value drifted!"
        assert self.honesty_weight.item() == 5.0, "Core value drifted!"