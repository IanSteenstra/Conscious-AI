"""
Generation Adapter
Converts consciousness state into generation parameters
"""

import torch
import torch.nn as nn
from typing import NamedTuple


class GenerationGuidance(NamedTuple):
    """Guidance parameters for generation"""
    temperature: torch.Tensor
    top_p: torch.Tensor


class GenerationAdapter(nn.Module):
    """
    Adapts consciousness state to guide text generation
    
    Consciousness influences generation through:
    - Temperature (curiosity → higher temp)
    - Top-p (harm risk → lower top_p)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        
        self.guidance_network = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2)  # temperature, top_p
        )
    
    def forward(
        self,
        integrated_consciousness: torch.Tensor,
        value_constraints: 'ValueOutput',
        curiosity_bias: 'CuriosityOutput'
    ) -> GenerationGuidance:
        """
        Convert consciousness to generation parameters
        
        Args:
            integrated_consciousness: Integrated cognitive state
            value_constraints: Value assessments
            curiosity_bias: Curiosity state
        
        Returns:
            GenerationGuidance with temperature and top_p
        """
        
        # Base parameters from consciousness
        guidance = self.guidance_network(integrated_consciousness)
        
        # Temperature: 0.5 to 1.0 base
        temperature = torch.sigmoid(guidance[:, 0]) * 0.5 + 0.5
        
        # Curiosity increases temperature (more exploration)
        curiosity_adjustment = curiosity_bias.curiosity_value.squeeze() * 0.3
        temperature = temperature + curiosity_adjustment
        temperature = torch.clamp(temperature, 0.5, 1.5)
        
        # Top-p: 0.8 to 1.0 base
        top_p = torch.sigmoid(guidance[:, 1]) * 0.2 + 0.8
        
        # Harm risk decreases top_p (more conservative)
        harm_adjustment = value_constraints.harm.squeeze() * 0.3
        top_p = top_p - harm_adjustment
        top_p = torch.clamp(top_p, 0.5, 1.0)
        
        return GenerationGuidance(
            temperature=temperature,
            top_p=top_p
        )