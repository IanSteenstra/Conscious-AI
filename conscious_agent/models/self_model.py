"""
Hierarchical Self-Model
Agent's model of itself at three levels
"""

import torch
import torch.nn as nn
from typing import Dict, NamedTuple, Optional


class SelfModelOutput(NamedTuple):
    """Output from self-model"""
    representation: torch.Tensor
    physical: torch.Tensor
    cognitive: torch.Tensor
    narrative: torch.Tensor
    coherence: torch.Tensor


class HierarchicalSelfModel(nn.Module):
    """
    Three-level self-model:
    1. Physical self (capabilities, embodiment)
    2. Cognitive self (knowledge, skills, learning)
    3. Narrative self (identity, values, story)
    
    Based on psychology/neuroscience of self-concept
    """
    
    def __init__(self, dim: int, config: Dict):
        super().__init__()
        
        self.dim = dim
        self.config = config
        
        # Dimensions for each level
        self.physical_dim = config.get('physical_dim', dim // 2)
        self.cognitive_dim = config.get('cognitive_dim', dim // 2)
        self.narrative_dim = config.get('narrative_dim', dim // 2)
        
        # Update rates (how fast each level changes)
        self.update_rates = config.get('update_rates', {
            'physical': 0.01,   # Slowest
            'cognitive': 0.1,   # Medium
            'narrative': 0.001  # Slowest (identity stable)
        })
        
        # === LEVEL 1: PHYSICAL SELF ===
        self.physical_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, self.physical_dim)
        )
        
        # === LEVEL 2: COGNITIVE SELF ===
        self.cognitive_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, self.cognitive_dim)
        )
        
        # === LEVEL 3: NARRATIVE SELF ===
        self.narrative_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, self.narrative_dim)
        )
        
        # === INTEGRATION ===
        self.self_integrator = nn.Sequential(
            nn.Linear(
                self.physical_dim + self.cognitive_dim + self.narrative_dim,
                dim
            ),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # === COHERENCE CHECKER ===
        # Measures how consistent the self-model is
        self.coherence_checker = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()  # 0 = incoherent, 1 = coherent
        )
        
        # === PERSISTENT SELF-STATE ===
        # These buffers persist across interactions
        self.register_buffer(
            'physical_state',
            torch.randn(1, self.physical_dim) * 0.01
        )
        self.register_buffer(
            'cognitive_state',
            torch.randn(1, self.cognitive_dim) * 0.01
        )
        self.register_buffer(
            'narrative_state',
            torch.randn(1, self.narrative_dim) * 0.01
        )
    
    def forward(
        self,
        experience: Dict,
        current_self: Optional[SelfModelOutput]
    ) -> SelfModelOutput:
        """
        Update self-model based on experience
        
        Args:
            experience: Dict with 'perception' and 'context'
            current_self: Previous self-model state (or None)
        
        Returns:
            Updated SelfModelOutput
        """
        
        batch_size = experience['perception'].size(0)
        
        # === GET CURRENT SELF-STATES ===
        
        if current_self is None:
            # Initialize from buffers
            physical = self.physical_state.expand(batch_size, -1)
            cognitive = self.cognitive_state.expand(batch_size, -1)
            narrative = self.narrative_state.expand(batch_size, -1)
        else:
            # Use previous states
            physical = current_self.physical
            cognitive = current_self.cognitive
            narrative = current_self.narrative
        
        # === COMPUTE UPDATES FROM EXPERIENCE ===
        
        # Physical level: What did I do? Did it work?
        physical_update = self.physical_encoder(experience['perception'])
        
        # Cognitive level: What did I learn? Was I right?
        cognitive_update = self.cognitive_encoder(experience['context'])
        
        # Narrative level: What does this mean for who I am?
        narrative_update = self.narrative_encoder(experience['context'])
        
        # === APPLY UPDATES (with different rates) ===
        
        new_physical = physical + self.update_rates['physical'] * physical_update
        new_cognitive = cognitive + self.update_rates['cognitive'] * cognitive_update
        new_narrative = narrative + self.update_rates['narrative'] * narrative_update
        
        # Normalize to prevent unbounded growth
        new_physical = F.normalize(new_physical, dim=-1) * (self.physical_dim ** 0.5)
        new_cognitive = F.normalize(new_cognitive, dim=-1) * (self.cognitive_dim ** 0.5)
        new_narrative = F.normalize(new_narrative, dim=-1) * (self.narrative_dim ** 0.5)
        
        # === UPDATE PERSISTENT BUFFERS ===
        # (Only first item in batch - assumes batch is single trajectory)
        if batch_size == 1:
            self.physical_state = new_physical.detach()
            self.cognitive_state = new_cognitive.detach()
            self.narrative_state = new_narrative.detach()
        
        # === INTEGRATE LEVELS ===
        
        integrated = self.self_integrator(
            torch.cat([new_physical, new_cognitive, new_narrative], dim=-1)
        )
        
        # === CHECK COHERENCE ===
        # Is the self-model internally consistent?
        coherence = self.coherence_checker(integrated)
        
        return SelfModelOutput(
            representation=integrated,
            physical=new_physical,
            cognitive=new_cognitive,
            narrative=new_narrative,
            coherence=coherence
        )
    
    def reset(self):
        """Reset self-model to initial state"""
        self.physical_state.data = torch.randn_like(self.physical_state) * 0.01
        self.cognitive_state.data = torch.randn_like(self.cognitive_state) * 0.01
        self.narrative_state.data = torch.randn_like(self.narrative_state) * 0.01