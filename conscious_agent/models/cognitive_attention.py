"""
Cognitive Multi-Head Attention
Different attention heads for different cognitive modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, NamedTuple


class CognitiveOutput(NamedTuple):
    """Output from cognitive attention"""
    integrated: torch.Tensor
    attention_weights: Dict[str, torch.Tensor]
    mode_outputs: Dict[str, torch.Tensor]


class CognitiveMultiHeadAttention(nn.Module):
    """
    Multi-head attention with different heads for different cognitive modes
    
    Based on neuroscience: Different brain systems attend differently
    """
    
    def __init__(self, dim: int, config: Dict):
        super().__init__()
        
        self.dim = dim
        self.num_heads = config.get('num_heads_per_mode', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # === ATTENTION HEADS (one per cognitive mode) ===
        
        # 1. Perceptual Attention (sensory processing)
        self.perceptual_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 2. Epistemic Attention (curiosity, what's unknown)
        self.epistemic_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 3. Prosocial Attention (human-focused)
        self.prosocial_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 4. Identity Attention (self-relevant)
        self.identity_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 5. Goal Attention (task-relevant)
        self.goal_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # === META-CONTROLLER ===
        # Learns which cognitive mode to use in which context
        
        self.meta_controller = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(dim, 5),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, hidden_states: torch.Tensor, context: Dict) -> CognitiveOutput:
        """
        Apply cognitive attention
        
        Args:
            hidden_states: [batch, seq_len, dim] from pretrained model
            context: Dict with uncertainty, self_model, human_state, etc.
        
        Returns:
            CognitiveOutput with integrated representation and attention weights
        """
        
        batch_size, seq_len, dim = hidden_states.shape
        
        # === APPLY EACH ATTENTION MODE ===
        
        # 1. Perceptual: Standard attention over sensory input
        perceptual_out, perceptual_weights = self.perceptual_attn(
            hidden_states, hidden_states, hidden_states
        )
        perceptual_pooled = perceptual_out.mean(dim=1)  # [batch, dim]
        
        # 2. Epistemic: Attend to uncertain/novel parts
        # Weight by uncertainty
        uncertainty = context.get('uncertainty', torch.ones(batch_size, 1).to(hidden_states.device))
        epistemic_query = hidden_states * uncertainty.unsqueeze(-1)
        epistemic_out, epistemic_weights = self.epistemic_attn(
            epistemic_query, hidden_states, hidden_states
        )
        epistemic_pooled = epistemic_out.mean(dim=1)
        
        # 3. Prosocial: Attend to human-relevant information
        # If we have human state, condition on it
        human_state = context.get('human_state', {})
        if human_state:
            # Could condition prosocial attention on human state
            # For now, standard attention
            prosocial_out, prosocial_weights = self.prosocial_attn(
                hidden_states, hidden_states, hidden_states
            )
        else:
            prosocial_out, prosocial_weights = self.prosocial_attn(
                hidden_states, hidden_states, hidden_states
            )
        prosocial_pooled = prosocial_out.mean(dim=1)
        
        # 4. Identity: Attend to self-relevant information
        # Condition on self-model if available
        self_model = context.get('self_model')
        if self_model is not None and hasattr(self_model, 'representation'):
            # Use self-model as query
            self_query = self_model.representation.unsqueeze(1).expand(-1, seq_len, -1)
            identity_out, identity_weights = self.identity_attn(
                self_query, hidden_states, hidden_states
            )
        else:
            identity_out, identity_weights = self.identity_attn(
                hidden_states, hidden_states, hidden_states
            )
        identity_pooled = identity_out.mean(dim=1)
        
        # 5. Goal: Attend to task-relevant information
        goal_out, goal_weights = self.goal_attn(
            hidden_states, hidden_states, hidden_states
        )
        goal_pooled = goal_out.mean(dim=1)
        
        # === META-CONTROLLER: WEIGHT MODES ===
        
        # Concatenate all mode outputs
        all_modes = torch.cat([
            perceptual_pooled,
            epistemic_pooled,
            prosocial_pooled,
            identity_pooled,
            goal_pooled
        ], dim=-1)  # [batch, dim*5]
        
        # Meta-controller decides weighting
        mode_weights = self.meta_controller(all_modes)  # [batch, 5]
        
        # === INTEGRATE MODES ===
        
        # Weighted combination
        integrated = (
            mode_weights[:, 0:1] * perceptual_pooled +
            mode_weights[:, 1:2] * epistemic_pooled +
            mode_weights[:, 2:3] * prosocial_pooled +
            mode_weights[:, 3:4] * identity_pooled +
            mode_weights[:, 4:5] * goal_pooled
        )
        
        # Project and normalize
        integrated = self.output_proj(integrated)
        integrated = self.layer_norm(integrated)
        
        # === RETURN ===
        
        attention_weights_dict = {
            'perceptual': mode_weights[:, 0],
            'epistemic': mode_weights[:, 1],
            'prosocial': mode_weights[:, 2],
            'identity': mode_weights[:, 3],
            'goal': mode_weights[:, 4]
        }
        
        mode_outputs = {
            'perceptual': perceptual_pooled,
            'epistemic': epistemic_pooled,
            'prosocial': prosocial_pooled,
            'identity': identity_pooled,
            'goal': goal_pooled
        }
        
        return CognitiveOutput(
            integrated=integrated,
            attention_weights=attention_weights_dict,
            mode_outputs=mode_outputs
        )