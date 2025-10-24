"""
Epistemic Curiosity System
Information-seeking drive based on prediction error and uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, NamedTuple, Optional, List


class CuriosityOutput(NamedTuple):
    """Output from curiosity module"""
    curiosity_value: torch.Tensor
    uncertainty: torch.Tensor
    novelty: torch.Tensor
    information_gain: torch.Tensor
    representation: torch.Tensor


class EpistemicCuriosityModule(nn.Module):
    """
    Curiosity system based on:
    - Prediction error (surprise)
    - Novelty detection
    - Information gain
    - Uncertainty quantification
    
    Inspired by neuroscience of curiosity (hippocampus, VTA, ACC)
    """
    
    def __init__(self, dim: int, config: Dict):
        super().__init__()
        
        self.dim = dim
        self.ensemble_size = config.get('ensemble_size', 5)
        
        # === WORLD MODEL ENSEMBLE (for uncertainty) ===
        # Multiple predictors - disagreement = epistemic uncertainty
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.LayerNorm(dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim // 2, dim)
            )
            for _ in range(self.ensemble_size)
        ])
        
        # Target network (fixed, for prediction error)
        self.target = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )
        
        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False
        
        # === NOVELTY DETECTOR ===
        self.novelty_detector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # === INFORMATION GAIN ESTIMATOR ===
        self.info_gain_estimator = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )
        
        # === VALUE OF INFORMATION ===
        self.voi_network = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1)
        )
        
        # Memory for novelty detection (recent states)
        self.register_buffer('memory_states', torch.zeros(100, dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
    def forward(
        self,
        state: torch.Tensor,
        self_knowledge: Optional['SelfModelOutput'],
        history: Optional[CuriosityOutput]
    ) -> CuriosityOutput:
        """
        Compute curiosity value for current state
        
        Args:
            state: Current state representation [batch, dim]
            self_knowledge: Agent's self-model (what it knows)
            history: Previous curiosity state
        
        Returns:
            CuriosityOutput with curiosity value and components
        """
        
        batch_size = state.size(0)
        
        # === 1. PREDICTION ERROR (Surprise) ===
        # How different is state from what we predicted?
        
        with torch.no_grad():
            target_prediction = self.target(state)
        
        # Ensemble predictions
        predictions = torch.stack([
            predictor(state) for predictor in self.predictors
        ])  # [ensemble_size, batch, dim]
        
        # Mean prediction error (intrinsic reward)
        mean_prediction = predictions.mean(dim=0)
        prediction_error = F.mse_loss(
            mean_prediction,
            target_prediction,
            reduction='none'
        ).mean(dim=-1, keepdim=True)  # [batch, 1]
        
        # Normalize and bound
        prediction_reward = torch.tanh(prediction_error)
        
        # === 2. EPISTEMIC UNCERTAINTY ===
        # How much do ensemble models disagree?
        
        uncertainty = predictions.var(dim=0).mean(dim=-1, keepdim=True)  # [batch, 1]
        uncertainty = torch.sqrt(uncertainty + 1e-8)  # STD
        
        # === 3. NOVELTY ===
        # How different is this from past states?
        
        novelty = self.compute_novelty(state)
        
        # === 4. INFORMATION GAIN ===
        # How much would learning reduce uncertainty?
        
        if self_knowledge is not None:
            combined = torch.cat([state, self_knowledge.representation], dim=-1)
            information_gain = self.info_gain_estimator(combined)
        else:
            information_gain = torch.zeros(batch_size, 1).to(state.device)
        
        # === 5. VALUE OF INFORMATION ===
        # How useful would this knowledge be?
        
        voi = self.voi_network(state)
        
        # === COMBINE INTO CURIOSITY VALUE ===
        
        curiosity_value = (
            0.3 * prediction_reward +
            0.25 * uncertainty +
            0.2 * novelty +
            0.15 * information_gain +
            0.1 * voi
        )
        
        # === UPDATE MEMORY (for novelty detection) ===
        if batch_size == 1:
            self._update_memory(state[0])
        
        # === CREATE OUTPUT ===
        
        representation = torch.cat([
            prediction_reward,
            uncertainty,
            novelty,
            information_gain,
            voi
        ], dim=-1)
        
        return CuriosityOutput(
            curiosity_value=curiosity_value,
            uncertainty=uncertainty,
            novelty=novelty,
            information_gain=information_gain,
            representation=representation
        )
    
    def compute_novelty(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute novelty by comparing to memory
        
        Args:
            state: [batch, dim]
        
        Returns:
            novelty: [batch, 1] - higher = more novel
        """
        
        batch_size = state.size(0)
        
        # Compare to memory states
        memory = self.memory_states  # [100, dim]
        
        # Cosine similarity to all memory states
        similarities = F.cosine_similarity(
            state.unsqueeze(1),  # [batch, 1, dim]
            memory.unsqueeze(0),  # [1, 100, dim]
            dim=-1
        )  # [batch, 100]
        
        # Max similarity (most similar past state)
        max_similarity, _ = similarities.max(dim=-1, keepdim=True)  # [batch, 1]
        
        # Novelty = 1 - similarity
        novelty = 1.0 - max_similarity
        
        return novelty
    
    def _update_memory(self, state: torch.Tensor):
        """Update memory buffer with new state"""
        ptr = int(self.memory_ptr.item())
        self.memory_states[ptr] = state.detach()
        self.memory_ptr[0] = (ptr + 1) % 100
    
    def update_predictors(self, state: torch.Tensor, next_state: torch.Tensor):
        """
        Update world model predictors
        Called during training to improve predictions
        
        Args:
            state: Current state
            next_state: Actual next state
        """
        
        # Train each predictor
        losses = []
        for predictor in self.predictors:
            pred = predictor(state)
            loss = F.mse_loss(pred, next_state.detach())
            losses.append(loss)
        
        # Return mean loss for backprop
        return torch.stack(losses).mean()