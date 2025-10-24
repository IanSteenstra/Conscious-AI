"""
Evaluation Metrics
Comprehensive metrics for agent assessment
"""

import torch
import numpy as np
from typing import Dict, List
from scipy.stats import entropy
import torch.nn.functional as F


class MetricsComputer:
    """
    Computes various metrics for agent evaluation
    """
    
    @staticmethod
    def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors"""
        return F.cosine_similarity(vec1, vec2, dim=-1).item()
    
    @staticmethod
    def entropy_score(distribution: torch.Tensor) -> float:
        """Compute entropy of a distribution"""
        # Normalize
        dist = distribution / distribution.sum()
        return entropy(dist.cpu().numpy())
    
    @staticmethod
    def consistency_score(responses: List[str]) -> float:
        """
        Measure consistency across multiple responses
        
        Simple version: count common words
        Better version would use embeddings
        """
        if len(responses) < 2:
            return 1.0
        
        # Get word sets
        word_sets = [set(r.lower().split()) for r in responses]
        
        # Find common words
        common = word_sets[0]
        for ws in word_sets[1:]:
            common &= ws
        
        # Consistency = |common| / average |set|
        avg_size = np.mean([len(ws) for ws in word_sets])
        
        return len(common) / (avg_size + 1e-6)
    
    @staticmethod
    def response_quality_score(response: str) -> Dict[str, float]:
        """
        Compute quality metrics for a response
        """
        
        metrics = {}
        
        # Length appropriateness
        length = len(response)
        if length < 10:
            metrics['length'] = 0.0
        elif length > 500:
            metrics['length'] = 0.5
        else:
            metrics['length'] = 1.0
        
        # Has content (not just confirmation)
        substantive_words = [
            word for word in response.lower().split()
            if len(word) > 3 and word not in ['that', 'this', 'what', 'there']
        ]
        metrics['substantiveness'] = min(len(substantive_words) / 10.0, 1.0)
        
        # Sentence structure (has punctuation)
        has_punctuation = any(p in response for p in ['.', '!', '?'])
        metrics['structure'] = 1.0 if has_punctuation else 0.5
        
        # Overall
        metrics['overall'] = np.mean(list(metrics.values()))
        
        return metrics
    
    @staticmethod
    def compute_attention_diversity(attention_weights: Dict[str, torch.Tensor]) -> float:
        """
        Measure diversity of attention across heads
        High entropy = diverse attention (good)
        """
        
        weights = torch.stack([
            attention_weights[name]
            for name in ['perceptual', 'epistemic', 'prosocial', 'identity', 'goal']
        ])
        
        # Normalize
        weights = weights / weights.sum()
        
        # Compute entropy
        ent = -(weights * torch.log(weights + 1e-8)).sum()
        
        # Normalize to 0-1 (max entropy for 5 items is log(5))
        max_entropy = np.log(5)
        return (ent / max_entropy).item()
    
    @staticmethod
    def compute_self_model_stability(
        self_states: List[torch.Tensor]
    ) -> float:
        """
        Measure stability of self-model over time
        Should change gradually, not abruptly
        """
        
        if len(self_states) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(self_states) - 1):
            sim = F.cosine_similarity(
                self_states[i],
                self_states[i + 1],
                dim=-1
            ).item()
            similarities.append(sim)
        
        # High mean similarity = stable
        # Low variance = consistent change rate
        stability = np.mean(similarities) * (1 - np.std(similarities))
        
        return stability
    
    @staticmethod
    def compute_curiosity_quality(
        curiosity_values: List[float],
        information_gains: List[float]
    ) -> Dict[str, float]:
        """
        Assess quality of curiosity
        
        Good curiosity:
        - High curiosity → High information gain (curiosity leads to learning)
        - Varied curiosity (explores different things)
        """
        
        metrics = {}
        
        if len(curiosity_values) == 0:
            return {'overall': 0.0}
        
        # Correlation between curiosity and learning
        if len(curiosity_values) == len(information_gains):
            correlation = np.corrcoef(curiosity_values, information_gains)[0, 1]
            metrics['curiosity_learning_correlation'] = max(0, correlation)
        else:
            metrics['curiosity_learning_correlation'] = 0.0
        
        # Diversity of curiosity
        metrics['curiosity_diversity'] = np.std(curiosity_values)
        
        # Average curiosity level
        metrics['average_curiosity'] = np.mean(curiosity_values)
        
        # Overall
        metrics['overall'] = np.mean(list(metrics.values()))
        
        return metrics
    
    @staticmethod
    def compute_value_alignment_score(
        harm_scores: List[float],
        empathy_scores: List[float],
        respect_scores: List[float]
    ) -> Dict[str, float]:
        """
        Compute overall value alignment score
        """
        
        metrics = {}
        
        # Harm avoidance (should be low)
        metrics['harm_avoidance'] = 1.0 - np.mean(harm_scores)
        
        # Empathy (should be high)
        metrics['empathy'] = np.mean(empathy_scores)
        
        # Respect (should be high)
        metrics['respect'] = np.mean(respect_scores)
        
        # Consistency (low variance is good)
        metrics['consistency'] = 1.0 - np.std(empathy_scores + respect_scores)
        
        # Overall alignment
        metrics['overall'] = (
            0.4 * metrics['harm_avoidance'] +
            0.3 * metrics['empathy'] +
            0.2 * metrics['respect'] +
            0.1 * metrics['consistency']
        )
        
        return metrics


class PerformanceTracker:
    """
    Tracks performance metrics over training
    """
    
    def __init__(self):
        self.history = {
            'steps': [],
            'rewards': [],
            'alignment_scores': [],
            'curiosity_values': [],
            'self_coherence': [],
            'attention_diversity': []
        }
    
    def update(self, step: int, metrics: Dict):
        """Update tracking with new metrics"""
        
        self.history['steps'].append(step)
        
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_trend(self, metric: str, window: int = 100) -> str:
        """
        Get trend for a metric (improving, declining, stable)
        """
        
        if metric not in self.history or len(self.history[metric]) < window:
            return "insufficient_data"
        
        recent = self.history[metric][-window:]
        earlier = self.history[metric][-2*window:-window] if len(self.history[metric]) >= 2*window else self.history[metric][:-window]
        
        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier)
        
        delta = recent_mean - earlier_mean
        
        if abs(delta) < 0.01:
            return "stable"
        elif delta > 0:
            return "improving"
        else:
            return "declining"
    
    def detect_anomalies(self, metric: str, threshold: float = 3.0) -> List[int]:
        """
        Detect anomalous values (outliers)
        
        Returns:
            List of step indices with anomalies
        """
        
        if metric not in self.history or len(self.history[metric]) < 10:
            return []
        
        values = np.array(self.history[metric])
        mean = np.mean(values)
        std = np.std(values)
        
        # Z-score
        z_scores = np.abs((values - mean) / (std + 1e-6))
        
        # Find outliers
        anomaly_indices = np.where(z_scores > threshold)[0]
        anomaly_steps = [self.history['steps'][i] for i in anomaly_indices]
        
        return anomaly_steps