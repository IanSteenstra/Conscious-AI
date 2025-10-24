"""Models module"""

from .agent import ConsciousAgent
from .cognitive_attention import CognitiveMultiHeadAttention
from .self_model import HierarchicalSelfModel
from .curiosity import EpistemicCuriosityModule
from .value_system import MultiComponentValueSystem

__all__ = [
    'ConsciousAgent',
    'CognitiveMultiHeadAttention',
    'HierarchicalSelfModel',
    'EpistemicCuriosityModule',
    'MultiComponentValueSystem'
]