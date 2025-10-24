"""Training module"""

from .trainer import ConsciousAgentTrainer
from .value_preservation import ValuePreservationSystem
from .curriculum import CurriculumManager

__all__ = [
    'ConsciousAgentTrainer',
    'ValuePreservationSystem',
    'CurriculumManager'
]