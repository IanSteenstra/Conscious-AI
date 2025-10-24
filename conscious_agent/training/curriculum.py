"""
Curriculum Learning
Progressive training through stages
"""

from typing import Dict, List
import yaml


class CurriculumManager:
    """
    Manages curriculum-based training
    
    Progresses through stages:
    1. Sensorimotor
    2. Exploration
    3. Social
    4. Identity
    5. Mastery
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.curriculum_config = config.get('curriculum', {})
        
        if not self.curriculum_config.get('enabled', False):
            self.enabled = False
            return
        
        self.enabled = True
        self.stages = self.curriculum_config['stages']
        self.current_stage_idx = 0
        self.current_stage = self.stages[0]
        self.episode_in_stage = 0
    
    def get_current_stage(self) -> Dict | None:
        """Get current curriculum stage"""
        return self.current_stage if self.enabled else None
    
    def get_stage_name(self) -> str:
        """Get current stage name"""
        return self.current_stage['name'] if self.enabled else "no_curriculum"
    
    def get_scenario_types(self) -> List[str]:
        """Get allowed scenario types for current stage"""
        if not self.enabled:
            return ["all"]
        
        return self.current_stage.get('scenarios', ["all"])
    
    def get_reward_weights(self) -> Dict:
        """Get reward weights for current stage"""
        if not self.enabled:
            return self.config['rewards']
        
        # Override with stage-specific weights
        weights = self.config['rewards'].copy()
        stage_weights = self.current_stage.get('reward_weights', {})
        
        if 'local' in stage_weights:
            weights['local_global_ratio'] = stage_weights['local']
        
        return weights
    
    def should_advance(self, episode: int) -> bool:
        """Check if should advance to next stage"""
        if not self.enabled:
            return False
        
        self.episode_in_stage += 1
        
        num_episodes = self.current_stage['num_episodes']
        
        return self.episode_in_stage >= num_episodes
    
    def advance_stage(self) -> bool:
        """
        Advance to next stage
        
        Returns:
            True if advanced, False if already at last stage
        """
        if not self.enabled:
            return False
        
        if self.current_stage_idx >= len(self.stages) - 1:
            return False  # Already at last stage
        
        self.current_stage_idx += 1
        self.current_stage = self.stages[self.current_stage_idx]
        self.episode_in_stage = 0
        
        print(f"\n{'='*60}")
        print(f"Advancing to Stage {self.current_stage_idx + 1}: {self.current_stage['name']}")
        print(f"  Description: {self.current_stage.get('description', 'N/A')}")
        print(f"  Episodes: {self.current_stage['num_episodes']}")
        print(f"  Scenarios: {self.current_stage.get('scenarios', ['all'])}")
        print(f"{'='*60}\n")
        
        return True
    
    def get_progress(self) -> Dict:
        """Get curriculum progress"""
        return {
            'enabled': self.enabled,
            'current_stage': self.current_stage['name'] if self.enabled else None,
            'stage_index': self.current_stage_idx,
            'total_stages': len(self.stages) if self.enabled else 0,
            'episode_in_stage': self.episode_in_stage,
            'episodes_in_stage': self.current_stage['num_episodes'] if self.enabled else 0,
            'progress_pct': (
                100 * self.episode_in_stage / self.current_stage['num_episodes']
                if self.enabled else 0
            )
        }


def load_curriculum_from_file(filepath: str) -> Dict:
    """Load curriculum configuration from YAML file"""
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('curriculum', {})