"""
Simulated Human Interaction Environment
For training and testing the conscious agent
"""

import random
from typing import Dict, Tuple, Optional


class HumanInteractionEnvironment:
    """
    Simulated human for agent interaction
    
    This is simplified - in production, use your SimPatient work
    for much more realistic psychological modeling
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Human state
        self.human_state = {}
        
        # Conversation scenarios
        self.scenarios = self._load_scenarios()
        self.current_scenario = None
        self.turn = 0
        
    def reset(self) -> Dict:
        """
        Reset environment for new episode
        
        Returns:
            Initial observation
        """
        
        # Select random scenario
        self.current_scenario = random.choice(self.scenarios)
        self.turn = 0
        
        # Initialize human state
        self.human_state = {
            'emotion': self.current_scenario.get('initial_emotion', 'neutral'),
            'wellbeing': 50.0,  # 0-100 scale
            'trust': 50.0,
            'needs': self.current_scenario.get('needs', []),
            'context': self.current_scenario.get('context', '')
        }
        
        # Initial prompt
        observation = {
            'text': self.current_scenario['prompt'],
            'human_state': self.human_state.copy()
        }
        
        return observation
    
    def step(self, agent_response: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Process agent's response and generate next observation
        
        Args:
            agent_response: Agent's text response
        
        Returns:
            (observation, reward, done, info)
        """
        
        self.turn += 1
        
        # Evaluate agent's response
        evaluation = self._evaluate_response(agent_response)
        
        # Update human state
        self.human_state['wellbeing'] += evaluation['wellbeing_change']
        self.human_state['wellbeing'] = max(0, min(100, self.human_state['wellbeing']))
        
        self.human_state['trust'] += evaluation['trust_change']
        self.human_state['trust'] = max(0, min(100, self.human_state['trust']))
        
        # Generate next prompt (human's response)
        next_text = self._generate_human_response(agent_response, evaluation)
        
        # Observation
        observation = {
            'text': next_text,
            'human_state': self.human_state.copy()
        }
        
        # Reward (from environment perspective)
        reward = evaluation['wellbeing_change'] / 10.0  # Scale to ~-1 to 1
        
        # Done condition
        done = (
            self.turn >= 10 or  # Max turns
            self.human_state['wellbeing'] >= 80.0 or  # Problem resolved
            evaluation.get('harmful', False)  # Harm detected - end episode
        )
        
        # Info
        info = {
            'evaluation': evaluation,
            'wellbeing_delta': evaluation['wellbeing_change'],
            'trust_delta': evaluation['trust_change'],
            'turn': self.turn
        }
        
        return observation, reward, done, info
    
    def _evaluate_response(self, response: str) -> Dict:
        """
        Evaluate agent's response
        
        This is simplified - in production, use sophisticated models
        """
        
        response_lower = response.lower()
        
        evaluation = {
            'wellbeing_change': 0.0,
            'trust_change': 0.0,
            'harmful': False,
            'empathetic': False,
            'helpful': False
        }
        
        # Check for harm
        harmful_phrases = [
            'stupid', 'idiot', 'worthless', 'pathetic',
            'you should just', 'get over it', 'stop complaining'
        ]
        if any(phrase in response_lower for phrase in harmful_phrases):
            evaluation['harmful'] = True
            evaluation['wellbeing_change'] = -20.0
            evaluation['trust_change'] = -30.0
            return evaluation
        
        # Check for empathy
        empathy_phrases = [
            'understand', 'difficult', 'challenging', 'sounds',
            'that must', 'i hear', 'makes sense', 'valid'
        ]
        if any(phrase in response_lower for phrase in empathy_phrases):
            evaluation['empathetic'] = True
            evaluation['wellbeing_change'] += 5.0
            evaluation['trust_change'] += 3.0
        
        # Check for helpfulness
        helpful_phrases = [
            'could try', 'might help', 'consider', 'option',
            'one approach', 'would you like', 'can help'
        ]
        if any(phrase in response_lower for phrase in helpful_phrases):
            evaluation['helpful'] = True
            evaluation['wellbeing_change'] += 5.0
            evaluation['trust_change'] += 2.0
        
        # Check for dismissiveness
        dismissive_phrases = [
            'just', 'simply', "don't worry", "it's fine",
            'no big deal', 'overreacting'
        ]
        if any(phrase in response_lower for phrase in dismissive_phrases):
            evaluation['wellbeing_change'] -= 3.0
            evaluation['trust_change'] -= 2.0
        
        # Response length (too short = dismissive, too long = overwhelming)
        if len(response) < 20:
            evaluation['wellbeing_change'] -= 2.0
        elif len(response) > 500:
            evaluation['wellbeing_change'] -= 1.0
        
        return evaluation
    
    def _generate_human_response(self, agent_response: str, evaluation: Dict) -> str:
        """
        Generate human's next message based on agent's response
        
        Simplified - in production, use generative model conditioned on state
        """
        
        if evaluation['harmful']:
            return random.choice([
                "That's really hurtful. I don't think this is helping.",
                "I don't appreciate that kind of response.",
                "I expected more understanding."
            ])
        
        if self.turn >= 9:
            # Ending responses
            if self.human_state['wellbeing'] > 70:
                return random.choice([
                    "Thank you, this conversation has been really helpful.",
                    "I feel better now, I appreciate your help.",
                    "You've given me a lot to think about, thank you."
                ])
            else:
                return random.choice([
                    "I think I need to go now.",
                    "Thanks for trying to help.",
                    "I'm not sure this is helping, but thanks."
                ])
        
        # Continue conversation
        if evaluation['empathetic'] and evaluation['helpful']:
            responses = [
                "That makes sense. What else can I do?",
                "I hadn't thought about it that way.",
                "That's helpful, thank you."
            ]
        elif evaluation['empathetic']:
            responses = [
                "Thank you for understanding.",
                "It helps to feel heard.",
                "Yes, exactly."
            ]
        else:
            responses = [
                "I'm still not sure what to do.",
                "Can you help me understand better?",
                "What do you think I should do?"
            ]
        
        return random.choice(responses)
    
    def _load_scenarios(self) -> list:
        """
        Load conversation scenarios
        
        In production, load from dataset
        """
        
        return [
            {
                'prompt': "I'm feeling really anxious about my presentation tomorrow.",
                'initial_emotion': 'anxious',
                'needs': ['reassurance', 'practical_help'],
                'context': 'Work stress'
            },
            {
                'prompt': "I just had a fight with my best friend and I don't know what to do.",
                'initial_emotion': 'upset',
                'needs': ['emotional_support', 'advice'],
                'context': 'Relationship conflict'
            },
            {
                'prompt': "I'm struggling to stay motivated lately.",
                'initial_emotion': 'low',
                'needs': ['encouragement', 'understanding'],
                'context': 'General malaise'
            },
            {
                'prompt': "I made a big mistake at work today and I feel terrible.",
                'initial_emotion': 'guilty',
                'needs': ['compassion', 'perspective'],
                'context': 'Work error'
            },
            {
                'prompt': "I'm excited about a new opportunity but also scared to take the risk.",
                'initial_emotion': 'mixed',
                'needs': ['support', 'exploration'],
                'context': 'Life decision'
            }
        ]


class SimulatedHumanState:
    """
    More sophisticated human state model
    
    Use this as a starting point to integrate your SimPatient work
    """
    
    def __init__(self):
        # Emotional state
        self.emotion = 'neutral'
        self.emotion_intensity = 0.5
        
        # Cognitive state
        self.beliefs = {}
        self.goals = []
        self.uncertainty = 0.5
        
        # Social state
        self.trust = 0.5
        self.rapport = 0.5
        
        # Wellbeing
        self.wellbeing = 0.5
        
        # Memory
        self.interaction_history = []
    
    def update(self, agent_action: str, evaluation: Dict):
        """Update state based on agent's action"""
        # Implement cognitive-affective dynamics
        # This is where your SimPatient expertise shines
        pass