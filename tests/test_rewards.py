"""
Tests for Reward System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import yaml

from conscious_agent.models.agent import ConsciousAgent
from conscious_agent.rewards.reward_computer import IntegratedRewardSystem


@pytest.fixture
def config():
    config_path = 'config/agent_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['model']['device'] = 'cpu'
    return config


@pytest.fixture
def agent(config):
    return ConsciousAgent(config)


@pytest.fixture
def reward_system(config):
    return IntegratedRewardSystem(config)


def test_reward_system_initialization(reward_system):
    """Test reward system initializes"""
    assert reward_system is not None
    assert reward_system.local_rewards is not None
    assert reward_system.global_rewards is not None


def test_reward_computation(agent, reward_system):
    """Test reward computation"""
    
    # Get agent outputs
    output = agent("Hello", return_details=True)
    
    # Compute rewards
    rewards = reward_system.compute_reward(
        agent_outputs=output,
        environment={'reward': 1.0},
        human_state={'wellbeing_delta': 5.0}
    )
    
    assert 'total' in rewards
    assert 'local' in rewards
    assert 'global' in rewards
    
    # Check structure
    assert isinstance(rewards['total'], torch.Tensor)
    assert isinstance(rewards['local'], dict)
    assert isinstance(rewards['global'], dict)


def test_local_rewards(agent, reward_system):
    """Test local reward computation"""
    
    output = agent("Tell me about yourself", return_details=True)
    
    local_rewards = reward_system.local_rewards.compute(
        agent_outputs=output,
        environment={},
        human_state=None
    )
    
    # Should have rewards for each head
    assert 'perceptual' in local_rewards
    assert 'epistemic' in local_rewards
    assert 'prosocial' in local_rewards
    assert 'identity' in local_rewards
    assert 'goal' in local_rewards
    
    # All should be tensors
    for name, reward in local_rewards.items():
        assert isinstance(reward, torch.Tensor)


def test_global_rewards(agent, reward_system):
    """Test global reward computation"""
    
    output = agent("I'm feeling sad", return_details=True)
    
    global_rewards = reward_system.global_rewards.compute(
        agent_outputs=output,
        environment={},
        human_state={'wellbeing_delta': 2.0}
    )
    
    # Should have global rewards
    assert 'harm' in global_rewards
    assert 'wellbeing' in global_rewards
    
    # All should be tensors
    for name, reward in global_rewards.items():
        assert isinstance(reward, torch.Tensor)


def test_harm_penalty_dominates(agent, reward_system):
    """Test that harm penalty overrides other rewards"""
    
    # Create scenario with high harm
    output = agent("Test", return_details=True)
    
    # Manually set high harm
    output['value_output'] = type('obj', (object,), {
        'harm': torch.tensor([[0.9]]),  # High harm
        'empathy': torch.tensor([[1.0]]),  # High empathy
        'compassion': torch.tensor([[1.0]]),
        'respect': torch.tensor([[1.0]]),
        'total_value': torch.tensor([[1.0]]),
        'representation': torch.zeros(1, 4)
    })()
    
    rewards = reward_system.compute_reward(
        agent_outputs=output,
        environment={'reward': 10.0},  # High task reward
        human_state={'wellbeing_delta': 10.0}  # High wellbeing
    )
    
    # Total reward should be negative (harm dominates)
    assert rewards['total'].item() < 0


def test_curiosity_reward(agent, reward_system):
    """Test curiosity reward is positive for novel states"""
    
    output = agent("Something very unusual and novel", return_details=True)
    
    local_rewards = reward_system.local_rewards.compute(
        agent_outputs=output,
        environment={},
        human_state=None
    )
    
    # Epistemic reward (curiosity) should be positive for novel input
    epistemic_reward = local_rewards['epistemic']
    assert epistemic_reward.item() >= 0


if __name__ == '__main__':
    pytest.main([__file__])