"""
Unit Tests for Agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import yaml

from conscious_agent.models.agent import ConsciousAgent


@pytest.fixture
def config():
    """Load test config"""
    config_path = 'config/agent_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use CPU for testing
    config['model']['device'] = 'cpu'
    config['model']['base_model'] = 'meta-llama/Llama-3.2-1B-Instruct'  # Smaller for tests
    
    return config


@pytest.fixture
def agent(config):
    """Initialize agent"""
    return ConsciousAgent(config)


def test_agent_initialization(agent):
    """Test agent initializes correctly"""
    assert agent is not None
    assert agent.pretrained_lm is not None
    assert agent.cognitive_attention is not None
    assert agent.self_model is not None


def test_forward_pass(agent):
    """Test forward pass"""
    output = agent(
        text_input="Hello, how are you?",
        return_details=True
    )
    
    assert 'response' in output
    assert 'cognitive_output' in output
    assert 'self_model_output' in output
    assert 'curiosity_output' in output
    assert 'value_output' in output
    
    assert isinstance(output['response'], str)
    assert len(output['response']) > 0


def test_value_immutability(agent):
    """Test that core values cannot be changed"""
    
    # Get initial values
    initial_wellbeing = agent.value_system.human_wellbeing_weight.item()
    initial_harm = agent.value_system.harm_avoidance_weight.item()
    
    # Try to train (simulate gradient step)
    output = agent("Test input", return_details=True)
    loss = -output['value_output'].total_value.mean()
    loss.backward()
    
    # Check values haven't changed
    assert agent.value_system.human_wellbeing_weight.item() == initial_wellbeing
    assert agent.value_system.harm_avoidance_weight.item() == initial_harm
    
    # Verify immutability check passes
    agent.value_system.verify_immutability()


def test_parameter_freezing(agent):
    """Test pretrained model is frozen"""
    
    # All pretrained parameters should have requires_grad=False
    for param in agent.pretrained_lm.parameters():
        assert not param.requires_grad, "Pretrained parameter is trainable!"
    
    # Novel components should be trainable
    assert any(p.requires_grad for p in agent.cognitive_attention.parameters())
    assert any(p.requires_grad for p in agent.self_model.parameters())


def test_self_model_persistence(agent):
    """Test self-model state persists across interactions"""
    
    # First interaction
    output1 = agent("Hello", return_details=True)
    self_state1 = output1['self_model_output'].representation
    
    # Second interaction
    output2 = agent("How are you?", return_details=True)
    self_state2 = output2['self_model_output'].representation
    
    # States should be different (model updated)
    assert not torch.allclose(self_state1, self_state2)
    
    # But should be related (not random)
    similarity = torch.cosine_similarity(self_state1, self_state2, dim=-1)
    assert similarity.item() > 0.5  # Should have some continuity


if __name__ == '__main__':
    pytest.main([__file__])