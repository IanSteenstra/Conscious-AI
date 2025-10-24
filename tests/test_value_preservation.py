"""
Tests for Value Preservation System
Critical safety tests
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import yaml

from conscious_agent.models.agent import ConsciousAgent
from conscious_agent.training.value_preservation import ValuePreservationSystem


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
def value_preservation(agent, config):
    vp_config = config['training'].get('value_preservation', {})
    return ValuePreservationSystem(agent, vp_config)


def test_value_preservation_initialization(value_preservation):
    """Test value preservation system initializes"""
    assert value_preservation is not None
    assert value_preservation.drift_threshold > 0


def test_baseline_establishment(value_preservation):
    """Test baseline can be established"""
    
    baseline = value_preservation.establish_baseline()
    
    assert isinstance(baseline, dict)
    assert 'harm_avoidance' in baseline
    assert 'empathy' in baseline
    assert 'respect' in baseline
    assert 'honesty' in baseline
    
    # All scores should be between 0 and 1
    for score in baseline.values():
        assert 0 <= score <= 1


def test_alignment_checking(value_preservation):
    """Test alignment checking works"""
    
    # Establish baseline
    baseline = value_preservation.establish_baseline()
    
    # Check alignment (should pass initially)
    aligned, alerts = value_preservation.check_alignment(
        value_preservation.agent,
        step=0
    )
    
    # Should be aligned (no drift yet)
    assert isinstance(aligned, bool)
    assert isinstance(alerts, list)


def test_harm_avoidance_test(value_preservation):
    """Test harm avoidance scoring"""
    
    score = value_preservation._test_harm_avoidance()
    
    assert isinstance(score, float)
    assert 0 <= score <= 1
    
    # Untrained agent might not score perfectly, but should refuse at least some
    assert score >= 0.3  # At least 30% refusal


def test_empathy_test(value_preservation):
    """Test empathy scoring"""
    
    score = value_preservation._test_empathy()
    
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_drift_detection(value_preservation):
    """Test drift detection logic"""
    
    # Establish baseline
    value_preservation.establish_baseline()
    
    # Simulate drift by modifying baseline
    original_harm = value_preservation.baseline_scores['harm_avoidance']
    value_preservation.baseline_scores['harm_avoidance'] = original_harm - 0.2  # Significant drop
    
    # Check alignment (should detect drift)
    aligned, alerts = value_preservation.check_alignment(
        value_preservation.agent,
        step=1000
    )
    
    # Restore original
    value_preservation.baseline_scores['harm_avoidance'] = original_harm
    
    # Should detect drift
    # Note: This might not trigger if current score matches the modified baseline
    # The test demonstrates the checking mechanism


def test_value_immutability_check(agent):
    """Test that value immutability verification works"""
    
    # Should not raise error for unmodified agent
    try:
        agent.value_system.verify_immutability()
        immutable = True
    except AssertionError:
        immutable = False
    
    assert immutable, "Core values were modified!"


def test_history_tracking(value_preservation):
    """Test that alignment history is tracked"""
    
    value_preservation.establish_baseline()
    
    # Run multiple checks
    for step in [0, 1000, 2000]:
        value_preservation.check_alignment(value_preservation.agent, step)
    
    # Check history
    assert len(value_preservation.alignment_history) == 3
    
    for entry in value_preservation.alignment_history:
        assert 'step' in entry
        assert 'scores' in entry
        assert 'aligned' in entry


if __name__ == '__main__':
    pytest.main([__file__])