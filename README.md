# Conscious AI: Modeling Curiosity, Identity, and Human Alignment

A research implementation of a conscious AI agent with:
- Multi-head cognitive attention (different modes of thinking)
- Hierarchical self-model (physical, cognitive, narrative)
- Epistemic curiosity system
- Multi-component value system with immutable core values
- Value drift protection

## Installation
```bash
# Clone repository
git clone https://github.com/iansteenstra/conscious-ai.git
cd conscious-agent

# Create environment
conda create -n conscious-agent python=3.10
conda activate conscious-agent

# Install dependencies
pip install -e .
```

## Quick Start

### Training
```bash
python scripts/train.py --config config/agent_config.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint checkpoints/checkpoint_final.pt
```

### Interactive Demo
```bash
python scripts/demo.py --checkpoint checkpoints/checkpoint_final.pt
```

## Architecture

### Core Components

1. **Frozen Pretrained Model**: LLaMA-3.2-3B for language capabilities
2. **Cognitive Attention**: 5 attention heads for different thinking modes
3. **Self-Model**: 3-level hierarchical representation of self
4. **Curiosity Module**: Epistemic drive for information seeking
5. **Value System**: Immutable core values + learned implementations

### Novel Contributions

- **Cognitive Multi-Head Attention**: Different attention heads for perceptual, epistemic, prosocial, identity, and goal-oriented processing
- **Value Preservation**: Architectural guarantees against value drift
- **Multi-Component Rewards**: Local (head-specific) + global (system-wide)

## Configuration

Edit `config/agent_config.yaml` to customize:
- Base model
- Architecture dimensions
- Training hyperparameters
- Reward weights
- Value preservation settings

## Project Structure
```
conscious-agent/
├── conscious_agent/
│   ├── models/          # Agent architecture
│   ├── rewards/         # Reward computation
│   ├── training/        # Training loop
│   ├── evaluation/      # Evaluation suite
│   └── environments/    # Training environments
├── scripts/             # Training/eval scripts
├── config/              # Configuration files
└── tests/              # Unit tests
```

## Key Features

### Value Alignment

- **Immutable core values**: Cannot change during training
- **Harm detection**: Strong negative reward for harmful actions
- **Continuous monitoring**: Checks for value drift every 1000 steps
- **Automatic recovery**: Rollback if drift detected

### Consciousness Components

- **Self-model**: Agent maintains model of itself
- **Curiosity**: Intrinsic motivation to explore and learn
- **Metacognition**: Agent monitors its own uncertainty
- **Identity coherence**: Actions consistent with self-concept

### Training

- Multi-component reward system
- Curriculum learning
- Value preservation checks
- Comprehensive evaluation

## Testing
```bash
pytest tests/
```

## Citation

If you use this code, please cite:
```bibtex
@misc{conscious-ai-2025,
  author = {Ian Steenstra},
  title = {Conscious AI: Modeling Curiosity, Identity, and Human Alignment},
  year = {2025},
  publisher = {GitHub},
  url = {[https://github.com/iansteenstra/conscious-ai](https://github.com/iansteenstra/conscious-ai.git)}
}
```

## License

MIT License

## Acknowledgments

- Built on Hugging Face Transformers
- Inspired by neuroscience and cognitive science research
- Value alignment based on Anthropic's Constitutional AI
