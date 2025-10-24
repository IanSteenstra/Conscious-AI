#!/bin/bash

# Quick start script for conscious agent

echo "=========================================="
echo "Conscious Agent Quick Start"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go to project root (parent of scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Project root: $PROJECT_ROOT"
echo ""

# Navigate to project root
cd "$PROJECT_ROOT"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda."
    exit 1
fi

# Create environment
echo "Creating conda environment..."
conda create -n conscious-agent python=3.10 -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate conscious-agent

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Download pretrained model (will be cached)
echo "Downloading pretrained model (this may take a few minutes)..."
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')"

# Run tests
echo "Running tests..."
pytest tests/ -v

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Train agent: python scripts/train.py"
echo "  2. Evaluate: python scripts/evaluate.py --checkpoint checkpoints/checkpoint_final.pt"
echo "  3. Interactive demo: python scripts/demo.py"
echo ""
echo "Starting demo with untrained agent..."
echo "(Type 'quit' to exit demo)"
echo ""

# Run demo
python scripts/demo.py