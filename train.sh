#!/bin/bash

# Script to train the RL agent

cd "$(dirname "$0")"

echo "🤖 Starting AI Training..."
echo "================================"

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "⚠️  Python 3.12 not found, trying 'python'..."
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3.12"
fi

# Run training
$PYTHON_CMD -m pipeline.train

echo ""
echo "✅ Training complete!"
echo "Model saved to: models/rl_model.pth"
echo ""
echo "To evaluate the model, run:"
echo "  bash eval.sh"
