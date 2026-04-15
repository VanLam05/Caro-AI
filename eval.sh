#!/bin/bash

# Script to evaluate the trained model

cd "$(dirname "$0")"

echo "📊 Evaluating Trained Model..."
echo "==============================="

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "⚠️  Python 3.12 not found, trying 'python'..."
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3.12"
fi

# Check if model exists
if [ ! -f "models/rl_model.pth" ]; then
    echo "❌ Model not found at models/rl_model.pth"
    echo "Please run training first:"
    echo "  bash train.sh"
    exit 1
fi

# Run inference
$PYTHON_CMD -m pipeline.inference

echo ""
echo "✅ Evaluation complete!"
