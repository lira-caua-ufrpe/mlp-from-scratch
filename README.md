# MLP from Scratch - Advanced AI Topics

Implementation of a Multi-Layer Perceptron (MLP) from scratch using a computational graph structure, with real-time visualization of feed-forward and backpropagation.

## Features

- **MLP from Scratch**: No PyTorch/TensorFlow - pure NumPy implementation
- **Graph-Based Architecture**: Neurons as nodes, weights as edges in a computational graph
- **Feed-Forward & Backpropagation**: Full implementation with automatic differentiation
- **Heart Disease Dataset**: UCI Cleveland dataset (80/20 train/test split)
- **Real-Time Visualization**: Live graph with weight updates, training curves, weight distributions
- **GitFlow Workflow**: Proper version control with feature branches
- **Modular Design**: Clean package structure for extensibility

## Project Structure

```
workspace/
├── src/
│   ├── mlp/                 # Core MLP implementation
│   │   ├── graph.py         # Computational graph (nodes, edges)
│   │   ├── mlp.py           # Main MLP class
│   │   └── __init__.py
│   ├── data/                # Data loading & preprocessing
│   │   ├── loader.py        # Heart Disease & Diabetes loaders
│   │   └── __init__.py
│   ├── visualization/       # Real-time visualization
│   │   ├── dashboard.py     # Graph & metrics dashboards
│   │   └── __init__.py
│   ├── utils/               # Utilities
│   └── __init__.py
├── scripts/
│   └── train.py             # Training harness
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── outputs/                 # Generated visualizations & models
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run default Heart Disease experiment
python scripts/train.py
```

Or programmatically:

```python
from scripts.train import ExperimentHarness, ExperimentConfig

config = ExperimentConfig(
    layer_sizes=[13, 64, 32, 1],
    activations=["relu", "relu", "sigmoid"],
    epochs=200,
    batch_size=32,
    learning_rate=0.01
)

harness = ExperimentHarness(config)
results = harness.run_full_experiment()
```

## Architecture

### Computational Graph

The MLP is represented as a directed acyclic graph:
- **Nodes**: Neurons with value, gradient, bias, activation
- **Edges**: Weighted connections with weight and gradient
- **Layers**: Ordered collections of nodes

### Forward Pass

```
Input Layer → Hidden Layers → Output Layer
     ↓            ↓              ↓
   x₁,x₂...   h = σ(Wx+b)    ŷ = σ(Wₕh+b)
```

### Backpropagation

Gradients flow backward through the graph:
1. Compute loss gradient at output
2. Apply chain rule through activation derivatives
3. Accumulate gradients on edges (weights) and nodes (biases)
4. Update parameters: `w ← w - η·∇w`

## Visualization

The dashboard provides:
- **Graph View**: Network topology with real-time weight colors/widths
- **Metrics View**: Training/validation loss & accuracy curves
- **Weight History**: Distribution evolution over epochs
- **Animation**: GIF generation of training progress

## GitFlow Workflow

```bash
# Start feature branch
git checkout develop
git checkout -b feature/mlp-implementation

# Work on feature...
git add .
git commit -m "feat: implement MLP computational graph"

# Finish feature
git checkout develop
git merge --no-ff feature/mlp-implementation
git branch -d feature/mlp-implementation

# Release
git checkout main
git merge --no-ff develop
git tag -a v0.1.0 -m "Release v0.1.0"
```

## Dataset

**Heart Disease (Cleveland)**:
- 303 samples, 13 features
- Binary classification: disease (1) vs no disease (0)
- Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

**Diabetes 130-US Hospitals** (bonus challenge):
- 100k+ samples, 50+ features
- Predict 30-day readmission
- Leaderboard competition

## Requirements

- Python 3.10+
- NumPy, Pandas, Scikit-learn
- Matplotlib, NetworkX (visualization)

## License

MIT License - Educational project for Advanced AI Topics course.