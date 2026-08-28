# Architecture Documentation

## Overview

This document describes the internal architecture of the MLP from Scratch implementation.

## Core Components

### 1. Computational Graph (`src/mlp/graph.py`)

The foundation of the implementation. Represents the neural network as a directed acyclic graph.

#### Node
Represents a single neuron:
- `node_id`: Unique identifier
- `layer_idx`: Layer index (0 = input)
- `position`: Position within layer
- `value`: Forward pass output (activation)
- `gradient`: Backward pass gradient (∂L/∂value)
- `bias`: Neuron bias term
- `bias_gradient`: Gradient for bias
- `activation`: Activation function name

#### Edge
Represents a weighted connection:
- `edge_id`: Unique identifier
- `source`: Source Node
- `target`: Target Node
- `weight`: Connection weight
- `gradient`: Weight gradient (∂L/∂weight)

#### ComputationalGraph
Manages the full network topology:
- `build_mlp()`: Construct from layer specifications
- `forward_pass()`: Execute feed-forward
- `backward_pass()`: Execute backpropagation
- `update_weights()`: Apply gradient descent step
- `get_weights_snapshot()`: For visualization
- `get_topology()`: For graph structure visualization

### 2. MLP Class (`src/mlp/mlp.py`)

High-level API wrapping the computational graph.

#### Key Methods
- `forward(X)`: Batch forward pass
- `predict(X)`: Class predictions
- `predict_proba(X)`: Probability outputs
- `train_step(batch)`: Single SGD step
- `fit()`: Full training loop with callbacks
- `validate()`: Validation evaluation
- `save_checkpoint()` / `load_checkpoint()`: Persistence

#### Training Loop
```python
for epoch in epochs:
    shuffle data
    for batch in batches:
        forward_pass(batch)
        loss = backward_pass(targets)
        update_weights(learning_rate)
    validate()
    callback(epoch, metrics, weights_snapshot)
```

### 3. Data Loading (`src/data/loader.py`)

#### Heart Disease
- Downloads from UCI repository
- Handles missing values (marked as "?")
- Binary target conversion (0/1)
- Standard scaling

#### Diabetes 130-US Hospitals
- Manual download required (large dataset)
- Categorical encoding
- Missing value imputation
- Readmission target (<30 days = 1)

### 4. Visualization (`src/visualization/dashboard.py`)

#### GraphVisualizer
- NetworkX for graph structure
- Matplotlib for rendering
- Layered layout algorithm
- Weight/gradient color coding
- Real-time updates via `draw()`

#### TrainingVisualizer
- Dual-axis plots (loss + accuracy)
- Train/validation comparison
- Live updates

#### WeightHistoryVisualizer
- Histogram of weight distributions
- Overlaid snapshots across epochs

#### LiveTrainingDashboard
- Combines all visualizers
- Callback integration with training loop
- Frame saving for animation
- GIF generation

## Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Dataset   │────▶│ Preprocessing │────▶│  Train/Val  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Results    │◀────│  Evaluation  │◀────│   Model     │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│Animation/GIF│◀────│  Dashboard   │◀────│  Training   │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Activation Functions

| Function | Formula | Derivative |
|----------|---------|------------|
| ReLU | max(0, x) | 1 if x > 0 else 0 |
| Sigmoid | 1/(1+e⁻ˣ) | σ(x)(1-σ(x)) |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 1-tanh²(x) |
| Linear | x | 1 |

## Loss Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| MSE | ½(y-ŷ)² | Regression |
| BCE | -[y log ŷ + (1-y) log(1-ŷ)] | Binary Classification |
| CrossEntropy | -Σ yᵢ log ŷᵢ | Multi-class Classification |

## Weight Initialization

| Method | Formula | Best For |
|--------|---------|----------|
| Xavier | U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out))) | Sigmoid/Tanh |
| He | N(0, √(2/fan_in)) | ReLU |
| Random | N(0, 0.01) | General |

## Extending the Framework

### Adding New Activation
```python
# In graph.py _activate() and _activate_derivative()
elif activation == "gelu":
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

### Adding New Loss
```python
# In mlp.py compute_loss() and graph.py backward_pass()
elif loss_fn == "huber":
    # Implementation
```

### Custom Callbacks
```python
def my_callback(epoch, metrics, weights):
    # Log to wandb, tensorboard, etc.
    pass

mlp.fit(..., callback=my_callback)
```