# Experiment Guide

## Running Experiments

### Basic Heart Disease Experiment

```bash
python scripts/train.py
```

This runs the default configuration:
- Architecture: 13 → 64 → 32 → 1
- Activations: ReLU, ReLU, Sigmoid
- Loss: Binary Cross-Entropy
- Optimizer: SGD (lr=0.01)
- Epochs: 200
- Batch size: 32

### Custom Configuration

```python
from scripts.train import ExperimentHarness, ExperimentConfig

config = ExperimentConfig(
    layer_sizes=[13, 128, 64, 32, 1],
    activations=["relu", "relu", "relu", "sigmoid"],
    weight_init="he",
    loss_fn="bce",
    learning_rate=0.005,
    epochs=300,
    batch_size=64,
    test_size=0.2,
    val_split=0.15,
    snapshot_interval=25,
    live_viz=True,      # Requires matplotlib interactive backend
    save_viz=True,
    dataset="heart_disease",
    scale_data=True
)

harness = ExperimentHarness(config, output_dir="outputs/my_experiment")
results = harness.run_full_experiment()
```

### With Live Visualization

```python
config = ExperimentConfig(..., live_viz=True, save_viz=True)
harness = ExperimentHarness(config)
# In another terminal, or use matplotlib interactive backend
results = harness.run_full_experiment()
```

The dashboard shows:
1. **Graph View**: Network with weight magnitudes (blue=positive, red=negative, width=magnitude)
2. **Metrics View**: Loss & accuracy curves (train vs validation)
3. **Weight History**: Distribution evolution

Frames saved to `outputs/visualization/` and combined into `training_animation.gif`.

## Architecture Exploration

### Wide vs Deep

```python
# Wide shallow
config = ExperimentConfig(layer_sizes=[13, 256, 1], activations=["relu", "sigmoid"])

# Deep narrow
config = ExperimentConfig(layer_sizes=[13, 32, 32, 32, 32, 1], 
                          activations=["relu"]*4 + ["sigmoid"])
```

### Activation Comparison

```python
for act in ["relu", "tanh", "sigmoid"]:
    config = ExperimentConfig(
        layer_sizes=[13, 64, 32, 1],
        activations=[act, act, "sigmoid"]
    )
    # Run and compare
```

### Weight Initialization

```python
for init in ["xavier", "he", "random"]:
    config = ExperimentConfig(..., weight_init=init)
```

### Learning Rate Schedule

```python
# Manual decay in callback
def lr_callback(epoch, metrics, weights):
    if epoch > 50 and epoch % 25 == 0:
        harness.mlp.learning_rate *= 0.5
        print(f"LR decayed to {harness.mlp.learning_rate}")

harness.mlp.fit(..., callback=lr_callback)
```

## Diabetes Challenge (Bonus)

### Data Preparation

1. Download from UCI: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
2. Place `dataset_diabetes.csv` in `data/` folder
3. Run:

```python
config = ExperimentConfig(
    layer_sizes=[50, 128, 64, 1],  # Adjust input size after preprocessing
    activations=["relu", "relu", "sigmoid"],
    epochs=100,
    batch_size=256,
    dataset="diabetes"
)
```

Note: Diabetes dataset has ~50 features after preprocessing. Adjust `layer_sizes[0]` accordingly.

### Leaderboard Tips

1. **Feature Engineering**: Use domain knowledge (medication counts, diagnosis codes)
2. **Class Imbalance**: Readmission <30 days is ~11% - use class weights or oversampling
3. **Architecture**: Try deeper networks with dropout (implement in graph)
4. **Ensemble**: Train multiple seeds, average predictions
5. **Hyperparameter Search**: LR, batch size, architecture

## Reproducibility

```python
config = ExperimentConfig(..., seed=42)
# All numpy randomness controlled
```

Set `PYTHONHASHSEED=42` before running for full reproducibility.

## Outputs

Each experiment generates:
- `outputs/experiment_<dataset>_<timestamp>.json`: Full results
- `outputs/model_<dataset>_<timestamp>.pkl`: Model checkpoint
- `outputs/visualization/graph_epoch_XXXX.png`: Graph frames
- `outputs/visualization/metrics_epoch_XXXX.png`: Metrics frames
- `outputs/visualization/weights_epoch_XXXX.png`: Weight histograms
- `outputs/training_animation.gif`: Combined animation

## Analyzing Results

```python
import json
import matplotlib.pyplot as plt

with open("outputs/experiment_heart_disease_1234567890.json") as f:
    data = json.load(f)

history = data["results"]["history"]
plt.plot(history["train_loss"], label="Train")
plt.plot(history["val_loss"], label="Val")
plt.legend()
plt.show()

test_acc = data["results"]["test"]["accuracy"]
print(f"Test Accuracy: {test_acc:.4f}")
```

## Common Issues

| Issue | Solution |
|-------|----------|
| NaN loss | Reduce learning rate, check data scaling |
| Low accuracy | Increase capacity, check preprocessing, try different init |
| Slow training | Reduce batch size, use smaller architecture |
| Visualization not showing | Install `tkinter` (Linux: `sudo apt install python3-tk`), use `save_viz=True` |
| Memory error (Diabetes) | Use generator/batch loading, reduce batch size |

## Extending for Research

### Custom Layers
Add to `graph.py`:
```python
def add_dropout_layer(self, rate=0.5):
    # Add dropout nodes/edges
```

### Batch Normalization
Track running mean/var in nodes, normalize in forward.

### Optimizers
Replace `update_weights()` with Adam, RMSprop, etc.

### Regularization
Add L2 penalty to loss in `backward_pass()`.