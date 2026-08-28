# API Reference

## MLP

### `MLP(layer_sizes, activations, weight_init="xavier", loss_fn="mse", learning_rate=0.01, seed=None)`

Main MLP class.

**Parameters:**
- `layer_sizes` (List[int]): Layer dimensions [input, hidden..., output]
- `activations` (List[str]): Activation per layer (len = len(layer_sizes) - 1)
- `weight_init` (str): "xavier", "he", or "random"
- `loss_fn` (str): "mse", "bce", "cross_entropy"
- `learning_rate` (float): SGD learning rate
- `seed` (int): Random seed

**Methods:**

#### `forward(X) -> np.ndarray`
Batch forward pass.
- `X`: (n_samples, n_features)
- Returns: (n_samples, n_outputs)

#### `predict(X) -> np.ndarray`
Class predictions.
- Returns: (n_samples,) for binary, (n_samples,) class indices for multi-class

#### `predict_proba(X) -> np.ndarray`
Probability outputs.

#### `fit(X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=True, snapshot_interval=10, callback=None) -> Dict`
Train the model.
- `callback(epoch, metrics, weights_snapshot)`: Called each epoch

#### `validate(X_val, y_val) -> Tuple[float, float]`
Returns (loss, accuracy).

#### `save_checkpoint(path) / load_checkpoint(path)`
Persist/load model.

#### `get_graph() -> ComputationalGraph`
Access underlying graph.

#### `get_topology() -> Dict`
Graph structure for visualization.

---

## ComputationalGraph

### `ComputationalGraph()`

Graph container.

**Methods:**

#### `build_mlp(layer_sizes, activations, weight_init="xavier")`
Build complete MLP topology.

#### `forward_pass(inputs) -> np.ndarray`
Single sample forward pass.

#### `backward_pass(targets, loss_fn="mse") -> float`
Backpropagation. Returns loss.

#### `update_weights(learning_rate)`
Gradient descent step.

#### `get_weights_snapshot() -> Dict`
Current weights/gradients for visualization.

#### `get_topology() -> Dict`
Structure: layers, edges, input/output nodes.

---

## Node

### `Node(node_id, layer_idx, position, is_input=False, is_output=False, activation="relu")`

**Attributes:**
- `value`: Forward output
- `gradient`: Backward gradient
- `bias`: Bias term
- `bias_gradient`: Bias gradient

---

## Edge

### `Edge(edge_id, source, target, weight)`

**Attributes:**
- `weight`: Connection weight
- `gradient`: Weight gradient

---

## Data Loading

### `load_heart_disease(data_dir="data", download=True) -> (X, y)`
Load Cleveland Heart Disease dataset.
- `X`: (n_samples, 13)
- `y`: (n_samples,) binary

### `load_diabetes(data_dir="data", download=False) -> (X, y)`
Load Diabetes 130-US dataset (manual download required).

### `preprocess_data(X, y, test_size=0.2, random_state=42, scale=True) -> (X_train, X_test, y_train, y_test, scaler)`
Split and scale.

### `get_data_info(X, y) -> Dict`
Dataset statistics.

---

## Visualization

### `GraphVisualizer(graph, figsize=(14, 8))`
Real-time graph visualization.

**Methods:**
- `draw(epoch, loss, show_weights=True, show_gradients=False)`
- `save_frame(path)`
- `close()`

### `TrainingVisualizer(figsize=(12, 5))`
Training curves.

**Methods:**
- `update(epoch, train_loss, train_acc, val_loss=None, val_acc=None)`
- `draw()`
- `save(path)`
- `close()`

### `WeightHistoryVisualizer(figsize=(10, 6))`
Weight distributions.

**Methods:**
- `add_snapshot(epoch, weights_snapshot)`
- `draw()`
- `save(path)`
- `close()`

### `LiveTrainingDashboard(mlp, save_dir="outputs/visualization")`
Combined dashboard.

**Methods:**
- `on_epoch_end(epoch, metrics, weights_snapshot)`: Training callback
- `enable_live() / disable_live()`
- `create_animation(output_path, fps=5)`
- `close_all()`

---

## ExperimentHarness

### `ExperimentHarness(config, output_dir="outputs")`
Full experiment pipeline.

**Methods:**
- `prepare_data()`: Load and preprocess
- `build_model()`: Create MLP + dashboard
- `train() -> Dict`: Run training
- `evaluate() -> Dict`: Test evaluation
- `inference_demo(n_samples=5)`: Show predictions
- `save_results(filename=None) -> str`
- `save_model(filename=None) -> str`
- `run_full_experiment() -> Dict`: Complete pipeline

---

## ExperimentConfig

### `ExperimentConfig(...)`
Dataclass for experiment configuration.

**Fields:**
- `layer_sizes`: List[int]
- `activations`: List[str]
- `weight_init`: str
- `loss_fn`: str
- `learning_rate`: float
- `seed`: int
- `epochs`: int
- `batch_size`: int
- `test_size`: float
- `val_split`: float
- `snapshot_interval`: int
- `live_viz`: bool
- `save_viz`: bool
- `dataset`: str ("heart_disease" or "diabetes")
- `scale_data`: bool

**Methods:**
- `to_dict() -> Dict`
- `from_dict(d) -> ExperimentConfig`