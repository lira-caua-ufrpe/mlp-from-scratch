"""Main MLP class - high-level API for training and inference."""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Callable
import numpy as np
from .graph import ComputationalGraph


class MLP:
    """Multi-Layer Perceptron implemented from scratch using computational graph."""

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        weight_init: str = "xavier",
        loss_fn: str = "mse",
        learning_rate: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Initialize MLP.

        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activations: List of activation functions for each layer (except input)
            weight_init: Weight initialization strategy ('xavier', 'he', 'random')
            loss_fn: Loss function ('mse', 'bce', 'cross_entropy')
            learning_rate: Initial learning rate
            seed: Random seed for reproducibility
        """
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"Expected {len(layer_sizes) - 1} activations, got {len(activations)}"
            )

        if seed is not None:
            np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weight_init = weight_init
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.seed = seed

        self.graph = ComputationalGraph()
        self.graph.build_mlp(layer_sizes, activations, weight_init)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "weights_snapshots": [],
            "gradients_snapshots": [],
        }
        self.epoch = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for batch of inputs."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self.graph.forward_pass(x) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (for classification)."""
        outputs = self.forward(X)
        if outputs.shape[1] == 1:
            return (outputs > 0.5).astype(int).flatten()
        return np.argmax(outputs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.forward(X)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss for batch."""
        if self.loss_fn == "mse":
            return np.mean(0.5 * (y_true - y_pred) ** 2)
        elif self.loss_fn == "bce":
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        elif self.loss_fn == "cross_entropy":
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return 0.0

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy for batch."""
        if y_pred.shape[1] == 1:
            preds = (y_pred > 0.5).astype(int)
            return np.mean(preds.flatten() == y_true.flatten())
        preds = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        return np.mean(preds == true_labels)

    def train_step(
        self, X_batch: np.ndarray, y_batch: np.ndarray
    ) -> Tuple[float, float]:
        """Single training step on a batch."""
        total_loss = 0.0
        for x, y in zip(X_batch, y_batch):
            self.graph.forward_pass(x)
            loss = self.graph.backward_pass(y, self.loss_fn)
            total_loss += loss

        self.graph.update_weights(self.learning_rate)
        avg_loss = total_loss / len(X_batch)
        return avg_loss, self.compute_accuracy(y_batch, self.forward(X_batch))

    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
        """Validate on validation set."""
        y_pred = self.forward(X_val)
        loss = self.compute_loss(y_val, y_pred)
        acc = self.compute_accuracy(y_val, y_pred)
        return loss, acc

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        snapshot_interval: int = 10,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Train the MLP.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for SGD
            verbose: Print progress
            snapshot_interval: Save weights snapshot every N epochs
            callback: Optional callback function(epoch, metrics, graph_snapshot)

        Returns:
            Training history dictionary
        """
        n_samples = len(X_train)
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_losses = []
            epoch_accs = []

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                loss, acc = self.train_step(X_batch, y_batch)
                epoch_losses.append(loss)
                epoch_accs.append(acc)

            avg_train_loss = np.mean(epoch_losses)
            avg_train_acc = np.mean(epoch_accs)

            self.history["train_loss"].append(avg_train_loss)
            self.history["train_acc"].append(avg_train_acc)

            val_loss, val_acc = 0.0, 0.0
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.validate(X_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

            if epoch % snapshot_interval == 0:
                snapshot = self.graph.get_weights_snapshot()
                self.history["weights_snapshots"].append(
                    {"epoch": epoch, "weights": snapshot}
                )

            self.epoch = epoch

            if verbose:
                val_str = (
                    f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                    if X_val is not None
                    else ""
                )
                print(
                    f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}{val_str}"
                )

            if callback:
                callback(
                    epoch,
                    {
                        "train_loss": avg_train_loss,
                        "train_acc": avg_train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    self.graph.get_weights_snapshot(),
                )

        return self.history

    def get_graph(self) -> ComputationalGraph:
        """Get the underlying computational graph."""
        return self.graph

    def get_topology(self) -> Dict:
        """Get network topology for visualization."""
        return self.graph.get_topology()

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        import pickle

        checkpoint = {
            "layer_sizes": self.layer_sizes,
            "activations": self.activations,
            "weight_init": self.weight_init,
            "loss_fn": self.loss_fn,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "epoch": self.epoch,
            "history": self.history,
            "graph_state": self.graph.get_weights_snapshot(),
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MLP":
        """Load model from checkpoint."""
        import pickle

        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        mlp = cls(
            layer_sizes=checkpoint["layer_sizes"],
            activations=checkpoint["activations"],
            weight_init=checkpoint["weight_init"],
            loss_fn=checkpoint["loss_fn"],
            learning_rate=checkpoint["learning_rate"],
            seed=checkpoint["seed"],
        )
        mlp.epoch = checkpoint["epoch"]
        mlp.history = checkpoint["history"]

        for edge_id, data in checkpoint["graph_state"]["edges"].items():
            if edge_id in mlp.graph.edges:
                mlp.graph.edges[edge_id].weight = data["weight"]
        for node_id, data in checkpoint["graph_state"]["nodes"].items():
            if node_id in mlp.graph.nodes:
                mlp.graph.nodes[node_id].bias = data["bias"]

        return mlp
