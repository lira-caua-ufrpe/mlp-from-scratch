"""Main training harness for MLP experiments."""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

from src.mlp.mlp import MLP
from src.data.loader import load_heart_disease, preprocess_data, get_data_info
from src.visualization.dashboard import LiveTrainingDashboard


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    # Model architecture
    layer_sizes: list
    activations: list
    weight_init: str = "xavier"
    loss_fn: str = "bce"
    learning_rate: float = 0.01
    seed: int = 42

    # Training
    epochs: int = 200
    batch_size: int = 32
    test_size: float = 0.2
    val_split: float = 0.1

    # Visualization
    snapshot_interval: int = 10
    live_viz: bool = False
    save_viz: bool = True

    # Data
    dataset: str = "heart_disease"
    scale_data: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentConfig":
        return cls(**d)


class ExperimentHarness:
    """Harness for running MLP experiments with full logging and visualization."""

    def __init__(self, config: ExperimentConfig, output_dir: str = "outputs"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mlp = None
        self.dashboard = None
        self.scaler = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.data_info = None
        self.results = {}

    def prepare_data(self) -> None:
        """Load and preprocess data."""
        print(f"Loading {self.config.dataset} dataset...")

        if self.config.dataset == "heart_disease":
            X, y = load_heart_disease(download=True)
        elif self.config.dataset == "diabetes":
            from src.data.loader import load_diabetes

            X, y = load_diabetes(download=False)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")

        self.data_info = get_data_info(X, y)
        print(f"Dataset info: {json.dumps(self.data_info, indent=2)}")

        X_train_val, self.X_test, y_train_val, self.y_test = preprocess_data(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            scale=False,
        )

        val_size = self.config.val_split / (1 - self.config.test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = preprocess_data(
            X_train_val,
            y_train_val,
            test_size=val_size,
            random_state=self.config.seed,
            scale=self.config.scale_data,
        )

        if self.config.scale_data:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_val = self.scaler.transform(self.X_val)
            self.X_test = self.scaler.transform(self.X_test)

        print(
            f"Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}"
        )

    def build_model(self) -> None:
        """Build MLP model."""
        print(f"Building MLP: {self.config.layer_sizes} with {self.config.activations}")
        self.mlp = MLP(
            layer_sizes=self.config.layer_sizes,
            activations=self.config.activations,
            weight_init=self.config.weight_init,
            loss_fn=self.config.loss_fn,
            learning_rate=self.config.learning_rate,
            seed=self.config.seed,
        )

        if self.config.save_viz:
            self.dashboard = LiveTrainingDashboard(self.mlp)
            if self.config.live_viz:
                self.dashboard.enable_live()

    def train(self) -> Dict[str, Any]:
        """Run training."""
        print(f"\nStarting training for {self.config.epochs} epochs...")
        start_time = time.time()

        def callback(epoch, metrics, weights_snapshot):
            if self.dashboard:
                self.dashboard.on_epoch_end(epoch, metrics, weights_snapshot)

        history = self.mlp.fit(
            self.X_train,
            self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=True,
            snapshot_interval=self.config.snapshot_interval,
            callback=callback,
        )

        train_time = time.time() - start_time
        print(f"\nTraining completed in {train_time:.2f}s")

        self.results["history"] = history
        self.results["train_time"] = train_time
        return history

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on test set."""
        print("\nEvaluating on test set...")

        y_pred = self.mlp.predict(self.X_test)
        y_proba = self.mlp.predict_proba(self.X_test)

        test_loss = self.mlp.compute_loss(
            self.y_test.reshape(-1, 1) if self.y_test.ndim == 1 else self.y_test,
            y_proba,
        )
        test_acc = self.mlp.compute_accuracy(
            self.y_test.reshape(-1, 1) if self.y_test.ndim == 1 else self.y_test,
            y_proba,
        )

        from sklearn.metrics import classification_report, confusion_matrix

        report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred).tolist()

        self.results["test"] = {
            "loss": float(test_loss),
            "accuracy": float(test_acc),
            "classification_report": report,
            "confusion_matrix": cm,
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist(),
        }

        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Confusion Matrix:\n{np.array(cm)}")

        return self.results["test"]

    def inference_demo(self, n_samples: int = 5) -> None:
        """Show inference on sample test cases."""
        print(f"\n=== Inference Demo ({n_samples} samples) ===")
        indices = np.random.choice(
            len(self.X_test), min(n_samples, len(self.X_test)), replace=False
        )

        for idx in indices:
            x = self.X_test[idx:idx + 1]
            y_true = self.y_test[idx]
            y_proba = self.mlp.predict_proba(x)[0]
            y_pred = self.mlp.predict(x)[0]

            print(f"\nSample {idx}:")
            print(f"  True label: {y_true}")
            print(
                f"  Predicted: {y_pred} (prob: {y_proba[0] if len(y_proba) == 1 else y_proba})"
            )
            print(f"  Correct: {'✓' if y_pred == y_true else '✗'}")

    def save_results(self, filename: str = None) -> str:
        """Save experiment results."""
        if filename is None:
            filename = f"experiment_{self.config.dataset}_{int(time.time())}.json"

        save_data = {
            "config": self.config.to_dict(),
            "data_info": self.data_info,
            "results": self.results,
            "topology": self.mlp.get_topology() if self.mlp else None,
        }

        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"\nResults saved to {filepath}")
        return str(filepath)

    def save_model(self, filename: str = None) -> str:
        """Save model checkpoint."""
        if filename is None:
            filename = f"model_{self.config.dataset}_{int(time.time())}.pkl"

        filepath = self.output_dir / filename
        self.mlp.save_checkpoint(str(filepath))
        print(f"Model saved to {filepath}")
        return str(filepath)

    def run_full_experiment(self) -> Dict[str, Any]:
        """Run complete experiment pipeline."""
        self.prepare_data()
        self.build_model()
        self.train()
        self.evaluate()
        self.inference_demo()
        self.save_results()
        self.save_model()

        if self.dashboard:
            self.dashboard.create_animation()
            self.dashboard.close_all()

        return self.results


def run_heart_disease_experiment():
    """Run default Heart Disease experiment."""
    config = ExperimentConfig(
        layer_sizes=[13, 64, 32, 1],
        activations=["relu", "relu", "sigmoid"],
        weight_init="xavier",
        loss_fn="bce",
        learning_rate=0.01,
        epochs=200,
        batch_size=32,
        test_size=0.2,
        val_split=0.1,
        snapshot_interval=20,
        live_viz=False,
        save_viz=True,
        dataset="heart_disease",
        scale_data=True,
    )

    harness = ExperimentHarness(config)
    return harness.run_full_experiment()


if __name__ == "__main__":
    run_heart_disease_experiment()
