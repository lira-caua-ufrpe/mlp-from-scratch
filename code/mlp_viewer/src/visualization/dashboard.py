"""Real-time visualization for MLP training - graph structure and weight updates."""

from __future__ import annotations
from typing import Dict
import numpy as np
from pathlib import Path

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class GraphVisualizer:
    """Visualize MLP as a computational graph with real-time weight updates."""

    def __init__(self, graph, figsize: tuple = (14, 8)):
        self.graph = graph
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.pos = None
        self.G = None
        self._setup_graph()

    def _setup_graph(self):
        """Build NetworkX graph from computational graph."""
        if not HAS_NETWORKX:
            raise ImportError(
                "networkx required for visualization. Install with: pip install networkx"
            )

        self.G = nx.DiGraph()
        topology = self.graph.get_topology()

        for layer_idx, node_ids in topology["layers"].items():
            for pos, node_id in enumerate(node_ids):
                node = self.graph.nodes[node_id]
                self.G.add_node(
                    node_id,
                    layer=layer_idx,
                    position=pos,
                    value=node.value,
                    bias=node.bias,
                    activation=node.activation,
                    is_input=node.is_input,
                    is_output=node.is_output,
                )

        for source_id, target_id in topology["edges"]:
            edge = None
            for e in self.graph.edges.values():
                if e.source.node_id == source_id and e.target.node_id == target_id:
                    edge = e
                    break
            if edge:
                self.G.add_edge(
                    source_id, target_id, weight=edge.weight, gradient=edge.gradient
                )

        self.pos = self._compute_layout()

    def _compute_layout(self) -> Dict:
        """Compute node positions for layered layout."""
        pos = {}
        topology = self.graph.get_topology()
        max_layer_size = max(len(nodes) for nodes in topology["layers"].values())

        for layer_idx, node_ids in topology["layers"].items():
            layer_size = len(node_ids)
            for i, node_id in enumerate(node_ids):
                x = layer_idx * 3.0
                y = (i - layer_size / 2) * (2.0 * max_layer_size / max(layer_size, 1))
                pos[node_id] = (x, y)
        return pos

    def update_weights(self):
        """Update edge weights and node values from graph."""
        if not HAS_NETWORKX:
            return

        for node_id in self.G.nodes():
            node = self.graph.nodes[node_id]
            self.G.nodes[node_id]["value"] = node.value
            self.G.nodes[node_id]["bias"] = node.bias
            self.G.nodes[node_id]["gradient"] = node.gradient

        for u, v in self.G.edges():
            for e in self.graph.edges.values():
                if e.source.node_id == u and e.target.node_id == v:
                    self.G.edges[u, v]["weight"] = e.weight
                    self.G.edges[u, v]["gradient"] = e.gradient
                    break

    def draw(
        self,
        epoch: int = 0,
        loss: float = 0.0,
        show_weights: bool = True,
        show_gradients: bool = False,
        weight_threshold: float = 0.01,
    ):
        """Draw the graph with current weights."""
        if not HAS_MPL or not HAS_NETWORKX:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)

        self.ax.clear()
        self.update_weights()

        node_colors = []
        node_sizes = []
        for node_id in self.G.nodes():
            node = self.G.nodes[node_id]
            if node["is_input"]:
                node_colors.append("#3498db")
                node_sizes.append(800)
            elif node["is_output"]:
                node_colors.append("#e74c3c")
                node_sizes.append(800)
            else:
                val = node["value"]
                intensity = min(abs(val) / 2.0, 1.0)
                if val >= 0:
                    node_colors.append((1, 1 - intensity, 1 - intensity))
                else:
                    node_colors.append((1 - intensity, 1 - intensity, 1))
                node_sizes.append(600)

        nx.draw_networkx_nodes(
            self.G,
            self.pos,
            node_color=node_colors,
            node_size=node_sizes,
            ax=self.ax,
            alpha=0.9,
        )

        edge_weights = []
        edge_colors = []
        edge_widths = []

        for u, v in self.G.edges():
            w = self.G.edges[u, v]["weight"]
            g = self.G.edges[u, v]["gradient"]

            if abs(w) < weight_threshold:
                continue

            edge_weights.append((u, v))
            if show_gradients:
                intensity = min(abs(g) * 10, 1.0)
                edge_colors.append(
                    (1, 1 - intensity, 1 - intensity)
                    if g >= 0
                    else (1 - intensity, 1 - intensity, 1)
                )
            else:
                intensity = min(abs(w) * 2, 1.0)
                edge_colors.append(
                    (0.2, 0.2, 0.8, intensity) if w >= 0 else (0.8, 0.2, 0.2, intensity)
                )
            edge_widths.append(max(abs(w) * 3, 0.5))

        if edge_weights:
            nx.draw_networkx_edges(
                self.G,
                self.pos,
                edgelist=edge_weights,
                edge_color=edge_colors,
                width=edge_widths,
                ax=self.ax,
                alpha=0.7,
                arrows=True,
                arrowsize=15,
            )

        if show_weights:
            edge_labels = {}
            for u, v in edge_weights:
                w = self.G.edges[u, v]["weight"]
                edge_labels[(u, v)] = f"{w:.2f}"
            nx.draw_networkx_edge_labels(
                self.G, self.pos, edge_labels=edge_labels, font_size=7, ax=self.ax
            )

        node_labels = {}
        for node_id in self.G.nodes():
            node = self.G.nodes[node_id]
            if node["is_input"]:
                node_labels[node_id] = f"x{node['position']}"
            elif node["is_output"]:
                node_labels[node_id] = f"y{node['position']}"
            else:
                node_labels[node_id] = f"h{node['layer']}_{node['position']}"

        nx.draw_networkx_labels(
            self.G,
            self.pos,
            labels=node_labels,
            font_size=9,
            font_weight="bold",
            ax=self.ax,
        )

        layer_info = {}
        for node_id in self.G.nodes():
            layer = self.G.nodes[node_id]["layer"]
            if layer not in layer_info:
                layer_info[layer] = []
            layer_info[layer].append(node_id)

        for layer_idx, nodes in layer_info.items():
            x = layer_idx * 3.0
            y_max = max(self.pos[n][1] for n in nodes) + 1.5
            label = (
                "Input"
                if layer_idx == 0
                else (
                    "Output"
                    if layer_idx == max(layer_info.keys())
                    else f"Hidden {layer_idx}"
                )
            )
            self.ax.text(
                x,
                y_max,
                label,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="gray",
            )

        self.ax.set_title(
            f"MLP Computational Graph | Epoch: {epoch} | Loss: {loss:.4f}",
            fontsize=14,
            fontweight="bold",
        )
        self.ax.axis("off")
        self.fig.tight_layout()

    def save_frame(self, path: str):
        """Save current frame to file."""
        if self.fig:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")

    def close(self):
        """Close the figure."""
        if self.fig and HAS_MPL:
            import matplotlib.pyplot as plt

            plt.close(self.fig)


class TrainingVisualizer:
    """Real-time training metrics visualization."""

    def __init__(self, figsize: tuple = (12, 5)):
        self.figsize = figsize
        self.fig = None
        self.axes = None
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "epochs": [],
        }

    def update(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float = None,
        val_acc: float = None,
    ):
        """Update metrics history."""
        self.history["epochs"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
        if val_acc is not None:
            self.history["val_acc"].append(val_acc)

    def draw(self):
        """Draw training curves."""
        if not HAS_MPL:
            return

        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 2, figsize=self.figsize)

        for ax in self.axes:
            ax.clear()

        epochs = self.history["epochs"]

        self.axes[0].plot(
            epochs, self.history["train_loss"], "b-", label="Train Loss", linewidth=2
        )
        if self.history["val_loss"]:
            self.axes[0].plot(
                epochs[: len(self.history["val_loss"])],
                self.history["val_loss"],
                "r-",
                label="Val Loss",
                linewidth=2,
            )
        self.axes[0].set_xlabel("Epoch")
        self.axes[0].set_ylabel("Loss")
        self.axes[0].set_title("Training Loss")
        self.axes[0].legend()
        self.axes[0].grid(True, alpha=0.3)

        self.axes[1].plot(
            epochs, self.history["train_acc"], "b-", label="Train Acc", linewidth=2
        )
        if self.history["val_acc"]:
            self.axes[1].plot(
                epochs[: len(self.history["val_acc"])],
                self.history["val_acc"],
                "r-",
                label="Val Acc",
                linewidth=2,
            )
        self.axes[1].set_xlabel("Epoch")
        self.axes[1].set_ylabel("Accuracy")
        self.axes[1].set_title("Training Accuracy")
        self.axes[1].legend()
        self.axes[1].grid(True, alpha=0.3)

        self.fig.tight_layout()

    def save(self, path: str):
        if self.fig:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")

    def close(self):
        if self.fig and HAS_MPL:
            import matplotlib.pyplot as plt

            plt.close(self.fig)


class WeightHistoryVisualizer:
    """Visualize weight distributions over training."""

    def __init__(self, figsize: tuple = (10, 6)):
        self.figsize = figsize
        self.fig = None
        self.ax = None
        self.snapshots = []

    def add_snapshot(self, epoch: int, weights_snapshot: Dict):
        """Add a weights snapshot."""
        all_weights = []
        for edge_data in weights_snapshot["edges"].values():
            all_weights.append(edge_data["weight"])
        self.snapshots.append({"epoch": epoch, "weights": np.array(all_weights)})

    def draw(self):
        """Draw weight distribution evolution."""
        if not HAS_MPL or not self.snapshots:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)

        self.ax.clear()

        for snap in self.snapshots:
            self.ax.hist(
                snap["weights"],
                bins=30,
                alpha=0.3,
                label=f"Epoch {snap['epoch']}",
                density=True,
            )

        self.ax.set_xlabel("Weight Value")
        self.ax.set_ylabel("Density")
        self.ax.set_title("Weight Distribution Evolution")
        self.ax.legend(fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()

    def save(self, path: str):
        if self.fig:
            self.fig.savefig(path, dpi=150, bbox_inches="tight")

    def close(self):
        if self.fig and HAS_MPL:
            import matplotlib.pyplot as plt

            plt.close(self.fig)


class LiveTrainingDashboard:
    """Combined dashboard for live training visualization."""

    def __init__(self, mlp, save_dir: str = "outputs/visualization"):
        self.mlp = mlp
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.graph_viz = GraphVisualizer(mlp.get_graph())
        self.metrics_viz = TrainingVisualizer()
        self.weight_viz = WeightHistoryVisualizer()

        self.epoch = 0
        self.is_live = False

    def on_epoch_end(self, epoch: int, metrics: Dict, weights_snapshot: Dict):
        """Callback for epoch end."""
        self.epoch = epoch
        self.metrics_viz.update(
            epoch,
            metrics["train_loss"],
            metrics["train_acc"],
            metrics.get("val_loss"),
            metrics.get("val_acc"),
        )

        if epoch % 10 == 0:
            self.weight_viz.add_snapshot(epoch, weights_snapshot)

        self.graph_viz.draw(epoch, metrics["train_loss"])
        self.metrics_viz.draw()
        self.weight_viz.draw()

        if self.is_live:
            import matplotlib.pyplot as plt

            plt.pause(0.01)

        self._save_frame(epoch)

    def _save_frame(self, epoch: int):
        """Save visualization frames."""
        self.graph_viz.save_frame(str(self.save_dir / f"graph_epoch_{epoch:04d}.png"))
        self.metrics_viz.save(str(self.save_dir / f"metrics_epoch_{epoch:04d}.png"))
        self.weight_viz.save(str(self.save_dir / f"weights_epoch_{epoch:04d}.png"))

    def enable_live(self):
        """Enable live matplotlib updates."""
        self.is_live = True
        import matplotlib.pyplot as plt

        plt.ion()

    def disable_live(self):
        """Disable live updates."""
        self.is_live = False
        import matplotlib.pyplot as plt

        plt.ioff()

    def create_animation(
        self, output_path: str = "outputs/training_animation.gif", fps: int = 5
    ):
        """Create animation from saved frames."""
        import glob
        from PIL import Image

        frames = sorted(glob.glob(str(self.save_dir / "graph_epoch_*.png")))
        if not frames:
            print("No frames to animate")
            return

        images = [Image.open(f) for f in frames]
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=1000 // fps,
            loop=0,
        )
        print(f"Animation saved to {output_path}")

    def close_all(self):
        """Close all visualizers."""
        self.graph_viz.close()
        self.metrics_viz.close()
        self.weight_viz.close()
