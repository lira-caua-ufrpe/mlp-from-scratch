"""Graph structure for MLP - nodes and edges representing neurons and connections."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np
import uuid


@dataclass
class Node:
    """Represents a neuron in the computational graph."""
    node_id: str
    layer_idx: int
    position: int
    value: float = 0.0
    gradient: float = 0.0
    bias: float = 0.0
    bias_gradient: float = 0.0
    is_input: bool = False
    is_output: bool = False
    activation: str = "relu"
    
    def __post_init__(self):
        if not self.node_id:
            self.node_id = f"n_{self.layer_idx}_{self.position}_{uuid.uuid4().hex[:8]}"
    
    def reset_gradients(self):
        self.gradient = 0.0
        self.bias_gradient = 0.0


@dataclass
class Edge:
    """Represents a weighted connection between neurons."""
    edge_id: str
    source: Node
    target: Node
    weight: float
    gradient: float = 0.0
    
    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = f"e_{self.source.node_id}_to_{self.target.node_id}_{uuid.uuid4().hex[:8]}"
    
    def reset_gradient(self):
        self.gradient = 0.0


class ComputationalGraph:
    """Directed acyclic graph representing the MLP topology."""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.layers: Dict[int, List[Node]] = {}
        self.input_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        self._layer_count = 0
    
    def add_layer(self, size: int, activation: str = "relu", is_input: bool = False, is_output: bool = False) -> List[Node]:
        """Add a new layer to the graph."""
        layer_idx = self._layer_count
        nodes = []
        
        for pos in range(size):
            node = Node(
                node_id="",
                layer_idx=layer_idx,
                position=pos,
                is_input=is_input,
                is_output=is_output,
                activation=activation if not is_input else "linear"
            )
            self.nodes[node.node_id] = node
            nodes.append(node)
            
            if is_input:
                self.input_nodes.append(node)
            if is_output:
                self.output_nodes.append(node)
        
        self.layers[layer_idx] = nodes
        self._layer_count += 1
        return nodes
    
    def connect_layers(self, source_layer_idx: int, target_layer_idx: int, 
                       weight_init: str = "xavier") -> List[Edge]:
        """Fully connect two adjacent layers."""
        source_nodes = self.layers[source_layer_idx]
        target_nodes = self.layers[target_layer_idx]
        
        fan_in = len(source_nodes)
        fan_out = len(target_nodes)
        
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            weights = np.random.uniform(-limit, limit, (fan_out, fan_in))
        elif weight_init == "he":
            limit = np.sqrt(2.0 / fan_in)
            weights = np.random.normal(0, limit, (fan_out, fan_in))
        else:
            weights = np.random.randn(fan_out, fan_in) * 0.01
        
        edges = []
        for i, target in enumerate(target_nodes):
            for j, source in enumerate(source_nodes):
                edge = Edge(
                    edge_id="",
                    source=source,
                    target=target,
                    weight=float(weights[i, j])
                )
                self.edges[edge.edge_id] = edge
                edges.append(edge)
        
        return edges
    
    def build_mlp(self, layer_sizes: List[int], activations: List[str],
                  weight_init: str = "xavier") -> None:
        """Build complete MLP from layer specification."""
        for i, (size, act) in enumerate(zip(layer_sizes, activations)):
            is_input = (i == 0)
            is_output = (i == len(layer_sizes) - 1)
            self.add_layer(size, activation=act, is_input=is_input, is_output=is_output)
        
        for i in range(len(layer_sizes) - 1):
            self.connect_layers(i, i + 1, weight_init)
    
    def forward_pass(self, inputs: np.ndarray) -> np.ndarray:
        """Execute forward pass through the graph."""
        if len(inputs) != len(self.input_nodes):
            raise ValueError(f"Expected {len(self.input_nodes)} inputs, got {len(inputs)}")
        
        for i, node in enumerate(self.input_nodes):
            node.value = float(inputs[i])
        
        for layer_idx in range(1, self._layer_count):
            for node in self.layers[layer_idx]:
                weighted_sum = node.bias
                for edge in self.get_incoming_edges(node):
                    weighted_sum += edge.source.value * edge.weight
                
                node.value = self._activate(weighted_sum, node.activation)
        
        return np.array([node.value for node in self.output_nodes])
    
    def get_incoming_edges(self, node: Node) -> List[Edge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges.values() if e.target.node_id == node.node_id]
    
    def get_outgoing_edges(self, node: Node) -> List[Edge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges.values() if e.source.node_id == node.node_id]
    
    def _activate(self, x: float, activation: str) -> float:
        if activation == "relu":
            return max(0.0, x)
        elif activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "linear":
            return x
        elif activation == "softmax":
            return x
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _activate_derivative(self, x: float, activation: str) -> float:
        if activation == "relu":
            return 1.0 if x > 0 else 0.0
        elif activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        elif activation == "tanh":
            return 1.0 - np.tanh(x) ** 2
        elif activation == "linear":
            return 1.0
        else:
            return 1.0
    
    def backward_pass(self, targets: np.ndarray, loss_fn: str = "mse") -> float:
        """Execute backward pass (backpropagation) through the graph."""
        if len(targets) != len(self.output_nodes):
            raise ValueError(f"Expected {len(self.output_nodes)} targets, got {len(targets)}")
        
        for node in self.nodes.values():
            node.reset_gradients()
        for edge in self.edges.values():
            edge.reset_gradient()
        
        loss = 0.0
        for i, node in enumerate(self.output_nodes):
            if loss_fn == "mse":
                loss += 0.5 * (node.value - targets[i]) ** 2
                node.gradient = node.value - targets[i]
            elif loss_fn == "bce":
                eps = 1e-15
                y_pred = np.clip(node.value, eps, 1 - eps)
                loss += -(targets[i] * np.log(y_pred) + (1 - targets[i]) * np.log(1 - y_pred))
                node.gradient = (y_pred - targets[i]) / (y_pred * (1 - y_pred) + eps)
        
        for layer_idx in range(self._layer_count - 1, 0, -1):
            for node in self.layers[layer_idx]:
                act_deriv = self._activate_derivative(node.value, node.activation)
                node.gradient *= act_deriv
                node.bias_gradient += node.gradient
                
                for edge in self.get_incoming_edges(node):
                    edge.gradient += edge.source.value * node.gradient
                    edge.source.gradient += edge.weight * node.gradient
        
        return loss
    
    def update_weights(self, learning_rate: float) -> None:
        """Update all weights and biases using gradients."""
        for edge in self.edges.values():
            edge.weight -= learning_rate * edge.gradient
        for node in self.nodes.values():
            if not node.is_input:
                node.bias -= learning_rate * node.bias_gradient
    
    def get_weights_snapshot(self) -> Dict:
        """Get current weights for visualization."""
        return {
            "edges": {e.edge_id: {"source": e.source.node_id, "target": e.target.node_id, 
                                    "weight": e.weight, "gradient": e.gradient} 
                      for e in self.edges.values()},
            "nodes": {n.node_id: {"layer": n.layer_idx, "position": n.position,
                                    "value": n.value, "gradient": n.gradient,
                                    "bias": n.bias, "bias_gradient": n.bias_gradient}
                      for n in self.nodes.values()}
        }
    
    def get_topology(self) -> Dict:
        """Get graph topology for visualization."""
        return {
            "layers": {idx: [n.node_id for n in nodes] for idx, nodes in self.layers.items()},
            "edges": [(e.source.node_id, e.target.node_id) for e in self.edges.values()],
            "input_nodes": [n.node_id for n in self.input_nodes],
            "output_nodes": [n.node_id for n in self.output_nodes]
        }