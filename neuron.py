"""Neuron - nó fundamental da rede neural como grafo."""

from __future__ import annotations
from dataclasses import dataclass, field
import uuid


@dataclass
class Neuron:
    """Representa um neurônio (nó) no grafo da rede neural."""

    bias: float = 0.0
    input_sum: float = 0.0
    output: float = 0.0
    delta: float = 0.0
    neuron_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    layer_index: int = -1
    position_in_layer: int = -1

    def reset_state(self) -> None:
        """Reseta valores temporários de forward/backward."""
        self.input_sum = 0.0
        self.output = 0.0
        self.delta = 0.0

    def __repr__(self) -> str:
        return f"Neuron(id={self.neuron_id}, layer={self.layer_index}, pos={self.position_in_layer}, bias={self.bias:.3f})"
