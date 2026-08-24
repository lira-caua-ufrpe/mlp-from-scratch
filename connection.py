"""Connection - aresta direcionada entre neurônios com peso."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import random
import uuid

if TYPE_CHECKING:
    from neuron import Neuron


@dataclass
class Connection:
    """Conexão direcionada source -> target com peso inicial aleatório."""
    source: Neuron
    target: Neuron
    weight: float = field(default_factory=lambda: random.uniform(-1.0, 1.0))
    gradient: float = 0.0
    connection_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def reset_gradient(self) -> None:
        self.gradient = 0.0

    def __repr__(self) -> str:
        return f"Connection({self.source.neuron_id}->{self.target.neuron_id}, w={self.weight:.3f}, grad={self.gradient:.3f})"