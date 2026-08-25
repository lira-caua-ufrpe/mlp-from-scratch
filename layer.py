"""Layer - agrupamento de neurônios com operações de adição/remoção."""

from __future__ import annotations
from typing import List, Optional
from neuron import Neuron


class Layer:
    """Camada contendo uma lista de neurônios."""

    def __init__(self, layer_index: int) -> None:
        self.layer_index = layer_index
        self.neurons: List[Neuron] = []

    def add_neuron(self, bias: float = 0.0) -> Neuron:
        """Adiciona neurônio no final da camada."""
        position = len(self.neurons)
        neuron = Neuron(
            bias=bias, layer_index=self.layer_index, position_in_layer=position
        )
        self.neurons.append(neuron)
        return neuron

    def add_neurons(self, count: int, bias: float = 0.0) -> List[Neuron]:
        """Adiciona múltiplos neurônios."""
        return [self.add_neuron(bias) for _ in range(count)]

    def remove_neuron(self, position: int) -> Optional[Neuron]:
        """Remove neurônio na posição (se não for input/output restrito)."""
        if 0 <= position < len(self.neurons):
            removed = self.neurons.pop(position)
            self._reindex()
            return removed
        return None

    def remove_last_neuron(self) -> Optional[Neuron]:
        """Remove último neurônio da camada."""
        if self.neurons:
            removed = self.neurons.pop()
            self._reindex()
            return removed
        return None

    def _reindex(self) -> None:
        """Atualiza position_in_layer após remoção."""
        for idx, neuron in enumerate(self.neurons):
            neuron.position_in_layer = idx

    def get_neuron(self, position: int) -> Optional[Neuron]:
        if 0 <= position < len(self.neurons):
            return self.neurons[position]
        return None

    def size(self) -> int:
        return len(self.neurons)

    def reset_states(self) -> None:
        for neuron in self.neurons:
            neuron.reset_state()

    def __repr__(self) -> str:
        return f"Layer(index={self.layer_index}, neurons={len(self.neurons)})"
