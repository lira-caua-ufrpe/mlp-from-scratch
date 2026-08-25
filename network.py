"""Network - MLP como grafo explícito com forward, backprop e treino."""

from __future__ import annotations
from typing import List, Optional, Callable
import math
import random

from neuron import Neuron
from connection import Connection
from layer import Layer


class Network:
    """
    Rede neural multicamada representada como grafo explícito.
    - Camadas: lista de Layer
    - Conexões: todas as arestas source->target entre camadas adjacentes
    - Forward passo-a-passo ou completo
    - Backpropagation para 1 neurônio de saída (MSE)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "sigmoid",
        weight_range: tuple = (-1.0, 1.0),
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            layer_sizes: Tamanho de cada camada [input, hidden..., output]
            activation: "sigmoid", "relu", "identity"
            weight_range: (min, max) para inicialização dos pesos
            seed: Semente para reprodutibilidade
        """
        if seed is not None:
            random.seed(seed)

        if len(layer_sizes) < 2:
            raise ValueError("Pelo menos 2 camadas (input e output)")

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.weight_range = weight_range
        self.layers: List[Layer] = []
        self.connections: List[Connection] = []
        self.last_gradients: List[tuple] = (
            []
        )  # (connection, gradient) para visualização

        self._build_layers()
        self._build_connections()

    def _build_layers(self) -> None:
        """Cria camadas e neurônios conforme layer_sizes."""
        for idx, size in enumerate(self.layer_sizes):
            layer = Layer(layer_index=idx)
            layer.add_neurons(size)
            self.layers.append(layer)

    def _build_connections(self) -> None:
        """Conecta totalmente cada camada à próxima (adjacente)."""
        self.connections.clear()
        for i in range(len(self.layers) - 1):
            source_layer = self.layers[i]
            target_layer = self.layers[i + 1]
            for source_neuron in source_layer.neurons:
                for target_neuron in target_layer.neurons:
                    conn = Connection(source=source_neuron, target=target_neuron)
                    # Re-inicializa peso no range especificado
                    conn.weight = random.uniform(*self.weight_range)
                    self.connections.append(conn)

    # ==================== ATIVAÇÕES ====================

    def _activate(self, x: float) -> float:
        if self.activation_name == "sigmoid":
            return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))
        elif self.activation_name == "relu":
            return max(0.0, x)
        elif self.activation_name == "identity":
            return x
        else:
            raise ValueError(f"Ativação desconhecida: {self.activation_name}")

    def _activate_derivative(self, output: float) -> float:
        """Derivada da ativação calculada a partir da saída (output)."""
        if self.activation_name == "sigmoid":
            return output * (1.0 - output)
        elif self.activation_name == "relu":
            return 1.0 if output > 0 else 0.0
        elif self.activation_name == "identity":
            return 1.0
        return 1.0

    # ==================== FORWARD ====================

    def set_input(self, values: List[float]) -> None:
        """Define valores de entrada na camada de input."""
        input_layer = self.layers[0]
        if len(values) != len(input_layer.neurons):
            raise ValueError(
                f"Esperado {len(input_layer.neurons)} entradas, recebido {len(values)}"
            )
        for neuron, val in zip(input_layer.neurons, values):
            neuron.output = float(val)

    def forward_step(self) -> List[Neuron]:
        """
        Executa UM passo de forward: processa a próxima camada não processada.
        Retorna lista de neurônios processados neste passo.
        """
        # Encontra primeira camada com neurônios não processados (output == 0 e não é input)
        for layer in self.layers[1:]:  # pula input
            unprocessed = [n for n in layer.neurons if n.output == 0.0]
            if unprocessed:
                processed = []
                for neuron in unprocessed:
                    # Soma entradas ponderadas + bias
                    total = neuron.bias
                    for conn in self._incoming_connections(neuron):
                        total += conn.source.output * conn.weight
                    neuron.input_sum = total
                    neuron.output = self._activate(total)
                    processed.append(neuron)
                return processed
        return []  # Todas processadas

    def forward_all(self, inputs: List[float]) -> List[float]:
        """Forward completo: set input + processa todas camadas."""
        self.reset_states()
        self.set_input(inputs)
        while True:
            processed = self.forward_step()
            if not processed:
                break
        return self.get_output()

    def forward_neuron_by_neuron(
        self, inputs: List[float], callback: Optional[Callable] = None
    ) -> List[float]:
        """
        Forward passo-a-passo neurônio a neurônio (para visualização).
        Callback(neuron) chamado após cada neurônio processado.
        """
        self.reset_states()
        self.set_input(inputs)
        while True:
            processed = self.forward_step()
            if not processed:
                break
            if callback:
                for n in processed:
                    callback(n)
        return self.get_output()

    def get_output(self) -> List[float]:
        """Retorna outputs da camada de saída."""
        return [n.output for n in self.layers[-1].neurons]

    def _incoming_connections(self, neuron: Neuron) -> List[Connection]:
        return [c for c in self.connections if c.target is neuron]

    def _outgoing_connections(self, neuron: Neuron) -> List[Connection]:
        return [c for c in self.connections if c.source is neuron]

    def reset_states(self) -> None:
        for layer in self.layers:
            layer.reset_states()
        for conn in self.connections:
            conn.reset_gradient()
        self.last_gradients.clear()

    # ==================== BACKPROP (1 output neuron) ====================

    def backprop(self, target: float) -> float:
        """
        Backpropagation para 1 neurônio de saída usando MSE.
        loss = 0.5 * (output - target)^2
        Retorna o loss.
        """
        output_neuron = self.layers[-1].neurons[0]
        output = output_neuron.output

        # Loss MSE
        loss = 0.5 * (output - target) ** 2

        # Delta do output: dL/doutput * doutput/dinput_sum
        # dL/doutput = (output - target)
        # doutput/dinput_sum = activation_derivative(output)
        output_neuron.delta = (output - target) * self._activate_derivative(output)

        # Propaga deltas para trás (camadas ocultas -> input)
        for layer in reversed(self.layers[1:-1]):  # exclui input e output
            for neuron in layer.neurons:
                # Soma: delta = sum( w * delta_target ) * activation_derivative(output)
                delta_sum = 0.0
                for conn in self._outgoing_connections(neuron):
                    delta_sum += conn.weight * conn.target.delta
                neuron.delta = delta_sum * self._activate_derivative(neuron.output)

        # Calcula gradientes dos pesos: dL/dw = source.output * target.delta
        self.last_gradients.clear()
        for conn in self.connections:
            conn.gradient = conn.source.output * conn.target.delta
            self.last_gradients.append((conn, conn.gradient))

        return loss

    def update_weights(self, learning_rate: float) -> None:
        """Atualiza pesos e biases usando gradientes calculados."""
        for conn in self.connections:
            conn.weight -= learning_rate * conn.gradient
        for layer in self.layers[1:]:  # não atualiza bias da input
            for neuron in layer.neurons:
                neuron.bias -= learning_rate * neuron.delta

    # ==================== TREINO ====================

    def train_step(
        self, inputs: List[float], target: float, learning_rate: float
    ) -> float:
        """Um passo de treino: forward + backprop + update."""
        self.forward_all(inputs)
        loss = self.backprop(target)
        self.update_weights(learning_rate)
        return loss

    def train_epoch(
        self,
        dataset_inputs: List[List[float]],
        dataset_targets: List[float],
        learning_rate: float,
    ) -> tuple:
        """
        Treina uma época completa no dataset.
        Retorna (loss_medio, acuracia) com threshold 0.5.
        """
        total_loss = 0.0
        correct = 0
        for inputs, target in zip(dataset_inputs, dataset_targets):
            loss = self.train_step(inputs, target, learning_rate)
            total_loss += loss
            pred = 1 if self.get_output()[0] > 0.5 else 0
            if pred == int(target):
                correct += 1
        avg_loss = total_loss / len(dataset_inputs) if dataset_inputs else 0.0
        accuracy = correct / len(dataset_inputs) if dataset_inputs else 0.0
        return avg_loss, accuracy

    # ==================== ARQUITETURA DINÂMICA ====================

    def add_hidden_layer(self, position: int, num_neurons: int) -> Layer:
        """Adiciona camada oculta na posição (entre input e output)."""
        if position < 1 or position >= len(self.layers):
            raise ValueError("Posição deve ser entre 1 e len(layers)-1")
        new_layer = Layer(layer_index=position)
        new_layer.add_neurons(num_neurons)
        self.layers.insert(position, new_layer)
        self._reindex_layers()
        self._rebuild_connections()
        return new_layer

    def remove_hidden_layer(self, position: int) -> bool:
        """Remove camada oculta na posição."""
        if position < 1 or position >= len(self.layers) - 1:
            return False
        self.layers.pop(position)
        self._reindex_layers()
        self._rebuild_connections()
        return True

    def add_neuron_to_hidden_layer(self, layer_index: int) -> Optional[Neuron]:
        """Adiciona neurônio a camada oculta."""
        if layer_index <= 0 or layer_index >= len(self.layers) - 1:
            return None
        neuron = self.layers[layer_index].add_neuron()
        self._rebuild_connections()
        return neuron

    def remove_neuron_from_hidden_layer(self, layer_index: int, position: int) -> bool:
        """Remove neurônio de camada oculta."""
        if layer_index <= 0 or layer_index >= len(self.layers) - 1:
            return False
        removed = self.layers[layer_index].remove_neuron(position)
        if removed:
            self._rebuild_connections()
            return True
        return False

    def _reindex_layers(self) -> None:
        for idx, layer in enumerate(self.layers):
            layer.layer_index = idx
            for pos, neuron in enumerate(layer.neurons):
                neuron.layer_index = idx
                neuron.position_in_layer = pos

    def _rebuild_connections(self) -> None:
        self.connections.clear()
        for i in range(len(self.layers) - 1):
            source_layer = self.layers[i]
            target_layer = self.layers[i + 1]
            for source_neuron in source_layer.neurons:
                for target_neuron in target_layer.neurons:
                    conn = Connection(source=source_neuron, target=target_neuron)
                    conn.weight = random.uniform(*self.weight_range)
                    self.connections.append(conn)

    # ==================== INSPEÇÃO ====================

    def topology_summary(self) -> str:
        """Resumo textual da topologia."""
        lines = []
        lines.append(f"layers: {', '.join(str(len(layer.neurons)) for layer in self.layers)}")
        lines.append(f"connections: {len(self.connections)}")
        lines.append(f"activation: {self.activation_name}")
        return " | ".join(lines)

    def get_topology_data(self) -> dict:
        """Dados para visualização (graph_utils/qt_viewer)."""
        return {
            "layers": [
                {
                    "index": layer.layer_index,
                    "neurons": [
                        {
                            "id": n.neuron_id,
                            "position": n.position_in_layer,
                            "bias": n.bias,
                            "output": n.output,
                            "delta": n.delta,
                            "input_sum": n.input_sum,
                        }
                        for n in layer.neurons
                    ],
                }
                for layer in self.layers
            ],
            "connections": [
                {
                    "source_id": c.source.neuron_id,
                    "target_id": c.target.neuron_id,
                    "weight": c.weight,
                    "gradient": c.gradient,
                }
                for c in self.connections
            ],
            "activation": self.activation_name,
        }

    def __repr__(self) -> str:
        return f"Network({self.topology_summary()})"
