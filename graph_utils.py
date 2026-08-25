"""graph_utils - Utilitários para visualização do grafo (posições, cores, espessuras)."""

from __future__ import annotations
from typing import List, Dict, Tuple


def compute_layer_positions(
    layer_sizes: List[int],
    canvas_width: float = 800,
    canvas_height: float = 600,
    margin: float = 80,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Calcula posições (x, y) para cada neurônio em cada camada.
    Layout: camadas horizontais (x), neurônios verticais (y) centralizados.
    """
    positions = {}
    num_layers = len(layer_sizes)

    # Espaçamento horizontal
    usable_width = canvas_width - 2 * margin
    layer_spacing = usable_width / max(1, num_layers - 1)

    for layer_idx, size in enumerate(layer_sizes):
        x = margin + layer_idx * layer_spacing
        # Espaçamento vertical
        usable_height = canvas_height - 2 * margin
        if size == 1:
            neuron_spacing = 0
            start_y = canvas_height / 2
        else:
            neuron_spacing = usable_height / (size - 1)
            start_y = margin

        layer_positions = []
        for neuron_idx in range(size):
            y = start_y + neuron_idx * neuron_spacing
            layer_positions.append((x, y))
        positions[layer_idx] = layer_positions

    return positions


def weight_color(weight: float, max_abs: float = 1.0) -> Tuple[int, int, int]:
    """
    Cor baseada no peso: azul (positivo) -> vermelho (negativo).
    Intensidade proporcional à magnitude.
    """
    if max_abs == 0:
        max_abs = 1.0
    normalized = max(-1.0, min(1.0, weight / max_abs))
    intensity = int(200 * abs(normalized)) + 55  # 55-255
    if normalized >= 0:
        return (55, 55, intensity)  # Azul
    else:
        return (intensity, 55, 55)  # Vermelho


def gradient_color(gradient: float, max_abs: float = 1.0) -> Tuple[int, int, int]:
    """
    Cor para gradientes: verde (positivo) -> magenta (negativo).
    """
    if max_abs == 0:
        max_abs = 1.0
    normalized = max(-1.0, min(1.0, gradient / max_abs))
    intensity = int(200 * abs(normalized)) + 55
    if normalized >= 0:
        return (55, intensity, 55)  # Verde
    else:
        return (intensity, 55, intensity)  # Magenta


def edge_thickness(
    weight: float, max_abs: float = 1.0, min_t: float = 1.0, max_t: float = 4.0
) -> float:
    """Espessura da aresta proporcional à magnitude do peso."""
    if max_abs == 0:
        max_abs = 1.0
    normalized = abs(weight) / max_abs
    return min_t + (max_t - min_t) * normalized


def max_weight_magnitude(connections: List[dict]) -> float:
    """Encontra magnitude máxima de peso para normalização."""
    if not connections:
        return 1.0
    return max(abs(c.get("weight", 0)) for c in connections)


def max_gradient_magnitude(connections: List[dict]) -> float:
    """Encontra magnitude máxima de gradiente para normalização."""
    if not connections:
        return 1.0
    return max(abs(c.get("gradient", 0)) for c in connections)


def neuron_color(
    output: float, is_input: bool = False, is_output: bool = False
) -> Tuple[int, int, int]:
    """Cor do neurônio baseada na saída (azul/vermelho) ou tipo."""
    if is_input:
        return (100, 200, 255)  # Azul claro para input
    if is_output:
        return (255, 150, 100)  # Laranja para output
    # Hidden: baseado no output (sigmoid 0-1, relu 0+)
    intensity = int(200 * max(0.0, min(1.0, output))) + 55
    return (intensity, intensity, intensity)  # Escala de cinza


def format_weight(value: float) -> str:
    """Formata peso para exibição."""
    if abs(value) < 0.001:
        return "0.000"
    return f"{value:.3f}"


def format_bias(value: float) -> str:
    """Formata bias para exibição."""
    return f"b={value:.2f}"
