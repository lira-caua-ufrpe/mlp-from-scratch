"""qt_viewer.py - Interface PyQt6 interativa para visualização e treino da MLP como grafo."""

from __future__ import annotations
import sys
import math
import random
import time
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from PyQt6.QtCore import (
    Qt, QTimer, QPointF, QRectF, QEvent, pyqtSignal, QObject, QThread
)
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QFontMetrics, QWheelEvent,
    QMouseEvent, QPaintEvent, QResizeEvent, QAction
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QComboBox,
    QLineEdit, QCheckBox, QGroupBox, QFormLayout, QScrollArea,
    QSplitter, QMessageBox, QFileDialog, QTextEdit, QDialog,
    QDialogButtonBox, QTabWidget
)

from neuron import Neuron
from connection import Connection
from layer import Layer
from network import Network
from loader import CSVLoader
import graph_utils as gu


# ==================== WORKER THREAD PARA TREINO ====================

class TrainingWorker(QObject):
    """Worker para treino em background (não bloqueia GUI)."""
    progress = pyqtSignal(int, int, float, float)  # sample_idx, total, loss, acc
    epoch_done = pyqtSignal(float, float)  # avg_loss, accuracy
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, network: Network, train_x: List[List[float]], train_y: List[float],
                 learning_rate: float, epochs: int = 1):
        super().__init__()
        self.network = network
        self.train_x = train_x
        self.train_y = train_y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._running = True

    def run(self) -> None:
        try:
            for epoch in range(self.epochs):
                if not self._running:
                    break
                total_loss = 0.0
                correct = 0
                for i, (x, y) in enumerate(zip(self.train_x, self.train_y)):
                    if not self._running:
                        break
                    loss = self.network.train_step(x, y, self.learning_rate)
                    total_loss += loss
                    pred = 1 if self.network.get_output()[0] > 0.5 else 0
                    if pred == int(y):
                        correct += 1
                    if i % 10 == 0:
                        self.progress.emit(i, len(self.train_x), loss, correct / (i + 1))
                avg_loss = total_loss / len(self.train_x) if self.train_x else 0.0
                accuracy = correct / len(self.train_x) if self.train_x else 0.0
                self.epoch_done.emit(avg_loss, accuracy)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def stop(self) -> None:
        self._running = False


# ==================== GRAPH VIEW WIDGET ====================

@dataclass
class ViewState:
    """Estado da visualização (zoom, pan)."""
    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    dragging: bool = False
    last_mouse: Tuple[int, int] = (0, 0)


class GraphView(QWidget):
    """Widget customizado para desenhar o grafo da rede."""

    neuron_clicked = pyqtSignal(object)  # Neuron

    def __init__(self, network: Network, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.network = network
        self.view = ViewState()
        self.setMinimumSize(800, 500)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Configurações visuais
        self.show_weights = True
        self.show_gradients = False
        self.show_bias = True
        self.show_feature_names = True
        self.highlighted_neuron: Optional[Neuron] = None
        self.gradient_flash_timer: Optional[QTimer] = None

        # Cache de posições
        self._positions: Dict[str, QPointF] = {}
        self._recompute_positions()

    def _recompute_positions(self) -> None:
        """Recalcula posições dos neurônios baseado no tamanho atual."""
        layer_sizes = [len(l.neurons) for l in self.network.layers]
        pos_dict = gu.compute_layer_positions(
            layer_sizes,
            canvas_width=self.width(),
            canvas_height=self.height(),
            margin=100
        )
        self._positions.clear()
        for layer_idx, layer_positions in pos_dict.items():
            layer = self.network.layers[layer_idx]
            for neuron_idx, (x, y) in enumerate(layer_positions):
                if neuron_idx < len(layer.neurons):
                    neuron = layer.neurons[neuron_idx]
                    self._positions[neuron.neuron_id] = QPointF(x, y)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._recompute_positions()
        super().resizeEvent(event)

    def world_to_screen(self, point: QPointF) -> QPointF:
        return QPointF(
            point.x() * self.view.scale + self.view.offset_x,
            point.y() * self.view.scale + self.view.offset_y
        )

    def screen_to_world(self, point: QPointF) -> QPointF:
        return QPointF(
            (point.x() - self.view.offset_x) / self.view.scale,
            (point.y() - self.view.offset_y) / self.view.scale
        )

    # ---------- Mouse Events ----------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or \
           (event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.view.dragging = True
            self.view.last_mouse = (event.position().x(), event.position().y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            # Clique em neurônio
            world_pos = self.screen_to_world(QPointF(event.position()))
            for neuron_id, pos in self._positions.items():
                screen_pos = self.world_to_screen(pos)
                dist = math.hypot(screen_pos.x() - event.position().x(),
                                  screen_pos.y() - event.position().y())
                if dist < 20 / self.view.scale:
                    # Encontra neurônio
                    for layer in self.network.layers:
                        for n in layer.neurons:
                            if n.neuron_id == neuron_id:
                                self.highlighted_neuron = n
                                self.neuron_clicked.emit(n)
                                self.update()
                                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.view.dragging:
            dx = event.position().x() - self.view.last_mouse[0]
            dy = event.position().y() - self.view.last_mouse[1]
            self.view.offset_x += dx
            self.view.offset_y += dy
            self.view.last_mouse = (event.position().x(), event.position().y())
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or \
           (event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self.view.dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        # Zoom no cursor
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        old_scale = self.view.scale
        self.view.scale *= factor
        self.view.scale = max(0.1, min(5.0, self.view.scale))

        # Ajusta offset para zoom no cursor
        cursor_pos = QPointF(event.position())
        world_before = self.screen_to_world(cursor_pos)
        self.view.offset_x = cursor_pos.x() - world_before.x() * self.view.scale
        self.view.offset_y = cursor_pos.y() - world_before.y() * self.view.scale
        self.update()
        super().wheelEvent(event)

    # ---------- Painting ----------

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Fundo
        painter.fillRect(self.rect(), QColor(245, 245, 250))

        # Desenha conexões
        self._draw_connections(painter)

        # Desenha neurônios
        self._draw_neurons(painter)

        # Legenda
        self._draw_legend(painter)

    def _draw_connections(self, painter: QPainter) -> None:
        connections_data = []
        for conn in self.network.connections:
            source_pos = self._positions.get(conn.source.neuron_id)
            target_pos = self._positions.get(conn.target.neuron_id)
            if source_pos and target_pos:
                connections_data.append({
                    "source": source_pos,
                    "target": target_pos,
                    "weight": conn.weight,
                    "gradient": conn.gradient
                })

        max_w = gu.max_weight_magnitude(connections_data)
        max_g = gu.max_gradient_magnitude(connections_data)

        for conn_data in connections_data:
            src = self.world_to_screen(conn_data["source"])
            dst = self.world_to_screen(conn_data["target"])

            weight = conn_data["weight"]
            gradient = conn_data["gradient"]

            if self.show_gradients and abs(gradient) > 1e-6:
                color = QColor(*gu.gradient_color(gradient, max_g))
                thickness = gu.edge_thickness(gradient, max_g, 2.0, 5.0)
            else:
                color = QColor(*gu.weight_color(weight, max_w))
                thickness = gu.edge_thickness(weight, max_w)

            pen = QPen(color, thickness)
            painter.setPen(pen)
            painter.drawLine(src, dst)

            # Label do peso
            if self.show_weights:
                mid_x = (src.x() + dst.x()) / 2
                mid_y = (src.y() + dst.y()) / 2
                painter.setPen(QPen(QColor(0, 0, 0, 180), 1))
                painter.setFont(QFont("Arial", 7))
                text = gu.format_weight(weight)
                painter.drawText(QPointF(mid_x + 2, mid_y - 2), text)

    def _draw_neurons(self, painter: QPainter) -> None:
        painter.setFont(QFont("Arial", 8))
        fm = QFontMetrics(painter.font())

        for layer_idx, layer in enumerate(self.network.layers):
            is_input = (layer_idx == 0)
            is_output = (layer_idx == len(self.network.layers) - 1)

            for neuron in layer.neurons:
                pos = self._positions.get(neuron.neuron_id)
                if not pos:
                    continue
                screen_pos = self.world_to_screen(pos)

                # Raio do círculo
                radius = 18 / self.view.scale
                radius = max(12, min(28, radius))

                # Cor
                if neuron is self.highlighted_neuron:
                    color = QColor(255, 255, 100)  # Amarelo highlight
                else:
                    r, g, b = gu.neuron_color(neuron.output, is_input, is_output)
                    color = QColor(r, g, b)

                # Desenha círculo
                painter.setPen(QPen(QColor(60, 60, 60), 1.5))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(screen_pos, radius, radius)

                # Bias label
                if self.show_bias:
                    painter.setPen(QPen(QColor(0, 0, 0), 1))
                    bias_text = gu.format_bias(neuron.bias)
                    text_rect = fm.boundingRect(bias_text)
                    painter.drawText(
                        QPointF(screen_pos.x() - text_rect.width() / 2,
                                screen_pos.y() + radius + 14),
                        bias_text
                    )

                # Output value (pequeno)
                if not is_input:
                    painter.setPen(QPen(QColor(80, 80, 80), 1))
                    out_text = f"{neuron.output:.2f}"
                    text_rect = fm.boundingRect(out_text)
                    painter.drawText(
                        QPointF(screen_pos.x() - text_rect.width() / 2,
                                screen_pos.y() - radius - 4),
                        out_text
                    )

        # Feature names (esquerda da input layer)
        if self.show_feature_names and self.network.layers:
            input_layer = self.network.layers[0]
            for neuron in input_layer.neurons:
                pos = self._positions.get(neuron.neuron_id)
                if not pos:
                    continue
                screen_pos = self.world_to_screen(pos)
                if neuron.position_in_layer < len(self.network.loader.feature_names) if hasattr(self.network, 'loader') and self.network.loader else False:
                    feat_name = self.network.loader.feature_names[neuron.position_in_layer] if hasattr(self.network, 'loader') and self.network.loader else f"x{neuron.position_in_layer}"
                else:
                    feat_name = f"x{neuron.position_in_layer}"
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.setFont(QFont("Arial", 7, QFont.Weight.Bold))
                painter.drawText(
                    QPointF(screen_pos.x() - 80, screen_pos.y() + 4),
                    feat_name
                )

        # Output label (direita da output layer)
        if self.network.layers:
            output_layer = self.network.layers[-1]
            for neuron in output_layer.neurons:
                pos = self._positions.get(neuron.neuron_id)
                if not pos:
                    continue
                screen_pos = self.world_to_screen(pos)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
                painter.drawText(
                    QPointF(screen_pos.x() + 25, screen_pos.y() + 4),
                    "ŷ"
                )

    def _draw_legend(self, painter: QPainter) -> None:
        painter.setFont(QFont("Arial", 7))
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        y = 15
        items = [
            ("■ Peso + (azul)", QColor(55, 55, 255)),
            ("■ Peso - (vermelho)", QColor(255, 55, 55)),
            ("■ Grad + (verde)", QColor(55, 255, 55)),
            ("■ Grad - (magenta)", QColor(255, 55, 255)),
            ("● Input", QColor(100, 200, 255)),
            ("● Hidden", QColor(180, 180, 180)),
            ("● Output", QColor(255, 150, 100)),
            ("★ Highlight", QColor(255, 255, 100)),
        ]
        for text, color in items:
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color, 1))
            painter.drawRect(10, y, 10, 10)
            painter.drawText(25, y + 9, text)
            y += 14

    def flash_gradients(self) -> None:
        """Mostra gradientes por 1.2s depois volta para pesos."""
        self.show_gradients = True
        self.update()
        if self.gradient_flash_timer:
            self.gradient_flash_timer.stop()
        self.gradient_flash_timer = QTimer.singleShot(1200, self._end_gradient_flash)

    def _end_gradient_flash(self) -> None:
        self.show_gradients = False
        self.update()

    def set_highlighted_neuron(self, neuron: Optional[Neuron]) -> None:
        self.highlighted_neuron = neuron
        self.update()


# ==================== LOSS HISTORY DIALOG ====================

class LossHistoryDialog(QDialog):
    """Janela modal com histórico textual de loss (últimos 200)."""

    def __init__(self, loss_history: List[float], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Histórico de Loss (últimos 200)")
        self.resize(400, 500)
        layout = QVBoxLayout(self)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFontFamily("Monospace")
        layout.addWidget(self.text_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self.update_history(loss_history)

    def update_history(self, loss_history: List[float]) -> None:
        recent = loss_history[-200:]
        lines = [f"{i+1:4d}: {loss:.6f}" for i, loss in enumerate(recent)]
        self.text_edit.setPlainText("\n".join(lines))


# ==================== MAIN WINDOW ====================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MLP Graph Viewer - Rede Neural como Grafo")
        self.resize(1200, 800)

        # Estado
        self.network: Optional[Network] = None
        self.loader: Optional[CSVLoader] = None
        self.train_x: List[List[float]] = []
        self.train_y: List[float] = []
        self.test_x: List[List[float]] = []
        self.test_y: List[float] = []
        self.means: List[float] = []
        self.stds: List[float] = []
        self.current_sample_idx: int = 0
        self.loss_history: List[float] = []
        self.auto_play_timer: Optional[QTimer] = None
        self.training_worker: Optional[TrainingWorker] = None
        self.training_thread: Optional[QThread] = None

        self._setup_ui()
        self._load_default_dataset()

    def _setup_ui(self) -> None:
        # Central widget com splitter
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # LEFT: Graph View
        self.graph_view = GraphView(Network([2, 1]))  # placeholder
        self.graph_view.neuron_clicked.connect(self._on_neuron_clicked)
        graph_container = QWidget()
        graph_layout = QVBoxLayout(graph_container)
        graph_layout.addWidget(QLabel("Visualização do Grafo (scroll=zoom, arraste=pan, clique=highlight)"))
        graph_layout.addWidget(self.graph_view)
        splitter.addWidget(graph_container)

        # RIGHT: Controls
        controls = QWidget()
        controls.setFixedWidth(350)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(5, 5, 5, 5)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._create_controls_panel())
        controls_layout.addWidget(scroll)

        splitter.addWidget(controls)
        splitter.setSizes([850, 350])

        # Status bar
        self.statusBar().showMessage("Pronto")

    def _create_controls_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # --- Dataset ---
        dataset_group = QGroupBox("Dataset")
        dlayout = QVBoxLayout(dataset_group)

        self.btn_load_heart = QPushButton("Carregar heart.csv")
        self.btn_load_heart.clicked.connect(lambda: self._load_dataset("rsc/heart.csv"))
        dlayout.addWidget(self.btn_load_heart)

        self.btn_load_diabetes = QPushButton("Carregar diabetes.csv")
        self.btn_load_diabetes.clicked.connect(lambda: self._load_dataset("rsc/diabetes.csv"))
        dlayout.addWidget(self.btn_load_diabetes)

        self.btn_load_custom = QPushButton("Carregar CSV personalizado...")
        self.btn_load_custom.clicked.connect(self._load_custom_csv)
        dlayout.addWidget(self.btn_load_custom)

        self.dataset_info = QLabel("Nenhum dataset carregado")
        self.dataset_info.setWordWrap(True)
        dlayout.addWidget(self.dataset_info)

        layout.addWidget(dataset_group)

        # --- Architecture ---
        arch_group = QGroupBox("Arquitetura")
        alayout = QFormLayout(arch_group)

        self.spin_hidden_layers = QSpinBox()
        self.spin_hidden_layers.setRange(0, 5)
        self.spin_hidden_layers.setValue(1)
        self.spin_hidden_layers.valueChanged.connect(self._on_arch_change)
        alayout.addRow("Camadas ocultas:", self.spin_hidden_layers)

        self.spin_hidden_neurons = QSpinBox()
        self.spin_hidden_neurons.setRange(1, 64)
        self.spin_hidden_neurons.setValue(8)
        self.spin_hidden_neurons.valueChanged.connect(self._on_arch_change)
        alayout.addRow("Neurônios/camada:", self.spin_hidden_neurons)

        self.combo_activation = QComboBox()
        self.combo_activation.addItems(["sigmoid", "relu", "identity"])
        self.combo_activation.currentTextChanged.connect(self._on_activation_change)
        alayout.addRow("Ativação:", self.combo_activation)

        self.btn_rebuild = QPushButton("Reconstruir Rede")
        self.btn_rebuild.clicked.connect(self._rebuild_network)
        alayout.addRow(self.btn_rebuild)

        layout.addWidget(arch_group)

        # --- Sample Input ---
        input_group = QGroupBox("Entrada (Manual)")
        ilayout = QVBoxLayout(input_group)

        self.edit_input = QLineEdit()
        self.edit_input.setPlaceholderText("Valores separados por vírgula (ex: 0.5, -0.2, 1.0...)")
        ilayout.addWidget(self.edit_input)

        self.btn_forward = QPushButton("▶ Forward Pass (completo)")
        self.btn_forward.clicked.connect(self._do_forward)
        ilayout.addWidget(self.btn_forward)

        self.btn_forward_step = QPushButton("Passo a Passo (1 neurônio)")
        self.btn_forward_step.clicked.connect(self._do_forward_step)
        ilayout.addWidget(self.btn_forward_step)

        self.btn_auto_play = QPushButton("▶ Auto Play (passo a passo)")
        self.btn_auto_play.setCheckable(True)
        self.btn_auto_play.toggled.connect(self._toggle_auto_play)
        ilayout.addWidget(self.btn_auto_play)

        self.spin_auto_speed = QSpinBox()
        self.spin_auto_speed.setRange(50, 2000)
        self.spin_auto_speed.setValue(300)
        self.spin_auto_speed.setSuffix(" ms")
        self.spin_auto_speed.valueChanged.connect(self._update_auto_speed)
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Velocidade:"))
        speed_layout.addWidget(self.spin_auto_speed)
        ilayout.addLayout(speed_layout)

        layout.addWidget(input_group)

        # --- Dataset Navigation ---
        nav_group = QGroupBox("Navegação Dataset")
        nlayout = QVBoxLayout(nav_group)

        self.btn_next_sample = QPushButton("Próximo Sample")
        self.btn_next_sample.clicked.connect(self._next_sample)
        nlayout.addWidget(self.btn_next_sample)

        self.btn_prev_sample = QPushButton("Sample Anterior")
        self.btn_prev_sample.clicked.connect(self._prev_sample)
        nlayout.addWidget(self.btn_prev_sample)

        self.btn_fast_forward = QPushButton("Fast Forward (todos samples)")
        self.btn_fast_forward.clicked.connect(self._fast_forward_all)
        nlayout.addWidget(self.btn_fast_forward)

        self.sample_info = QLabel("Sample: -/-")
        nlayout.addWidget(self.sample_info)

        layout.addWidget(nav_group)

        # --- Training ---
        train_group = QGroupBox("Treino")
        tlayout = QFormLayout(train_group)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setValue(0.1)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setSingleStep(0.01)
        tlayout.addRow("Learning Rate:", self.spin_lr)

        self.btn_train_step = QPushButton("Train Step (1 sample)")
        self.btn_train_step.clicked.connect(self._train_step)
        tlayout.addRow(self.btn_train_step)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 100)
        self.spin_epochs.setValue(1)
        tlayout.addRow("Épocas:", self.spin_epochs)

        self.btn_train_epoch = QPushButton("Train Época")
        self.btn_train_epoch.clicked.connect(self._train_epoch)
        tlayout.addRow(self.btn_train_epoch)

        self.btn_stop_training = QPushButton("Parar Treino")
        self.btn_stop_training.clicked.connect(self._stop_training)
        self.btn_stop_training.setEnabled(False)
        tlayout.addRow(self.btn_stop_training)

        self.btn_loss_history = QPushButton("Ver Loss (últimos 200)")
        self.btn_loss_history.clicked.connect(self._show_loss_history)
        tlayout.addRow(self.btn_loss_history)

        self.train_status = QLabel("Loss: - | Acc: -")
        tlayout.addRow(self.train_status)

        layout.addWidget(train_group)

        # --- Visualization Options ---
        vis_group = QGroupBox("Visualização")
        vlayout = QVBoxLayout(vis_group)

        self.chk_show_weights = QCheckBox("Mostrar pesos nas arestas")
        self.chk_show_weights.setChecked(True)
        self.chk_show_weights.toggled.connect(lambda v: setattr(self.graph_view, 'show_weights', v) or self.graph_view.update())
        vlayout.addWidget(self.chk_show_weights)

        self.chk_show_bias = QCheckBox("Mostrar bias nos neurônios")
        self.chk_show_bias.setChecked(True)
        self.chk_show_bias.toggled.connect(lambda v: setattr(self.graph_view, 'show_bias', v) or self.graph_view.update())
        vlayout.addWidget(self.chk_show_bias)

        self.chk_show_features = QCheckBox("Mostrar nomes das features")
        self.chk_show_features.setChecked(True)
        self.chk_show_features.toggled.connect(lambda v: setattr(self.graph_view, 'show_feature_names', v) or self.graph_view.update())
        vlayout.addWidget(self.chk_show_features)

        layout.addWidget(vis_group)

        layout.addStretch()
        return panel

    # ==================== DATASET LOADING ====================

    def _load_default_dataset(self) -> None:
        # Tenta carregar heart.csv se existir
        import os
        if os.path.exists("rsc/heart.csv"):
            self._load_dataset("rsc/heart.csv")

    def _load_dataset(self, filepath: str) -> None:
        try:
            # heart.csv não tem header; diabetes.csv tem
            has_header = "diabetes" in filepath.lower()
            self.loader = CSVLoader(filepath, has_header=has_header)
            features, targets = self.loader.load()

            # Split
            self.train_x, self.train_y, self.test_x, self.test_y = \
                self.loader.split_train_test(features, targets, test_ratio=0.2)

            # Normaliza
            self.train_x, self.test_x, self.means, self.stds = \
                self.loader.normalize_features(self.train_x, self.test_x)

            self.current_sample_idx = 0
            self._rebuild_network()
            self._update_dataset_info()
            self._update_sample_info()
            self.statusBar().showMessage(f"Carregado: {filepath} ({len(self.train_x)} treino, {len(self.test_x)} teste)")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao carregar dataset:\n{e}")

    def _load_custom_csv(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(self, "Selecionar CSV", "", "CSV Files (*.csv)")
        if filepath:
            self._load_dataset(filepath)

    def _update_dataset_info(self) -> None:
        if self.loader:
            info = self.loader.get_info()
            text = (f"Features: {info['num_features']}\n"
                    f"Samples: {info['num_samples']}\n"
                    f"Target: {info['target_name']}\n"
                    f"Treino: {len(self.train_x)} | Teste: {len(self.test_x)}")
            self.dataset_info.setText(text)

    # ==================== NETWORK MANAGEMENT ====================

    def _rebuild_network(self) -> None:
        if not self.loader:
            return

        hidden_layers = self.spin_hidden_layers.value()
        hidden_neurons = self.spin_hidden_neurons.value()
        activation = self.combo_activation.currentText()

        layer_sizes = [self.loader.num_features] + [hidden_neurons] * hidden_layers + [1]

        self.network = Network(layer_sizes, activation=activation, seed=42)
        # Anexa loader para nomes de features
        self.network.loader = self.loader

        # Atualiza graph view
        self.graph_view.network = self.network
        self.graph_view._recompute_positions()
        self.graph_view.update()

        self.statusBar().showMessage(f"Rede reconstruída: {self.network.topology_summary()}")

    def _on_arch_change(self) -> None:
        self._rebuild_network()

    def _on_activation_change(self) -> None:
        if self.network:
            self.network.activation_name = self.combo_activation.currentText()
            self.graph_view.update()

    # ==================== SAMPLE NAVIGATION ====================

    def _update_sample_info(self) -> None:
        if self.train_x:
            self.sample_info.setText(f"Sample: {self.current_sample_idx + 1}/{len(self.train_x)}")
            # Preenche input manual
            x = self.train_x[self.current_sample_idx]
            self.edit_input.setText(", ".join(f"{v:.3f}" for v in x))

    def _next_sample(self) -> None:
        if self.train_x:
            self.current_sample_idx = (self.current_sample_idx + 1) % len(self.train_x)
            self._update_sample_info()
            self._do_forward()

    def _prev_sample(self) -> None:
        if self.train_x:
            self.current_sample_idx = (self.current_sample_idx - 1) % len(self.train_x)
            self._update_sample_info()
            self._do_forward()

    # ==================== FORWARD ====================

    def _do_forward(self) -> None:
        if not self.network or not self.train_x:
            return
        x = self.train_x[self.current_sample_idx]
        y = self.train_y[self.current_sample_idx]

        self.network.reset_states()
        self.network.set_input(x)
        while True:
            processed = self.network.forward_step()
            if not processed:
                break
            # Atualiza highlight do último processado
            if processed:
                self.graph_view.set_highlighted_neuron(processed[-1])
                self.graph_view.update()
                QApplication.processEvents()

        output = self.network.get_output()[0]
        target = y
        error = abs(output - target)
        self.statusBar().showMessage(f"Forward: out={output:.3f} target={target} error={error:.3f}")

    def _do_forward_step(self) -> None:
        if not self.network or not self.train_x:
            return
        x = self.train_x[self.current_sample_idx]

        # Se rede está em estado inicial, faz set_input
        if all(n.output == 0 for layer in self.network.layers[1:] for n in layer.neurons):
            self.network.reset_states()
            self.network.set_input(x)

        processed = self.network.forward_step()
        if processed:
            self.graph_view.set_highlighted_neuron(processed[-1])
            self.graph_view.update()
            # Se terminou, mostra resultado
            if not any(n.output == 0 for layer in self.network.layers[1:] for n in layer.neurons):
                output = self.network.get_output()[0]
                target = self.train_y[self.current_sample_idx]
                self.statusBar().showMessage(f"Forward completo: out={output:.3f} target={target}")

    def _toggle_auto_play(self, checked: bool) -> None:
        if checked:
            self.btn_auto_play.setText("⏸ Parar Auto Play")
            self._auto_play_step()
        else:
            self.btn_auto_play.setText("▶ Auto Play (passo a passo)")
            if self.auto_play_timer:
                self.auto_play_timer.stop()

    def _auto_play_step(self) -> None:
        if not self.btn_auto_play.isChecked():
            return
        if not self.network or not self.train_x:
            self._toggle_auto_play(False)
            return

        # Se precisa resetar
        if all(n.output == 0 for layer in self.network.layers[1:] for n in layer.neurons):
            x = self.train_x[self.current_sample_idx]
            self.network.reset_states()
            self.network.set_input(x)

        processed = self.network.forward_step()
        if processed:
            self.graph_view.set_highlighted_neuron(processed[-1])
            self.graph_view.update()

        # Verifica se terminou
        if not any(n.output == 0 for layer in self.network.layers[1:] for n in layer.neurons):
            output = self.network.get_output()[0]
            target = self.train_y[self.current_sample_idx]
            self.statusBar().showMessage(f"Auto: out={output:.3f} target={target}")
            # Próximo sample
            self.current_sample_idx = (self.current_sample_idx + 1) % len(self.train_x)
            self._update_sample_info()
            self.network.reset_states()

        # Agenda próximo passo
        self.auto_play_timer = QTimer.singleShot(self.spin_auto_speed.value(), self._auto_play_step)

    def _update_auto_speed(self, value: int) -> None:
        # Timer será recriado no próximo step
        pass

    def _fast_forward_all(self) -> None:
        if not self.network or not self.test_x:
            return
        correct = 0
        for x, y in zip(self.test_x, self.test_y):
            self.network.forward_all(x)
            pred = 1 if self.network.get_output()[0] > 0.5 else 0
            if pred == int(y):
                correct += 1
        acc = correct / len(self.test_x)
        self.statusBar().showMessage(f"Fast Forward Teste: Acurácia = {acc:.2%} ({correct}/{len(self.test_x)})")
        QMessageBox.information(self, "Fast Forward", f"Acurácia no conjunto de teste: {acc:.2%}")

    # ==================== TRAINING ====================

    def _train_step(self) -> None:
        if not self.network or not self.train_x:
            return
        x = self.train_x[self.current_sample_idx]
        y = self.train_y[self.current_sample_idx]
        lr = self.spin_lr.value()

        self.network.reset_states()
        self.network.set_input(x)
        # Forward completo
        while self.network.forward_step():
            pass
        loss = self.network.backprop(y)
        self.network.update_weights(lr)
        self.loss_history.append(loss)

        # Flash gradientes
        self.graph_view.flash_gradients()

        # Atualiza status
        output = self.network.get_output()[0]
        self.train_status.setText(f"Loss: {loss:.6f} | Out: {output:.3f} | Target: {y}")
        self.graph_view.update()

    def _train_epoch(self) -> None:
        if not self.network or not self.train_x:
            return

        epochs = self.spin_epochs.value()
        lr = self.spin_lr.value()

        self.btn_train_epoch.setEnabled(False)
        self.btn_train_step.setEnabled(False)
        self.btn_stop_training.setEnabled(True)
        self.btn_auto_play.setEnabled(False)

        self.training_worker = TrainingWorker(self.network, self.train_x, self.train_y, lr, epochs)
        self.training_thread = QThread()
        self.training_worker.moveToThread(self.training_thread)

        self.training_worker.epoch_done.connect(self._on_epoch_done)
        self.training_worker.progress.connect(self._on_train_progress)
        self.training_worker.finished.connect(self._on_training_finished)
        self.training_worker.error.connect(self._on_training_error)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_thread.finished.connect(self.training_thread.deleteLater)

        self.training_thread.start()

    def _on_train_progress(self, sample_idx: int, total: int, loss: float, acc: float) -> None:
        self.train_status.setText(f"Sample {sample_idx+1}/{total} | Loss: {loss:.4f} | Acc: {acc:.2%}")

    def _on_epoch_done(self, avg_loss: float, accuracy: float) -> None:
        self.loss_history.append(avg_loss)
        self.train_status.setText(f"Época concluída | Loss médio: {avg_loss:.4f} | Acc: {accuracy:.2%}")

    def _on_training_finished(self) -> None:
        self.btn_train_epoch.setEnabled(True)
        self.btn_train_step.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.btn_auto_play.setEnabled(True)
        if self.training_thread:
            self.training_thread.quit()
            self.training_thread.wait()
        self.statusBar().showMessage("Treino finalizado")

    def _stop_training(self) -> None:
        if self.training_worker:
            self.training_worker.stop()

    def _on_training_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Erro no Treino", msg)
        self._on_training_finished()

    def _show_loss_history(self) -> None:
        if not self.loss_history:
            QMessageBox.information(self, "Loss History", "Nenhum histórico de loss ainda.")
            return
        dialog = LossHistoryDialog(self.loss_history, self)
        dialog.exec()

    # ==================== NEURON CLICK ====================

    def _on_neuron_clicked(self, neuron: Neuron) -> None:
        self.statusBar().showMessage(
            f"Neuron {neuron.neuron_id} | Layer {neuron.layer_index} | Pos {neuron.position_in_layer} | "
            f"Bias: {neuron.bias:.3f} | Output: {neuron.output:.3f} | Delta: {neuron.delta:.3f}"
        )


# ==================== ENTRY POINT ====================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Look consistente cross-platform

    # Fonte padrão
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()