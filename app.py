"""Streamlit app for MLP from Scratch - Interactive Training & Visualization.

Uses Professor's Network implementation (graph-based, step-by-step forward, MSE backprop).
"""

import sys

sys.path.insert(0, ".")  # noqa: E402

import streamlit as st  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import time  # noqa: E402

from network import Network  # noqa: E402
from loader import Loader  # noqa: E402

# Page config
st.set_page_config(
    page_title="MLP from Scratch",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
if "network" not in st.session_state:
    st.session_state.network = None
if "loader" not in st.session_state:
    st.session_state.loader = None
if "train_x" not in st.session_state:
    st.session_state.train_x = None
if "train_y" not in st.session_state:
    st.session_state.train_y = None
if "test_x" not in st.session_state:
    st.session_state.test_x = None
if "test_y" not in st.session_state:
    st.session_state.test_y = None
if "training_history" not in st.session_state:
    st.session_state.training_history = {"loss": [], "acc": []}
if "current_sample_idx" not in st.session_state:
    st.session_state.current_sample_idx = 0
if "is_training" not in st.session_state:
    st.session_state.is_training = False


def reset_training_state():
    st.session_state.training_history = {"loss": [], "acc": []}


def load_heart_disease():
    """Load and preprocess Heart Disease dataset."""
    loader = Loader("rsc/heart.csv")
    # Loader assumes first row is header, last column is target
    rows = loader.rows

    # Parse features and targets
    X = []
    y = []
    for row in rows:
        try:
            features = [float(v) for v in row[:-1]]
            target = float(row[-1])
            X.append(features)
            y.append(target)
        except (ValueError, IndexError):
            continue

    X = np.array(X)
    y = np.array(y)

    # Split 80/20
    split_idx = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], loader


# Sidebar - Configuration
st.sidebar.markdown("## ⚙️ Configuração")

# Dataset
dataset_option = st.sidebar.selectbox(
    "Dataset",
    ["Heart Disease (UCI)"],
    help="Heart Disease dataset from UCI (last column = target)",
)

# Architecture
st.sidebar.markdown("### Arquitetura")
input_size = 13  # Heart Disease has 13 features
hidden_layers_str = st.sidebar.text_input(
    "Camadas ocultas (separadas por vírgula)", value="8", help="Ex: 8, 4"
)
try:
    hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(",") if x.strip()]
except Exception:
    hidden_layers = [8]
    st.sidebar.error("Formato inválido. Use: 8, 4")

activations = st.sidebar.multiselect(
    "Ativação por camada oculta",
    ["relu", "sigmoid", "identity"],
    default=["relu"] * len(hidden_layers),
    help="Uma por camada oculta",
)
if len(activations) != len(hidden_layers):
    activations = ["relu"] * len(hidden_layers)

output_activation = st.sidebar.selectbox(
    "Ativação de saída", ["sigmoid", "identity", "relu"], index=0
)

# Training hyperparameters
st.sidebar.markdown("### Hiperparâmetros")
learning_rate = st.sidebar.number_input(
    "Learning Rate", 0.0001, 1.0, 0.01, step=0.001, format="%.4f"
)
epochs = st.sidebar.slider("Épocas", 1, 500, 50, step=10)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

# Preprocessing
preprocess_mode = st.sidebar.selectbox(
    "Pré-processamento", ["none", "normalize", "standardize"], index=1
)

# Main header
st.markdown(
    '<div class="main-header">🧠 MLP from Scratch - Professor\'s Implementation</div>',
    unsafe_allow_html=True,
)

# Tabs
tab_data, tab_arch, tab_train, tab_results, tab_inference = st.tabs(
    ["📊 Dados", "🏗️ Arquitetura", "🚀 Treino", "📈 Resultados", "🔮 Inferência"]
)


# ==================== TAB: DADOS ====================
with tab_data:
    st.subheader("Dataset Information")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("📥 Carregar Heart Disease", type="primary"):
            with st.spinner("Carregando dados..."):
                try:
                    X_train, X_test, y_train, y_test, loader = load_heart_disease()
                    st.session_state.loader = loader
                    st.session_state.train_x = X_train
                    st.session_state.train_y = y_train
                    st.session_state.test_x = X_test
                    st.session_state.test_y = y_test
                    st.success(f"Carregado: {len(X_train)} treino, {len(X_test)} teste")
                except Exception as e:
                    st.error(f"Erro: {e}")

    if st.session_state.train_x is not None:
        X_train = st.session_state.train_x
        y_train = st.session_state.train_y
        X_test = st.session_state.test_x
        y_test = st.session_state.test_y

        col1, col2, col3 = st.columns(3)
        col1.metric("Treino", len(X_train))
        col2.metric("Teste", len(X_test))
        col3.metric("Features", X_train.shape[1])

        st.write("**Distribuição de Classes (Treino):**")
        unique, counts = np.unique(y_train, return_counts=True)
        dist_df = pd.DataFrame({"Classe": unique, "Contagem": counts})
        st.bar_chart(dist_df.set_index("Classe"))

        st.write("**Pré-visualização (primeiras 5 amostras):**")
        feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
        df_preview = pd.DataFrame(X_train[:5], columns=feature_names)
        df_preview["target"] = y_train[:5]
        st.dataframe(df_preview)


# ==================== TAB: ARQUITETURA ====================
with tab_arch:
    st.subheader("Arquitetura da Rede")

    layer_sizes = [input_size] + hidden_layers + [1]
    all_activations = activations + [output_activation]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("**Configuração:**")
        arch_df = pd.DataFrame(
            {
                "Camada": [f"Input ({input_size})"]
                + [
                    f"Hidden {i + 1} ({h})" for i, h in enumerate(hidden_layers)
                ]  # noqa: E226
                + [f"Output (1)"],  # noqa: F541
                "Neurônios": layer_sizes,
                "Ativação": ["identity"] + all_activations,
            }
        )
        st.dataframe(arch_df, hide_index=True)

        total_params = sum(
            (layer_sizes[i] + 1) * layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )
        st.metric("Parâmetros Totais", f"{total_params:,}")

    with col2:
        if st.button("🎨 Visualizar Arquitetura (Grafo)"):
            net = Network(layer_sizes)
            net.set_activation(activations[0] if activations else "relu")
            for i, act in enumerate(activations[1:], 1):
                pass  # Only single activation for all layers in professor's impl
            net.set_activation(output_activation)

            from graph_utils import graph_data

            data = graph_data(net)

            # Plot using matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))

            # Extract positions and draw
            nodes = data["nodes"]
            edges = data["edges"]

            # Draw edges
            node_pos = {n["id"]: (n["x"], n["y"]) for n in nodes}
            for edge in edges:
                src = node_pos[edge["source"]]
                dst = node_pos[edge["target"]]
                color = edge["color"]
                width = edge["width"]
                ax.plot(
                    [src[0], dst[0]],
                    [src[1], dst[1]],
                    color=color,
                    linewidth=width,
                    alpha=0.7,
                )

            # Draw nodes
            for node in nodes:
                color = node["color"]
                size = node["width"] * 50
                ax.scatter(
                    node["x"], node["y"], s=size, c=color, edgecolors="black", zorder=5
                )

            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title("Arquitetura da Rede (Grafo)")
            st.pyplot(fig)
            plt.close(fig)

        if st.button("🏗️ Construir Rede"):
            st.session_state.network = Network(layer_sizes)
            st.session_state.network.set_activation(output_activation)
            st.success(f"Rede criada: {st.session_state.network.summary()}")


# ==================== TAB: TREINO ====================
with tab_train:
    st.subheader("Treinamento")

    if st.session_state.train_x is None:
        st.warning("⚠️ Carregue os dados primeiro na aba **Dados**")

    net = st.session_state.network
    if net is None:
        st.info("Clique em **Construir Rede** na aba Arquitetura primeiro")

    train_disabled = (
        st.session_state.train_x is None or net is None or st.session_state.is_training
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(
            "▶️ Iniciar Treino",
            type="primary",
            disabled=train_disabled,
            use_container_width=True,
        ):
            st.session_state.is_training = True
            reset_training_state()

            # Prepare data
            X_train = st.session_state.train_x
            y_train = st.session_state.train_y
            X_test = st.session_state.test_x
            y_test = st.session_state.test_y

            # Apply preprocessing
            net.set_preprocess_mode(preprocess_mode)
            net.load_dataset_rows([[*x, y] for x, y in zip(X_train, y_train)])

            # Placeholders for live updates
            progress_bar = st.progress(0)
            status_text = st.empty()

            metrics_col1, metrics_col2 = st.columns(2)
            train_loss_ph = metrics_col1.empty()
            test_loss_ph = metrics_col2.empty()

            loss_chart = st.line_chart()

            try:
                total_loss = 0.0
                correct = 0

                for epoch in range(epochs):
                    if not st.session_state.is_training:
                        break

                    epoch_loss = 0.0
                    epoch_correct = 0

                    for x, target in zip(
                        st.session_state.train_x, st.session_state.train_y
                    ):
                        if not st.session_state.is_training:
                            break

                        outputs, error, loss = net.train_step(x, target, learning_rate)
                        epoch_loss += loss

                        pred = 1 if outputs[0] >= 0.5 else 0
                        if pred == int(target):
                            epoch_correct += 1

                    avg_loss = epoch_loss / len(st.session_state.train_x)
                    accuracy = epoch_correct / len(st.session_state.train_x)

                    # Validation on test set
                    test_loss = 0.0
                    test_correct = 0
                    for x, target in zip(
                        st.session_state.test_x, st.session_state.test_y
                    ):
                        out = net.forward_propagation(net.transform_inputs(x))
                        test_loss += 0.5 * (out[0] - target) ** 2
                        pred = 1 if out[0] >= 0.5 else 0
                        if pred == int(target):
                            test_correct += 1
                    test_loss /= len(st.session_state.test_x)
                    test_acc = test_correct / len(st.session_state.test_x)

                    # Update history
                    st.session_state.training_history["loss"].append(avg_loss)
                    st.session_state.training_history["acc"].append(accuracy)

                    # Update UI
                    train_loss_ph.metric("Train Loss", f"{avg_loss:.4f}")
                    test_loss_ph.metric("Test Loss", f"{test_loss:.4f}")

                    hist = st.session_state.training_history
                    loss_df = pd.DataFrame(
                        {
                            "Época": range(1, len(hist["loss"]) + 1),
                            "Train Loss": hist["loss"],
                            "Test Loss": [test_loss] * len(hist["loss"]),  # simplified
                        }
                    )
                    loss_chart.line_chart(loss_df.set_index("Época"))

                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.text(
                        f"Época {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {accuracy:.2%}"
                    )

                    # Allow UI to update
                    time.sleep(0.01)

                st.session_state.is_training = False
                st.success("✅ Treino concluído!")
                st.balloons()

            except Exception as e:
                st.session_state.is_training = False
                st.error(f"Erro durante treino: {e}")

    with col2:
        if st.button("⏹️ Parar Treino", disabled=not st.session_state.is_training):
            st.session_state.is_training = False
            st.warning("Treino interrompido")


# ==================== TAB: RESULTADOS ====================
with tab_results:
    st.subheader("Resultados do Treino")

    net = st.session_state.network
    if net is None or st.session_state.train_x is None:
        st.info("Execute o treino primeiro na aba **Treino**")
    else:
        # Final evaluation
        X_test = st.session_state.test_x
        y_test = st.session_state.test_y

        correct = 0
        test_loss = 0.0
        for x, target in zip(X_test, y_test):
            out = net.forward_propagation(net.transform_inputs(x))
            pred = 1 if out[0] >= 0.5 else 0
            if pred == int(target):
                correct += 1
            test_loss += 0.5 * (out[0] - target) ** 2

        test_loss /= len(X_test)
        test_acc = correct / len(X_test)

        col1, col2, col3 = st.columns(3)
        col1.metric("Test Loss", f"{test_loss:.4f}")
        col2.metric("Test Accuracy", f"{test_acc:.2%}")
        col3.metric("Épocas Treinadas", len(st.session_state.training_history["loss"]))

        # Training curves
        hist = st.session_state.training_history
        if hist["loss"]:
            col1, col2 = st.columns(2)
            with col1:
                loss_df = pd.DataFrame(
                    {
                        "Época": range(1, len(hist["loss"]) + 1),
                        "Train Loss": hist["loss"],
                    }
                )
                st.line_chart(loss_df.set_index("Época"))
            with col2:
                acc_df = pd.DataFrame(
                    {"Época": range(1, len(hist["acc"]) + 1), "Train Acc": hist["acc"]}
                )
                st.line_chart(acc_df.set_index("Época"))

        # Network summary
        st.write("**Resumo da Rede:**")
        st.code(net.summary())


# ==================== TAB: INFERÊNCIA ====================
with tab_inference:
    st.subheader("Inferência - Teste Manual")

    net = st.session_state.network
    if net is None:
        st.info("Construa e treine um modelo primeiro")
    else:
        st.write("Insira valores para as 13 features:")

        cols = st.columns(4)
        feature_names = [
            "idade",
            "sexo",
            "cp",
            "pressao",
            "colesterol",
            "acucar",
            "ecg",
            "freq_max",
            "angina",
            "oldpeak",
            "slope",
            "vessels",
            "thal",
        ]

        inputs = []
        for i, name in enumerate(feature_names):
            with cols[i % 4]:
                val = st.number_input(name, value=0.0, step=0.1, format="%.2f")
                inputs.append(val)

        if st.button("🔮 Prever", type="primary"):
            x = np.array(inputs)
            out = net.forward_propagation(net.transform_inputs(x))
            pred = 1 if out[0] >= 0.5 else 0
            prob = out[0]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predição", "Doença (1)" if pred == 1 else "Sem Doença (0)")
            with col2:
                st.metric("Probabilidade", f"{prob:.2%}")

            st.progress(prob)
            st.caption(f"Confiança: {prob:.1%}")

        st.divider()
        st.write("**Teste em lote (amostras aleatórias do teste):**")
        n_samples = st.slider("Número de amostras", 1, 20, 5)

        if st.button("🎲 Testar Amostras Aleatórias"):
            X_test = st.session_state.test_x
            y_test = st.session_state.test_y

            indices = np.random.choice(
                len(X_test), min(n_samples, len(X_test)), replace=False
            )
            results = []
            for idx in indices:
                x = X_test[idx]
                y_true = y_test[idx]
                out = net.forward_propagation(net.transform_inputs(x))
                prob = out[0]
                pred = 1 if prob >= 0.5 else 0
                results.append(
                    {
                        "Índice": idx,
                        "Real": int(y_true),
                        "Predito": int(pred),
                        "Probabilidade": f"{prob:.2%}",
                        "Correto": "✅" if pred == y_true else "❌",
                    }
                )

            st.dataframe(pd.DataFrame(results), hide_index=True)


# Footer
st.divider()
st.caption(
    "MLP from Scratch - Professor's Graph-based Implementation | PyQt6 + Streamlit"
)

# Auto-refresh for live training
if st.session_state.is_training:
    time.sleep(0.1)
    st.rerun()
