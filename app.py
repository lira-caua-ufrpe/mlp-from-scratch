"""Streamlit app for MLP from Scratch - Interactive Training & Visualization."""

import sys
sys.path.insert(0, "src")  # noqa: E402

import streamlit as st  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import time  # noqa: E402

from mlp.mlp import MLP  # noqa: E402
from data.loader import load_heart_disease, preprocess_data, get_data_info  # noqa: E402
from visualization.dashboard import GraphVisualizer  # noqa: E402

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
if "training_history" not in st.session_state:
    st.session_state.training_history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
if "mlp_model" not in st.session_state:
    st.session_state.mlp_model = None
if "training_data" not in st.session_state:
    st.session_state.training_data = None
if "is_training" not in st.session_state:
    st.session_state.is_training = False


def reset_training_state():
    st.session_state.training_history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    st.session_state.mlp_model = None


# Sidebar - Configuration
st.sidebar.markdown("## ⚙️ Configuração")

# Dataset selection
dataset_option = st.sidebar.selectbox(
    "Dataset",
    ["Heart Disease (UCI)", "Dados Sintéticos"],
    help="Heart Disease baixa automaticamente do UCI. Sintético gera dados aleatórios.",
)

# Architecture
st.sidebar.markdown("### Arquitetura")
input_size = 13 if dataset_option == "Heart Disease (UCI)" else 10
hidden_layers_str = st.sidebar.text_input(
    "Camadas ocultas (separadas por vírgula)", value="64, 32", help="Ex: 64, 32, 16"
)
try:
    hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(",") if x.strip()]
except Exception:
    hidden_layers = [64, 32]
    st.sidebar.error("Formato inválido. Use: 64, 32, 16")

activations = st.sidebar.multiselect(
    "Ativações por camada oculta",
    ["relu", "tanh", "sigmoid"],
    default=["relu"] * len(hidden_layers),
    help="Uma por camada oculta",
)
if len(activations) != len(hidden_layers):
    activations = ["relu"] * len(hidden_layers)

output_activation = st.sidebar.selectbox(
    "Ativação de saída", ["sigmoid", "linear"], index=0
)

# Training hyperparameters
st.sidebar.markdown("### Hiperparâmetros")
learning_rate = st.sidebar.number_input(
    "Learning Rate", 0.0001, 0.5, 0.01, step=0.001, format="%.4f"
)
epochs = st.sidebar.slider("Épocas", 10, 1000, 100, step=10)
batch_size = st.sidebar.select_slider("Batch Size", [8, 16, 32, 64, 128, 256], value=32)
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
val_split = st.sidebar.slider("Validation Split (%)", 5, 30, 15) / 100
weight_init = st.sidebar.selectbox("Inicialização", ["xavier", "he", "random"], index=0)
loss_fn = st.sidebar.selectbox("Loss Function", ["bce", "mse"], index=0)
seed = st.sidebar.number_input("Random Seed", 0, 9999, 42)

# Visualization options
st.sidebar.markdown("### Visualização")
show_graph = st.sidebar.checkbox("Mostrar Grafo", value=True)
show_weights = st.sidebar.checkbox("Mostrar Pesos no Grafo", value=True)
update_interval = st.sidebar.slider("Atualizar a cada N épocas", 1, 20, 5)

# Main header
st.markdown(
    '<div class="main-header">🧠 MLP from Scratch - Treino Interativo</div>',
    unsafe_allow_html=True,
)

# Tabs
tab_train, tab_data, tab_architecture, tab_results, tab_inference = st.tabs(
    ["🚀 Treino", "📊 Dados", "🏗️ Arquitetura", "📈 Resultados", "🔮 Inferência"]
)


# ==================== TAB: DADOS ====================
with tab_data:
    st.subheader("Dataset Information")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("📥 Carregar / Gerar Dados", type="primary"):
            with st.spinner("Carregando dados..."):
                try:
                    if dataset_option == "Heart Disease (UCI)":
                        X, y = load_heart_disease(download=True)
                        st.success(
                            f"Heart Disease carregado: {X.shape[0]} amostras, {X.shape[1]} features"
                        )
                    else:
                        np.random.seed(seed)
                        X = np.random.randn(1000, input_size)
                        y = (X[:, 0] + X[:, 1] > 0).astype(float)
                        st.success(
                            f"Dados sintéticos gerados: {X.shape[0]} amostras, {X.shape[1]} features"
                        )

                    info = get_data_info(X, y)
                    st.session_state.training_data = {"X": X, "y": y, "info": info}

                except Exception as e:
                    st.error(f"Erro: {e}")

    if st.session_state.training_data:
        info = st.session_state.training_data["info"]
        X = st.session_state.training_data["X"]
        y = st.session_state.training_data["y"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Amostras", info["n_samples"])
        col2.metric("Features", info["n_features"])
        col3.metric("Classes", len(info["classes"]))

        st.write("**Distribuição de Classes:**")
        dist_df = pd.DataFrame(
            list(info["class_distribution"].items()), columns=["Classe", "Contagem"]
        )
        st.bar_chart(dist_df.set_index("Classe"))

        st.write("**Amostra dos dados (primeiras 5 linhas):**")
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        df_preview = pd.DataFrame(X[:5], columns=feature_names)
        df_preview["target"] = y[:5]
        st.dataframe(df_preview)

        st.write("**Estatísticas das Features:**")
        stats_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Média": info["feature_stats"]["mean"],
                "Std": info["feature_stats"]["std"],
                "Min": info["feature_stats"]["min"],
                "Max": info["feature_stats"]["max"],
            }
        )
        st.dataframe(stats_df)


# ==================== TAB: ARQUITETURA ====================
with tab_architecture:
    st.subheader("Arquitetura da Rede")

    layer_sizes = [input_size] + hidden_layers + [1]
    all_activations = activations + [output_activation]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("**Configuração:**")
        arch_df = pd.DataFrame(
            {
                "Camada": (
                    [f"Input ({input_size})"]
                    + [f"Hidden {i + 1} ({h})" for i, h in enumerate(hidden_layers)]  # noqa: F541
                    + [f"Output (1)"]  # noqa: F541
                ),
                "Neurônios": layer_sizes,
                "Ativação": ["linear"] + all_activations,
            }
        )
        st.dataframe(arch_df, hide_index=True)

        total_params = sum(
            (layer_sizes[i] + 1) * layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )
        st.metric("Parâmetros Totais", f"{total_params:,}")

    with col2:
        # Draw static architecture graph
        if st.button("🎨 Visualizar Arquitetura"):
            dummy_mlp = MLP(
                layer_sizes, all_activations, weight_init=weight_init, seed=seed
            )
            graph_viz = GraphVisualizer(dummy_mlp.get_graph())

            fig, ax = plt.subplots(figsize=(12, 6))
            graph_viz.draw(epoch=0, loss=0.0, show_weights=show_weights)
            st.pyplot(fig)
            plt.close(fig)


# ==================== TAB: TREINO ====================
with tab_train:
    st.subheader("Treinamento")

    # Check if data loaded
    data_ready = st.session_state.training_data is not None

    if not data_ready:
        st.warning("⚠️ Carregue os dados primeiro na aba **Dados**")

    # Train button
    train_disabled = not data_ready or st.session_state.is_training

    if st.button(
        "▶️ Iniciar Treino",
        type="primary",
        disabled=train_disabled,
        use_container_width=True,
    ):
        st.session_state.is_training = True
        reset_training_state()

        # Prepare data
        X = st.session_state.training_data["X"]
        y = st.session_state.training_data["y"]

        X_train, X_test, y_train, y_test, scaler = preprocess_data(
            X, y, test_size=test_size, random_state=seed, scale=True
        )

        val_size = val_split / (1 - test_size)
        X_train, X_val, y_train, y_val, _ = preprocess_data(
            X_train, y_train, test_size=val_size, random_state=seed, scale=True
        )

        # Reshape for binary classification
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # Build model
        layer_sizes = [input_size] + hidden_layers + [1]
        all_activations = activations + [output_activation]

        mlp = MLP(
            layer_sizes=layer_sizes,
            activations=all_activations,
            weight_init=weight_init,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            seed=seed,
        )

        st.session_state.mlp_model = mlp
        st.session_state.training_data["X_test"] = X_test
        st.session_state.training_data["y_test"] = y_test
        st.session_state.training_data["scaler"] = scaler

        # Placeholders for live updates
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Metrics placeholders
        col1, col2, col3, col4 = st.columns(4)
        train_loss_ph = col1.empty()
        val_loss_ph = col2.empty()
        train_acc_ph = col3.empty()
        val_acc_ph = col4.empty()

        # Charts
        loss_chart = st.line_chart()
        acc_chart = st.line_chart()

        # Graph visualization
        graph_placeholder = st.empty()
        graph_viz = GraphVisualizer(mlp.get_graph())

        # Training callback
        def streamlit_callback(epoch, metrics, weights_snapshot):
            # Update history
            st.session_state.training_history["epochs"].append(epoch)
            st.session_state.training_history["train_loss"].append(
                metrics["train_loss"]
            )
            st.session_state.training_history["train_acc"].append(metrics["train_acc"])
            if metrics.get("val_loss") is not None:
                st.session_state.training_history["val_loss"].append(
                    metrics["val_loss"]
                )
            if metrics.get("val_acc") is not None:
                st.session_state.training_history["val_acc"].append(metrics["val_acc"])

            # Update metrics
            train_loss_ph.metric("Train Loss", f"{metrics['train_loss']:.4f}")
            val_loss_ph.metric("Val Loss", f"{metrics.get('val_loss', 0):.4f}")
            train_acc_ph.metric("Train Acc", f"{metrics['train_acc']:.2%}")
            val_acc_ph.metric("Val Acc", f"{metrics.get('val_acc', 0):.2%}")

            # Update charts
            hist = st.session_state.training_history
            loss_df = pd.DataFrame(
                {
                    "Época": hist["epochs"],
                    "Train Loss": hist["train_loss"],
                    "Val Loss": (
                        hist["val_loss"]
                        if hist["val_loss"]
                        else [0] * len(hist["epochs"])
                    ),
                }
            )
            acc_df = pd.DataFrame(
                {
                    "Época": hist["epochs"],
                    "Train Acc": hist["train_acc"],
                    "Val Acc": (
                        hist["val_acc"]
                        if hist["val_acc"]
                        else [0] * len(hist["epochs"])
                    ),
                }
            )
            loss_chart.line_chart(loss_df.set_index("Época"))
            acc_chart.line_chart(acc_df.set_index("Época"))

            # Update graph visualization
            if show_graph and epoch % update_interval == 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                graph_viz.draw(epoch, metrics["train_loss"], show_weights=show_weights)
                graph_placeholder.pyplot(fig)
                plt.close(fig)

            # Progress
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(
                f"Época {epoch + 1}/{epochs} - Loss: {metrics['train_loss']:.4f}"
            )

        # Run training
        try:
            with st.spinner("Treinando..."):
                history = mlp.fit(
                    X_train,
                    y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=False,
                    callback=streamlit_callback,
                )

            st.session_state.is_training = False
            st.success("✅ Treino concluído!")
            st.balloons()

        except Exception as e:
            st.session_state.is_training = False
            st.error(f"Erro durante treino: {e}")

    if st.session_state.is_training:
        st.info("🔄 Treinando... aguarde")


# ==================== TAB: RESULTADOS ====================
with tab_results:
    st.subheader("Resultados do Treino")

    if st.session_state.mlp_model is None:
        st.info("Execute o treino primeiro na aba **Treino**")
    else:
        mlp = st.session_state.mlp_model
        X_test = st.session_state.training_data["X_test"]
        y_test = st.session_state.training_data["y_test"]

        # Final evaluation
        y_pred = mlp.predict(X_test)
        y_proba = mlp.predict_proba(X_test)

        test_loss = mlp.compute_loss(y_test, y_proba)
        test_acc = mlp.compute_accuracy(y_test, y_proba)

        col1, col2, col3 = st.columns(3)
        col1.metric("Test Loss", f"{test_loss:.4f}")
        col2.metric("Test Accuracy", f"{test_acc:.2%}")
        col3.metric(
            "Épocas Treinadas", len(st.session_state.training_history["epochs"])
        )

        # Confusion Matrix
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test.flatten(), y_pred.flatten())

        st.write("**Matriz de Confusão:**")
        cm_df = pd.DataFrame(
            cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]
        )
        st.dataframe(cm_df)

        # Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write("**Relatório de Classificação:**")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"))

        # Training curves
        hist = st.session_state.training_history
        if hist["epochs"]:
            col1, col2 = st.columns(2)
            with col1:
                loss_df = pd.DataFrame(
                    {
                        "Época": hist["epochs"],
                        "Train": hist["train_loss"],
                        "Val": (
                            hist["val_loss"]
                            if hist["val_loss"]
                            else [None] * len(hist["epochs"])
                        ),
                    }
                )
                st.line_chart(loss_df.set_index("Época"))
            with col2:
                acc_df = pd.DataFrame(
                    {
                        "Época": hist["epochs"],
                        "Train": hist["train_acc"],
                        "Val": (
                            hist["val_acc"]
                            if hist["val_acc"]
                            else [None] * len(hist["epochs"])
                        ),
                    }
                )
                st.line_chart(acc_df.set_index("Época"))

        # Final graph visualization
        if show_graph:
            st.write("**Grafo Final (Pesos Aprendidos):**")
            graph_viz = GraphVisualizer(mlp.get_graph())
            fig, ax = plt.subplots(figsize=(14, 8))
            graph_viz.draw(
                epoch=len(hist["epochs"]),
                loss=hist["train_loss"][-1] if hist["train_loss"] else 0,
                show_weights=show_weights,
            )
            st.pyplot(fig)
            plt.close(fig)

        # Weight distribution
        st.write("**Distribuição dos Pesos Finais:**")
        snapshot = mlp.get_graph().get_weights_snapshot()
        weights = [e["weight"] for e in snapshot["edges"].values()]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(weights, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Valor do Peso")
        ax.set_ylabel("Frequência")
        ax.set_title("Distribuição dos Pesos")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        # Download model
        if st.button("💾 Salvar Modelo"):
            import pickle  # noqa: F401
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mlp_model_{timestamp}.pkl"
            mlp.save_checkpoint(filename)
            st.success(f"Modelo salvo como `{filename}`")


# ==================== TAB: INFERÊNCIA ====================
with tab_inference:
    st.subheader("Inferência - Teste Manual")

    if st.session_state.mlp_model is None:
        st.info("Treine um modelo primeiro na aba **Treino**")
    else:
        mlp = st.session_state.mlp_model
        scaler = st.session_state.training_data.get("scaler")

        st.write(
            "Insira valores para as features (serão normalizados automaticamente):"
        )

        # Create input form
        cols = st.columns(4)
        inputs = []
        for i in range(input_size):
            with cols[i % 4]:
                val = st.number_input(
                    f"Feature {i}", value=0.0, step=0.1, format="%.2f"
                )
                inputs.append(val)

        if st.button("🔮 Prever", type="primary"):
            x = np.array(inputs).reshape(1, -1)
            if scaler:
                x = scaler.transform(x)

            proba = mlp.predict_proba(x)[0]
            pred = mlp.predict(x)[0]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predição", "Classe 1" if pred == 1 else "Classe 0")
            with col2:
                st.metric(
                    "Probabilidade",
                    f"{proba[0] if isinstance(proba, np.ndarray) else proba:.2%}",
                )

            # Show probability bar
            prob_val = proba[0] if isinstance(proba, np.ndarray) else proba
            st.progress(prob_val)
            st.caption(f"Confiança: {prob_val:.1%}")

        st.divider()
        st.write("**Teste em Lote (amostras do conjunto de teste):**")
        n_samples = st.slider("Número de amostras", 1, 20, 5)

        if st.button("🎲 Testar Amostras Aleatórias"):
            X_test = st.session_state.training_data["X_test"]
            y_test = st.session_state.training_data["y_test"]

            indices = np.random.choice(
                len(X_test), min(n_samples, len(X_test)), replace=False
            )

            results = []
            for idx in indices:
                x = X_test[idx:idx + 1]
                y_true = y_test[idx][0]
                y_proba = mlp.predict_proba(x)[0]
                y_pred = mlp.predict(x)[0]

                prob = y_proba[0] if isinstance(y_proba, np.ndarray) else y_proba
                results.append(
                    {
                        "Índice": idx,
                        "Real": int(y_true),
                        "Predito": int(y_pred),
                        "Probabilidade": f"{prob:.2%}",
                        "Correto": "✅" if y_pred == y_true else "❌",
                    }
                )

            st.dataframe(pd.DataFrame(results), hide_index=True)


# Footer
st.divider()
st.caption(
    "MLP from Scratch - Advanced AI Topics | Built with Streamlit, NumPy, Matplotlib"
)


# Auto-refresh for live training (optional)
if st.session_state.is_training:
    time.sleep(0.1)
    st.rerun()
