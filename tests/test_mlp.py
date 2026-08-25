"""Unit tests for MLP implementation."""

import sys
sys.path.insert(0, "src")  # noqa: E402

import numpy as np  # noqa: E402
from mlp.graph import ComputationalGraph  # noqa: E402
from mlp.mlp import MLP  # noqa: E402
from data.loader import preprocess_data, one_hot_encode, get_data_info  # noqa: E402


def test_graph_creation():
    """Test basic graph construction."""
    graph = ComputationalGraph()
    graph.build_mlp([3, 4, 2], ["relu", "sigmoid"])

    assert len(graph.layers) == 3
    assert len(graph.layers[0]) == 3  # input
    assert len(graph.layers[1]) == 4  # hidden
    assert len(graph.layers[2]) == 2  # output
    assert len(graph.input_nodes) == 3
    assert len(graph.output_nodes) == 2
    assert len(graph.edges) == 3 * 4 + 4 * 2  # 12 + 8 = 20
    print("✓ Graph creation test passed")


def test_forward_pass():
    """Test forward pass computation."""
    graph = ComputationalGraph()
    graph.build_mlp([2, 3, 1], ["relu", "linear"])

    # Set known weights for verification
    for edge in graph.edges.values():
        edge.weight = 1.0
    for node in graph.nodes.values():
        node.bias = 0.0

    output = graph.forward_pass(np.array([1.0, 2.0]))
    assert output.shape == (1,)
    print("✓ Forward pass test passed")


def test_backward_pass():
    """Test backpropagation gradients."""
    graph = ComputationalGraph()
    graph.build_mlp([2, 2, 1], ["linear", "linear"])

    # Simple linear network: y = W2 * (W1 * x)
    for edge in graph.edges.values():
        edge.weight = 1.0
    for node in graph.nodes.values():
        node.bias = 0.0

    x = np.array([1.0, 1.0])
    y_true = np.array([2.0])  # Expected: 1*1 + 1*1 = 2

    graph.forward_pass(x)
    _ = graph.backward_pass(y_true, loss_fn="mse")

    # Check gradients exist
    for edge in graph.edges.values():
        assert edge.gradient != 0
    for node in graph.nodes.values():
        if not node.is_input:
            assert node.bias_gradient != 0
    print("✓ Backward pass test passed")


def test_mlp_training():
    """Test MLP training on simple problem."""
    # XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    mlp = MLP(
        layer_sizes=[2, 4, 1],
        activations=["relu", "sigmoid"],
        learning_rate=0.5,
        seed=42,
    )

    _ = mlp.fit(X, y, epochs=500, batch_size=4, verbose=False)

    predictions = mlp.predict(X)
    accuracy = np.mean(predictions.flatten() == y.flatten())

    assert accuracy >= 0.75, f"XOR accuracy too low: {accuracy}"
    print(f"✓ MLP training test passed (accuracy: {accuracy})")


def test_data_preprocessing():
    """Test data preprocessing pipeline."""
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y, test_size=0.2)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert scaler is not None
    assert abs(X_train.mean()) < 0.1  # roughly zero mean
    assert abs(X_train.std() - 1.0) < 0.1  # roughly unit std
    print("✓ Data preprocessing test passed")


def test_one_hot():
    """Test one-hot encoding."""
    y = np.array([0, 1, 2, 1, 0])
    y_oh = one_hot_encode(y, 3)

    assert y_oh.shape == (5, 3)
    assert np.allclose(y_oh.sum(axis=1), 1.0)
    assert np.argmax(y_oh[0]) == 0
    assert np.argmax(y_oh[1]) == 1
    assert np.argmax(y_oh[2]) == 2
    print("✓ One-hot encoding test passed")


def test_get_data_info():
    """Test data info extraction."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])

    info = get_data_info(X, y)

    assert info["n_samples"] == 3
    assert info["n_features"] == 2
    assert info["classes"] == [0, 1]
    assert info["class_distribution"] == {0: 2, 1: 1}
    print("✓ Data info test passed")


def test_weight_snapshot():
    """Test weight snapshot for visualization."""
    mlp = MLP([2, 3, 1], ["relu", "sigmoid"], seed=42)
    mlp.forward(np.array([[1, 2]]))

    snapshot = mlp.get_graph().get_weights_snapshot()

    assert "edges" in snapshot
    assert "nodes" in snapshot
    assert len(snapshot["edges"]) == 2 * 3 + 3 * 1
    assert len(snapshot["nodes"]) == 2 + 3 + 1
    print("✓ Weight snapshot test passed")


def test_topology():
    """Test topology extraction."""
    mlp = MLP([2, 3, 1], ["relu", "sigmoid"])
    topology = mlp.get_topology()

    assert "layers" in topology
    assert "edges" in topology
    assert "input_nodes" in topology
    assert "output_nodes" in topology
    assert len(topology["layers"]) == 3
    print("✓ Topology test passed")


def run_all_tests():
    """Run all tests."""
    print("Running MLP tests...\n")

    test_graph_creation()
    test_forward_pass()
    test_backward_pass()
    test_mlp_training()
    test_data_preprocessing()
    test_one_hot()
    test_get_data_info()
    test_weight_snapshot()
    test_topology()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
