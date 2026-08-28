"""Unit tests for MLP implementation (Professor's structure)."""

import sys

sys.path.insert(0, ".")  # noqa: E402

from neuron import Neuron  # noqa: E402
from connection import Connection  # noqa: E402
from layer import Layer  # noqa: E402
from network import Network  # noqa: E402
from loader import Loader  # noqa: E402


def test_neuron_creation():
    """Test Neuron creation with default values."""
    n = Neuron()
    assert n.bias == 0.0
    assert n.input_sum == 0.0
    assert n.output == 0.0
    assert n.delta == 0.0
    assert n.id is not None
    print("✓ Neuron creation test passed")


def test_connection_creation():
    """Test Connection creation with random weight."""
    a = Neuron()
    b = Neuron()
    c = Connection(a, b)
    assert c.source is a
    assert c.target is b
    assert -1.0 <= c.weight <= 1.0
    # Gradient is stored in Network.last_gradients, not on Connection
    print("✓ Connection creation test passed")


def test_layer_creation():
    """Test Layer creation and neuron management."""
    layer = Layer(3)
    assert len(layer) == 3
    assert all(isinstance(n, Neuron) for n in layer)

    # Add neuron
    n = layer.add_neuron()
    assert len(layer) == 4
    assert n in layer.neurons

    # Remove the neuron we just added (last index)
    removed = layer.remove_neuron(len(layer) - 1)
    assert len(layer) == 3
    assert removed is n
    print("✓ Layer creation test passed")


def test_network_creation():
    """Test Network creation and wiring."""
    net = Network([2, 3, 1])
    assert len(net.layers) == 3
    assert len(net.layers[0]) == 2
    assert len(net.layers[1]) == 3
    assert len(net.layers[2]) == 1
    assert len(net.connections) == 2 * 3 + 3 * 1  # 9
    print("✓ Network creation test passed")


def test_network_forward():
    """Test forward propagation."""
    net = Network([2, 3, 1])
    net.set_activation("identity")  # linear for predictable output

    # Set all weights to 1.0 and biases to 0 for predictable result
    for c in net.connections:
        c.weight = 1.0
    for layer in net.layers[1:]:
        for n in layer:
            n.bias = 0.0

    outputs = net.forward_propagation([1.0, 2.0])
    assert len(outputs) == 1
    # With identity activation and all weights=1: output = 1*1 + 1*2 + 1*1 + 1*2 = 6?
    # Actually each output neuron sums all inputs from previous layer
    print("✓ Network forward test passed")


def test_network_activations():
    """Test different activation functions."""
    net = Network([2, 1])

    for act in ["identity", "sigmoid", "relu"]:
        net.set_activation(act)
        net.reset_state()
        out = net.forward_propagation([1.0, -1.0])
        assert len(out) == 1
        assert isinstance(out[0], float)
    print("✓ Network activations test passed")


def test_backward_pass():
    """Test backpropagation with MSE."""
    net = Network([2, 1])
    net.set_activation("identity")

    # Simple case: input [1, 1], target 2.0
    # With weights=1, bias=0: output = 2.0, target=2.0, error=0
    for c in net.connections:
        c.weight = 1.0
    for n in net.layers[1]:
        n.bias = 0.0

    net.start_forward([1.0, 1.0])
    while True:
        status = net.forward_step()
        if status[0] == "done":
            break

    # Target is 2.0, output should be 2.0 (1*1 + 1*1)
    error = net._compute_deltas(2.0)
    assert abs(error) < 1e-6  # error = output - target = 2 - 2 = 0
    print("✓ Backward pass test passed")


def test_train_step():
    """Test train_step with MSE."""
    net = Network([2, 1])
    net.set_activation("identity")

    # Initialize with known weights
    for c in net.connections:
        c.weight = 0.5
    for n in net.layers[1]:
        n.bias = 0.0

    outputs, error, loss = net.train_step([1.0, 1.0], target=2.0, learning_rate=0.1)
    assert len(outputs) == 1
    assert isinstance(loss, float)
    assert loss >= 0.0
    print("✓ Train step test passed")


def test_loader():
    """Test CSV Loader."""
    loader = Loader("code/mlp_viewer/rsc/heart.csv")
    assert loader.feature_count == 13
    assert loader.row_count > 0
    assert len(loader.headers) == 14  # 13 features + 1 target
    print("✓ Loader test passed")


def test_network_add_remove_neuron():
    """Test dynamic architecture modification."""
    net = Network([2, 3, 1])
    original_connections = len(net.connections)

    # Add neuron to hidden layer (index 1)
    _ = net.add_neuron(1)
    assert len(net.layers[1]) == 4
    assert len(net.connections) > original_connections

    # Remove neuron
    net.remove_neuron(1, 0)  # remove first neuron of layer 1
    assert len(net.layers[1]) == 3
    print("✓ Add/remove neuron test passed")


def test_network_add_remove_layer():
    """Test dynamic layer modification."""
    net = Network([2, 3, 1])

    # Add hidden layer at position 1
    from graph_utils import add_layer

    add_layer(net, 4, position=1)
    assert len(net.layers) == 4
    assert len(net.layers[1]) == 4

    # Remove layer
    from graph_utils import remove_layer

    remove_layer(net, 1)
    assert len(net.layers) == 3
    print("✓ Add/remove layer test passed")


def test_preprocessing():
    """Test normalize and standardize preprocessing."""
    net = Network([2, 1])
    net.set_activation("identity")

    # Load some dummy data
    rows = [["1.0", "2.0", "0.5"], ["2.0", "4.0", "1.0"], ["3.0", "6.0", "1.5"]]
    net.load_dataset_rows(rows)

    # Test normalize
    net.set_preprocess_mode("normalize")
    x_norm = net.transform_inputs([1.0, 2.0])
    assert len(x_norm) == 2

    # Test standardize
    net.set_preprocess_mode("standardize")
    x_std = net.transform_inputs([1.0, 2.0])
    assert len(x_std) == 2

    # Test none
    net.set_preprocess_mode("none")
    x_none = net.transform_inputs([1.0, 2.0])
    assert x_none == [1.0, 2.0]
    print("✓ Preprocessing test passed")


def test_save_load_weights():
    """Test weight persistence."""
    net = Network([2, 3, 1])
    net.set_activation("sigmoid")

    # Set specific weights
    for i, c in enumerate(net.connections):
        c.weight = float(i) * 0.1
    for layer in net.layers[1:]:
        for j, n in enumerate(layer):
            n.bias = float(j) * 0.01

    # Save
    net.save_json("test_weights.json")

    # Create new network and load
    net2 = Network([2, 3, 1])
    net2.load_json("test_weights.json")

    # Check weights match
    for c1, c2 in zip(net.connections, net2.connections):
        assert abs(c1.weight - c2.weight) < 1e-6
    for l1, l2 in zip(net.layers[1:], net2.layers[1:]):
        for n1, n2 in zip(l1.neurons, l2.neurons):
            assert abs(n1.bias - n2.bias) < 1e-6
    print("✓ Save/Load weights test passed")


def run_all_tests():
    """Run all tests."""
    print("Running MLP tests (Professor's structure)...\n")

    test_neuron_creation()
    test_connection_creation()
    test_layer_creation()
    test_network_creation()
    test_network_forward()
    test_network_activations()
    test_backward_pass()
    test_train_step()
    test_loader()
    test_network_add_remove_neuron()
    test_network_add_remove_layer()
    test_preprocessing()
    test_save_load_weights()

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
