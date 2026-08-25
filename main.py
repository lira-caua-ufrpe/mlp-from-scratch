"""main.py - Exemplo mínimo não-gráfico da rede neural."""

from __future__ import annotations
from network import Network
from loader import CSVLoader


def main() -> None:
    # Carrega dados
    loader = CSVLoader("rsc/heart.csv")
    features, targets = loader.load()
    print(
        f"dados carregados: features={loader.num_features} linhas={loader.num_samples}"
    )

    # Split treino/teste
    train_x, train_y, test_x, test_y = loader.split_train_test(
        features, targets, test_ratio=0.2
    )

    # Normaliza
    train_x, test_x, means, stds = loader.normalize_features(train_x, test_x)

    # Cria rede: input=13, hidden=1, output=1
    layer_sizes = [loader.num_features, 1]
    net = Network(layer_sizes, activation="sigmoid", seed=42)

    print(f"rede criada: {net.topology_summary()}")

    # Treino rápido (1 época)
    loss, acc = net.train_epoch(train_x, train_y, learning_rate=0.1)
    print(f"época 1: loss={loss:.4f} acc={acc:.2%}")

    # Teste
    correct = 0
    for x, y in zip(test_x, test_y):
        net.forward_all(x)
        pred = 1 if net.get_output()[0] > 0.5 else 0
        if pred == int(y):
            correct += 1
    print(f"teste: acurácia={correct}/{len(test_x)} = {correct / len(test_x):.2%}")


if __name__ == "__main__":
    main()
