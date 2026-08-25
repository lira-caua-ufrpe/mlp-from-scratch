"""loader - Carregador CSV simples (última coluna = target)."""

from __future__ import annotations
from typing import List, Tuple
import csv
import random


class CSVLoader:
    """
    Carrega dataset CSV assumindo:
    - Primeira linha = cabeçalho (ignorado)
    - Última coluna = target (rótulo)
    - Demais colunas = features
    """

    def __init__(self, filepath: str, has_header: bool = True) -> None:
        self.filepath = filepath
        self.has_header = has_header
        self.headers: List[str] = []
        self.num_features: int = 0
        self.num_samples: int = 0
        self._raw_rows: List[List[str]] = []

    def load(self) -> Tuple[List[List[float]], List[float]]:
        """
        Carrega e retorna (features, targets).
        features: List[List[float]] - cada amostra é lista de features
        targets: List[float] - cada target é float
        """
        with open(self.filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            raise ValueError("CSV vazio")

        if self.has_header:
            self.headers = rows[0]
            data_rows = rows[1:]
        else:
            self.headers = [f"feat_{i}" for i in range(len(rows[0]) - 1)] + ["target"]
            data_rows = rows

        # Filtra linhas vazias
        data_rows = [r for r in data_rows if r and any(c.strip() for c in r)]

        self.num_features = len(data_rows[0]) - 1 if data_rows else 0
        self.num_samples = len(data_rows)
        self._raw_rows = data_rows

        features = []
        targets = []
        for row in data_rows:
            try:
                feature_vals = [float(v) for v in row[:-1]]
                target_val = float(row[-1])
                features.append(feature_vals)
                targets.append(target_val)
            except ValueError:
                # Pula linhas com valores não numéricos
                continue

        return features, targets

    @property
    def feature_names(self) -> List[str]:
        return (
            self.headers[:-1]
            if self.headers
            else [f"feat_{i}" for i in range(self.num_features)]
        )

    def get_info(self) -> dict:
        return {
            "filepath": self.filepath,
            "num_features": self.num_features,
            "num_samples": self.num_samples,
            "headers": self.headers,
            "feature_names": self.headers[:-1],
            "target_name": self.headers[-1] if self.headers else "target",
        }

    def split_train_test(
        self,
        features: List[List[float]],
        targets: List[float],
        test_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
        """Split estratificado simples."""
        if shuffle:
            combined = list(zip(features, targets))
            random.seed(seed)
            random.shuffle(combined)
            features, targets = zip(*combined)

        split_idx = int(len(features) * (1 - test_ratio))
        return (
            list(features[:split_idx]),
            list(targets[:split_idx]),
            list(features[split_idx:]),
            list(targets[split_idx:]),
        )

    def normalize_features(
        self, train_features: List[List[float]], test_features: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
        """
        Normalização z-score (média 0, desvio 1) usando estatísticas do treino.
        Retorna (train_norm, test_norm, means, stds).
        """
        if not train_features:
            return train_features, test_features, [], []

        num_feats = len(train_features[0])
        means = [0.0] * num_feats
        stds = [1.0] * num_feats

        # Média
        for i in range(num_feats):
            means[i] = sum(row[i] for row in train_features) / len(train_features)

        # Desvio padrão
        for i in range(num_feats):
            var = sum((row[i] - means[i]) ** 2 for row in train_features) / len(
                train_features
            )
            stds[i] = math.sqrt(var) if var > 0 else 1.0

        def normalize(rows: List[List[float]]) -> List[List[float]]:
            return [
                [(row[i] - means[i]) / stds[i] for i in range(num_feats)]
                for row in rows
            ]

        return normalize(train_features), normalize(test_features), means, stds


import math  # noqa: E402
