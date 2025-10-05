"""Multimodal retrieval utilities combining graph structure and visual features."""
from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from dierpian.graphs.pmi_attribute_graph import PMIAttributeGraph

__all__ = ["MultiModalRetriever"]


SIMILARITY_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "cosine": lambda x, y: cosine_similarity(x, y),
    "dot": lambda x, y: x @ y.T,
    "euclidean": lambda x, y: -euclidean_distances(x, y),  # Negated so that larger is better
}


class MultiModalRetriever:
    """Perform similarity search by fusing PMI graph features with image embeddings.

    Parameters
    ----------
    similarity:
        Metric used to score candidate matches. Supported values are
        ``"cosine"``, ``"dot"`` and ``"euclidean"``. Larger scores are always
        considered better.
    graph_weight:
        Relative weight assigned to PMI-based features during concatenation.
    visual_weight:
        Relative weight assigned to visual features.
    """

    def __init__(self, similarity: str = "cosine", graph_weight: float = 0.5, visual_weight: float = 0.5) -> None:
        if similarity not in SIMILARITY_FUNCS:
            raise ValueError(f"Unsupported similarity metric: {similarity}")
        self.similarity = similarity
        self.graph_weight = graph_weight
        self.visual_weight = visual_weight

        self.graph: PMIAttributeGraph | None = None
        self.graph_features: np.ndarray | None = None
        self.visual_features: np.ndarray | None = None
        self.feature_matrix: np.ndarray | None = None
        self.metadata: list[dict[str, object]] = []
        self.tokens: list[list[str]] = []
        self.image_root: Path | None = None

    @staticmethod
    def _normalise_metadata(metadata: Sequence[Mapping[str, object] | object]) -> list[dict[str, object]]:
        normalised: list[dict[str, object]] = []
        for row in metadata:
            if isinstance(row, Mapping):
                normalised.append(dict(row))
            elif is_dataclass(row):
                normalised.append(asdict(row))
            elif hasattr(row, "_asdict"):
                normalised.append(dict(row._asdict()))
            else:
                normalised.append(dict(vars(row)))
        return normalised

    @staticmethod
    def _extract_tokens(metadata_row: Mapping[str, object]) -> list[str]:
        return [f"{key}={value}" for key, value in metadata_row.items() if key not in {"filename", "seed"}]

    @staticmethod
    def _load_image_feature(path: Path, bins: int = 8) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        histogram = np.array(image.histogram(), dtype=np.float32)
        # Normalise histogram
        histogram /= histogram.sum() + 1e-8
        if bins == 256:
            return histogram
        # Compress histogram by reshaping
        histogram = histogram.reshape(3, 256)
        factor = 256 // bins
        histogram = histogram.reshape(3, bins, factor).sum(axis=2)
        return histogram.flatten()

    def _build_graph_features(self, tokens: list[list[str]], graph: PMIAttributeGraph) -> np.ndarray:
        adjacency = graph.adjacency_matrix()
        vocab = list(adjacency.index)
        vocab_index = {token: idx for idx, token in enumerate(vocab)}
        features = np.zeros((len(tokens), len(vocab)), dtype=np.float32)
        for row_idx, row_tokens in enumerate(tokens):
            for token in row_tokens:
                if token in vocab_index:
                    features[row_idx, vocab_index[token]] += 1.0
                    features[row_idx] += adjacency.loc[token].to_numpy(dtype=np.float32)
        # Normalise rows
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return features / norms

    def _build_visual_features(self, image_paths: Sequence[Path]) -> np.ndarray:
        feats = [self._load_image_feature(path) for path in image_paths]
        feats = np.stack(feats, axis=0)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        return feats / norms

    def fit(
        self,
        metadata: Sequence[Mapping[str, object] | object],
        image_root: str | Path,
        graph: PMIAttributeGraph | None = None,
    ) -> "MultiModalRetriever":
        """Prepare the retriever by computing graph and visual features."""

        self.metadata = self._normalise_metadata(metadata)
        self.tokens = [self._extract_tokens(row) for row in self.metadata]
        self.image_root = Path(image_root)

        if graph is None:
            graph = PMIAttributeGraph()
            graph.fit(self.metadata)
        else:
            graph.fit(self.metadata)

        self.graph = graph
        self.graph_features = self._build_graph_features(self.tokens, graph)

        image_paths = [self.image_root / row["filename"] for row in self.metadata]
        self.visual_features = self._build_visual_features(image_paths)

        self.feature_matrix = np.concatenate(
            [
                self.graph_features * self.graph_weight,
                self.visual_features * self.visual_weight,
            ],
            axis=1,
        )
        return self

    def _compute_similarity(self, query_matrix: np.ndarray) -> np.ndarray:
        if self.feature_matrix is None:
            raise ValueError("Retriever has not been fitted")
        return SIMILARITY_FUNCS[self.similarity](query_matrix, self.feature_matrix)

    def query_by_index(self, index: int, top_k: int = 5) -> list[int]:
        """Return indices of the top ``k`` most similar items to ``index``."""

        if self.feature_matrix is None:
            raise ValueError("Retriever has not been fitted")
        if index < 0 or index >= len(self.feature_matrix):
            raise IndexError("Query index out of range")
        query_vec = self.feature_matrix[index][None, :]
        scores = self._compute_similarity(query_vec)[0]
        ranking = np.argsort(scores)[::-1]
        return ranking[:top_k].tolist()

    def query_by_tokens(self, tokens: Iterable[str], top_k: int = 5) -> list[int]:
        """Run a search conditioned on attribute tokens (e.g., ``"shape=circle"``)."""

        if self.graph is None or self.feature_matrix is None:
            raise ValueError("Retriever has not been fitted")
        adjacency = self.graph.adjacency_matrix()
        vocab = list(adjacency.index)
        vocab_index = {token: idx for idx, token in enumerate(vocab)}
        query = np.zeros((1, len(vocab)), dtype=np.float32)
        for token in tokens:
            if token in vocab_index:
                query[0, vocab_index[token]] += 1.0
                query += adjacency.loc[token].to_numpy(dtype=np.float32)[None, :]
        query_norm = np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
        graph_part = (query / query_norm) * self.graph_weight
        zero_visual = np.zeros((1, self.visual_features.shape[1]), dtype=np.float32)
        query_vec = np.concatenate([graph_part, zero_visual], axis=1)
        scores = self._compute_similarity(query_vec)[0]
        ranking = np.argsort(scores)[::-1]
        return ranking[:top_k].tolist()

    def evaluate(
        self,
        query_indices: Sequence[int],
        top_k: int = 5,
        attribute: str = "shape",
    ) -> dict[str, float]:
        """Evaluate retrieval accuracy and latency for a subset of queries.

        Returns a dictionary containing ``accuracy`` (proportion of queries where
        the correct attribute matches appear within ``top_k``), ``latency_ms``
        (average per-query wall time) and ``num_queries``.
        """

        if self.feature_matrix is None:
            raise ValueError("Retriever has not been fitted")

        hits = 0
        timings: list[float] = []
        for idx in query_indices:
            target_value = self.metadata[idx].get(attribute)
            start = time.perf_counter()
            retrieved = self.query_by_index(idx, top_k=top_k)
            elapsed = (time.perf_counter() - start) * 1000
            timings.append(elapsed)
            if any(self.metadata[r].get(attribute) == target_value for r in retrieved if r != idx):
                hits += 1
        accuracy = hits / len(query_indices) if query_indices else 0.0
        latency = float(np.mean(timings)) if timings else 0.0
        return {"accuracy": accuracy, "latency_ms": latency, "num_queries": len(query_indices)}

    def to_networkx(self) -> nx.Graph:
        """Return the underlying NetworkX graph for custom processing."""

        if self.graph is None:
            raise ValueError("Graph has not been initialised; call `fit` first")
        return self.graph.graph
