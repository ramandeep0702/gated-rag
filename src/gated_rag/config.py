"""Typed config schema + loader.

These dataclasses mirror configs/default.yaml field-for-field. The YAML is the source of
truth; this module gives typed access and one place to validate invariants (e.g. embedding.dim
matches the chosen model, retrieval.backend has a matching sub-config).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CorpusConfig:
    source: str
    hf_dataset: str
    hf_split: str
    limit: Optional[int]


@dataclass
class CatalogConfig:
    catalog: str
    schema: str
    bronze_table: str
    silver_table: str
    gold_table: str


@dataclass
class ChunkingConfig:
    strategy: str            # "token" | "structure"
    max_tokens: int
    overlap_tokens: int
    tokenizer: str
    min_chunk_chars: int


@dataclass
class EmbeddingConfig:
    backend: str
    model: str
    batch_size: int
    normalize: bool
    dim: int


@dataclass
class FaissConfig:
    index_type: str
    metric: str
    persist_path: str


@dataclass
class VectorSearchConfig:
    endpoint_name: str
    index_name: str
    embedding_source: str


@dataclass
class RetrievalConfig:
    backend: str             # "faiss" | "vector_search"
    top_k: int
    faiss: FaissConfig
    vector_search: VectorSearchConfig


@dataclass
class GateConfig:
    enabled: bool
    on_failure: str          # "fail" | "warn"
    thresholds: dict[str, float]


@dataclass
class EvalConfig:
    golden_path: str
    k_values: list[int]
    metrics: list[str]
    gate: GateConfig


@dataclass
class MlflowConfig:
    enabled: bool
    experiment: str


@dataclass
class RunConfig:
    seed: int
    mlflow: MlflowConfig


@dataclass
class Config:
    corpus: CorpusConfig
    catalog: CatalogConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    eval: EvalConfig
    run: RunConfig
    raw: dict[str, Any] = field(default_factory=dict)  # original parsed YAML, for round-tripping


def load_config(path: str = "configs/default.yaml") -> Config:
    """Parse YAML at `path` into a typed Config.

    Hint: yaml.safe_load -> build nested dataclasses -> call validate(cfg) before returning.
    """
    # TODO: read file, yaml.safe_load, construct dataclasses, keep the raw dict on Config.raw.
    raise NotImplementedError


def validate(cfg: Config) -> None:
    """Fail fast on incoherent config.

    Hint: assert retrieval.backend in {faiss, vector_search}; chunking.strategy in {token, structure};
    embedding.dim consistent with the model; gate.thresholds keys are recognized metrics.
    """
    # TODO: raise ValueError with an actionable message on any violation.
    raise NotImplementedError
