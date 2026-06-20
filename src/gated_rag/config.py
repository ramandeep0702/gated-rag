"""Typed config schema + loader.

These dataclasses mirror configs/default.yaml field-for-field. The YAML is the source of
truth; this module gives typed access and one place to validate invariants (e.g. embedding.dim
matches the chosen model, retrieval.backend has a matching sub-config).
"""
from __future__ import annotations

import re
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
    query_prefix: str = ""   # instruction prepended to queries only (BGE-style asymmetric retrieval)


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


# A threshold key is a recognized metric if it is mrr, groundedness, or recall_at_<k>.
_RECALL_KEY = re.compile(r"^recall_at_\d+$")
_KNOWN_METRICS = {"mrr", "groundedness"}


def _is_known_metric_key(name: str) -> bool:
    return name in _KNOWN_METRICS or bool(_RECALL_KEY.match(name))


def load_config(path: str = "configs/default.yaml") -> Config:
    """Parse YAML at `path` into a typed Config.

    yaml.safe_load -> build nested dataclasses -> validate(cfg) before returning. The original
    parsed dict is kept on Config.raw for round-tripping / passing through to notebooks.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    r = raw["retrieval"]
    e = raw["eval"]
    cfg = Config(
        corpus=CorpusConfig(**raw["corpus"]),
        catalog=CatalogConfig(**raw["catalog"]),
        chunking=ChunkingConfig(**raw["chunking"]),
        embedding=EmbeddingConfig(**raw["embedding"]),
        retrieval=RetrievalConfig(
            backend=r["backend"],
            top_k=r["top_k"],
            faiss=FaissConfig(**r["faiss"]),
            vector_search=VectorSearchConfig(**r["vector_search"]),
        ),
        eval=EvalConfig(
            golden_path=e["golden_path"],
            k_values=e["k_values"],
            metrics=e["metrics"],
            gate=GateConfig(**e["gate"]),
        ),
        run=RunConfig(
            seed=raw["run"]["seed"],
            mlflow=MlflowConfig(**raw["run"]["mlflow"]),
        ),
        raw=raw,
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Fail fast on incoherent config. Raises ValueError with an actionable message."""
    if cfg.retrieval.backend not in {"faiss", "vector_search"}:
        raise ValueError(
            f"retrieval.backend must be 'faiss' or 'vector_search', got {cfg.retrieval.backend!r}"
        )
    if cfg.chunking.strategy not in {"token", "structure"}:
        raise ValueError(
            f"chunking.strategy must be 'token' or 'structure', got {cfg.chunking.strategy!r}"
        )
    if cfg.chunking.overlap_tokens >= cfg.chunking.max_tokens:
        raise ValueError(
            f"chunking.overlap_tokens ({cfg.chunking.overlap_tokens}) must be < "
            f"max_tokens ({cfg.chunking.max_tokens}) or the window never advances"
        )
    if cfg.embedding.dim <= 0:
        raise ValueError(f"embedding.dim must be positive, got {cfg.embedding.dim}")
    if cfg.embedding.batch_size <= 0:
        raise ValueError(f"embedding.batch_size must be positive, got {cfg.embedding.batch_size}")
    if cfg.retrieval.top_k <= 0:
        raise ValueError(f"retrieval.top_k must be positive, got {cfg.retrieval.top_k}")
    if cfg.retrieval.faiss.metric not in {"ip", "l2"}:
        raise ValueError(
            f"retrieval.faiss.metric must be 'ip' or 'l2', got {cfg.retrieval.faiss.metric!r}"
        )
    # 'ip' similarity only equals cosine when vectors are normalized — catch the mismatch early.
    if cfg.retrieval.faiss.metric == "ip" and not cfg.embedding.normalize:
        raise ValueError(
            "retrieval.faiss.metric == 'ip' assumes normalized vectors for cosine similarity, "
            "but embedding.normalize is false. Set normalize: true or use metric: l2."
        )
    unknown = [k for k in cfg.eval.gate.thresholds if not _is_known_metric_key(k)]
    if unknown:
        raise ValueError(
            f"eval.gate.thresholds has unrecognized metric keys {unknown}; "
            "expected 'mrr', 'groundedness', or 'recall_at_<k>'."
        )
    if cfg.eval.gate.on_failure not in {"fail", "warn"}:
        raise ValueError(
            f"eval.gate.on_failure must be 'fail' or 'warn', got {cfg.eval.gate.on_failure!r}"
        )
