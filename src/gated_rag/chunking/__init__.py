"""Chunking strategies.

Two strategies behind one dispatch, selected by chunking.strategy in config. The token-vs-structure
trade-off for clause-numbered contracts is framed in docs/design.md and is your call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..config import ChunkingConfig


@dataclass(frozen=True)
class Chunk:
    """A chunk of one contract, carrying enough to cite it later."""
    contract_id: str
    chunk_id: str                      # stable id, e.g. f"{contract_id}:{ordinal}"
    text: str
    char_span: tuple[int, int]         # (start, end) offsets into the source contract text
    section: Optional[str] = None      # clause/section label when known (structure strategy)


def chunk_contract(contract_id: str, text: str, cfg: ChunkingConfig) -> list[Chunk]:
    """Dispatch to the configured strategy.

    Hint: if cfg.strategy == 'token' -> _token_chunks; 'structure' -> _structure_chunks; else ValueError.
    """
    # TODO: dispatch on cfg.strategy.
    raise NotImplementedError


def _token_chunks(contract_id: str, text: str, cfg: ChunkingConfig) -> list[Chunk]:
    """Fixed-size token windows with overlap.

    Hint: tiktoken.get_encoding(cfg.tokenizer); slide max_tokens with overlap_tokens; map token
    windows back to char_span; drop chunks shorter than cfg.min_chunk_chars.
    """
    # TODO: sliding-window tokenization -> Chunk list.
    raise NotImplementedError


def _structure_chunks(contract_id: str, text: str, cfg: ChunkingConfig) -> list[Chunk]:
    """Structure-aware chunking on clause/section boundaries.

    Hint: detect numbered clause headings (regex), split there, then sub-split oversized clauses
    with _token_chunks so no chunk exceeds max_tokens.
    """
    # TODO: boundary detection -> per-clause chunks, sub-split if too long.
    raise NotImplementedError
