"""Chunking strategies.

Two strategies behind one dispatch, selected by chunking.strategy in config. The token-vs-structure
trade-off for clause-numbered contracts is framed in docs/design.md and is your call.

Both strategies emit `Chunk`s carrying a `char_span` (offsets into the source contract text) so a
retrieved chunk can be cited and scored against CUAD's annotated answer spans.
"""
from __future__ import annotations

import re
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
    """Dispatch to the configured strategy."""
    if cfg.strategy == "token":
        return _token_chunks(contract_id, text, cfg)
    if cfg.strategy == "structure":
        return _structure_chunks(contract_id, text, cfg)
    raise ValueError(f"unknown chunking.strategy: {cfg.strategy!r}")


def _token_windows(text: str, cfg: ChunkingConfig) -> list[tuple[str, tuple[int, int]]]:
    """Sliding token windows over `text` -> [(chunk_text, (char_start, char_end)), ...].

    Windows of cfg.max_tokens tokens, advancing by (max_tokens - overlap_tokens). Char spans are
    recovered by decoding the token prefix (exact for ASCII-dominant legal text). Windows whose
    trimmed text is shorter than cfg.min_chunk_chars are dropped as boilerplate/whitespace.
    """
    import tiktoken

    enc = tiktoken.get_encoding(cfg.tokenizer)
    tokens = enc.encode(text)
    n = len(tokens)
    if n == 0:
        return []

    step = max(1, cfg.max_tokens - cfg.overlap_tokens)
    windows: list[tuple[str, tuple[int, int]]] = []
    for start in range(0, n, step):
        end = min(start + cfg.max_tokens, n)
        chunk_text = enc.decode(tokens[start:end])
        char_start = len(enc.decode(tokens[:start]))
        char_end = char_start + len(chunk_text)
        if len(chunk_text.strip()) >= cfg.min_chunk_chars:
            windows.append((chunk_text, (char_start, char_end)))
        if end == n:
            break
    return windows


def _token_chunks(contract_id: str, text: str, cfg: ChunkingConfig) -> list[Chunk]:
    """Fixed-size token windows with overlap (model-agnostic baseline)."""
    return [
        Chunk(
            contract_id=contract_id,
            chunk_id=f"{contract_id}:{ordinal}",
            text=chunk_text,
            char_span=span,
        )
        for ordinal, (chunk_text, span) in enumerate(_token_windows(text, cfg))
    ]


# Numbered clause / section / article headings. CUAD formatting is inconsistent, so this is a
# best-effort boundary detector; oversized sections are still token-sub-split below.
_HEADING = re.compile(
    r"^[ \t]*("
    r"(?:\d+\.)+\d*"          # 1.  /  1.1  /  2.3.4
    r"|\d+\.\s"               # 1. with trailing space
    r"|ARTICLE\s+[\w.]+"      # ARTICLE 5 / ARTICLE V
    r"|SECTION\s+[\w.]+"      # SECTION 9
    r")\s",
    re.MULTILINE | re.IGNORECASE,
)


def _structure_chunks(contract_id: str, text: str, cfg: ChunkingConfig) -> list[Chunk]:
    """Structure-aware chunking on clause/section boundaries.

    Split at detected headings; emit one chunk per section, but token-sub-split any section that
    exceeds max_tokens so no chunk blows the budget. Falls back to pure token windows when no
    headings are found.
    """
    starts = [m.start() for m in _HEADING.finditer(text)]
    if not starts:
        return _token_chunks(contract_id, text, cfg)

    # Section boundaries: [first_heading, next_heading, ..., end]. Any preamble before the first
    # heading is captured as its own leading section.
    bounds = ([0] if starts[0] > 0 else []) + starts + [len(text)]

    chunks: list[Chunk] = []
    ordinal = 0
    for i in range(len(bounds) - 1):
        sec_start, sec_end = bounds[i], bounds[i + 1]
        section_text = text[sec_start:sec_end]
        if len(section_text.strip()) < cfg.min_chunk_chars:
            continue
        label = section_text.strip().split("\n", 1)[0][:60]

        for sub_text, (cs, ce) in _token_windows(section_text, cfg):
            chunks.append(
                Chunk(
                    contract_id=contract_id,
                    chunk_id=f"{contract_id}:{ordinal}",
                    text=sub_text,
                    char_span=(sec_start + cs, sec_start + ce),
                    section=label,
                )
            )
            ordinal += 1
    return chunks
