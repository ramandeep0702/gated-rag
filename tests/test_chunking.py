"""Chunking strategies — char-span correctness + size discipline."""
from __future__ import annotations

from gated_rag.chunking import chunk_contract
from gated_rag.config import ChunkingConfig

TOKEN_CFG = ChunkingConfig(
    strategy="token", max_tokens=32, overlap_tokens=8, tokenizer="cl100k_base", min_chunk_chars=10
)
STRUCTURE_CFG = ChunkingConfig(
    strategy="structure", max_tokens=32, overlap_tokens=8, tokenizer="cl100k_base", min_chunk_chars=10
)

SAMPLE = (
    "1. Definitions. This Agreement is made between the parties identified below. "
    "The term 'Confidential Information' means any data disclosed by one party to the other. "
    "2. Term. This Agreement begins on the Effective Date and continues for three years. "
    "3. Termination. Either party may terminate upon thirty days written notice to the other party."
)


def test_token_chunks_have_valid_spans_into_source():
    chunks = chunk_contract("C1", SAMPLE, TOKEN_CFG)
    assert chunks, "expected at least one chunk"
    for ch in chunks:
        start, end = ch.char_span
        assert 0 <= start < end <= len(SAMPLE)
        assert ch.contract_id == "C1"
        # span text and chunk text should align (ASCII-dominant legal text)
        assert SAMPLE[start:end] == ch.text


def test_token_chunks_overlap_advances():
    chunks = chunk_contract("C1", SAMPLE, TOKEN_CFG)
    starts = [c.char_span[0] for c in chunks]
    assert starts == sorted(starts)
    assert len(set(c.chunk_id for c in chunks)) == len(chunks)  # unique ids


def test_structure_chunks_label_sections():
    chunks = chunk_contract("C1", SAMPLE, STRUCTURE_CFG)
    assert chunks
    # at least one chunk should be tagged with a numbered-clause section label
    assert any(c.section and c.section[0].isdigit() for c in chunks)
    for ch in chunks:
        start, end = ch.char_span
        assert 0 <= start < end <= len(SAMPLE)


def test_empty_text_yields_no_chunks():
    assert chunk_contract("C1", "", TOKEN_CFG) == []
