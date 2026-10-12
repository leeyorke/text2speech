"""Tests for sentence-aware text splitting."""

from __future__ import annotations

import pytest

from text2speech.textio import split_text


def assert_within_limit_and_conserved(chunks, text, max_chars):
    """Hard cuts may split words, so only character-level conservation is guaranteed."""
    assert all(len(c) <= max_chars for c in chunks)
    total = sum(len(c) for c in chunks)
    assert total <= len(text)
    # Only whitespace at the seams may be lost (each seam keeps a single space).
    assert len(text) - total <= 4 * len(chunks) + 4


class TestSplitText:
    def test_short_text_is_single_chunk(self):
        text = "Hello world. This is short."
        assert split_text(text, 100) == [text]

    def test_empty_and_blank(self):
        assert split_text("", 100) == []
        assert split_text("   \n\n  ", 100) == []

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            split_text("abc", 0)

    def test_exact_boundary_is_single_chunk(self):
        text = "x" * 50
        assert split_text(text, 50) == [text]

    def test_english_sentence_boundaries(self):
        text = "One two three. Four five six! Seven eight nine? Ten eleven twelve."
        chunks = split_text(text, 30)
        assert len(chunks) > 1
        assert all(len(c) <= 30 for c in chunks)
        assert " ".join(chunks).split() == text.split()

    def test_chinese_sentence_boundaries(self):
        text = "今天天气很好。我们出去散步吧！你觉得怎么样？我记得你说过要早点回家。"
        chunks = split_text(text, 12)
        assert len(chunks) > 1
        assert all(len(c) <= 12 for c in chunks)
        assert "".join(chunks) == text  # Chinese has no seam whitespace

    def test_paragraph_boundaries_preferred(self):
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
        chunks = split_text(text, 30)
        assert len(chunks) >= 2
        assert all(len(c) <= 30 for c in chunks)
        assert " ".join(chunks).split() == text.split()

    def test_oversized_sentence_is_hard_cut(self):
        text = "A" * 250  # one gigantic "sentence"
        chunks = split_text(text, 100)
        assert chunks == ["A" * 100, "A" * 100, "A" * 50]
        assert sum(len(c) for c in chunks) == len(text)

    def test_mixed_oversized_and_normal(self):
        text = "short one. " + "B" * 120 + " tail end."
        chunks = split_text(text, 50)
        assert_within_limit_and_conserved(chunks, text, 50)
        assert chunks[0] == "short one."

    def test_semicolon_and_ellipsis_boundaries(self):
        text = "first part; second part; third part… fourth part."
        chunks = split_text(text, 20)
        assert len(chunks) > 1
        assert all(len(c) <= 20 for c in chunks)
        assert " ".join(chunks).split() == text.split()

    def test_long_realistic_document(self):
        sentences = [f"This is sentence number {i} with some words in it." for i in range(200)]
        text = " ".join(sentences)
        chunks = split_text(text, 2000)
        assert len(chunks) > 5
        assert all(len(c) <= 2000 for c in chunks)
        assert " ".join(chunks).split() == text.split()

    @pytest.mark.parametrize("max_chars", [10, 50, 200, 1000])
    def test_never_exceeds_limit(self, max_chars):
        text = ("Para one with words. " * 5) + "\n\n" + ("第二段中文内容。" * 8) + "\n\n" + "tail."
        chunks = split_text(text, max_chars)
        assert_within_limit_and_conserved(chunks, text, max_chars)
