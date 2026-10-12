"""Tests for txt/markdown text extraction."""

from __future__ import annotations

import pytest

from text2speech.textio import (
    TextInputError,
    decode_bytes,
    detect_kind,
    extract_text,
    markdown_to_text,
    normalize_text,
)


class TestMarkdownToText:
    def test_headings_and_paragraphs(self):
        md = "# Title\n\nSome **bold** and *italic* text."
        text = markdown_to_text(md)
        assert "Title" in text
        assert "Some bold and italic text." in text
        assert "#" not in text
        assert "**" not in text

    def test_links_and_images(self):
        md = "See [OpenRouter](https://openrouter.ai) and ![logo](https://x/y.png)."
        text = markdown_to_text(md)
        assert "See OpenRouter and" in text
        assert "https://" not in text

    def test_lists(self):
        md = "- one\n- two\n\n1. first\n2. second"
        text = markdown_to_text(md)
        assert "one" in text and "two" in text
        assert "first" in text and "second" in text
        assert "-" not in text

    def test_code_blocks_dropped_by_default(self):
        md = "Before.\n\n```python\nprint('secret code')\n```\n\nAfter."
        text = markdown_to_text(md)
        assert "Before." in text
        assert "After." in text
        assert "secret code" not in text
        assert "```" not in text

    def test_code_blocks_kept_when_requested(self):
        md = "Before.\n\n```python\nprint('kept code')\n```\n\nAfter."
        text = markdown_to_text(md, include_code_blocks=True)
        assert "kept code" in text

    def test_inline_code_and_strikethrough(self):
        md = "Use `pip install uv` now, ~~old~~ new."
        text = markdown_to_text(md)
        assert "Use pip install uv now, old new." == text

    def test_tables(self):
        md = "| Name | Role |\n| --- | --- |\n| Ada | engineer |\n| Alan | mathematician |"
        text = markdown_to_text(md)
        assert "Name" in text and "Role" in text
        assert "Ada" in text and "engineer" in text
        assert "---" not in text
        assert "|" not in text

    def test_blockquote(self):
        md = "> quoted wisdom\n\nplain"
        text = markdown_to_text(md)
        assert "quoted wisdom" in text
        assert ">" not in text

    def test_html_comments_removed(self):
        md = "visible <!-- hidden note -->"
        assert markdown_to_text(md) == "visible"

    def test_empty_input(self):
        assert markdown_to_text("") == ""

    def test_crlf_normalized(self):
        assert markdown_to_text("# a\r\n\r\nb") == "a\nb"


class TestDetectKind:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("notes.txt", "txt"),
            ("README.md", "markdown"),
            ("doc.markdown", "markdown"),
            ("/some/dir/weird.name.TXT", "txt"),
        ],
    )
    def test_known_extensions(self, name, expected):
        assert detect_kind(name) == expected

    def test_unknown_extension_raises(self):
        with pytest.raises(TextInputError):
            detect_kind("audio.mp3")

    def test_declared_kind_wins(self):
        assert detect_kind("noext", "markdown") == "markdown"


class TestDecodeBytes:
    def test_utf8_with_bom(self):
        assert decode_bytes("\ufeffhéllo".encode("utf-8")) == "héllo"

    def test_gbk_fallback(self):
        assert decode_bytes("中文".encode("gb18030")) == "中文"

    def test_latin1_fallback(self):
        assert decode_bytes("café".encode("latin-1")) == "café"


class TestExtractText:
    def test_from_txt_file(self):
        result = extract_text(file_name="a.txt", file_bytes=b"  line one\r\nline two  ")
        assert result.text == "line one\nline two"
        assert result.kind == "txt"
        assert result.characters == len("line one\nline two")

    def test_from_markdown_file_strips_syntax(self):
        md = b"# Heading\n\n**bold** body"
        result = extract_text(file_name="a.md", file_bytes=md)
        assert result.kind == "markdown"
        assert "Heading" in result.text and "bold body" in result.text
        assert "#" not in result.text

    def test_markdown_file_can_keep_code(self):
        md = b"before\n\n```\ncode here\n```\n\nafter"
        without = extract_text(file_name="a.md", file_bytes=md)
        assert "code here" not in without.text
        assert "before" in without.text and "after" in without.text
        with_code = extract_text(
            file_name="a.md", file_bytes=md, include_code_blocks=True
        )
        assert "code here" in with_code.text

    def test_code_only_markdown_gives_helpful_error(self):
        with pytest.raises(TextInputError, match="code blocks"):
            extract_text(file_name="a.md", file_bytes=b"```\nonly code\n```")

    def test_raw_text_used_as_is(self):
        result = extract_text(raw_text="# not parsed, just text")
        assert result.text == "# not parsed, just text"
        assert result.kind == "txt"

    def test_no_input_raises(self):
        with pytest.raises(TextInputError):
            extract_text(raw_text="   ")

    def test_empty_file_raises(self):
        with pytest.raises(TextInputError):
            extract_text(file_name="a.txt", file_bytes=b"")

    def test_max_chars_enforced(self):
        with pytest.raises(TextInputError, match="too long"):
            extract_text(raw_text="x" * 50, max_chars=10)

    def test_normalize_text(self):
        assert normalize_text("  a  \r\n  b  ") == "a\nb"
