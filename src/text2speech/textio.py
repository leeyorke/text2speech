"""Reading and normalizing txt / markdown input into text suitable for TTS."""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass

import markdown as md_lib

# filename extension -> logical kind
EXTENSION_KINDS = {
    ".txt": "txt",
    ".text": "txt",
    ".md": "markdown",
    ".markdown": "markdown",
}

KIND_LABELS = {"txt": "Plain text", "markdown": "Markdown"}


class TextInputError(ValueError):
    """Raised when the input text/file is unusable."""


@dataclass(slots=True)
class ExtractedText:
    text: str
    kind: str
    source_name: str

    @property
    def characters(self) -> int:
        return len(self.text)


def detect_kind(filename: str | None, declared: str | None = None) -> str:
    """Determine the logical kind ('txt' / 'markdown') of an upload."""
    if declared in ("txt", "markdown"):
        return declared
    name = (filename or "").lower()
    for ext, kind in EXTENSION_KINDS.items():
        if name.endswith(ext):
            return kind
    raise TextInputError(
        f"Unsupported file type '{filename or declared}'. Supported: .txt, .md, .markdown"
    )


def decode_bytes(data: bytes) -> str:
    """Decode file bytes, tolerating BOMs and common Windows encodings."""
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    for encoding in ("gb18030", "big5", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


_FENCED_CODE_RE = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n.*?^[ \t]*\1[ \t]*$", re.S | re.M)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.M)
_BLOCKQUOTE_RE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.M)
_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+", re.M)
_ORDERED_RE = re.compile(r"^[ \t]*\d+[.)][ \t]+", re.M)
_TABLE_SEPARATOR_RE = re.compile(r"^[ \t]*\|?[ \t]*:?-{1,}:?[ \t]*(\|[ \t]*:?-{1,}:?[ \t]*)+\|?[ \t]*$", re.M)
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|(.*)\|[ \t]*$", re.M)
_HR_RE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.M)
_BOLD_RE = re.compile(r"(\*\*|__)(.*?)\1", re.S)
_ITALIC_RE = re.compile(r"(\*|_)(?=\S)(.*?\S)\1", re.S)
_STRIKE_RE = re.compile(r"~~(.*?)~~", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_TAG_RE = re.compile(r"<[^>]+>")


def markdown_to_text(markdown_text: str, *, include_code_blocks: bool = False) -> str:
    """Convert markdown to plain, speakable text.

    Fenced code blocks are dropped by default (code read aloud is noise);
    set ``include_code_blocks=True`` to keep their content.
    """
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    if not include_code_blocks:
        text = _FENCED_CODE_RE.sub("\n", text)

    text = _HTML_COMMENT_RE.sub("", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _TABLE_SEPARATOR_RE.sub("", text)
    text = _TABLE_ROW_RE.sub(lambda m: " " + m.group(1).replace("|", " ") + " ", text)
    text = _HR_RE.sub("", text)
    text = _HEADING_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _ORDERED_RE.sub("", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _STRIKE_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\2", text)

    # Let the markdown parser handle whatever structure remains, then flatten.
    html_text = md_lib.markdown(text, extensions=["extra", "sane_lists"])
    plain = _TAG_RE.sub(" ", html_text)
    plain = html_lib.unescape(plain)

    plain = re.sub(r"[ \t]+", " ", plain)
    plain = re.sub(r" ?\n ?", "\n", plain)
    plain = re.sub(r"\n{3,}", "\n\n", plain)
    return plain.strip()


_SENTENCE_END_CHARS = "。！？；…!?;."
# A sentence boundary: end punctuation (optionally followed by closing quotes/brackets)
# then whitespace, or a newline of any kind.
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[" + re.escape(_SENTENCE_END_CHARS) + r"\"'”’)\]】])[\s]+|[\r\n]+"
)
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


def split_text(text: str, max_chars: int) -> list[str]:
    """Split text into sentence-aware chunks of at most ``max_chars`` characters.

    Boundaries are preferred in this order: paragraph breaks, then sentence-ending
    punctuation, then whitespace; a single sentence longer than ``max_chars`` is
    hard-cut. The concatenation of all chunks preserves the original text (modulo
    whitespace at the seams).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not text or not text.strip():
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in _PARAGRAPH_SPLIT_RE.split(text):
        if not paragraph.strip():
            continue
        if len(paragraph) <= max_chars:
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush()
                current = paragraph
            continue

        for sentence in _SENTENCE_BOUNDARY_RE.split(paragraph):
            if not sentence.strip():
                continue
            if len(sentence) > max_chars:
                # A single sentence longer than the limit: flush what we have and
                # hard-cut the sentence into exact-size pieces.
                flush()
                for start in range(0, len(sentence), max_chars):
                    chunks.append(sentence[start : start + max_chars])
                continue
            candidate = f"{current} {sentence}" if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush()
                current = sentence
    flush()
    return chunks


def normalize_text(text: str) -> str:
    """Normalize raw pasted text (uniform newlines, trim per-line whitespace)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def extract_text(
    *,
    raw_text: str | None = None,
    file_name: str | None = None,
    file_bytes: bytes | None = None,
    declared_kind: str | None = None,
    include_code_blocks: bool = False,
    max_chars: int | None = None,
) -> ExtractedText:
    """Produce speakable text from either raw text or an uploaded .txt/.md file."""
    if file_bytes is not None:
        if not file_bytes:
            raise TextInputError("Uploaded file is empty.")
        kind = detect_kind(file_name, declared_kind)
        decoded = decode_bytes(file_bytes)
        if kind == "markdown":
            text = markdown_to_text(decoded, include_code_blocks=include_code_blocks)
        else:
            text = normalize_text(decoded)
        source_name = file_name or "input.txt"
    elif raw_text is not None and raw_text.strip():
        # Pasted text is used as-is; only markdown *files* get markdown treatment.
        text = normalize_text(raw_text)
        kind = declared_kind if declared_kind in ("txt", "markdown") else "txt"
        source_name = "pasted text"
    else:
        raise TextInputError("No input provided. Paste text or upload a .txt/.md file.")

    if not text.strip():
        if kind == "markdown" and not include_code_blocks:
            raise TextInputError(
                "Input contained no speakable text. Markdown code blocks are excluded "
                "by default — enable 'Read code blocks aloud' to include them."
            )
        raise TextInputError("Input contained no speakable text after processing.")

    if max_chars is not None and len(text) > max_chars:
        raise TextInputError(
            f"Input is too long: {len(text)} characters (limit {max_chars}). "
            "Split it into smaller chunks or raise T2S_MAX_TEXT_CHARS."
        )
    return ExtractedText(text=text, kind=kind, source_name=source_name)
