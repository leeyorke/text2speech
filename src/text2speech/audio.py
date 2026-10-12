"""Merging per-chunk audio parts back into a single file.

WAV (and raw PCM) can be merged sample-accurately; container formats such as MP3
are byte-concatenated, which players handle as a continuous stream.
"""

from __future__ import annotations

import io
import wave


class AudioMergeError(RuntimeError):
    """Raised when audio parts cannot be merged (e.g. mismatched WAV parameters)."""


def merge_audio(parts: list[bytes], fmt: str) -> bytes:
    """Merge audio parts produced for the same request into one byte string."""
    if not parts:
        raise AudioMergeError("No audio parts to merge.")
    if len(parts) == 1:
        return parts[0]

    fmt = (fmt or "").lower().lstrip(".")
    if fmt == "wav":
        return _merge_wav(parts)
    # mp3 / opus / aac / flac / pcm: concatenating the byte streams yields a
    # playable file (WAV would need its header rewritten, hence the branch above).
    return b"".join(parts)


def _wave_format_key(params: wave._wave_params) -> tuple:
    """Format identity of a WAV file: everything except the frame count."""
    return (params.nchannels, params.sampwidth, params.framerate, params.comptype)


def _merge_wav(parts: list[bytes]) -> bytes:
    """Concatenate WAV payloads frame-by-frame, keeping the first file's format."""
    params = None
    frames: list[bytes] = []
    for index, part in enumerate(parts, start=1):
        try:
            with wave.open(io.BytesIO(part), "rb") as reader:
                part_params = reader.getparams()
                if params is None:
                    params = part_params
                elif _wave_format_key(part_params) != _wave_format_key(params):
                    # channels / sample width / sample rate / compression must match.
                    raise AudioMergeError(
                        f"Audio part {index} has different WAV parameters "
                        f"({part_params.nchannels}ch {part_params.framerate}Hz "
                        f"{part_params.sampwidth * 8}-bit vs "
                        f"{params.nchannels}ch {params.framerate}Hz "
                        f"{params.sampwidth * 8}-bit); cannot merge. "
                        "Use a single provider/format for the whole text."
                    )
                frames.append(reader.readframes(reader.getnframes()))
        except wave.Error as exc:
            raise AudioMergeError(f"Audio part {index} is not a valid WAV file: {exc}") from exc

    assert params is not None
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams(params)
        for chunk in frames:
            writer.writeframes(chunk)
    return output.getvalue()
