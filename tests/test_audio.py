"""Tests for merging per-chunk audio parts."""

from __future__ import annotations

import io
import struct
import wave

import pytest

from text2speech.audio import AudioMergeError, merge_audio


def make_wav(seconds: float = 0.5, rate: int = 8000, freq: int = 440) -> bytes:
    frames = int(rate * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for i in range(frames):
            value = int(10000 * ((i // 100) % 2) - 5000)  # square-ish wave
            wav.writeframes(struct.pack("<h", value))
    return buffer.getvalue()


def make_wav_with_params(nchannels=1, sampwidth=2, rate=8000, frames=100) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(nchannels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x01" * (frames * nchannels * sampwidth // 2))
    return buffer.getvalue()


class TestMergeWav:
    def test_single_part_returned_as_is(self):
        part = make_wav(0.2)
        assert merge_audio([part], "wav") == part

    def test_two_wavs_merge_into_valid_wav(self):
        first, second = make_wav(0.5), make_wav(0.3)
        merged = merge_audio([first, second], "wav")

        with wave.open(io.BytesIO(merged), "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == 8000
            assert w.getnframes() == 4000 + 2400  # 0.5s + 0.3s at 8kHz

    def test_frames_are_concatenated_in_order(self):
        first, second = make_wav(0.1), make_wav(0.1)
        merged = merge_audio([first, second], "wav")
        with wave.open(io.BytesIO(first), "rb") as w1, wave.open(
            io.BytesIO(second), "rb"
        ) as w2, wave.open(io.BytesIO(merged), "rb") as wm:
            assert wm.readframes(wm.getnframes()) == w1.readframes(w1.getnframes()) + w2.readframes(
                w2.getnframes()
            )

    def test_mismatched_sample_rate_raises(self):
        parts = [make_wav(0.1, rate=8000), make_wav(0.1, rate=24000)]
        with pytest.raises(AudioMergeError, match="different WAV parameters"):
            merge_audio(parts, "wav")

    def test_mismatched_channels_raises(self):
        parts = [make_wav_with_params(nchannels=1), make_wav_with_params(nchannels=2)]
        with pytest.raises(AudioMergeError, match="different WAV parameters"):
            merge_audio(parts, "wav")

    def test_invalid_wav_raises(self):
        with pytest.raises(AudioMergeError, match="not a valid WAV"):
            merge_audio([b"not a wav file at all", make_wav(0.1)], "wav")

    def test_empty_parts_raises(self):
        with pytest.raises(AudioMergeError, match="No audio parts"):
            merge_audio([], "wav")


class TestMergeOtherFormats:
    def test_mp3_is_byte_concatenated(self):
        parts = [b"ID3-part-one", b"ID3-part-two", b"ID3-part-three"]
        assert merge_audio(parts, "mp3") == b"".join(parts)

    def test_pcm_is_byte_concatenated(self):
        parts = [b"\x00\x01", b"\x02\x03"]
        assert merge_audio(parts, "pcm") == b"\x00\x01\x02\x03"

    def test_opus_flac_aac_concatenated(self):
        for fmt in ("opus", "flac", "aac"):
            assert merge_audio([b"a", b"b"], fmt) == b"ab"

    def test_format_with_leading_dot(self):
        assert merge_audio([b"x", b"y"], ".mp3") == b"xy"
