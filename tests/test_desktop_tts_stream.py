#!/usr/bin/env python3
"""Tests for POST /tts/stream in desktop_tts.py (token-window streaming).

Same rules as test_desktop_tts.py: no Piper, no Chatterbox, no GPU; fakes
only, importable under plain system python3 (numpy is the one dependency).
Run: python3 -m pytest ./tests -v
"""
import http.client
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import desktop_tts  # noqa: E402
from test_desktop_tts import (  # noqa: E402,F401  (fixtures + fakes)
    _FakeChatterboxModel, loaded_piper, tts_server,
)

np = pytest.importorskip("numpy")


def _post(port, path, obj):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, body=json.dumps(obj), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


def test_wav_stream_header_is_44_bytes_with_unknown_length_sentinel():
    h = desktop_tts._wav_stream_header(24000)
    assert len(h) == 44 and h[:4] == b"RIFF" and h[8:12] == b"WAVE"
    assert struct.unpack("<I", h[4:8])[0] == 0xFFFFFFFF
    assert struct.unpack("<I", h[40:44])[0] == 0xFFFFFFFF
    assert struct.unpack("<IHHIIHH", h[16:36]) == (16, 1, 1, 24000, 48000, 2, 16)


def test_chunk_blender_preserves_length_and_continuity():
    ov = 100
    b = desktop_tts.ChunkBlender(ov)
    # A constant signal split into overlapping pieces must come back constant
    # with total length == the un-overlapped length.
    pieces = [np.ones(400 + ov, dtype="float32"), np.ones(400 + ov, dtype="float32"),
              np.ones(300, dtype="float32")]
    out = [b.push(pieces[0], False), b.push(pieces[1], False), b.push(pieces[2], True)]
    joined = np.concatenate(out)
    assert len(joined) == 400 + 400 + 300
    assert np.allclose(joined, 1.0, atol=1e-5)


def test_stream_falls_back_to_plain_tts_when_chatterbox_not_loaded(tts_server):
    resp, body = _post(tts_server, "/tts/stream", {"text": "hello"})
    assert resp.status == 200
    assert resp.getheader("X-TTS-Engine") == "piper"
    assert resp.getheader("Content-Length")
    assert body.startswith(b"RIFF")


def test_stream_empty_text_is_400(tts_server):
    resp, _ = _post(tts_server, "/tts/stream", {"text": "  "})
    assert resp.status == 400


def test_stream_yields_wav_header_then_pcm(monkeypatch, tts_server):
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", _FakeChatterboxModel())
    monkeypatch.setattr(desktop_tts, "stream_pcm_chunks",
                        lambda text, cancel=None: iter([b"\x01\x00" * 50, b"\x02\x00" * 30]))
    resp, body = _post(tts_server, "/tts/stream", {"text": "hello"})
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "audio/wav"
    assert resp.getheader("X-TTS-Engine") == "chatterbox-stream"
    assert resp.getheader("X-Sample-Rate") == "24000"
    assert body[:44] == desktop_tts._wav_stream_header(24000)
    assert len(body) == 44 + 160


def test_stream_pcm_format_has_no_header(monkeypatch, tts_server):
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", _FakeChatterboxModel())
    monkeypatch.setattr(desktop_tts, "stream_pcm_chunks",
                        lambda text, cancel=None: iter([b"\x01\x00" * 50]))
    resp, body = _post(tts_server, "/tts/stream", {"text": "hello", "format": "pcm"})
    assert resp.getheader("Content-Type") == "audio/L16"
    assert body == b"\x01\x00" * 50


def test_stream_failure_before_first_chunk_falls_back_to_tts(monkeypatch, tts_server):
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", _FakeChatterboxModel())

    def boom(text, cancel=None):
        raise RuntimeError("cuda oom")
        yield b""  # pragma: no cover

    monkeypatch.setattr(desktop_tts, "stream_pcm_chunks", boom)
    resp, body = _post(tts_server, "/tts/stream", {"text": "hello"})
    assert resp.status == 200
    assert resp.getheader("X-TTS-Engine") == "chatterbox"  # the plain /tts path (fake model)
    assert body.startswith(b"RIFF")


def test_stream_disabled_flag_serves_plain_tts(monkeypatch, tts_server):
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", _FakeChatterboxModel())
    monkeypatch.setattr(desktop_tts, "STREAM_ENABLED", False)
    resp, _ = _post(tts_server, "/tts/stream", {"text": "hello"})
    assert resp.getheader("X-TTS-Engine") == "chatterbox"
    assert resp.getheader("Content-Length")


def test_stream_sentence_windowing_schedule(monkeypatch):
    """Fake tokens + fake decoder: total samples == (tokens + 3 silence) * 960,
    the first window is 4 new + 2 lookahead tokens, later windows start with
    the configured context, and the last window is the final one."""
    monkeypatch.setattr(desktop_tts, "STREAM_FIRST_TOKENS", 4)
    monkeypatch.setattr(desktop_tts, "STREAM_CHUNK_TOKENS", 5)
    monkeypatch.setattr(desktop_tts, "STREAM_CONTEXT_TOKENS", 2)
    monkeypatch.setattr(desktop_tts, "STREAM_LOOKAHEAD_TOKENS", 2)
    monkeypatch.setattr(desktop_tts, "STREAM_OVERLAP_SAMPLES", 96)
    windows = []

    def fake_iter(model, text, cancel=None):
        yield from range(23)

    def fake_decode(model, tokens, final):
        windows.append((list(tokens), final))
        n = len(tokens) + (3 if final else 0)
        return np.full(n * 960, 0.5, dtype="float32")

    monkeypatch.setattr(desktop_tts, "_t3_token_iter", fake_iter)
    monkeypatch.setattr(desktop_tts, "_decode_window", fake_decode)
    chunks = list(desktop_tts._stream_sentence_pcm(type("M", (), {"sr": 24000})(), "x"))
    assert sum(len(c) for c in chunks) // 2 == (23 + 3) * 960
    assert windows[0] == (list(range(0, 6)), False)
    assert windows[1][0][0] == 2  # 2 context tokens before the second span (starts at 4)
    assert windows[-1][1] is True
    assert len(windows) >= 4
