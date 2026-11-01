#!/usr/bin/env python3
"""Tests for ./desktop_tts.py.

No network, no real Piper or Chatterbox: both engines are fake objects —
Piper's `synthesize_wav` writes a tiny fixed WAV, Chatterbox's `generate`
returns a tiny fixed tensor — so this module must be importable and
testable under plain system python3, which has neither piper-tts nor
chatterbox-tts installed (those live only in ~/.mvg/venvs/chatterbox
on the desktop, ELBERETH 2026-09-27).
Run: python3 -m pytest ./tests -v
"""
import http.client
import io
import json
import sys
import threading
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import desktop_tts  # noqa: E402


class _FakePiperVoice:
    """Stands in for piper.PiperVoice — writes silence, but a real WAV."""

    def synthesize_wav(self, text, wav_file, syn_config=None, set_wav_format=True,
                        include_alignments=False):
        wav_file.setframerate(22050)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(b"\x00\x00" * 100)
        return None


class _FakeTensor:
    """Stands in for the torch.Tensor chatterbox.generate() returns —
    shape (1, N) float32. Only the methods desktop_tts actually calls."""

    def __init__(self, values):
        self._values = values

    def squeeze(self, _dim):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np
        return np.array(self._values, dtype="float32")


class _FakeChatterboxModel:
    sr = 24000

    def __init__(self, fail=False):
        self.fail = fail

    def generate(self, text):
        if self.fail:
            raise RuntimeError("boom")
        return _FakeTensor([0.0, 0.25, -0.25, 0.5] * 25)


@pytest.fixture()
def loaded_piper(monkeypatch):
    monkeypatch.setattr(desktop_tts, "_piper_voice", _FakePiperVoice())
    monkeypatch.setattr(desktop_tts, "_piper_load_error", None)
    yield


@pytest.fixture()
def tts_server(monkeypatch, loaded_piper):
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", None)
    server = desktop_tts.Server(("127.0.0.1", 0), desktop_tts.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


def test_tts_returns_wav_via_piper_when_chatterbox_not_loaded(tts_server):
    conn = http.client.HTTPConnection("127.0.0.1", tts_server, timeout=10)
    payload = json.dumps({"text": "hello from the desktop"})
    conn.request("POST", "/tts", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "audio/wav"
    assert resp.getheader("X-TTS-Engine") == "piper"
    assert body.startswith(b"RIFF")
    with wave.open(io.BytesIO(body), "rb") as w:
        assert w.getframerate() == 22050
    conn.close()


def test_tts_prefers_chatterbox_when_loaded(monkeypatch, tts_server):
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", _FakeChatterboxModel())
    conn = http.client.HTTPConnection("127.0.0.1", tts_server, timeout=10)
    payload = json.dumps({"text": "hello from the desktop"})
    conn.request("POST", "/tts", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("X-TTS-Engine") == "chatterbox"
    assert body.startswith(b"RIFF")
    with wave.open(io.BytesIO(body), "rb") as w:
        assert w.getframerate() == 24000
    conn.close()


def test_tts_falls_back_to_piper_when_text_exceeds_chatterbox_max_chars(monkeypatch, tts_server):
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", _FakeChatterboxModel())
    monkeypatch.setattr(desktop_tts, "CHATTERBOX_MAX_CHARS", 10)
    conn = http.client.HTTPConnection("127.0.0.1", tts_server, timeout=10)
    payload = json.dumps({"text": "this text is longer than the cap"})
    conn.request("POST", "/tts", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("X-TTS-Engine") == "piper"
    assert body.startswith(b"RIFF")
    conn.close()


def test_tts_falls_back_to_piper_when_chatterbox_synth_fails(monkeypatch, tts_server):
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", _FakeChatterboxModel(fail=True))
    conn = http.client.HTTPConnection("127.0.0.1", tts_server, timeout=10)
    payload = json.dumps({"text": "hello from the desktop"})
    conn.request("POST", "/tts", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("X-TTS-Engine") == "piper"
    assert body.startswith(b"RIFF")
    conn.close()


def test_tts_empty_text_is_400(tts_server):
    conn = http.client.HTTPConnection("127.0.0.1", tts_server, timeout=10)
    payload = json.dumps({"text": ""})
    conn.request("POST", "/tts", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 400
    conn.close()


def test_tts_missing_text_is_400(tts_server):
    conn = http.client.HTTPConnection("127.0.0.1", tts_server, timeout=10)
    payload = json.dumps({})
    conn.request("POST", "/tts", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 400
    conn.close()


def test_health_ok_when_piper_loaded(tts_server):
    conn = http.client.HTTPConnection("127.0.0.1", tts_server, timeout=10)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert resp.status == 200
    assert data["status"] == "ok"
    assert data["piper_loaded"] is True
    assert data["chatterbox_loaded"] is False
    conn.close()


def test_health_error_when_nothing_loaded(monkeypatch):
    monkeypatch.setattr(desktop_tts, "_piper_voice", None)
    monkeypatch.setattr(desktop_tts, "_piper_load_error", "voice model not found: nope.onnx")
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", None)
    server = desktop_tts.Server(("127.0.0.1", 0), desktop_tts.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        data = json.loads(resp.read())
        assert resp.status == 503
        assert data["status"] == "error"
        conn.close()
    finally:
        server.shutdown()


def test_synthesize_wav_bytes_raises_when_nothing_loaded(monkeypatch):
    monkeypatch.setattr(desktop_tts, "_piper_voice", None)
    monkeypatch.setattr(desktop_tts, "_piper_load_error", "voice model not found")
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", None)
    with pytest.raises(RuntimeError):
        desktop_tts.synthesize_wav_bytes("hi")


def test_tts_no_cap_by_default_uses_chatterbox_for_long_text(monkeypatch, tts_server):
    monkeypatch.setattr(desktop_tts, "_chatterbox_model", _FakeChatterboxModel())
    monkeypatch.setattr(desktop_tts, "CHATTERBOX_MAX_CHARS", 0)
    conn = http.client.HTTPConnection("127.0.0.1", tts_server, timeout=10)
    conn.request("POST", "/tts", body=json.dumps({"text": "x" * 400}), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    assert resp.getheader("X-TTS-Engine") == "chatterbox"
    conn.close()


# --------------------------------------------------------------------------
# synthesize_wav_bytes_chunked -- sentence-parallel synthesis for long
# multi-sentence inputs (specs/2026-09-28-mandos-streaming.md).
# --------------------------------------------------------------------------

def test_synthesize_wav_bytes_chunked_short_text_single_call(monkeypatch):
    calls = []

    def fake(text):
        calls.append(text)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(b"\x00\x00" * 4)
        return buf.getvalue(), "fake"

    monkeypatch.setattr(desktop_tts, "synthesize_wav_bytes", fake)
    desktop_tts.synthesize_wav_bytes_chunked("Short.")
    assert calls == ["Short."]


def test_synthesize_wav_bytes_chunked_splits_long_multi_sentence_text(monkeypatch):
    calls = []

    def fake(text):
        calls.append(text)
        marker = text.encode("utf-8")[:8].ljust(8, b"_")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes(marker * 4)
        return buf.getvalue(), "fake"

    monkeypatch.setattr(desktop_tts, "synthesize_wav_bytes", fake)
    text = (
        "This sentence is long enough on its own. "
        "Here is a second sentence that adds more length. "
        "And a third sentence closes out the paragraph nicely."
    )
    assert len(text) >= desktop_tts.SENTENCE_CHUNK_MIN_CHARS
    wav, engine = desktop_tts.synthesize_wav_bytes_chunked(text)
    expected_sentences = desktop_tts.segment_full_text(text)
    assert sorted(calls) == sorted(expected_sentences)
    assert engine == "fake"
    with wave.open(io.BytesIO(wav), "rb") as w:
        combined = w.readframes(w.getnframes())
    expected = b"".join(
        s.encode("utf-8")[:8].ljust(8, b"_") * 4 for s in expected_sentences
    )
    assert combined == expected


def test_synthesize_wav_bytes_chunked_falls_back_on_format_mismatch(monkeypatch):
    def fake(text):
        rate = 16000 if len(text) % 2 else 22050
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"\x00\x00" * 4)
        return buf.getvalue(), "fake"

    monkeypatch.setattr(desktop_tts, "synthesize_wav_bytes", fake)
    text = (
        "This sentence is long enough on its own. "
        "Here is a second sentence that adds more length. "
        "And a third sentence closes out the paragraph nicely."
    )
    wav, engine = desktop_tts.synthesize_wav_bytes_chunked(text)
    with wave.open(io.BytesIO(wav), "rb") as w:
        assert w.getframerate() in (16000, 22050)  # fell back to one serial call
