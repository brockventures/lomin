#!/usr/bin/env python3
"""Tests for gateway.normalize_wav() (MIRRORMERE-ONE-FORMAT) and its use on
the wire in /ask/stream and /v1/audio/speech, so every audio payload the
gateway emits is 24 kHz mono 16-bit PCM WAV regardless of which TTS engine
(and therefore which native sample rate/channels/width) produced it.

Run: python3 -m pytest ./tests -v
"""
import base64
import http.client
import json
import struct
import sys
import threading
import time
import wave
from io import BytesIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402


def _make_wav(rate: int, channels: int, sampwidth: int, n_samples: int = 240,
              freq: float = 440.0) -> bytes:
    """A short deterministic tone (not silence -- makes resampling exercise
    real interpolation, not just zeros) at the given format."""
    import math

    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n_samples):
            t = i / rate
            val = int(16000 * math.sin(2 * math.pi * freq * t))
            for ch in range(channels):
                if sampwidth == 1:
                    frames.append((val >> 8) + 128 & 0xFF)
                elif sampwidth == 2:
                    frames += struct.pack("<h", val)
                elif sampwidth == 3:
                    v24 = val << 8
                    frames += v24.to_bytes(4, "little", signed=True)[:3]
                elif sampwidth == 4:
                    frames += struct.pack("<i", val << 16)
                else:
                    raise ValueError(sampwidth)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _wav_info(wav_bytes: bytes):
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()


# --------------------------------------------------------------------------
# normalize_wav() — pure function
# --------------------------------------------------------------------------

def test_normalize_wav_22050_mono_becomes_24k_mono_s16():
    src_rate = 22050
    src = _make_wav(rate=src_rate, channels=1, sampwidth=2, n_samples=240)
    out = gateway.normalize_wav(src)
    channels, sampwidth, rate, nframes = _wav_info(out)
    assert (channels, sampwidth, rate) == (1, 2, 24000)
    # Frame count must reflect actual resampling (240 * 24000/22050), not
    # just a header rewrite -- a header-only bug would leave nframes at 240
    # and play the audio at the wrong speed/pitch in the ear.
    assert abs(nframes - round(240 * 24000 / src_rate)) <= 1


def test_normalize_wav_44100_stereo_becomes_24k_mono_s16():
    src_rate = 44100
    src = _make_wav(rate=src_rate, channels=2, sampwidth=2, n_samples=240)
    out = gateway.normalize_wav(src)
    channels, sampwidth, rate, nframes = _wav_info(out)
    assert (channels, sampwidth, rate) == (1, 2, 24000)
    assert abs(nframes - round(240 * 24000 / src_rate)) <= 1


def test_normalize_wav_8bit_becomes_24k_mono_s16():
    src_rate = 16000
    src = _make_wav(rate=src_rate, channels=1, sampwidth=1, n_samples=240)
    out = gateway.normalize_wav(src)
    channels, sampwidth, rate, nframes = _wav_info(out)
    assert (channels, sampwidth, rate) == (1, 2, 24000)
    assert abs(nframes - round(240 * 24000 / src_rate)) <= 1


def test_normalize_wav_already_24k_mono_s16_is_passthrough():
    src = _make_wav(rate=24000, channels=1, sampwidth=2)
    out = gateway.normalize_wav(src)
    assert out == src  # fast path: byte-identical, not just format-identical


def test_normalize_wav_24bit_and_32bit_supported():
    for sampwidth in (3, 4):
        src = _make_wav(rate=22050, channels=1, sampwidth=sampwidth)
        out = gateway.normalize_wav(src)
        channels, out_sampwidth, rate, nframes = _wav_info(out)
        assert (channels, out_sampwidth, rate) == (1, 2, 24000)
        assert nframes > 0


def test_normalize_wav_rejects_non_mono_16bit_target():
    src = _make_wav(rate=22050, channels=1, sampwidth=2)
    with pytest.raises(ValueError):
        gateway.normalize_wav(src, target_channels=2)
    with pytest.raises(ValueError):
        gateway.normalize_wav(src, target_sampwidth=4)


def test_safe_normalize_wav_falls_back_on_malformed_input(monkeypatch):
    """If normalize_wav() blows up (malformed/unparseable WAV bytes),
    _safe_normalize_wav must log and return the ORIGINAL bytes rather than
    raising or dropping the sentence."""
    logged = []
    monkeypatch.setattr(gateway, "log_fallback", lambda *a, **kw: logged.append((a, kw)))
    garbage = b"not a wav file at all"
    out = gateway._safe_normalize_wav(garbage)
    assert out == garbage
    assert logged  # a failure was logged


# --------------------------------------------------------------------------
# On the wire: /v1/audio/speech always emits 24 kHz mono 16-bit WAV
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _dead_desktop_tts_by_default(monkeypatch):
    monkeypatch.setattr(gateway, "DESKTOP_TTS_URL", "http://127.0.0.1:1/tts")


@pytest.fixture()
def gateway_server():
    server = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


def test_speech_endpoint_normalizes_non_24k_engine_output(gateway_server, monkeypatch):
    fake_wav = _make_wav(rate=22050, channels=1, sampwidth=2)
    monkeypatch.setattr(gateway, "desktop_tts_speak", lambda text: (fake_wav, "chatterbox"))

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    payload = json.dumps({"model": "x", "input": "hello there", "voice": "agent"})
    conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    assert resp.status == 200
    channels, sampwidth, rate, _ = _wav_info(body)
    assert (channels, sampwidth, rate) == (1, 2, 24000)


# --------------------------------------------------------------------------
# On the wire: /ask/stream — two sentences from engines at different native
# rates both come out at 24 kHz mono 16-bit.
# --------------------------------------------------------------------------

def _read_sse_events(resp):
    events = []
    buf = b""
    while True:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            block, buf = buf.split(b"\n\n", 1)
            event, data = None, None
            for line in block.split(b"\n"):
                if line.startswith(b"event: "):
                    event = line[len(b"event: "):].decode()
                elif line.startswith(b"data: "):
                    data = json.loads(line[len(b"data: "):].decode())
            if event:
                events.append((event, data))
                if event == "done":
                    return events
    return events


def test_ask_stream_normalizes_sentences_from_different_engine_rates(gateway_server, monkeypatch):
    """Sentence 1 comes back as if synthesized by a 22.05 kHz engine (Piper),
    sentence 2 as if from a 24 kHz engine (the desktop Chatterbox clone) --
    both must be emitted on the wire at 24 kHz mono 16-bit."""

    def fake_stream_brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        on_partial("Piper said this.")
        return "Piper said this. Chatterbox said that."

    def mixed_rate_synth(text):
        if "Piper" in text:
            return _make_wav(rate=22050, channels=1, sampwidth=2), "piper"
        return _make_wav(rate=24000, channels=1, sampwidth=2), "chatterbox-desktop"

    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", fake_stream_brain)
    monkeypatch.setattr(gateway, "synthesize", mixed_rate_synth)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=15)
    payload = json.dumps({"prompt": "hello", "session_id": "s1", "node_id": "node-1"})
    conn.putrequest("POST", "/ask/stream")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    events = _read_sse_events(resp)
    conn.close()

    sentence_events = [d for e, d in events if e == "sentence"]
    assert len(sentence_events) == 2
    for d in sentence_events:
        wav_bytes = base64.b64decode(d["audio_b64"])
        channels, sampwidth, rate, _ = _wav_info(wav_bytes)
        assert (channels, sampwidth, rate) == (1, 2, 24000), d["engine"]
    assert {d["engine"] for d in sentence_events} == {"piper", "chatterbox-desktop"}
