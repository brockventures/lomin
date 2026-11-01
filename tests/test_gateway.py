#!/usr/bin/env python3
"""Tests for ./gateway.py.

No network: every upstream (whisper, desktop TTS, piper, brain) is faked —
either a local HTTP server on port 0, or a monkeypatched module function.
Run: python3 -m pytest ./tests -v
"""
import http.client
import json
import struct
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402


# --------------------------------------------------------------------------
# Unit tests: pure functions
# --------------------------------------------------------------------------

def test_strip_markdown_removes_common_constructs():
    raw = "**Hello** _there_, see `code` and [a link](http://x)\n# Heading\n- item"
    out = gateway.strip_markdown(raw)
    assert "*" not in out
    assert "`" not in out
    assert "[" not in out and "](" not in out
    assert "#" not in out
    assert "Hello" in out and "there" in out and "code" in out and "a link" in out


def test_strip_markdown_empty():
    assert gateway.strip_markdown("") == ""
    assert gateway.strip_markdown(None) is None


def test_pcm_to_wav_roundtrip():
    # 1000 samples of a simple ramp, 16-bit mono PCM
    samples = [i % 1000 for i in range(1000)]
    pcm = struct.pack("<%dh" % len(samples), *samples)
    wav_bytes = gateway.pcm_to_wav(pcm, sample_rate=16000)
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == len(samples)


def test_pcm_to_wav_trims_odd_trailing_byte():
    pcm = b"\x01\x02\x03"  # 3 bytes, not a multiple of 2
    wav_bytes = gateway.pcm_to_wav(pcm, sample_rate=16000)
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        assert w.getnframes() == 1  # only the first full sample kept


# --------------------------------------------------------------------------
# multipart parsing — byte-exact round trip against Go's multipart.Writer
# framing (mime/multipart.CreatePart), including WAV bytes that themselves
# contain "\r\n--" and "\r\n\r\n" sequences.
# --------------------------------------------------------------------------

def _build_multipart(boundary: str, audio_bytes: bytes) -> bytes:
    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        b'Content-Disposition: form-data; name="audio"; filename="audio.wav"\r\n'
    )
    parts.append(b"Content-Type: audio/wav\r\n\r\n")
    parts.append(audio_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts)


def test_multipart_parse_simple():
    audio = b"RIFF....WAVEfmt some pcm data"
    boundary = "boundary123"
    body = _build_multipart(boundary, audio)
    parsed = gateway.parse_multipart_audio(body, f"multipart/form-data; boundary={boundary}")
    assert parsed == audio


def test_multipart_parse_audio_with_embedded_crlf_dashdash_and_crlfcrlf():
    # Payload deliberately contains the byte sequences that could be
    # mistaken for multipart framing if split naively.
    audio = b"RIFF" + b"\r\n--" + b"notaboundary" + b"\r\n\r\n" + b"\x00\x01\xff\xfe" * 50
    boundary = "----WebKitFormBoundaryXYZ"
    body = _build_multipart(boundary, audio)
    parsed = gateway.parse_multipart_audio(body, f"multipart/form-data; boundary={boundary}")
    assert parsed == audio


def test_multipart_parse_no_boundary_header():
    assert gateway.parse_multipart_audio(b"whatever", "multipart/form-data") is None


def test_multipart_parse_no_audio_part():
    boundary = "b1"
    body = (
        f"--{boundary}\r\n".encode()
        + b'Content-Disposition: form-data; name="other"\r\n\r\n'
        + b"not audio"
        + f"\r\n--{boundary}--\r\n".encode()
    )
    assert gateway.parse_multipart_audio(body, f"multipart/form-data; boundary={boundary}") is None


# --------------------------------------------------------------------------
# Fake upstream servers
# --------------------------------------------------------------------------

class _FakeWhisperHandler(BaseHTTPRequestHandler):
    received_bodies = []

    def log_message(self, *_):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _FakeWhisperHandler.received_bodies.append(body)
        resp = json.dumps({"text": "hello from fake whisper", "ms": 12}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self):
        if self.path == "/health":
            resp = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)


@pytest.fixture()
def fake_whisper(monkeypatch):
    _FakeWhisperHandler.received_bodies = []
    server = HTTPServer(("127.0.0.1", 0), _FakeWhisperHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(gateway, "WHISPER_URL", f"http://127.0.0.1:{port}/transcribe")
    monkeypatch.setattr(gateway, "WHISPER_HEALTH_URL", f"http://127.0.0.1:{port}/health")
    yield server
    server.shutdown()


@pytest.fixture(autouse=True)
def _dead_desktop_tts_by_default(monkeypatch):
    """A real the desktop TTS server may be listening on this
    machine's 127.0.0.1:9097 (it's part of what this PR installs). Point
    every test at a dead port by default so a test that forgets to stub
    desktop_tts_speak fails loudly instead of quietly talking to a real
    server and passing for the wrong reason. Tests that want the real
    HTTP path point DESKTOP_TTS_URL at their own fake server instead."""
    monkeypatch.setattr(gateway, "DESKTOP_TTS_URL", "http://127.0.0.1:1/tts")


@pytest.fixture(autouse=True)
def _dead_desktop_embed_by_default(monkeypatch):
    """Same reasoning as _dead_desktop_tts_by_default, for the speaker-embed
    hop: a real the desktop embed server may be listening on this
    machine's 127.0.0.1:9096. Point every test at a dead port by default."""
    monkeypatch.setattr(gateway, "DESKTOP_EMBED_URL", "http://127.0.0.1:1/embed")


@pytest.fixture()
def gateway_server(monkeypatch):
    server = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


# --------------------------------------------------------------------------
# /stt
# --------------------------------------------------------------------------

def test_stt_multipart(gateway_server, fake_whisper):
    audio = b"RIFF" + b"fake wav bytes" * 20
    boundary = "----WebKitFormBoundaryABC"
    body = _build_multipart(boundary, audio)
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request(
        "POST", "/stt", body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert resp.status == 200
    assert data["text"] == "hello from fake whisper"
    assert _FakeWhisperHandler.received_bodies[-1] == audio
    conn.close()


def test_stt_raw_wav_body(gateway_server, fake_whisper):
    audio = b"RIFF" + b"raw body bytes" * 5
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("POST", "/stt", body=audio, headers={"Content-Type": "audio/wav"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert resp.status == 200
    assert data["text"] == "hello from fake whisper"
    assert _FakeWhisperHandler.received_bodies[-1] == audio
    conn.close()


def test_stt_upstream_failure(gateway_server, monkeypatch):
    monkeypatch.setattr(gateway, "WHISPER_URL", "http://127.0.0.1:1/transcribe")  # nothing listening
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("POST", "/stt", body=b"RIFFdata", headers={"Content-Type": "audio/wav"})
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 502
    conn.close()


def test_stt_no_audio(gateway_server):
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("POST", "/stt", body=b"", headers={"Content-Type": "audio/wav"})
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 400
    conn.close()


# --------------------------------------------------------------------------
# /ask (SSE)
# --------------------------------------------------------------------------

def _read_sse_events(sock_file, deadline=15):
    """Read raw bytes from a socket-like file until connection closes,
    parse event/data pairs."""
    raw = b""
    start = time.time()
    while time.time() - start < deadline:
        chunk = sock_file.read(4096)
        if not chunk:
            break
        raw += chunk
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        if event:
            events.append((event, json.loads(data) if data else {}))
    return events


def test_ask_happy_path(gateway_server, monkeypatch):
    def fake_brain(prompt, session_id, node_id, speaker, speaker_score, timeout):
        assert "hello there" in prompt
        assert node_id == "node-1"
        return "**Sure**, the weather is fine."

    monkeypatch.setattr(gateway, "BRAIN_CALL", fake_brain)
    monkeypatch.setattr(gateway, "STATUS_INTERVAL_SEC", 0.2)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=15)
    payload = json.dumps({
        "prompt": "hello there", "session_id": "s1", "effort": "low",
        "node_id": "node-1", "speaker": "mike", "speaker_score": 0.9,
    })
    conn.putrequest("POST", "/ask")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "text/event-stream"
    events = _read_sse_events(resp)
    kinds = [e for e, _ in events]
    assert "status" in kinds
    assert kinds[-2:] == ["reply", "done"]
    reply_data = [d for e, d in events if e == "reply"][0]
    assert reply_data["reply"] == "Sure, the weather is fine."
    conn.close()


def test_ask_unanswered_gets_apology(gateway_server, monkeypatch):
    monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "unanswered: timed_out")
    monkeypatch.setattr(gateway, "STATUS_INTERVAL_SEC", 0.2)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=15)
    payload = json.dumps({"prompt": "x", "node_id": "n"})
    conn.putrequest("POST", "/ask")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    events = _read_sse_events(resp)
    reply_data = [d for e, d in events if e == "reply"][0]
    assert reply_data["reply"] == gateway.APOLOGY
    conn.close()


def test_ask_unanswered_with_suffix_reason(gateway_server, monkeypatch):
    monkeypatch.setattr(
        gateway, "BRAIN_CALL",
        lambda *a, **k: "unanswered: timed_out (could not deliver: boom)",
    )
    monkeypatch.setattr(gateway, "STATUS_INTERVAL_SEC", 0.2)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=15)
    payload = json.dumps({"prompt": "x", "node_id": "n"})
    conn.putrequest("POST", "/ask")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    events = _read_sse_events(resp)
    reply_data = [d for e, d in events if e == "reply"][0]
    assert reply_data["reply"] == gateway.APOLOGY
    conn.close()


def test_ask_empty_reply_gets_apology(gateway_server, monkeypatch):
    monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "")
    monkeypatch.setattr(gateway, "STATUS_INTERVAL_SEC", 0.2)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=15)
    payload = json.dumps({"prompt": "x", "node_id": "n"})
    conn.putrequest("POST", "/ask")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    events = _read_sse_events(resp)
    reply_data = [d for e, d in events if e == "reply"][0]
    assert reply_data["reply"] == gateway.APOLOGY
    conn.close()


def test_ask_multiple_status_pulses_while_waiting(gateway_server, monkeypatch):
    done = threading.Event()

    def slow_brain(*a, **k):
        done.wait(0.7)
        return "ok reply"

    monkeypatch.setattr(gateway, "BRAIN_CALL", slow_brain)
    monkeypatch.setattr(gateway, "STATUS_INTERVAL_SEC", 0.2)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=15)
    payload = json.dumps({"prompt": "x", "node_id": "n"})
    conn.putrequest("POST", "/ask")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    events = _read_sse_events(resp)
    done.set()
    status_count = sum(1 for e, _ in events if e == "status")
    assert status_count >= 2
    conn.close()


# --------------------------------------------------------------------------
# /v1/audio/speech
# --------------------------------------------------------------------------

def _make_test_wav(text_marker=b"PIPERWAV"):
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(text_marker * 4)
    return buf.getvalue()


def test_speech_uses_desktop_clone_when_available(gateway_server, monkeypatch):
    # _make_test_wav defaults to 22050 Hz mono 16-bit — deliberately NOT the
    # 24 kHz wire format, so this also exercises /v1/audio/speech's
    # normalize_wav() pass (MIRRORMERE-ONE-FORMAT).
    fake_wav = _make_test_wav(b"ELEVEN01")
    monkeypatch.setattr(gateway, "desktop_tts_speak", lambda text: (fake_wav, "chatterbox"))

    def boom(text):
        raise AssertionError("piper should not be called when desktop tts succeeds")

    monkeypatch.setattr(gateway, "piper_speak", boom)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    payload = json.dumps({"model": "x", "input": "hello", "voice": "agent", "response_format": "mp3"})
    conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("X-TTS-Engine") == "chatterbox"
    assert resp.getheader("Content-Type") == "audio/wav"
    with wave.open(BytesIO(body), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
    conn.close()


def test_speech_uses_desktop_tts_without_local_piper(gateway_server, monkeypatch):
    fake_wav = _make_test_wav(b"DESKTOP1")
    monkeypatch.setattr(gateway, "desktop_tts_speak", lambda text: (fake_wav, "piper-desktop"))

    def piper_boom(text):
        raise AssertionError("local piper should not be called when desktop tts succeeds")

    monkeypatch.setattr(gateway, "piper_speak", piper_boom)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    payload = json.dumps({"model": "x", "input": "hello", "voice": "agent", "response_format": "mp3"})
    conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("X-TTS-Engine") == "piper-desktop"
    with wave.open(BytesIO(body), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
    conn.close()


# --------------------------------------------------------------------------
# synthesize_chunked — sentence-parallel synthesis for long, non-streaming
# /v1/audio/speech inputs (specs/2026-09-28-mandos-streaming.md).
# --------------------------------------------------------------------------

def _wav_text(wav_bytes: bytes) -> bytes:
    with wave.open(BytesIO(wav_bytes), "rb") as w:
        return w.readframes(w.getnframes())


def test_synthesize_chunked_short_text_not_chunked(monkeypatch):
    calls = []

    def fake_synth(text):
        calls.append(text)
        return _make_test_wav(text.encode()[:8].ljust(8, b"X")), "fake"

    monkeypatch.setattr(gateway, "synthesize", fake_synth)
    wav, engine = gateway.synthesize_chunked("Short.")
    assert calls == ["Short."]  # exactly one call, not split
    assert engine == "fake"


def test_synthesize_chunked_long_text_splits_and_concatenates_in_order(monkeypatch):
    calls = []
    lock = threading.Lock()

    def fake_synth(text):
        # Marker is a deterministic function of the TEXT (not arrival
        # order), so this stays correct however the thread pool interleaves
        # concurrent calls -- only the final concatenation order matters.
        with lock:
            calls.append(text)
        marker = text.encode("utf-8")[:8].ljust(8, b"_")
        return _make_test_wav(marker), "fake"

    monkeypatch.setattr(gateway, "synthesize", fake_synth)
    text = (
        "This sentence is long enough on its own. "
        "Here is a second sentence that adds more length. "
        "And a third sentence closes out the paragraph nicely."
    )
    assert len(text) >= gateway.SENTENCE_CHUNK_MIN_CHARS
    wav, engine = gateway.synthesize_chunked(text)
    expected_sentences = gateway.segment_full_text(text)
    assert sorted(calls) == sorted(expected_sentences)  # each sentence synthesized once
    assert engine == "fake"

    # Concatenated frame data preserves sentence ORDER even though
    # synthesis ran concurrently across a thread pool: rebuild the expected
    # combined bytes by concatenating each sentence's OWN deterministic
    # marker in source order, and compare against the actual output.
    expected_combined = b"".join(
        _wav_text(_make_test_wav(s.encode("utf-8")[:8].ljust(8, b"_")))
        for s in expected_sentences
    )
    assert _wav_text(wav) == expected_combined


def test_synthesize_chunked_falls_back_on_format_mismatch(monkeypatch):
    """A mid-request engine switch (e.g. desktop TTS failing over to Piper
    between two sentence calls) can leave sentences with different sample
    rates. synthesize_chunked must not silently concatenate mismatched
    formats -- it re-synthesizes the whole text as one serial call."""
    call_count = {"n": 0}

    def flaky_synth(text):
        call_count["n"] += 1
        # First pass (per-sentence, concurrent): alternate sample rates so
        # concatenation must fail. Second pass (the single-shot fallback
        # call with the FULL text): stable rate.
        if text == full_text:
            return _make_test_wav(b"FULLTEXT"), "fake"
        rate = 16000 if call_count["n"] % 2 else 22050
        buf = BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"XX" * 4)
        return buf.getvalue(), "fake"

    monkeypatch.setattr(gateway, "synthesize", flaky_synth)
    full_text = (
        "This sentence is long enough on its own. "
        "Here is a second sentence that adds more length. "
        "And a third sentence closes out the paragraph nicely."
    )
    wav, engine = gateway.synthesize_chunked(full_text)
    with wave.open(BytesIO(wav), "rb") as w:
        assert w.getframerate() in (16000, 22050)  # the fallback single-shot call


# --------------------------------------------------------------------------
# /ask/stream (SSE, MANDOS streaming)
# --------------------------------------------------------------------------

def _fake_stream_synth(text):
    """Deterministic per-sentence WAV whose frame bytes encode the
    sentence's own position marker, and a small variable delay so tests can
    force TTS completion order to differ from sentence order."""
    marker = f"S{len(text):04d}".encode()[:8].ljust(8, b"_")
    return _make_test_wav(marker), "fake"


def test_ask_stream_emits_sentences_as_they_complete_and_matches_final_reply(gateway_server, monkeypatch):
    def fake_stream_brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        assert node_id == "node-1"
        on_partial("First sentence here.")
        on_partial("First sentence here. Second one follows.")
        return "First sentence here. Second one follows. Third and final."

    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", fake_stream_brain)
    monkeypatch.setattr(gateway, "synthesize", _fake_stream_synth)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=15)
    payload = json.dumps({
        "prompt": "hello", "session_id": "s1", "node_id": "node-1",
        "speaker": "mike", "speaker_score": 0.9,
    })
    conn.putrequest("POST", "/ask/stream")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "text/event-stream"
    events = _read_sse_events(resp)
    conn.close()

    sentence_events = [d for e, d in events if e == "sentence"]
    sentence_texts = [d["text"] for d in sentence_events]
    assert sentence_texts == [
        "First sentence here.", "Second one follows.", "Third and final.",
    ]
    for d in sentence_events:
        assert d["audio_b64"]
        assert d["engine"] == "fake"

    reply_data = [d for e, d in events if e == "reply"][0]
    assert reply_data["reply"] == "First sentence here. Second one follows. Third and final."
    assert [e for e, _ in events][-1] == "done"
    # Final-text parity: the sentence stream reconstructs the full reply.
    assert " ".join(sentence_texts) == reply_data["reply"]


def test_ask_stream_preserves_order_under_out_of_order_tts_completion(gateway_server, monkeypatch):
    """Sentence 1's synthesis is slow; sentence 2's is fast. The stream
    must still emit them in sentence order, not completion order."""
    release_first = threading.Event()

    def slow_then_fast_synth(text):
        if text.startswith("Slow"):
            release_first.wait(2)
        return _fake_stream_synth(text)

    def fake_stream_brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        on_partial("Slow first sentence.")
        on_partial("Slow first sentence. Fast second sentence.")
        # Give the fast sentence's TTS a head start before the slow one is released.
        time.sleep(0.1)
        release_first.set()
        return "Slow first sentence. Fast second sentence. Done third."

    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", fake_stream_brain)
    monkeypatch.setattr(gateway, "synthesize", slow_then_fast_synth)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=15)
    payload = json.dumps({"prompt": "x", "node_id": "n"})
    conn.putrequest("POST", "/ask/stream")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    events = _read_sse_events(resp)
    conn.close()

    sentence_texts = [d["text"] for e, d in events if e == "sentence"]
    assert sentence_texts == [
        "Slow first sentence.", "Fast second sentence.", "Done third.",
    ]


def test_ask_stream_unanswered_gets_apology_sentence(gateway_server, monkeypatch):
    def fake_stream_brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        return "unanswered: timed_out"

    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", fake_stream_brain)
    monkeypatch.setattr(gateway, "synthesize", _fake_stream_synth)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=15)
    payload = json.dumps({"prompt": "x", "node_id": "n"})
    conn.putrequest("POST", "/ask/stream")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    events = _read_sse_events(resp)
    conn.close()

    sentence_texts = [d["text"] for e, d in events if e == "sentence"]
    assert sentence_texts == [gateway.APOLOGY]
    reply_data = [d for e, d in events if e == "reply"][0]
    assert reply_data["reply"] == gateway.APOLOGY


def test_speech_falls_back_to_local_piper_when_desktop_tts_also_down(gateway_server, monkeypatch):
    def desktop_boom(text):
        raise RuntimeError("desktop tts unreachable")

    fake_wav = _make_test_wav(b"PIPERFALL")
    monkeypatch.setattr(gateway, "desktop_tts_speak", desktop_boom)
    monkeypatch.setattr(gateway, "piper_speak", lambda text: fake_wav)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    payload = json.dumps({"model": "x", "input": "hello", "voice": "agent", "response_format": "mp3"})
    conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("X-TTS-Engine") == "piper"
    with wave.open(BytesIO(body), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
    conn.close()


def test_synthesize_chain_is_desktop_then_piper(monkeypatch):
    """The TTS chain is two hops: desktop TTS, then local Piper. Each hop's
    failure is logged with its reason."""
    logged = []

    def desktop_boom(text):
        raise RuntimeError("desktop down")

    monkeypatch.setattr(gateway, "desktop_tts_speak", desktop_boom)
    monkeypatch.setattr(gateway, "piper_speak", lambda text: b"PIPERWAV")
    monkeypatch.setattr(
        gateway, "log_fallback",
        lambda reason, tts_engine="piper", **extra: logged.append((reason, tts_engine)),
    )
    assert gateway.synthesize("hello") == (b"PIPERWAV", "piper")
    assert logged == [("desktop-tts:desktop down", "piper")]


class _FakeDesktopTTSHandler(BaseHTTPRequestHandler):
    received_bodies = []
    response_body = b"RIFF" + b"fakewav" * 10
    response_status = 200
    response_engine = "piper"

    def log_message(self, *_):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _FakeDesktopTTSHandler.received_bodies.append(body)
        self.send_response(_FakeDesktopTTSHandler.response_status)
        self.send_header("Content-Length", str(len(_FakeDesktopTTSHandler.response_body)))
        self.send_header("X-TTS-Engine", _FakeDesktopTTSHandler.response_engine)
        self.end_headers()
        self.wfile.write(_FakeDesktopTTSHandler.response_body)


@pytest.fixture()
def fake_desktop_tts(monkeypatch):
    _FakeDesktopTTSHandler.received_bodies = []
    _FakeDesktopTTSHandler.response_body = b"RIFF" + b"fakewav" * 10
    _FakeDesktopTTSHandler.response_status = 200
    _FakeDesktopTTSHandler.response_engine = "piper"
    server = HTTPServer(("127.0.0.1", 0), _FakeDesktopTTSHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(gateway, "DESKTOP_TTS_URL", f"http://127.0.0.1:{port}/tts")
    yield server
    server.shutdown()


def test_desktop_tts_speak_sends_json_and_returns_wav(fake_desktop_tts):
    wav, engine = gateway.desktop_tts_speak("hello")
    assert wav == _FakeDesktopTTSHandler.response_body
    assert engine == "piper-desktop"
    sent = json.loads(_FakeDesktopTTSHandler.received_bodies[-1])
    assert sent == {"text": "hello"}


def test_desktop_tts_speak_relays_chatterbox_engine(fake_desktop_tts):
    _FakeDesktopTTSHandler.response_engine = "chatterbox"
    wav, engine = gateway.desktop_tts_speak("hello")
    assert wav == _FakeDesktopTTSHandler.response_body
    assert engine == "chatterbox"


def test_speech_uses_real_desktop_hop(gateway_server, monkeypatch, fake_desktop_tts):
    def piper_boom(text):
        raise AssertionError("local piper should not be called when desktop tts succeeds")

    monkeypatch.setattr(gateway, "piper_speak", piper_boom)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    payload = json.dumps({"model": "x", "input": "hello", "voice": "agent"})
    conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("X-TTS-Engine") == "piper-desktop"
    assert body == _FakeDesktopTTSHandler.response_body
    conn.close()


def test_desktop_tts_speak_rejects_non_wav_body(fake_desktop_tts):
    _FakeDesktopTTSHandler.response_body = b'{"error": "not really audio"}'
    with pytest.raises(Exception):
        gateway.desktop_tts_speak("hello")


def test_speech_falls_through_to_local_piper_on_non_wav_desktop_body(
    gateway_server, monkeypatch, fake_desktop_tts
):
    _FakeDesktopTTSHandler.response_body = b'{"error": "not really audio"}'

    fake_piper_wav = _make_test_wav(b"LOCALFB1")
    monkeypatch.setattr(gateway, "piper_speak", lambda text: fake_piper_wav)

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    payload = json.dumps({"model": "x", "input": "hello", "voice": "agent"})
    conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert resp.getheader("X-TTS-Engine") == "piper"
    with wave.open(BytesIO(body), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
    conn.close()


def test_speech_response_is_always_wav_regardless_of_response_format(gateway_server, monkeypatch):
    fake_wav = _make_test_wav()
    monkeypatch.setattr(gateway, "desktop_tts_speak", lambda text: (fake_wav, "chatterbox"))
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    for fmt in ("mp3", "opus", "wav", "flac"):
        payload = json.dumps({"model": "x", "input": "hi", "voice": "agent", "response_format": fmt})
        conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        assert resp.getheader("Content-Type") == "audio/wav"
    conn.close()


def test_speech_both_engines_fail(gateway_server, monkeypatch):
    monkeypatch.setattr(gateway, "desktop_tts_speak", lambda text: (_ for _ in ()).throw(RuntimeError("no desktop")))
    monkeypatch.setattr(gateway, "piper_speak", lambda text: (_ for _ in ()).throw(RuntimeError("also no")))
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    payload = json.dumps({"model": "x", "input": "hi", "voice": "agent"})
    conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 502
    conn.close()


def test_speech_missing_input(gateway_server):
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    payload = json.dumps({"model": "x"})
    conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 400
    conn.close()


# --------------------------------------------------------------------------
# Per-node brain routing
# --------------------------------------------------------------------------

def test_resolve_target_uses_node_targets_file(monkeypatch, tmp_path):
    targets_path = tmp_path / "node-targets.json"
    targets_path.write_text(json.dumps({"deck-node": "deck-node"}))
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", targets_path)
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "absent-nodes.json")
    monkeypatch.delenv("MVG_NODE_TARGETS", raising=False)
    assert gateway.resolve_target("deck-node") == "deck-node"


def test_resolve_target_env_wins_over_file(monkeypatch, tmp_path):
    targets_path = tmp_path / "node-targets.json"
    targets_path.write_text(json.dumps({"deck-node": "deck-node"}))
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", targets_path)
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "absent-nodes.json")
    monkeypatch.setenv("MVG_NODE_TARGETS", json.dumps({"deck-node": "mac-node"}))
    assert gateway.resolve_target("deck-node") == "mac-node"


def test_resolve_target_unmapped_node_falls_back_to_default_target(monkeypatch, tmp_path):
    targets_path = tmp_path / "node-targets.json"
    targets_path.write_text(json.dumps({"deck-node": "deck-node"}))
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", targets_path)
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "absent-nodes.json")
    monkeypatch.delenv("MVG_NODE_TARGETS", raising=False)
    assert gateway.resolve_target("some-unknown-node") == gateway.DEFAULT_TARGET
    assert gateway.resolve_target("") == gateway.DEFAULT_TARGET


def test_example_nodes_json_routes_kitchen_speaker_to_agent(monkeypatch):
    monkeypatch.setattr(gateway, "NODES_PATH", Path(gateway.__file__).parent / "nodes.example.json")
    monkeypatch.delenv("MVG_NODE_TARGETS", raising=False)
    assert gateway.resolve_target("kitchen-speaker") == "agent"
    assert gateway.resolve_profile("kitchen-speaker")["open_room"] is False


def test_plugin_brain_call_routes_by_node_id(monkeypatch, tmp_path):
    targets_path = tmp_path / "node-targets.json"
    targets_path.write_text(json.dumps({"deck-node": "deck-node"}))
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", targets_path)
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "absent-nodes.json")
    monkeypatch.delenv("MVG_NODE_TARGETS", raising=False)

    calls = []

    def fake_brain(target, prompt, timeout):
        calls.append(target)
        return "ok"

    monkeypatch.setattr(gateway, "_plugin_brain_call", fake_brain)
    result = gateway.plugin_brain_call("hi", "s1", "deck-node", "mike", 0.9, 45)
    assert result == "ok"
    assert calls == ["deck-node"]


def test_plugin_brain_call_unmapped_node_uses_default_target(monkeypatch, tmp_path):
    targets_path = tmp_path / "node-targets.json"
    targets_path.write_text(json.dumps({"deck-node": "deck-node"}))
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", targets_path)
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "absent-nodes.json")
    monkeypatch.delenv("MVG_NODE_TARGETS", raising=False)

    calls = []
    monkeypatch.setattr(
        gateway, "_plugin_brain_call",
        lambda target, prompt, timeout: calls.append(target) or "ok",
    )
    gateway.plugin_brain_call("hi", "s1", "some-other-node", "mike", 0.9, 45)
    assert calls == [gateway.DEFAULT_TARGET]


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------

def test_health(gateway_server, fake_whisper, monkeypatch):
    monkeypatch.setattr(gateway, "PIPER_CONFIG_PATH", Path("/nonexistent/piper-tts.json"))
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("GET", "/health")
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert resp.status == 200
    assert data["status"] == "ok"
    assert data["upstreams"]["whisper"] is True
    conn.close()


def test_speech_prefers_desktop_clone_over_local_piper(gateway_server, monkeypatch):
    """Mike, 2026-09-28: the cloned voice is the default, ahead of local Piper."""
    fake_wav = _make_test_wav(b"CLONEWAV")
    piper_calls = []
    monkeypatch.setattr(gateway, "piper_speak", lambda text: piper_calls.append(1) or fake_wav)
    monkeypatch.setattr(gateway, "desktop_tts_speak", lambda text: (fake_wav, "chatterbox"))
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    payload = json.dumps({"model": "x", "input": "a long reply " * 30, "voice": "agent"})
    conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.getheader("X-TTS-Engine") == "chatterbox"
    assert piper_calls == []


# --------------------------------------------------------------------------
# /embed (SPEC-011 speaker identification proxy)
# --------------------------------------------------------------------------

class _FakeEmbedHandler(BaseHTTPRequestHandler):
    received_bodies = []
    response_body = b'{"embedding": [0.1, 0.2], "model": "speechbrain/spkrec-ecapa-voxceleb"}'
    response_status = 200
    delay_sec = 0

    def log_message(self, *_):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _FakeEmbedHandler.received_bodies.append(body)
        if _FakeEmbedHandler.delay_sec:
            time.sleep(_FakeEmbedHandler.delay_sec)
        self.send_response(_FakeEmbedHandler.response_status)
        self.send_header("Content-Length", str(len(_FakeEmbedHandler.response_body)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(_FakeEmbedHandler.response_body)


@pytest.fixture()
def fake_embed(monkeypatch):
    _FakeEmbedHandler.received_bodies = []
    _FakeEmbedHandler.response_body = b'{"embedding": [0.1, 0.2], "model": "speechbrain/spkrec-ecapa-voxceleb"}'
    _FakeEmbedHandler.response_status = 200
    _FakeEmbedHandler.delay_sec = 0
    server = HTTPServer(("127.0.0.1", 0), _FakeEmbedHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(gateway, "DESKTOP_EMBED_URL", f"http://127.0.0.1:{port}/embed")
    yield server
    server.shutdown()


def test_embed_proxies_200_byte_exact(gateway_server, fake_embed):
    wav = b"RIFF" + b"fakewavbytes" * 20
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("POST", "/embed", body=wav, headers={"Content-Type": "audio/wav"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 200
    assert body == _FakeEmbedHandler.response_body
    assert _FakeEmbedHandler.received_bodies[-1] == wav
    conn.close()


def test_embed_proxies_upstream_400(gateway_server, fake_embed):
    _FakeEmbedHandler.response_status = 400
    _FakeEmbedHandler.response_body = b'{"error": "audio too short: 0.10s < 0.5s"}'
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("POST", "/embed", body=b"RIFF short", headers={"Content-Type": "audio/wav"})
    resp = conn.getresponse()
    body = resp.read()
    assert resp.status == 400
    assert body == _FakeEmbedHandler.response_body
    conn.close()


def test_embed_dead_upstream_is_502(gateway_server):
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("POST", "/embed", body=b"RIFF" + b"x" * 100, headers={"Content-Type": "audio/wav"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert resp.status == 502
    assert "error" in data
    conn.close()


def test_embed_slow_upstream_times_out_as_502(gateway_server, fake_embed, monkeypatch):
    monkeypatch.setattr(gateway, "DESKTOP_EMBED_TIMEOUT", 0.2)
    _FakeEmbedHandler.delay_sec = 1.5
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("POST", "/embed", body=b"RIFF" + b"x" * 100, headers={"Content-Type": "audio/wav"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert resp.status == 502
    assert "error" in data
    conn.close()


def test_embed_no_body_is_400(gateway_server):
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("POST", "/embed", body=b"", headers={"Content-Type": "audio/wav"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    assert resp.status == 400
    assert "error" in data
    conn.close()
