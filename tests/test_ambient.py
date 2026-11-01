#!/usr/bin/env python3
"""Tests for gateway.py's /ambient ambient-wake gate and the /stt cache it
feeds. No network: Whisper is a local HTTP server (port 0); Jev and Haiku
stubbed one level below gateway.classify_addressed: lib/classify.py's own
jev_once / haiku_probability (so the real classify() runs), via _patch_jev /
_patch_haiku below. Silero is stubbed or skipped; no test needs the ONNX file
except one optional real-model test.

Run: python3 -m pytest ./tests -v
"""
import http.client
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402


# --------------------------------------------------------------------------
# Fake Whisper (returns whatever text the test queues up)
# --------------------------------------------------------------------------

try:
    # The household plugin's addressee classifier. Not part of this repo: without
    # it the classifier-backed tests skip (see needs_classify) and the rest run.
    import classify as _cl  # noqa: E402
except ImportError:
    _cl = None

needs_classify = pytest.mark.skipif(
    _cl is None, reason="needs the household classifier module (plugin 'classify')")


@pytest.fixture(autouse=True)
def _no_real_classifier_io(monkeypatch):
    """No test may reach the real AI Gateway or `claude`: every layer fails
    unless a test patches it with _patch_jev / _patch_haiku."""
    if _cl is None:
        return
    # With the classifier installed, the gateway's classifier hook is the module
    # the plugin would have supplied.
    monkeypatch.setattr(gateway, "_jev_classifier", _cl)
    monkeypatch.setattr(_cl, "load_key", lambda: "test-key")
    monkeypatch.setattr(_cl, "jev_choice_once",
                        lambda key, state, qk, instr, opts, timeout: (None, "stub: not patched", True, None, {}))
    monkeypatch.setattr(_cl, "haiku_choice",
                        lambda state, instr, opts, aliases=None, **kw: (None, "stub: not patched"))

    monkeypatch.setattr(_cl, "laya_probability",
                        lambda *a, **k: (None, None, "stub: laya not patched", None))

    def _no_post(*a, **k):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(_cl.requests, "post", _no_post)


@pytest.fixture(autouse=True)
def _speech_gate_unavailable_by_default(monkeypatch):
    """Existing tests post b"RIFFdata" (not a WAV): the gate fails open. Pin
    the model lookup so a dev box with a real model never changes that."""
    monkeypatch.setattr(gateway, "_silero_get", lambda: None)


def _patch_jev(monkeypatch, fn):
    """fn(state, timeout=None) -> (P(addressed_to_agent)|None, error), patched
    in at classify.jev_choice_once so the real classify() runs around it. The
    score becomes a Jev choice answer: addressed_to_agent when it is the
    larger of the two probabilities, else conversation_between_people."""
    if _cl is None:
        pytest.skip("needs the household classifier module (plugin 'classify')")
    def fake_jev_choice_once(key, state, question_key, instructions, options, timeout):
        assert question_key == "addressed_to_agent"
        assert set(options) == set(gateway.AMBIENT_OPTIONS)
        score, err = fn(state, timeout)
        if score is None:
            return None, err, True, None, {}
        choice = "addressed_to_agent" if score >= 0.5 else "conversation_between_people"
        probs = {"addressed_to_agent": score, "conversation_between_people": 1 - score}
        return choice, "", True, None, {"probabilities": probs,
                                        "confidence": max(probs.values())}
    monkeypatch.setattr(_cl, "jev_choice_once", fake_jev_choice_once)


def _patch_haiku(monkeypatch, fn):
    """fn(state, timeout=None) -> (score|None, error), patched in at
    classify.haiku_choice (the layer's `claude -p` hop). Haiku returns a
    choice only: score >= 0.6 means addressed_to_agent, else
    conversation_between_people."""
    if _cl is None:
        pytest.skip("needs the household classifier module (plugin 'classify')")
    def fake_haiku(state, instructions, options, aliases=None, *, timeout=None, **kw):
        score, err = fn(state, timeout)
        if score is None:
            return None, err
        return ("addressed_to_agent" if score >= 0.6
                else "conversation_between_people"), "stub"
    monkeypatch.setattr(_cl, "haiku_choice", fake_haiku)


class _FakeWhisperHandler(BaseHTTPRequestHandler):
    received_bodies = []
    next_text = "hello Lomin"

    def log_message(self, *_):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _FakeWhisperHandler.received_bodies.append(body)
        resp = json.dumps({"text": _FakeWhisperHandler.next_text}).encode()
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
    _FakeWhisperHandler.next_text = "hello Lomin"
    server = HTTPServer(("127.0.0.1", 0), _FakeWhisperHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(gateway, "WHISPER_URL", f"http://127.0.0.1:{port}/transcribe")
    monkeypatch.setattr(gateway, "WHISPER_HEALTH_URL", f"http://127.0.0.1:{port}/health")
    yield server
    server.shutdown()


@pytest.fixture()
def gateway_server():
    server = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture(autouse=True)
def _reset_ambient_state():
    """Rolling context / last-reply / STT cache are module-level globals —
    without this, whichever test runs first would poison every test after
    it, same reasoning as the module-state resets in test_gateway.py."""
    gateway._ambient_reset_state()
    yield
    gateway._ambient_reset_state()


@pytest.fixture(autouse=True)
def _dead_desktop_embed_by_default(monkeypatch):
    """Same reasoning as test_gateway.py's _dead_desktop_tts_by_default: a
    real the desktop embed server may be listening on this
    machine's 127.0.0.1:9096. Point every test at a dead port by default so
    a test that forgets to stub the embed hop fails loudly, or (for the
    speaker-ID best-effort path) simply proves the "upstream down -> speaker
    empty, never an error" behavior rather than accidentally talking to a
    real server."""
    monkeypatch.setattr(gateway, "DESKTOP_EMBED_URL", "http://127.0.0.1:1/embed")


@pytest.fixture(autouse=True)
def _no_speakers_file_by_default(monkeypatch, tmp_path):
    """Point every test at a speakers.json that doesn't exist, so
    identify_speaker() short-circuits (no fingerprints -> no embed call at
    all) unless a test explicitly writes one. Also resets the fingerprint
    store's mtime/size cache so a path change between tests is never served
    stale data."""
    monkeypatch.setattr(gateway, "SPEAKERS_PATH", tmp_path / "speakers-not-written.json")
    with gateway._fp_lock:
        gateway._fp_cache.update(path=None, mtime=None, size=None, data=None)
    yield


class _FakeEmbedHandler(BaseHTTPRequestHandler):
    received_bodies = []
    response_body = json.dumps(
        {"embedding": [1.0, 0.0, 0.0, 0.0], "model": "speechbrain/spkrec-ecapa-voxceleb"}
    ).encode()
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
    _FakeEmbedHandler.delay_sec = 0
    _FakeEmbedHandler.response_body = json.dumps(
        {"embedding": [1.0, 0.0, 0.0, 0.0], "model": "speechbrain/spkrec-ecapa-voxceleb"}
    ).encode()
    _FakeEmbedHandler.response_status = 200
    server = HTTPServer(("127.0.0.1", 0), _FakeEmbedHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(gateway, "DESKTOP_EMBED_URL", f"http://127.0.0.1:{port}/embed")
    yield server
    server.shutdown()


def _write_speakers(tmp_path, model="speechbrain/spkrec-ecapa-voxceleb", speakers=None):
    path = tmp_path / "speakers.json"
    path.write_text(json.dumps({"model": model, "speakers": speakers if speakers is not None else []}))
    return path


def _post_ambient(port, wav_bytes: bytes, node_id: str, content_type="audio/wav", path=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    p = path or f"/ambient?node_id={node_id}"
    conn.request("POST", p, body=wav_bytes, headers={"Content-Type": content_type})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return resp.status, data


def _wait_for_log_line(log_file, timeout=5.0):
    """log_request() is called AFTER the response is already sent and
    flushed, so a test's own read of the response can race the server
    thread's subsequent write of the log line. Poll briefly rather than
    assume the file exists (or has content) the instant the client's
    conn.getresponse().read() returns."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log_file.exists() and log_file.stat().st_size > 0:
            return
        time.sleep(0.02)


def _build_multipart_ambient(boundary: str, audio_bytes: bytes, node_id: str) -> bytes:
    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="node_id"\r\n\r\n')
    parts.append(node_id.encode())
    parts.append(f"\r\n--{boundary}\r\n".encode())
    parts.append(
        b'Content-Disposition: form-data; name="audio"; filename="audio.wav"\r\n'
    )
    parts.append(b"Content-Type: audio/wav\r\n\r\n")
    parts.append(audio_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts)


# --------------------------------------------------------------------------
# Hallucination / short-transcript filtering
# --------------------------------------------------------------------------

def test_is_hallucination_or_too_short():
    assert gateway.is_hallucination_or_too_short("Thank you.")
    assert gateway.is_hallucination_or_too_short("you")
    assert gateway.is_hallucination_or_too_short("Thanks for watching")
    assert gateway.is_hallucination_or_too_short("")
    assert gateway.is_hallucination_or_too_short("a")
    assert not gateway.is_hallucination_or_too_short("Lomin?")
    assert not gateway.is_hallucination_or_too_short("what's the weather")


@needs_classify
def test_ambient_skips_classification_on_hallucination(gateway_server, fake_whisper, monkeypatch):
    _FakeWhisperHandler.next_text = "Thank you."
    calls = []
    _patch_jev(monkeypatch, lambda state, timeout=None: (calls.append(1), (0.99, ""))[1])
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert status == 200
    assert data["engage"] is False
    assert data["classifier"] == "none"
    assert not calls  # classifier never invoked


# --------------------------------------------------------------------------
# Engage by name
# --------------------------------------------------------------------------

@needs_classify
def test_ambient_engage_by_name(gateway_server, fake_whisper, monkeypatch):
    _FakeWhisperHandler.next_text = "what's the weather tomorrow, Lomin"
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert status == 200
    assert data["engage"] is True
    assert data["classifier"] == "jev"
    assert data["score"] == 0.95
    assert data["transcript"] == "what's the weather tomorrow, Lomin"
    assert "stt" in data["ms"] and "classify" in data["ms"]


def _do_ask(port, prompt, node_id, monkeypatch, brain_reply):
    import gateway as gw
    monkeypatch.setattr(gw, "BRAIN_CALL", lambda *a, **k: brain_reply)
    monkeypatch.setattr(gw, "STATUS_INTERVAL_SEC", 0.2)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    payload = json.dumps({"prompt": prompt, "node_id": node_id})
    conn.putrequest("POST", "/ask")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    # Drain the SSE stream so the handler thread finishes before we move on.
    while resp.read(4096):
        pass
    conn.close()


# --------------------------------------------------------------------------
# End-to-end: a real /ask reply opens the /ambient follow-up window
# --------------------------------------------------------------------------

@needs_classify
def test_ask_reply_opens_ambient_followup_window(gateway_server, fake_whisper, monkeypatch):
    _do_ask(gateway_server, "what's the weather today", "node-1", monkeypatch, "It's sunny today.")

    seen_state = {}

    def fake_jev(state, timeout=None):
        seen_state["state"] = state
        return (0.9 if "OPEN" in state else 0.1), ""

    _patch_jev(monkeypatch, fake_jev)
    _FakeWhisperHandler.next_text = "and tomorrow?"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert "OPEN" in seen_state["state"]
    assert "sunny" in seen_state["state"]
    assert data["engage"] is True


@needs_classify
def test_ask_apology_does_not_open_ambient_followup_window(gateway_server, fake_whisper, monkeypatch):
    # unanswered: -> APOLOGY, not a real reply, so record_reply must not fire.
    _do_ask(gateway_server, "x", "node-1", monkeypatch, "unanswered: timed_out")

    def fake_jev(state, timeout=None):
        return (0.9 if "OPEN" in state else 0.1), ""

    _patch_jev(monkeypatch, fake_jev)
    _FakeWhisperHandler.next_text = "and tomorrow?"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is False


def _do_ask_stream(port, prompt, node_id, monkeypatch, brain_reply):
    """Same shape as _do_ask above, but drives /ask/stream (STREAM_BRAIN_CALL
    instead of BRAIN_CALL) — the MANDOS path. Ensures /ask/stream opens the
    /ambient follow-up window exactly like /ask does; the hub does not speak
    this endpoint yet, but when it does, the window must not silently stop
    working."""
    import gateway as gw

    def fake_stream_brain(prompt_, session_id, node_id_, speaker, speaker_score, timeout, on_partial):
        return brain_reply

    monkeypatch.setattr(gw, "STREAM_BRAIN_CALL", fake_stream_brain)
    monkeypatch.setattr(gw, "synthesize", lambda text: (b"RIFF" + b"\x00" * 40, "fake"))
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    payload = json.dumps({"prompt": prompt, "node_id": node_id})
    conn.putrequest("POST", "/ask/stream")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(len(payload)))
    conn.endheaders()
    conn.send(payload.encode())
    resp = conn.getresponse()
    while resp.read(4096):
        pass
    conn.close()


@needs_classify
def test_ask_stream_reply_opens_ambient_followup_window(gateway_server, fake_whisper, monkeypatch):
    """Parity with test_ask_reply_opens_ambient_followup_window above:
    /ask/stream is a second path to a real answer and must open the same
    follow-up window /ask does (gateway.py's own docstring: "the timestamp
    of Lomin's last /ask reply on that node" — /ask/stream is included)."""
    _do_ask_stream(gateway_server, "what's the weather today", "node-1", monkeypatch, "It's sunny today.")

    seen_state = {}

    def fake_jev(state, timeout=None):
        seen_state["state"] = state
        return (0.9 if "OPEN" in state else 0.1), ""

    _patch_jev(monkeypatch, fake_jev)
    _FakeWhisperHandler.next_text = "and tomorrow?"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert "OPEN" in seen_state["state"]
    assert "sunny" in seen_state["state"]
    assert data["engage"] is True


@needs_classify
def test_ask_stream_apology_does_not_open_ambient_followup_window(gateway_server, fake_whisper, monkeypatch):
    _do_ask_stream(gateway_server, "x", "node-1", monkeypatch, "unanswered: timed_out")

    def fake_jev(state, timeout=None):
        return (0.9 if "OPEN" in state else 0.1), ""

    _patch_jev(monkeypatch, fake_jev)
    _FakeWhisperHandler.next_text = "and tomorrow?"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is False


# --------------------------------------------------------------------------
# Follow-up inside the conversation window
# --------------------------------------------------------------------------

@needs_classify
def test_ambient_followup_inside_window_engages(gateway_server, fake_whisper, monkeypatch):
    gateway.record_reply("node-1", "It's sunny today.")

    seen_state = {}

    def fake_jev(state, timeout=None):
        seen_state["state"] = state
        return (0.9 if "OPEN" in state else 0.1), ""

    _patch_jev(monkeypatch, fake_jev)
    _FakeWhisperHandler.next_text = "and tomorrow?"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert status == 200
    assert "OPEN" in seen_state["state"]
    assert data["engage"] is True


def test_ambient_context_and_followup_window_are_per_node(gateway_server, fake_whisper, monkeypatch):
    """LAMMEN shares the gateway with other nodes: one node's room lines and
    open-conversation window must never reach another node's classifier."""
    gateway.record_reply("lammen", "It's sunny today.")
    gateway._record_ambient_transcript("lammen", "hello there", "mike")
    assert gateway._ambient_context_lines("lammen") == ["agent: It's sunny today.", "mike: hello there"]
    assert gateway._ambient_context_lines("bench-node") == []
    assert "OPEN" in gateway._conversation_open_note("lammen")
    assert "OPEN" not in gateway._conversation_open_note("bench-node")


@needs_classify
def test_ambient_followup_outside_window_does_not_force_engage(gateway_server, fake_whisper, monkeypatch):
    gateway.record_reply("node-1", "It's sunny today.")
    # Push the recorded reply outside the follow-up window.
    with gateway._ambient_lock:
        ts, reply = gateway._ambient_last_reply["node-1"]
        gateway._ambient_last_reply["node-1"] = (ts - (gateway.AMBIENT_FOLLOWUP_WINDOW_SEC + 5), reply)

    def fake_jev(state, timeout=None):
        return (0.9 if "OPEN" in state else 0.1), ""

    _patch_jev(monkeypatch, fake_jev)
    _FakeWhisperHandler.next_text = "and tomorrow?"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is False


# --------------------------------------------------------------------------
# No engage on a mention to someone else
# --------------------------------------------------------------------------

@needs_classify
def test_ambient_no_engage_on_mention_to_someone_else(gateway_server, fake_whisper, monkeypatch):
    _FakeWhisperHandler.next_text = "I told Lomin about it yesterday"
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.1, ""))
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is False
    assert data["classifier"] == "jev"


# --------------------------------------------------------------------------
# Jev -> Haiku fallback (error/timeout only, not a low score)
# --------------------------------------------------------------------------

@needs_classify
def test_ambient_falls_back_to_haiku_on_jev_error(gateway_server, fake_whisper, monkeypatch):
    _patch_jev(monkeypatch, lambda state, timeout=None: (None, "http 503"))
    _patch_haiku(monkeypatch, lambda state, timeout=None: (0.8, ""))
    _FakeWhisperHandler.next_text = "Lomin turn off the lights"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is True
    assert data["classifier"] == "haiku"


@needs_classify
def test_ambient_low_jev_score_never_calls_haiku(gateway_server, fake_whisper, monkeypatch):
    haiku_calls = []
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.1, ""))
    _patch_haiku(
        monkeypatch,
        lambda state, timeout=None: (haiku_calls.append(1), (0.9, ""))[1],
    )
    _FakeWhisperHandler.next_text = "that's almost done"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is False
    assert data["classifier"] == "jev"
    assert not haiku_calls


# --------------------------------------------------------------------------
# Both classifiers fail (Jev then Haiku)
# --------------------------------------------------------------------------

@needs_classify
def test_ambient_both_classifiers_fail_does_not_engage(gateway_server, fake_whisper, monkeypatch):
    _patch_jev(monkeypatch, lambda state, timeout=None: (None, "timeout"))
    _patch_haiku(monkeypatch, lambda state, timeout=None: (None, "connection refused"))
    _FakeWhisperHandler.next_text = "Lomin what's up"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert status == 200
    assert data["engage"] is False
    assert data["classifier"] == "none"


# --------------------------------------------------------------------------
# multipart form with node_id field
# --------------------------------------------------------------------------

@needs_classify
def test_ambient_multipart_node_id(gateway_server, fake_whisper, monkeypatch):
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    _FakeWhisperHandler.next_text = "hey Lomin"
    boundary = "----WebKitFormBoundaryXYZ"
    body = _build_multipart_ambient(boundary, b"RIFFdata", "node-mp")
    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request(
        "POST", "/ambient", body=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    assert data["engage"] is True
    # context recorded under the right node, labeled "unknown" (no speakers
    # configured by default — see _no_speakers_file_by_default)
    assert gateway._ambient_context_lines("node-mp") == ["unknown: hey Lomin"]


# --------------------------------------------------------------------------
# STT cache: /ambient engage populates it, /stt serves the hit
# --------------------------------------------------------------------------

@needs_classify
def test_stt_cache_hit_after_ambient_engage(gateway_server, fake_whisper, monkeypatch):
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    _FakeWhisperHandler.next_text = "Lomin set a timer"
    wav_bytes = b"RIFF" + b"same audio bytes" * 10
    status, data = _post_ambient(gateway_server, wav_bytes, "node-1")
    assert data["engage"] is True
    assert len(_FakeWhisperHandler.received_bodies) == 1

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("POST", "/stt", body=wav_bytes, headers={"Content-Type": "audio/wav"})
    resp = conn.getresponse()
    stt_data = json.loads(resp.read())
    conn.close()
    assert stt_data["text"] == "Lomin set a timer"
    # Whisper was not called a second time — served from cache.
    assert len(_FakeWhisperHandler.received_bodies) == 1


@needs_classify
def test_stt_cache_not_populated_when_ambient_does_not_engage(gateway_server, fake_whisper, monkeypatch):
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.1, ""))
    _FakeWhisperHandler.next_text = "I told Lomin about it yesterday"
    wav_bytes = b"RIFF" + b"other audio bytes" * 10
    status, data = _post_ambient(gateway_server, wav_bytes, "node-1")
    assert data["engage"] is False

    conn = http.client.HTTPConnection("127.0.0.1", gateway_server, timeout=10)
    conn.request("POST", "/stt", body=wav_bytes, headers={"Content-Type": "audio/wav"})
    resp = conn.getresponse()
    resp.read()
    conn.close()
    # Whisper called again — no cache entry since /ambient never engaged.
    assert len(_FakeWhisperHandler.received_bodies) == 2


# --------------------------------------------------------------------------
# Privacy: a non-engaging transcript is never persisted anywhere
# --------------------------------------------------------------------------

@needs_classify
def test_ambient_non_engaging_transcript_never_written_to_disk(gateway_server, fake_whisper, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "LOG_DIR", tmp_path)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.05, ""))
    secret_phrase = "XYZZY-DISTINCTIVE-SECRET-PHRASE-9182"
    _FakeWhisperHandler.next_text = f"I mentioned Lomin to {secret_phrase}"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is False

    log_file = tmp_path / "lomin.jsonl"
    _wait_for_log_line(log_file)

    for f in tmp_path.rglob("*"):
        if f.is_file():
            content = f.read_text(errors="ignore")
            assert secret_phrase not in content

    lines = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
    ambient_lines = [l for l in lines if l["endpoint"] == "/ambient"]
    assert ambient_lines
    for l in ambient_lines:
        assert "transcript" not in l
        assert "chars" in l


@needs_classify
def test_ambient_engaging_transcript_also_not_logged_as_text(gateway_server, fake_whisper, monkeypatch, tmp_path):
    """Even an engaged transcript's text is never written to the request
    log — only metadata. (The transcript itself only ever leaves this
    process in the HTTP response to the caller.)"""
    monkeypatch.setattr(gateway, "LOG_DIR", tmp_path)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    secret_phrase = "QUUX-ENGAGED-SECRET-PHRASE-4477"
    _FakeWhisperHandler.next_text = f"Lomin, {secret_phrase}"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is True

    log_file = tmp_path / "lomin.jsonl"
    _wait_for_log_line(log_file)
    for line in log_file.read_text().splitlines():
        if not line.strip():
            continue
        assert secret_phrase not in line


# --------------------------------------------------------------------------
# classify_addressed chain
# --------------------------------------------------------------------------

@needs_classify
def test_classify_addressed_jev_success(monkeypatch):
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.7, ""))
    called = []
    _patch_haiku(monkeypatch, lambda state, timeout=None: (called.append(1), (0.5, ""))[1])
    score, classifier, reason = gateway.classify_addressed("state")
    assert (score, classifier) == (0.7, "jev")
    assert not called


@needs_classify
def test_classify_addressed_both_fail(monkeypatch):
    _patch_jev(monkeypatch, lambda state, timeout=None: (None, "boom"))
    _patch_haiku(monkeypatch, lambda state, timeout=None: (None, "bust"))
    score, classifier, reason = gateway.classify_addressed("state")
    assert classifier == "none"
    assert score == 0.0
    assert "boom" in reason and "bust" in reason


# --------------------------------------------------------------------------
# Speaker ID on /ambient (SPEC-011) — identify_speaker() and its wiring into
# the /ambient response, rolling context, and speaker cache.
# --------------------------------------------------------------------------

@needs_classify
def test_ambient_identifies_matching_speaker(gateway_server, fake_whisper, fake_embed, monkeypatch, tmp_path):
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    monkeypatch.setattr(
        gateway, "SPEAKERS_PATH",
        _write_speakers(tmp_path, speakers=[{"id": "mike", "embedding": [1.0, 0.0, 0.0, 0.0]}]),
    )
    _FakeWhisperHandler.next_text = "what's the weather, Lomin"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert status == 200
    assert data["engage"] is True
    assert data["speaker"] == "mike"
    assert data["speaker_score"] >= gateway.SPEAKER_THRESHOLD
    assert gateway._ambient_context_lines("node-1") == ["mike: what's the weather, Lomin"]


@needs_classify
def test_ambient_no_match_below_threshold(gateway_server, fake_whisper, fake_embed, monkeypatch, tmp_path):
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    _FakeEmbedHandler.response_body = json.dumps(
        {"embedding": [0.0, 1.0, 0.0, 0.0], "model": "speechbrain/spkrec-ecapa-voxceleb"}
    ).encode()
    monkeypatch.setattr(
        gateway, "SPEAKERS_PATH",
        _write_speakers(tmp_path, speakers=[{"id": "mike", "embedding": [1.0, 0.0, 0.0, 0.0]}]),
    )
    _FakeWhisperHandler.next_text = "hey Lomin"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is True
    assert data["speaker"] == ""
    assert data["speaker_score"] == 0.0
    assert gateway._ambient_context_lines("node-1") == ["unknown: hey Lomin"]


@needs_classify
def test_ambient_embed_upstream_down_speaker_empty(gateway_server, fake_whisper, monkeypatch, tmp_path):
    """DESKTOP_EMBED_URL stays at the dead port (_dead_desktop_embed_by_default
    autouse fixture, no fake_embed here) — the ambient turn must still
    succeed with an empty speaker, never an error."""
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    monkeypatch.setattr(
        gateway, "SPEAKERS_PATH",
        _write_speakers(tmp_path, speakers=[{"id": "mike", "embedding": [1.0, 0.0, 0.0, 0.0]}]),
    )
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert status == 200
    assert data["engage"] is True
    assert data["speaker"] == ""
    assert data["speaker_score"] == 0.0


@needs_classify
def test_ambient_model_mismatch_speaker_empty(gateway_server, fake_whisper, fake_embed, monkeypatch, tmp_path):
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    monkeypatch.setattr(
        gateway, "SPEAKERS_PATH",
        _write_speakers(tmp_path, model="some-other-model",
                         speakers=[{"id": "mike", "embedding": [1.0, 0.0, 0.0, 0.0]}]),
    )
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["speaker"] == ""
    assert data["speaker_score"] == 0.0


@needs_classify
def test_ambient_speakers_file_missing_speaker_empty(gateway_server, fake_whisper, fake_embed, monkeypatch):
    """_no_speakers_file_by_default already points SPEAKERS_PATH at a file
    that doesn't exist — identify_speaker() must short-circuit before ever
    calling the (otherwise-healthy) fake embed server."""
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["speaker"] == ""
    assert data["speaker_score"] == 0.0
    assert _FakeEmbedHandler.received_bodies == []


@needs_classify
def test_speaker_cache_hit_avoids_second_embed_call(gateway_server, fake_whisper, fake_embed, monkeypatch, tmp_path):
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    monkeypatch.setattr(
        gateway, "SPEAKERS_PATH",
        _write_speakers(tmp_path, speakers=[{"id": "mike", "embedding": [1.0, 0.0, 0.0, 0.0]}]),
    )
    wav_bytes = b"RIFF" + b"same audio bytes for speaker cache" * 5

    status, data = _post_ambient(gateway_server, wav_bytes, "node-1")
    assert data["speaker"] == "mike"
    assert len(_FakeEmbedHandler.received_bodies) == 1

    status2, data2 = _post_ambient(gateway_server, wav_bytes, "node-1")
    assert data2["speaker"] == "mike"
    # Served from the speaker cache — no second call to the embed server.
    assert len(_FakeEmbedHandler.received_bodies) == 1


def test_match_speaker_skips_uncomparable_fingerprints():
    speakers = [
        {"id": "bad-dim", "embedding": [1.0, 0.0]},
        {"id": "mike", "embedding": [1.0, 0.0, 0.0, 0.0]},
    ]
    speaker_id, score = gateway._match_speaker([1.0, 0.0, 0.0, 0.0], speakers, 0.70)
    assert speaker_id == "mike"
    assert score == 1.0


def test_cosine_similarity_orthogonal_is_zero():
    assert gateway._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


@needs_classify
def test_ambient_speaker_label_never_written_to_disk(gateway_server, fake_whisper, fake_embed, monkeypatch, tmp_path):
    """Same privacy posture as the transcript text itself (see
    test_ambient_non_engaging_transcript_never_written_to_disk) — a matched
    speaker's name must not appear in any file under the log directory."""
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(gateway, "LOG_DIR", log_dir)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    monkeypatch.setattr(
        gateway, "SPEAKERS_PATH",
        _write_speakers(tmp_path, speakers=[{"id": "mike", "embedding": [1.0, 0.0, 0.0, 0.0]}]),
    )
    _FakeWhisperHandler.next_text = "what's the weather, Lomin"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["speaker"] == "mike"

    log_file = log_dir / "lomin.jsonl"
    _wait_for_log_line(log_file)
    for f in log_dir.rglob("*"):
        if f.is_file():
            assert "mike" not in f.read_text(errors="ignore")


@needs_classify
def test_ambient_speaker_id_never_delays_response_past_wait_bound(
    gateway_server, fake_whisper, fake_embed, monkeypatch, tmp_path
):
    """identify_speaker() itself may take much longer than
    AMBIENT_SPEAKER_WAIT_SEC (a slow desktop embed hop) — _handle_ambient
    must still respond within roughly that bound of STT finishing, with an
    empty speaker, rather than waiting for the slow call."""
    monkeypatch.setattr(gateway, "AMBIENT_SPEAKER_WAIT_SEC", 0.3)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    monkeypatch.setattr(
        gateway, "SPEAKERS_PATH",
        _write_speakers(tmp_path, speakers=[{"id": "mike", "embedding": [1.0, 0.0, 0.0, 0.0]}]),
    )
    _FakeEmbedHandler.delay_sec = 2.0

    t0 = time.time()
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    elapsed = time.time() - t0

    assert data["speaker"] == ""
    assert data["speaker_score"] == 0.0
    assert elapsed < 1.5, f"took {elapsed:.2f}s — should have returned within ~{gateway.AMBIENT_SPEAKER_WAIT_SEC}s of STT"


# --------------------------------------------------------------------------
# MAHANAXAR wiring: classify_addressed -> classify(), Haiku chain
# --------------------------------------------------------------------------

@needs_classify
def test_classify_addressed_calls_classify_with_spec_kwargs(monkeypatch):
    calls = []

    def fake_classify(caller, state, question, **kw):
        calls.append(dict(kw, caller=caller, state=state, question=question))
        if kw["layers"] == ("laya",):
            return _cl.Result()  # Laya down
        return _cl.Result(choice="addressed_to_agent", layer="jev",
                          probabilities={"addressed_to_agent": 0.9, "unclear": 0.1},
                          confidence=0.9)

    monkeypatch.setattr(_cl, "classify", fake_classify)
    score, classifier, _ = gateway.classify_addressed("state")
    assert (score, classifier) == (0.9, "jev")
    laya_kw, seen = calls
    assert laya_kw["kind"] == "choice"
    assert laya_kw["options"] == gateway.AMBIENT_OPTIONS
    assert laya_kw["laya_question"] == gateway.AMBIENT_LAYA_QUESTION
    assert "laya_criteria" not in laya_kw  # options double as Laya criteria
    assert laya_kw["laya_timeout"] == gateway.AMBIENT_LAYA_TIMEOUT == 0.5
    assert seen["kind"] == "choice"
    assert set(seen["options"]) == {"addressed_to_agent", "conversation_between_people",
                                    "background_media", "unclear"}
    assert seen["caller"] == "lomin.ambient"
    assert seen["question_key"] == "addressed_to_agent"
    assert seen["layers"] == ("jev", "haiku")
    assert "laya_state" not in seen
    assert seen["jev_retry"] is False
    assert seen["timeout"] == gateway.AMBIENT_JEV_TIMEOUT
    assert seen["haiku_timeout"] == gateway.AMBIENT_HAIKU_TIMEOUT
    assert gateway.AMBIENT_HAIKU_TIMEOUT == 12.0
    assert "local_model" not in seen


@needs_classify
def test_classify_addressed_jev_error_is_single_attempt(monkeypatch):
    calls = []
    monkeypatch.setattr(_cl, "jev_choice_once",
                        lambda *a, **k: (calls.append(1), (None, "http 503", True, None, {}))[1])
    _patch_haiku(monkeypatch, lambda state, timeout=None: (0.8, ""))
    score, classifier, _ = gateway.classify_addressed("s")
    assert classifier == "haiku" and len(calls) == 1  # no 1s-delay retry


@needs_classify
def test_haiku_addressed_choice_scores_one_and_other_choice_zero(monkeypatch):
    _patch_jev(monkeypatch, lambda state, timeout=None: (None, "boom"))
    _patch_haiku(monkeypatch, lambda state, timeout=None: (0.9, ""))
    assert gateway.classify_addressed("s")[:2] == (1.0, "haiku")
    _patch_haiku(monkeypatch, lambda state, timeout=None: (0.1, ""))
    assert gateway.classify_addressed("s")[:2] == (0.0, "haiku")


@needs_classify
def test_jev_non_addressed_choices_stay_below_threshold(monkeypatch):
    for choice in ("background_media", "unclear", "conversation_between_people"):
        def fake(key, state, qk, instr, options, timeout, _c=choice):
            return _c, "", True, None, {
                "probabilities": {"addressed_to_agent": 0.2, _c: 0.8}, "confidence": 0.8}
        monkeypatch.setattr(_cl, "jev_choice_once", fake)
        score, classifier, _ = gateway.classify_addressed("s")
        assert classifier == "jev" and score == 0.2 < gateway.AMBIENT_THRESHOLD


@needs_classify
def test_jev_choice_without_probabilities_scores_by_choice(monkeypatch):
    monkeypatch.setattr(_cl, "jev_choice_once",
                        lambda *a, **k: ("addressed_to_agent", "", True, None, {}))
    assert gateway.classify_addressed("s")[:2] == (1.0, "jev")


@needs_classify
def test_laya_choice_without_probabilities_is_ignored(monkeypatch):
    """Laya must report P(addressed_to_agent); a bare choice cannot take the
    fast path, Jev decides."""
    _patch_laya(monkeypatch, lambda state, question, **kw:
                (None, "addressed_to_agent", "", None))
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.2, ""))
    d = gateway.classify_addressed_detail("s", _TURNS)
    assert (d["classifier"], d["laya_score"], d["score"]) == ("jev", None, 0.2)


@needs_classify
def test_classify_addressed_one_usage_line_per_classify_call(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("CLASSIFY_USAGE_LOG_PATH", str(log))
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.9, ""))
    gateway.classify_addressed("s")
    lines = log.read_text().splitlines()
    assert len(lines) == 2  # Laya (down) and Jev each get their own line
    last = json.loads(lines[-1])
    assert last["caller"] == "lomin.ambient"
    assert last["kind"] == "choice" and last["choice"] == "addressed_to_agent"


# --------------------------------------------------------------------------
# Speech gate (Silero, pre-STT)
# --------------------------------------------------------------------------

import math  # noqa: E402
import struct  # noqa: E402
import wave as _wave  # noqa: E402
from io import BytesIO  # noqa: E402

try:
    import numpy as _np  # noqa: E402
except ImportError:  # pragma: no cover
    _np = None

needs_numpy = pytest.mark.skipif(_np is None, reason="numpy not installed")


def _wav(samples, rate=16000, channels=1):
    buf = BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))
    return buf.getvalue()


def _silence_wav(seconds=1.0, rate=16000):
    return _wav([0] * int(rate * seconds), rate)


def _noise_wav(seconds=1.0, rate=16000):
    import random
    rnd = random.Random(1)
    return _wav([rnd.randint(-3000, 3000) for _ in range(int(rate * seconds))], rate)


class _StubSession:
    """Stands in for the ONNX session: returns the next scripted probability."""

    def __init__(self, probs):
        self.probs = list(probs)
        self.calls = 0

    def run(self, _outs, feeds):
        p = self.probs[min(self.calls, len(self.probs) - 1)]
        self.calls += 1
        z = feeds["h"]
        return [[[p]], z, feeds["c"]]


def test_voiced_ms_and_decision():
    probs = [0.1, 0.6, 0.9, 0.5, 0.49]
    assert gateway.voiced_ms(probs) == 3 * gateway.SILERO_WINDOW_MS  # 0.6, 0.9, 0.5
    assert gateway.voiced_ms(probs, threshold=0.8) == 1 * gateway.SILERO_WINDOW_MS
    assert gateway.speech_gate_decision(416.0, 400)
    assert gateway.speech_gate_decision(400.0, 400)
    assert not gateway.speech_gate_decision(384.0, 400)


@needs_numpy
def test_silero_window_probs_stub_session_and_partial_window_dropped():
    sess = _StubSession([0.9])
    pcm = _np.zeros(512 * 3 + 100, dtype=_np.int16)
    assert gateway.silero_window_probs(sess, pcm) == [pytest.approx(0.9)] * 3
    assert sess.calls == 3


@needs_numpy
def test_speech_gate_rejects_when_stub_says_silent(monkeypatch):
    monkeypatch.setattr(gateway, "_silero_get", lambda: _StubSession([0.05]))
    status, voiced, vad_ms = gateway.speech_gate(_silence_wav())
    assert (status, voiced) == ("rejected", 0)
    assert vad_ms >= 0


@needs_numpy
def test_speech_gate_passes_when_stub_says_speech(monkeypatch):
    monkeypatch.setattr(gateway, "_silero_get", lambda: _StubSession([0.95]))
    status, voiced, _ = gateway.speech_gate(_silence_wav())
    assert status == "passed" and voiced >= gateway.MIN_SPEECH_MS


@needs_numpy
def test_speech_gate_resamples_and_downmixes(monkeypatch):
    sess = _StubSession([0.95])
    monkeypatch.setattr(gateway, "_silero_get", lambda: sess)
    # 1s stereo at 8 kHz -> 1s mono at 16 kHz -> 31 full windows
    stereo = _wav([100, -100] * 8000, rate=8000, channels=2)
    status, voiced, _ = gateway.speech_gate(stereo)
    assert status == "passed"
    assert sess.calls == 16000 // 512


def test_speech_gate_unavailable_paths(monkeypatch):
    monkeypatch.setattr(gateway, "_silero_get", lambda: None)
    assert gateway.speech_gate(_silence_wav())[:2] == ("unavailable", None)
    monkeypatch.setattr(gateway, "_silero_get", lambda: _StubSession([0.9]))
    assert gateway.speech_gate(b"RIFFdata")[:2] == ("unavailable", None)  # not a WAV

    def boom():
        raise RuntimeError("onnx exploded")
    monkeypatch.setattr(gateway, "_silero_get", boom)
    assert gateway.speech_gate(_silence_wav())[:2] == ("unavailable", None)
    monkeypatch.setattr(gateway, "SPEECH_GATE_ENABLED", False)
    assert gateway.speech_gate(_silence_wav())[:2] == ("unavailable", None)


def test_silero_get_fails_open_on_missing_model(monkeypatch, tmp_path):
    monkeypatch.undo()
    monkeypatch.setattr(gateway, "SILERO_MODEL_PATH", str(tmp_path / "nope.onnx"))
    monkeypatch.setattr(gateway, "_silero_session", None)
    monkeypatch.setattr(gateway, "_silero_failed_at", 0.0)
    assert gateway._silero_get() is None
    assert gateway.speech_gate(_silence_wav())[0] == "unavailable"


@needs_numpy
def test_ambient_no_speech_clip_skips_stt_and_classify(gateway_server, fake_whisper, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "_silero_get", lambda: _StubSession([0.02]))
    monkeypatch.setattr(gateway, "LOG_DIR", tmp_path)
    classify_calls = []
    monkeypatch.setattr(gateway, "classify_addressed",
                        lambda state: classify_calls.append(1) or (0.99, "jev", ""))
    _FakeWhisperHandler.received_bodies = []
    status, data = _post_ambient(gateway_server, _noise_wav(), "node-1")
    assert status == 200
    assert data["engage"] is False
    assert data["classifier"] == "no-speech"
    assert data["transcript"] == ""
    assert not _FakeWhisperHandler.received_bodies  # Whisper never called
    assert not classify_calls
    log_file = tmp_path / "lomin.jsonl"
    _wait_for_log_line(log_file)
    rec = json.loads(log_file.read_text().splitlines()[-1])
    assert rec["speech_gate"] == "rejected"
    assert rec["voiced_ms"] == 0
    assert rec["classifier"] == "no-speech"
    assert "vad_ms" in rec


@needs_numpy
@needs_classify
def test_ambient_speech_clip_passes_gate_and_logs_it(gateway_server, fake_whisper, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "_silero_get", lambda: _StubSession([0.95]))
    monkeypatch.setattr(gateway, "LOG_DIR", tmp_path)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    _FakeWhisperHandler.next_text = "Lomin, what time is it"
    status, data = _post_ambient(gateway_server, _silence_wav(), "node-1")
    assert data["engage"] is True and data["classifier"] == "jev"
    assert "vad" in data["ms"]
    log_file = tmp_path / "lomin.jsonl"
    _wait_for_log_line(log_file)
    rec = json.loads(log_file.read_text().splitlines()[-1])
    assert rec["speech_gate"] == "passed"
    assert rec["voiced_ms"] >= gateway.MIN_SPEECH_MS


@needs_classify
def test_ambient_gate_unavailable_falls_through_to_whisper(gateway_server, fake_whisper, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "_silero_get", lambda: None)
    monkeypatch.setattr(gateway, "LOG_DIR", tmp_path)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.95, ""))
    status, data = _post_ambient(gateway_server, _silence_wav(), "node-1")
    assert data["engage"] is True
    log_file = tmp_path / "lomin.jsonl"
    _wait_for_log_line(log_file)
    rec = json.loads(log_file.read_text().splitlines()[-1])
    assert rec["speech_gate"] == "unavailable"
    assert "voiced_ms" not in rec


@needs_classify
def test_ambient_jev_ok_error_fallback_and_both_fail_with_real_wav(gateway_server, fake_whisper, monkeypatch):
    # Not a wake phrase: "hello Lomin" would now short-circuit before Jev.
    _FakeWhisperHandler.next_text = "what's the weather tomorrow, Lomin"
    wav = _silence_wav()
    _patch_jev(monkeypatch, lambda s, t=None: (0.9, ""))
    assert _post_ambient(gateway_server, wav, "n")[1]["classifier"] == "jev"
    _patch_jev(monkeypatch, lambda s, t=None: (None, "http 503"))
    _patch_haiku(monkeypatch, lambda s, t=None: (0.8, ""))
    assert _post_ambient(gateway_server, wav, "n")[1]["classifier"] == "haiku"
    _patch_haiku(monkeypatch, lambda s, t=None: (None, "refused"))
    status, data = _post_ambient(gateway_server, wav, "n")
    assert data["classifier"] == "none" and data["engage"] is False


REAL_SILERO = next(iter(__import__("glob").glob(
    "/opt/lomin/venv/lib/python3*/site-packages/openwakeword/resources/models/silero_vad.onnx"
    )), None) or next(iter(__import__("glob").glob(
    str(Path.home() / ".mvg/wakeword-train/venv*/lib/python3*/site-packages/openwakeword/resources/models/silero_vad.onnx")
    )), None)


@needs_numpy
@pytest.mark.skipif(REAL_SILERO is None, reason="silero_vad.onnx not present")
def test_real_silero_model_rejects_silence_and_noise(monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.undo()  # drop the autouse _silero_get stub; use the real loader
    monkeypatch.setattr(gateway, "SILERO_MODEL_PATH", REAL_SILERO)
    monkeypatch.setattr(gateway, "_silero_session", None)
    monkeypatch.setattr(gateway, "_silero_failed_at", 0.0)
    for wav in (_silence_wav(2.0), _noise_wav(2.0)):
        status, voiced, vad_ms = gateway.speech_gate(wav)
        assert status == "rejected", (status, voiced)
    # silence session is loaded once and reused
    assert gateway._silero_session is not None


# --------------------------------------------------------------------------
# Laya primary
# --------------------------------------------------------------------------

def _patch_laya(monkeypatch, fn):
    """fn(state, question, **kw) -> (prob|None, choice, err, usage)."""
    if _cl is None:
        pytest.skip("needs the household classifier module (plugin 'classify')")
    monkeypatch.setattr(_cl, "laya_probability", fn)


_TURNS = [{"role": "user", "content": "mike: hi lomin"}]


def _laya_p(monkeypatch, p, record=None):
    def fake(state, question, **kw):
        if record is not None:
            record.update(state=state, question=question, **kw)
        if p is None:
            return None, None, "laya request failed: timeout", None
        probs = {"addressed_to_agent": p, "conversation_between_people": 1 - p}
        choice = max(probs, key=probs.get)
        if kw.get("meta") is not None:
            kw["meta"].update({"probabilities": probs, "confidence": probs[choice]})
        return probs[choice], choice, "", None
    _patch_laya(monkeypatch, fake)


@needs_classify
def test_laya_confident_yes_skips_jev(monkeypatch):
    seen = {}
    _laya_p(monkeypatch, 0.93, seen)
    called = []
    _patch_jev(monkeypatch, lambda state, timeout=None: (called.append(1), (0.1, ""))[1])
    d = gateway.classify_addressed_detail("jev state", _TURNS)
    assert (d["score"], d["classifier"], d["laya_score"]) == (0.93, "laya", 0.93)
    assert not called
    assert seen["state"] == _TURNS
    assert seen["question"] == gateway.AMBIENT_LAYA_QUESTION
    assert seen["kind"] == "choice"
    assert seen["criteria"] == gateway.AMBIENT_OPTIONS
    assert seen["laya_yes_option"] is None
    assert seen["timeout"] == 0.5


@needs_classify
def test_laya_low_jev_yes_overrides(monkeypatch):
    _laya_p(monkeypatch, 0.3)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.85, ""))
    d = gateway.classify_addressed_detail("s", _TURNS)
    assert (d["score"], d["classifier"], d["laya_score"]) == (0.85, "laya+jev", 0.3)


@needs_classify
def test_laya_low_jev_low_decides_no(monkeypatch):
    _laya_p(monkeypatch, 0.65)  # below 0.7 confident, above 0.6
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.2, ""))
    d = gateway.classify_addressed_detail("s", _TURNS)
    assert (d["score"], d["classifier"]) == (0.2, "laya+jev")
    assert d["score"] < gateway.AMBIENT_THRESHOLD


@needs_classify
def test_laya_low_jev_fails_haiku_answers(monkeypatch):
    _laya_p(monkeypatch, 0.1)
    _patch_jev(monkeypatch, lambda state, timeout=None: (None, "boom"))
    _patch_haiku(monkeypatch, lambda state, timeout=None: (0.75, ""))
    d = gateway.classify_addressed_detail("s", _TURNS)
    assert (d["score"], d["classifier"], d["laya_score"]) == (1.0, "laya+haiku", 0.1)


@needs_classify
def test_laya_timeout_falls_to_jev(monkeypatch):
    _laya_p(monkeypatch, None)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.8, ""))
    d = gateway.classify_addressed_detail("s", _TURNS)
    assert (d["score"], d["classifier"], d["laya_score"]) == (0.8, "jev", None)


@needs_classify
def test_laya_timeout_jev_fails_haiku(monkeypatch):
    _laya_p(monkeypatch, None)
    _patch_jev(monkeypatch, lambda state, timeout=None: (None, "boom"))
    _patch_haiku(monkeypatch, lambda state, timeout=None: (0.66, ""))
    score, classifier, _ = gateway.classify_addressed("s", _TURNS)
    assert (score, classifier) == (1.0, "haiku")  # Haiku: choice only, no probability


@needs_classify
def test_laya_answered_jev_and_haiku_fail_uses_laya_verdict(monkeypatch):
    _laya_p(monkeypatch, 0.65)
    _patch_jev(monkeypatch, lambda state, timeout=None: (None, "boom"))
    _patch_haiku(monkeypatch, lambda state, timeout=None: (None, "bust"))
    d = gateway.classify_addressed_detail("s", _TURNS)
    assert (d["score"], d["classifier"], d["laya_score"]) == (0.65, "laya", 0.65)
    assert d["score"] >= gateway.AMBIENT_THRESHOLD  # engages at 0.6


@needs_classify
def test_laya_low_and_everything_fails_below_threshold(monkeypatch):
    _laya_p(monkeypatch, 0.2)
    _patch_jev(monkeypatch, lambda state, timeout=None: (None, "boom"))
    _patch_haiku(monkeypatch, lambda state, timeout=None: (None, "bust"))
    d = gateway.classify_addressed_detail("s", _TURNS)
    assert d["classifier"] == "laya" and d["score"] < gateway.AMBIENT_THRESHOLD


@needs_classify
def test_no_layer_answers_is_none(monkeypatch):
    score, classifier, _ = gateway.classify_addressed("s", _TURNS)
    assert (score, classifier) == (0.0, "none")


def test_no_classifier_installed_engages_on_the_name_alone(monkeypatch):
    monkeypatch.setattr(gateway, "_jev_classifier", None)
    score, classifier, _ = gateway.classify_addressed("s", _TURNS)
    assert (score, classifier) == (1.0, "name-only")


@needs_classify
def test_laya_and_jev_each_write_a_usage_line(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("CLASSIFY_USAGE_LOG_PATH", str(log))
    _laya_p(monkeypatch, 0.2)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.9, ""))
    gateway.classify_addressed("s", _TURNS)
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert [r["layer_answered"] for r in recs] == ["laya", "jev"]


def test_ambient_turns_shape_with_open_conversation(monkeypatch):
    gateway._record_ambient_transcript("n1", "what is the weather", "mike")
    gateway._record_ambient_transcript("n1", "dinner soon", "")
    gateway.record_reply("n1", "Sunny and 70.")
    turns = gateway._build_ambient_turns("n1", "and tomorrow?", "mike")
    assert [t["content"] for t in turns] == [
        "mike: what is the weather",
        "unknown: dinner soon",
        "agent: Sunny and 70.",
        "mike: and tomorrow?",
    ]
    assert all(t["role"] == "user" for t in turns)


def test_ambient_turns_no_open_conversation_after_window(monkeypatch):
    gateway.record_reply("n2", "old reply")
    monkeypatch.setattr(gateway, "AMBIENT_FOLLOWUP_WINDOW_SEC", -1.0)
    turns = gateway._build_ambient_turns("n2", "hello", "")
    # Lomin's reply stays in the rolling context for the whole context window
    # even after the follow-up window closed.
    assert [t["content"] for t in turns] == ["agent: old reply", "unknown: hello"]
    assert not gateway._conversation_open("n2")


# --------------------------------------------------------------------------
# Wake phrase: a greeting + name at utterance start bypasses the classifiers
# --------------------------------------------------------------------------

def _spy_chain(monkeypatch):
    """Count calls into the classifier chain (decide_addressed's last step);
    the real chain still runs."""
    calls = []
    real = gateway.classify_addressed_detail

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(gateway, "classify_addressed_detail", spy)
    return calls


@pytest.mark.parametrize("text", [
    "hey Lomin what time is it",
    "Hey, Lomin.",
    "okay Lommin turn on the light",
])
@needs_classify
def test_wake_phrase_decides_without_classifier(monkeypatch, text):
    calls = _spy_chain(monkeypatch)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.0, ""))
    d = gateway.decide_addressed("node-wake", text)
    assert d == {"score": 1.0, "classifier": "wake-phrase",
                 "reason": "wake phrase at utterance start", "laya_score": None}
    assert calls == []


@pytest.mark.parametrize("text", [
    "hey Lomin what time is it",
    "Hey, Lomin.",
    "okay Lommin turn on the light",
])
@needs_classify
def test_wake_phrase_engages_over_http_without_classifier(
        gateway_server, fake_whisper, monkeypatch, text):
    jev_calls = []
    _patch_jev(monkeypatch, lambda state, timeout=None: (jev_calls.append(1), (0.0, ""))[1])
    _FakeWhisperHandler.next_text = text
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-wake")
    assert status == 200
    assert data["engage"] is True
    assert data["classifier"] == "wake-phrase"
    assert data["score"] == 1.0
    assert jev_calls == []


@pytest.mark.parametrize("text", [
    "I said hey lomin earlier to him",
    "that was famous",
    "almost there",
])
@needs_classify
def test_non_wake_phrases_take_the_normal_chain(monkeypatch, text):
    calls = _spy_chain(monkeypatch)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.1, ""))
    d = gateway.decide_addressed("node-nowake", text)
    assert d["classifier"] != "wake-phrase"
    assert calls == [1]


def test_wake_phrase_regex_boundaries():
    m = gateway.WAKE_PHRASE_RE.match
    assert m("  ...hi LOWMIN, hello")
    assert m("Hello Lomin")
    assert not m("hey lemon")
    assert not m("hey loaming")
    assert not m("hey Lomincow")
    assert not m("hey, there Lomin")
