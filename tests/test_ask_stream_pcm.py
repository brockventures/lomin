#!/usr/bin/env python3
"""/ask/stream sub-sentence PCM streaming (MVG_TTS_STREAM) against a fake
desktop TTS server. Run: python3 -m pytest ./tests -v"""
import base64
import http.client
import json
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402

PCM_MIN = 4800


def _wav(frames: bytes, rate=24000) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


class FakeTTS(BaseHTTPRequestHandler):
    """Streams each request's pcm as writes with a short delay between them.
    The audio for a sentence is bytes([n, 0]) repeated, where n is the first
    digit in the request text, so ordering is checkable."""
    protocol_version = "HTTP/1.0"
    requests: list = []
    mode = "pcm"          # pcm | wav | error
    gate: threading.Event | None = None
    sizes = (6000, 6000, 3000)
    closed_early = threading.Event()

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, body))
        if self.path == "/tts":
            wav = _wav(b"\x09\x00" * 500, rate=22050)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("X-TTS-Engine", "piper")
            self.end_headers()
            self.wfile.write(wav)
            return
        mode = type(self).mode
        if mode == "error":
            self.send_response(500)
            self.end_headers()
            return
        ordinal = int("".join(c for c in body["text"] if c.isdigit())[:1] or "0")
        if mode == "wav":
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("X-TTS-Engine", "piper")
            self.end_headers()
            self.wfile.write(_wav(bytes([ordinal, 0]) * 2000))
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/L16")
        self.send_header("X-TTS-Engine", "chatterbox-stream")
        self.send_header("X-Sample-Rate", "24000")
        self.end_headers()
        try:
            for i, n in enumerate(type(self).sizes):
                self.wfile.write(bytes([ordinal, 0]) * (n // 2))
                self.wfile.flush()
                if i == 0 and type(self).gate is not None:
                    type(self).gate.wait(5)
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            type(self).closed_early.set()


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "LOG_DIR", tmp_path)
    monkeypatch.setattr(gateway, "TTS_STREAM", True)
    monkeypatch.setattr(gateway, "DESKTOP_TTS_URL", "http://127.0.0.1:1/tts")
    monkeypatch.setattr(gateway, "DESKTOP_TTS_STREAM_URL", "")
    monkeypatch.setattr(gateway, "TTS_STREAM_MIN_CHUNK_BYTES", PCM_MIN)
    FakeTTS.requests = []
    FakeTTS.mode = "pcm"
    FakeTTS.gate = None
    FakeTTS.sizes = (6000, 6000, 3000)
    FakeTTS.closed_early = threading.Event()

    def boom(text):
        raise RuntimeError("no engine")
    monkeypatch.setattr(gateway, "piper_speak", boom)


@pytest.fixture()
def fake_tts(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTTS)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(gateway, "DESKTOP_TTS_URL", f"http://127.0.0.1:{server.server_address[1]}/tts")
    yield server
    server.shutdown()


@pytest.fixture()
def gw(monkeypatch):
    server = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def _brain(monkeypatch, text):
    def brain(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        on_partial(text)
        return text
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)


def _post(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    payload = json.dumps({"prompt": "hi", "session_id": "s", "node_id": "n"})
    conn.request("POST", "/ask/stream", body=payload, headers={"Content-Type": "application/json"})
    return conn, conn.getresponse()


def _events(resp):
    raw = resp.read().decode()
    out = []
    for block in raw.split("\n\n"):
        ev = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
        if ev:
            out.append((ev, data))
    return out


def _last_log(tmp_path):
    lines = (tmp_path / "lomin.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if '"/ask/stream"' in line][-1]


def test_chunks_in_order_text_on_first_chunk_only(gw, fake_tts, monkeypatch, tmp_path):
    _brain(monkeypatch, "Sentence 1 is here. Sentence 2 is here.")
    conn, resp = _post(gw)
    events = _events(resp)
    conn.close()
    sents = [d for e, d in events if e == "sentence"]
    assert len(sents) >= 4
    assert all(d["format"] == "pcm" and d["sample_rate"] == 24000 and d["channels"] == 1 for d in sents)
    pcm = [base64.b64decode(d["audio_b64"]) for d in sents]
    assert all(len(p) % 2 == 0 and p for p in pcm)
    marks = [p[0] for p in pcm]
    assert marks == sorted(marks) and set(marks) == {1, 2}
    texts = [d["text"] for d in sents]
    assert [t for t in texts if t] == ["Sentence 1 is here.", "Sentence 2 is here."]
    for m in (1, 2):
        idxs = [i for i, p in enumerate(pcm) if p[0] == m]
        assert texts[idxs[0]] and all(texts[i] == "" for i in idxs[1:])
        assert sum(len(pcm[i]) for i in idxs) == 15000
    assert all(path == "/tts/stream" and b["format"] == "pcm" for path, b in FakeTTS.requests)
    assert events[-1][0] == "done"
    rec = _last_log(tmp_path)
    assert rec["first_audio_ms"] is not None and rec["tts_first_ms"] is not None
    assert rec["tts_stream"] is True


def test_first_chunk_sent_before_sentence_finishes(gw, fake_tts, monkeypatch):
    """Audio is forwarded while the desktop is still synthesizing the same
    sentence (the gate holds the rest of it back)."""
    FakeTTS.gate = threading.Event()
    _brain(monkeypatch, "Sentence 1 is here.")
    conn, resp = _post(gw)
    seen = False
    buf = b""
    deadline = time.time() + 5
    while time.time() < deadline and not seen:
        piece = resp.read1(65536)
        if not piece:
            break
        buf += piece
        seen = b"event: sentence" in buf
    FakeTTS.gate.set()
    assert seen, "no sentence event arrived while upstream was still streaming"
    resp.read()
    conn.close()


def test_stream_error_falls_back_to_wav_converted_to_pcm(gw, fake_tts, monkeypatch):
    FakeTTS.mode = "error"
    _brain(monkeypatch, "Sentence 1 is here.")
    conn, resp = _post(gw)
    events = _events(resp)
    conn.close()
    sents = [d for e, d in events if e == "sentence"]
    assert sents and all(d["format"] == "pcm" for d in sents)
    assert sents[0]["text"] == "Sentence 1 is here."
    total = sum(len(base64.b64decode(d["audio_b64"])) for d in sents)
    assert total == round(500 * 24000 / 22050) * 2  # Piper's 22.05 kHz resampled
    assert events[-1][0] == "done"


def test_stream_returning_wav_is_converted_to_pcm(gw, fake_tts, monkeypatch):
    FakeTTS.mode = "wav"
    _brain(monkeypatch, "Sentence 3 is here.")
    conn, resp = _post(gw)
    sents = [d for e, d in _events(resp) if e == "sentence"]
    conn.close()
    assert sents and all(d["format"] == "pcm" for d in sents)
    assert base64.b64decode(sents[0]["audio_b64"])[0] == 3
    assert all(path == "/tts/stream" for path, _ in FakeTTS.requests)


def test_everything_down_skips_sentence_turn_still_completes(gw, monkeypatch):
    _brain(monkeypatch, "Sentence 1 is here.")
    conn, resp = _post(gw)
    events = _events(resp)
    conn.close()
    assert not [d for e, d in events if e == "sentence"]
    assert events[-1][0] == "done"
    assert [d for e, d in events if e == "reply"][0]["reply"] == "Sentence 1 is here."


def test_flag_off_uses_whole_sentence_wav(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "TTS_STREAM", False)
    monkeypatch.setattr(gateway, "synthesize", lambda t: (_wav(b"\x01\x00" * 300), "fake"))
    _brain(monkeypatch, "Sentence 1 is here. Sentence 2 is here.")
    conn, resp = _post(gw)
    sents = [d for e, d in _events(resp) if e == "sentence"]
    conn.close()
    assert [d["text"] for d in sents] == ["Sentence 1 is here.", "Sentence 2 is here."]
    assert all("format" not in d for d in sents)
    assert base64.b64decode(sents[0]["audio_b64"]).startswith(b"RIFF")
    assert FakeTTS.requests == []
    assert _last_log(tmp_path)["tts_stream"] is False


def test_client_disconnect_closes_upstream_stream(gw, fake_tts, monkeypatch, tmp_path):
    FakeTTS.sizes = (6000,) * 200  # long stream
    _brain(monkeypatch, "Sentence 1 is here.")
    import socket
    payload = json.dumps({"prompt": "hi", "session_id": "s", "node_id": "n"}).encode()
    sock = socket.create_connection(("127.0.0.1", gw), timeout=10)
    sock.sendall(b"POST /ask/stream HTTP/1.0\r\nContent-Type: application/json\r\n"
                 b"Content-Length: %d\r\n\r\n" % len(payload) + payload)
    buf = b""
    while b"event: sentence" not in buf:   # first audio arrives
        piece = sock.recv(65536)
        assert piece
        buf += piece
    sock.close()   # hub goes away
    assert FakeTTS.closed_early.wait(5), "upstream stream was not closed"
    deadline = time.time() + 5
    while time.time() < deadline:
        p = tmp_path / "lomin.jsonl"
        if p.exists() and "client_gone" in p.read_text():
            break
        time.sleep(0.05)
    assert _last_log(tmp_path)["outcome"] == "error:client_gone"


def test_stream_url_derivation_and_override(monkeypatch):
    monkeypatch.setattr(gateway, "DESKTOP_TTS_URL", "http://h:9097/tts")
    assert gateway._desktop_tts_stream_url() == "http://h:9097/tts/stream"
    monkeypatch.setattr(gateway, "DESKTOP_TTS_STREAM_URL", "http://x/s")
    assert gateway._desktop_tts_stream_url() == "http://x/s"
