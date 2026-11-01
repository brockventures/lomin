#!/usr/bin/env python3
"""MVG_TEXT_SENTENCES: text-only /ask/stream sentences and streamed
/v1/audio/speech, plus the early first-clause flush on /ask/stream.
Run: python3 -m pytest tests/test_text_sentences.py -v"""
import http.client
import json
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402
from test_ask_stream_pcm import FakeTTS, _events, _post  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "LOG_DIR", tmp_path)
    monkeypatch.setattr(gateway, "TEXT_SENTENCES", True)
    monkeypatch.setattr(gateway, "TTS_STREAM", True)
    monkeypatch.setattr(gateway, "FIRST_CLAUSE_EARLY", True)
    monkeypatch.setattr(gateway, "GAP_FILL", True)
    monkeypatch.setattr(gateway, "DESKTOP_TTS_URL", "http://127.0.0.1:1/tts")
    monkeypatch.setattr(gateway, "DESKTOP_TTS_STREAM_URL", "")
    monkeypatch.setattr(gateway, "TTS_STREAM_MIN_CHUNK_BYTES", 4800)
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
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeTTS)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(gateway, "DESKTOP_TTS_URL", f"http://127.0.0.1:{server.server_address[1]}/tts")
    yield server
    server.shutdown()


@pytest.fixture()
def gw():
    server = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def _brain(monkeypatch, *partials):
    def brain(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        for p in partials:
            on_partial(p)
        return partials[-1]
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)


def test_ask_stream_emits_text_only_sentences(gw, fake_tts, monkeypatch, tmp_path):
    _brain(monkeypatch, "Sentence 1 is here. Sentence 2 is here.")
    conn, resp = _post(gw)
    events = _events(resp)
    conn.close()
    sents = [d for e, d in events if e == "sentence"]
    assert [d["text"] for d in sents] == ["Sentence 1 is here.", "Sentence 2 is here."]
    assert all("audio_b64" not in d for d in sents)
    assert events[-1][0] == "done"
    assert FakeTTS.requests == []          # the gateway synthesized nothing
    rec = json.loads((tmp_path / "voice-latency.jsonl").read_text().splitlines()[-1])
    assert rec["text_sentences"] is True and rec["tts_engine"] == "text"
    assert rec["first_text_event_ms"] is not None


def test_text_mode_disables_gap_fill(gw, fake_tts, monkeypatch):
    """A gap-fill clip is audio; the hub rejects a turn that mixes audio and
    text-only sentences, so none may be sent in this mode."""
    monkeypatch.setattr(gateway, "GAP_FILL_AFTER_MS", 0)
    monkeypatch.setattr(gateway, "pick_gap_clip", lambda *a, **k: {
        "slug": "x", "text": "hmm", "pcm": b"\x01\x00" * 2400})

    def slow(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        time.sleep(0.6)
        on_partial("Done now.")
        return "Done now."
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", slow)
    conn, resp = _post(gw)
    events = _events(resp)
    conn.close()
    sents = [d for e, d in events if e == "sentence"]
    assert [d["text"] for d in sents] == ["Done now."]
    assert all("audio_b64" not in d for d in sents)


def test_flag_off_still_sends_audio(gw, fake_tts, monkeypatch):
    monkeypatch.setattr(gateway, "TEXT_SENTENCES", False)
    _brain(monkeypatch, "Sentence 1 is here.")
    conn, resp = _post(gw)
    events = _events(resp)
    conn.close()
    assert all("audio_b64" in d for e, d in events if e == "sentence")


def test_first_clause_emitted_without_waiting_for_more_text(gw, fake_tts, monkeypatch):
    """The first partial ends in a terminator with no trailing whitespace; the
    sentence must go out while the brain is still working, not at the end."""
    release = threading.Event()

    def brain(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        on_partial("Sure thing.")
        release.wait(5)
        on_partial("Sure thing. Here is the rest.")
        return "Sure thing. Here is the rest."
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)
    conn, resp = _post(gw)
    buf = b""
    deadline = time.time() + 5
    seen = False
    while time.time() < deadline and not seen:
        piece = resp.read1(65536)
        if not piece:
            break
        buf += piece
        seen = b"event: sentence" in buf
    release.set()
    assert seen, "first sentence was held back until the brain finished"
    conn.close()


def test_first_clause_early_can_be_disabled(monkeypatch):
    monkeypatch.setattr(gateway, "FIRST_CLAUSE_EARLY", False)
    seg = gateway.SentenceSegmenter(early_first=gateway.FIRST_CLAUSE_EARLY)
    assert seg.feed("Sure thing.") == []


def _speech(port, payload):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    conn.request("POST", "/v1/audio/speech", body=json.dumps(payload),
                 headers={"Content-Type": "application/json"})
    return conn, conn.getresponse()


def test_streaming_wav_header_is_what_the_hub_parses():
    h = gateway.streaming_wav_header(24000, 1)
    assert len(h) == 44 and h[:4] == b"RIFF" and h[8:12] == b"WAVE"
    assert h[12:16] == b"fmt " and struct.unpack("<H", h[20:22])[0] == 1
    assert h[36:40] == b"data"
    assert struct.unpack("<H", h[22:24])[0] == 1
    assert struct.unpack("<I", h[24:28])[0] == 24000


def test_speech_streams_wav_header_then_pcm(gw, fake_tts):
    conn, resp = _speech(gw, {"model": "m", "input": "Sentence 1 is here.", "voice": "v"})
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "audio/wav"
    assert resp.getheader("X-TTS-Engine") == "chatterbox-stream"
    body = resp.read()
    conn.close()
    assert body[:4] == b"RIFF" and body[36:40] == b"data"
    pcm = body[44:]
    assert len(pcm) == 15000 and pcm[0] == 1
    assert FakeTTS.requests and FakeTTS.requests[0][0] == "/tts/stream"


def test_speech_first_bytes_arrive_before_synthesis_finishes(gw, fake_tts):
    FakeTTS.gate = threading.Event()
    conn, resp = _speech(gw, {"input": "Sentence 1 is here."})
    first = resp.read(44 + 2)   # header plus some audio, while upstream is gated
    FakeTTS.gate.set()
    assert first[:4] == b"RIFF" and len(first) == 46
    resp.read()
    conn.close()


def test_speech_stream_opt_out_returns_complete_wav(gw, fake_tts, monkeypatch):
    wav = gateway.pcm_to_wav(b"\x03\x00" * 800, 24000)
    monkeypatch.setattr(gateway, "synthesize_chunked", lambda text: (wav, "test"))
    conn, resp = _speech(gw, {"input": "hello there", "stream": False})
    body = resp.read()
    conn.close()
    assert resp.status == 200 and resp.getheader("X-TTS-Engine") == "test"
    assert struct.unpack("<I", body[40:44])[0] == 1600  # real data size, not 0xFFFFFFFF


def test_speech_falls_back_when_desktop_unreachable(gw, monkeypatch):
    """Desktop stream down: the chain (desktop /tts -> Piper)
    still answers, converted to the same WAV-header-then-PCM stream."""
    wav = gateway.pcm_to_wav(b"\x05\x00" * 4000, 24000)

    def piper(text):
        return wav
    monkeypatch.setattr(gateway, "piper_speak", piper)
    conn, resp = _speech(gw, {"input": "hello there"})
    body = resp.read()
    conn.close()
    assert resp.status == 200
    assert body[:4] == b"RIFF" and body[36:40] == b"data"
    assert len(body) - 44 == 8000
    assert resp.getheader("X-TTS-Engine") == "piper"


def test_speech_total_failure_is_502(gw):
    conn, resp = _speech(gw, {"input": "hello there"})
    resp.read()
    conn.close()
    assert resp.status == 502
