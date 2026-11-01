#!/usr/bin/env python3
"""The PASS sentinel: a brain reply that is exactly "PASS" after strip() means
speak nothing. No TTS, no follow-up window, outcome "silent".

Run: python3 -m pytest tests/test_silent_sentinel.py -q
"""
import http.client
import json
import sys
import threading
from io import BytesIO
from pathlib import Path
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402


def _wav(frames: bytes, rate=24000) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture()
def tts_spy(monkeypatch):
    """Records every text handed to the TTS engine. Nothing is spoken for real."""
    calls = []

    def fake_synthesize(text):
        calls.append(text)
        return _wav(b"\x01\x00" * 2000), "spy"

    monkeypatch.setattr(gateway, "synthesize", fake_synthesize)
    return calls


@pytest.fixture()
def gw(monkeypatch):
    server = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def _post(port, path, node_id):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    payload = json.dumps({"prompt": "hi", "session_id": "s", "node_id": node_id,
                          "speaker": "mike", "speaker_score": 0.9})
    conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
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


def _last_outcome(tmp_path, endpoint):
    lines = (tmp_path / "lomin.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if f'"{endpoint}"' in line]
    return rows[-1]


def _stream_brain(monkeypatch, text):
    def brain(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        on_partial(text)
        return text
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)


def _ask_brain(monkeypatch, text):
    monkeypatch.setattr(gateway, "BRAIN_CALL",
                        lambda prompt, session_id, node_id, speaker, score, timeout: text)


# --- /ask/stream -----------------------------------------------------------

@pytest.mark.parametrize("brain_reply", ["PASS", "  PASS\n", "Pass.", "pass!"])
def test_stream_pass_is_silent(gw, monkeypatch, tts_spy, tmp_path, brain_reply):
    node = f"silent-stream-{brain_reply.strip() or 'x'}-{len(brain_reply)}"
    _stream_brain(monkeypatch, brain_reply)
    conn, resp = _post(gw, "/ask/stream", node)
    events = _events(resp)
    conn.close()

    kinds = [e for e, _ in events]
    assert "sentence" not in kinds
    assert "audio_chunk" not in kinds
    reply = [d for e, d in events if e == "reply"][0]
    assert reply["reply"] == ""
    assert tts_spy == []
    assert _last_outcome(tmp_path, "/ask/stream")["outcome"] == "silent"
    assert not gateway._conversation_open(node)


def test_stream_pass_with_trailing_words_is_spoken(gw, monkeypatch, tts_spy, tmp_path):
    node = "silent-stream-words"
    _stream_brain(monkeypatch, "PASS the lights")
    conn, resp = _post(gw, "/ask/stream", node)
    events = _events(resp)
    conn.close()

    reply = [d for e, d in events if e == "reply"][0]
    assert reply["reply"] == "PASS the lights"
    assert tts_spy, "non-sentinel reply must reach TTS"
    assert _last_outcome(tmp_path, "/ask/stream")["outcome"] == "ok"
    assert gateway._conversation_open(node)


def test_stream_empty_reply_still_apologises(gw, monkeypatch, tts_spy, tmp_path):
    node = "silent-stream-empty"
    _stream_brain(monkeypatch, "")
    conn, resp = _post(gw, "/ask/stream", node)
    events = _events(resp)
    conn.close()

    reply = [d for e, d in events if e == "reply"][0]
    assert reply["reply"] == gateway.APOLOGY
    assert tts_spy == [gateway.APOLOGY]
    assert _last_outcome(tmp_path, "/ask/stream")["outcome"] == "empty_reply"
    assert not gateway._conversation_open(node)


# --- /ask -------------------------------------------------------------------

@pytest.mark.parametrize("brain_reply", ["PASS", "  PASS\n", "Pass.", "pass!"])
def test_ask_pass_is_silent(gw, monkeypatch, tmp_path, brain_reply):
    node = f"silent-ask-{len(brain_reply)}"
    _ask_brain(monkeypatch, brain_reply)
    conn, resp = _post(gw, "/ask", node)
    events = _events(resp)
    conn.close()

    reply = [d for e, d in events if e == "reply"][0]
    assert reply["reply"] == ""
    assert _last_outcome(tmp_path, "/ask")["outcome"] == "silent"
    assert not gateway._conversation_open(node)


def test_ask_pass_with_trailing_words_is_spoken(gw, monkeypatch, tmp_path):
    node = "silent-ask-words"
    _ask_brain(monkeypatch, "PASS the lights")
    conn, resp = _post(gw, "/ask", node)
    events = _events(resp)
    conn.close()

    reply = [d for e, d in events if e == "reply"][0]
    assert reply["reply"] == "PASS the lights"
    assert _last_outcome(tmp_path, "/ask")["outcome"] == "ok"
    assert gateway._conversation_open(node)


def test_ask_empty_reply_still_apologises(gw, monkeypatch, tmp_path):
    node = "silent-ask-empty"
    _ask_brain(monkeypatch, "")
    conn, resp = _post(gw, "/ask", node)
    events = _events(resp)
    conn.close()

    reply = [d for e, d in events if e == "reply"][0]
    assert reply["reply"] == gateway.APOLOGY
    assert _last_outcome(tmp_path, "/ask")["outcome"] == "empty_reply"
    assert not gateway._conversation_open(node)
