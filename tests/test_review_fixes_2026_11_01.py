#!/usr/bin/env python3
"""Fixes from Zero's review of 2026-11-01: streamed markdown offsets, the
Laya fast path with no agent names configured, and the request-body cap.

Run: python3 -m pytest ./tests -v
"""
import io
import random

import gateway
import multipart
from text_utils import stable_markdown_prefix, strip_markdown, unspoken_delta


def _speak_stream(partials, final):
    """Mirror of _handle_ask_stream's on_partial + tail flush: returns the
    concatenation of every delta fed to the segmenter."""
    spoken, out = "", []
    for raw in partials:
        full = strip_markdown(stable_markdown_prefix(raw)) or ""
        d = unspoken_delta(full, spoken)
        if d:
            out.append(d)
            spoken = full
    out.append(unspoken_delta(strip_markdown(final) or "", spoken))
    return "".join(out)


def _prefixes(text, step=3):
    return [text[:i] for i in range(1, len(text), step)] + [text]


def test_link_collapsing_mid_stream_drops_nothing():
    final = "See [the docs](http://example.com/a/b) for more detail today."
    assert _speak_stream(_prefixes(final), final) == strip_markdown(final)


def test_code_span_and_fence_mid_stream_drop_nothing():
    final = "Run `ls -l` first. Then ```py\nx = 1\n``` and you are done."
    assert _speak_stream(_prefixes(final, 2), final) == strip_markdown(final)


def test_old_offset_slicing_would_have_dropped_text():
    # The bug Zero reported: slicing by the earlier stripped length.
    a, b = "See [the docs", "See [the docs](http://x.y/z) for more."
    seen = len(strip_markdown(a))
    assert strip_markdown(b)[seen:] != " for more."  # old path skips "fo"
    assert _speak_stream([a, b], b) == strip_markdown(b)


def test_stable_prefix_keeps_closed_markdown_and_plain_brackets():
    assert stable_markdown_prefix("A [b] c.") == "A [b] c."
    assert stable_markdown_prefix("Tail [x]") == "Tail "
    assert stable_markdown_prefix("x [y](http://z") == "x "
    assert stable_markdown_prefix("Use `ls") == "Use "


def test_no_agent_names_lets_laya_engage_on_open_conversation(monkeypatch):
    seen = {}
    monkeypatch.setattr(gateway, "_AGENT_NAME_RE", None)
    monkeypatch.setattr(gateway, "_conversation_open", lambda node_id: True)
    monkeypatch.setattr(gateway, "_build_ambient_state", lambda *a: "state")
    monkeypatch.setattr(gateway, "_build_ambient_turns", lambda *a: [])

    def fake_detail(state, turns, name_present=True):
        seen["name_present"] = name_present
        return {"score": 1.0, "classifier": "laya", "reason": "", "laya_score": 1.0}

    monkeypatch.setattr(gateway, "classify_addressed_detail", fake_detail)
    gateway.decide_addressed("node-1", "what time is it", "")
    assert seen["name_present"] is True


def test_names_configured_but_absent_still_restricts_laya(monkeypatch):
    import re
    seen = {}
    monkeypatch.setattr(gateway, "_AGENT_NAME_RE", re.compile(r"\blomin\b", re.I))
    monkeypatch.setattr(gateway, "_conversation_open", lambda node_id: True)
    monkeypatch.setattr(gateway, "_build_ambient_state", lambda *a: "state")
    monkeypatch.setattr(gateway, "_build_ambient_turns", lambda *a: [])
    monkeypatch.setattr(gateway, "classify_addressed_detail",
                        lambda s, t, name_present=True: seen.setdefault("np", name_present) or {})
    gateway.decide_addressed("node-1", "what time is it", "")
    assert seen["np"] is False


class _FakeHandler:
    _read_body = gateway.GatewayHandler._read_body
    _reject_body = gateway.GatewayHandler._reject_body

    def __init__(self, length, body=b""):
        self.headers = {"Content-Length": str(length)}
        self.rfile = io.BytesIO(body)
        self.sent = None
        self.close_connection = False

    def _send_json(self, code, obj):
        self.sent = (code, obj)


def test_oversized_body_is_refused_before_reading(monkeypatch):
    monkeypatch.setattr(gateway, "MAX_BODY_BYTES", 100)
    h = _FakeHandler(101, b"x" * 101)
    try:
        h._read_body()
    except gateway._BodyRejected:
        pass
    else:
        raise AssertionError("expected _BodyRejected")
    assert h.sent[0] == 413
    assert h.close_connection is True
    assert h.rfile.tell() == 0


def test_bad_content_length_is_400():
    h = _FakeHandler("abc")
    try:
        h._read_body()
    except gateway._BodyRejected:
        pass
    assert h.sent[0] == 400


def test_body_at_cap_is_read(monkeypatch):
    monkeypatch.setattr(gateway, "MAX_BODY_BYTES", 100)
    assert _FakeHandler(100, b"y" * 100)._read_body() == b"y" * 100


def test_multipart_iterating_parser_matches_split():
    rng = random.Random(7)
    wav = bytes(rng.getrandbits(8) for _ in range(50_000)) + b"\r\n--notthebound"
    b = "XyZb0und"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"node_id\"\r\n\r\nn1\r\n"
            f"--{b}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + wav + f"\r\n--{b}--\r\n".encode()
    parts = multipart.parse_multipart_parts(body, f"multipart/form-data; boundary={b}")
    assert parts["node_id"] == b"n1"
    assert parts["audio"] == wav


def test_stt_correlation_is_per_node(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "_last_stt", {})
    monkeypatch.setattr(gateway, "_last_stt_any", {})
    monkeypatch.setattr(gateway, "log_latency", lambda rec: None)
    gateway._note_stt("kitchen", 1.0, 120.0, 5, False)
    gateway._note_stt("office", 1.0, 340.0, 5, False)
    import time
    now = time.time()
    assert gateway._correlated_stt("kitchen", now)["stt_ms"] == 120.0
    assert gateway._correlated_stt("office", now)["stt_ms"] == 340.0
    assert gateway._correlated_stt("", now)["stt_ms"] == 340.0  # newest overall
    assert gateway._correlated_stt("garage", now) == {}
