#!/usr/bin/env python3
"""The follow-up window counts from the end of PLAYBACK, not text-ready.

Run: python3 -m pytest ./tests -v
"""
import gateway


def _clock(monkeypatch, t):
    box = {"t": t}
    monkeypatch.setattr(gateway, "_now", lambda: box["t"])
    return box


def _reset(node):
    gateway._ambient_last_reply.pop(node, None)
    gateway._ambient_context.pop(node, None)


def test_default_window_is_30s():
    assert gateway.AMBIENT_FOLLOWUP_WINDOW_SEC == 30.0


def test_spoken_reply_keeps_window_open_from_playback_end(monkeypatch):
    _reset("pb1")
    clk = _clock(monkeypatch, 1000.0)
    gateway.record_reply("pb1", "It will be sunny tomorrow.")
    # 15 s of audio sent starting at text-ready -> playback ends at 1015.
    gateway.extend_reply_playback("pb1", 1015.0)
    clk["t"] = 1000.0 + 20.0 + 15.0  # 20 s after playback end
    assert gateway._conversation_open("pb1")
    clk["t"] = 1015.0 + gateway.AMBIENT_FOLLOWUP_WINDOW_SEC + 1
    assert not gateway._conversation_open("pb1")


def test_text_only_reply_unchanged(monkeypatch):
    _reset("pb2")
    clk = _clock(monkeypatch, 1000.0)
    gateway.record_reply("pb2", "Hello.")
    gateway.extend_reply_playback("pb2", None)  # audio length unknown
    clk["t"] = 1000.0 + gateway.AMBIENT_FOLLOWUP_WINDOW_SEC - 1
    assert gateway._conversation_open("pb2")
    clk["t"] = 1000.0 + gateway.AMBIENT_FOLLOWUP_WINDOW_SEC + 1
    assert not gateway._conversation_open("pb2")


def test_extend_never_moves_earlier_and_keeps_context(monkeypatch):
    _reset("pb3")
    _clock(monkeypatch, 1000.0)
    gateway.record_reply("pb3", "Reply text.")
    gateway.extend_reply_playback("pb3", 990.0)
    assert gateway._ambient_last_reply["pb3"][0] == 1000.0
    gateway.extend_reply_playback("pb3", 1012.0)
    assert gateway._ambient_last_reply["pb3"] == (1012.0, "Reply text.")
    ctx = list(gateway._ambient_context["pb3"])
    assert ctx[-1][0] == 1000.0 and ctx[-1][1] == "agent"
