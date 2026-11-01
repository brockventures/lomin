#!/usr/bin/env python3
"""Echo filter: a transcript that repeats Amos's last reply is dropped.

Run: python3 -m pytest tests/test_reply_echo.py -v
"""
import gateway

REPLY = "Yes, I can hear you clearly. Reviewing the relevant notes now, one moment."


def _setup(monkeypatch, node="echo1", t=1000.0):
    gateway._ambient_last_reply.pop(node, None)
    box = {"t": t}
    monkeypatch.setattr(gateway, "_now", lambda: box["t"])
    gateway.record_reply(node, REPLY)
    return box


def test_fragment_of_reply_is_echo(monkeypatch):
    _setup(monkeypatch)
    assert gateway.is_reply_echo("echo1", "reviewing the relevant notes now")


def test_new_request_is_not_echo(monkeypatch):
    _setup(monkeypatch)
    assert not gateway.is_reply_echo("echo1", "what is the weather tomorrow")


def test_short_fragment_not_echo(monkeypatch):
    _setup(monkeypatch)
    assert not gateway.is_reply_echo("echo1", "yes")


def test_old_reply_not_echo(monkeypatch):
    box = _setup(monkeypatch)
    box["t"] += gateway.ECHO_WINDOW_SEC + 1
    assert not gateway.is_reply_echo("echo1", "reviewing the relevant notes now")


def test_other_node_not_echo(monkeypatch):
    _setup(monkeypatch)
    assert not gateway.is_reply_echo("other", "reviewing the relevant notes now")
