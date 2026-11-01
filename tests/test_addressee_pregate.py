#!/usr/bin/env python3
"""Addressee pre-gate: human-to-human chatter must never reach a classifier
(Mike, 2026-10-03: Laya said 0.92 to "You don't think I should do that now,
do you?" with nobody naming Lomin and no conversation open).

Reuses test_ambient.py's fixtures and helpers. Classifier layers are mocked.

Run: python3 -m pytest ./tests -v
"""
import pytest

import gateway
from test_ambient import (  # noqa: F401  (fixtures + helpers)
    _FakeWhisperHandler, _laya_p, _no_real_classifier_io, _patch_haiku,
    _patch_jev, _patch_laya, _post_ambient, _speech_gate_unavailable_by_default,
    fake_whisper, gateway_server, needs_classify,
)


def _count_classify(monkeypatch, laya_p=0.95, jev_p=0.1):
    """Mock the classifier layers; returns a dict of call counters."""
    calls = {"laya": 0, "jev": 0, "haiku": 0}

    def fake_laya(state, question, **kw):
        calls["laya"] += 1
        probs = {"addressed_to_agent": laya_p, "conversation_between_people": 1 - laya_p}
        choice = max(probs, key=probs.get)
        if kw.get("meta") is not None:
            kw["meta"].update({"probabilities": probs, "confidence": probs[choice]})
        return probs[choice], choice, "", None

    def fake_jev(state, timeout=None):
        calls["jev"] += 1
        return jev_p, ""

    def fake_haiku(state, timeout=None):
        calls["haiku"] += 1
        return jev_p, ""

    _patch_laya(monkeypatch, fake_laya)
    _patch_jev(monkeypatch, fake_jev)
    _patch_haiku(monkeypatch, fake_haiku)
    return calls


@pytest.mark.parametrize("text", [
    "Did you still want to move your desk out here? Yes.",
    "You don't think I should do that now, do you?",
])
@needs_classify
def test_pregate_blocks_unnamed_chatter_without_open_conversation(
        gateway_server, fake_whisper, monkeypatch, text):
    calls = _count_classify(monkeypatch, laya_p=0.9221)
    _FakeWhisperHandler.next_text = text
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert status == 200
    assert data["engage"] is False
    assert data["classifier"] == "pregate"
    assert data["score"] == 0.0
    assert calls == {"laya": 0, "jev": 0, "haiku": 0}


@needs_classify
def test_pregate_named_utterance_reaches_classifiers(gateway_server, fake_whisper, monkeypatch):
    calls = _count_classify(monkeypatch, laya_p=0.95)
    _FakeWhisperHandler.next_text = "Lomin, what's the weather"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is True
    assert data["classifier"] == "laya"
    assert calls["laya"] == 1 and calls["jev"] == 0


@needs_classify
def test_pregate_whisper_variant_lemon_reaches_classifiers(gateway_server, fake_whisper, monkeypatch):
    calls = _count_classify(monkeypatch, laya_p=0.95)
    _FakeWhisperHandler.next_text = "Lemon, turn on the lights"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is True
    assert calls["laya"] == 1


def test_name_variants_are_word_boundary_and_case_insensitive():
    for ok in ("lomin", "LOMIN, hi", "hey Lommin", "Lowmin?", "lemon please", "loaming", "Lomin's list"):
        assert gateway._name_mentioned(ok), ok
    for bad in ("Lomincow", "the lemonade stand", "Dame Lomina", "move your desk", ""):
        assert not gateway._name_mentioned(bad), bad
    for v in gateway.AGENT_NAME_VARIANTS:
        assert v in gateway.VOICE_ADDRESSED_INSTRUCTIONS


@needs_classify
def test_unnamed_followup_needs_jev_even_if_laya_says_yes(monkeypatch):
    gateway.record_reply("n9", "Sunny and 70.")
    calls = _count_classify(monkeypatch, laya_p=0.95, jev_p=0.1)
    d = gateway.decide_addressed("n9", "and tomorrow?", "mike")
    assert calls["laya"] == 1 and calls["jev"] == 1
    assert d["classifier"] == "laya+jev"
    assert d["score"] == 0.1 and d["laya_score"] == 0.95
    assert d["score"] < gateway.AMBIENT_THRESHOLD


@needs_classify
def test_unnamed_followup_engages_when_jev_confirms(monkeypatch):
    gateway.record_reply("n9", "Sunny and 70.")
    _count_classify(monkeypatch, laya_p=0.95, jev_p=0.9)
    d = gateway.decide_addressed("n9", "and tomorrow?", "mike")
    assert (d["score"], d["classifier"]) == (0.9, "laya+jev")


@needs_classify
def test_unnamed_followup_laya_alone_never_engages_when_jev_down(monkeypatch):
    gateway.record_reply("n9", "Sunny and 70.")
    _laya_p(monkeypatch, 0.95)
    _patch_jev(monkeypatch, lambda state, timeout=None: (None, "boom"))
    _patch_haiku(monkeypatch, lambda state, timeout=None: (None, "boom"))
    d = gateway.decide_addressed("n9", "and tomorrow?", "mike")
    assert d["classifier"] == "none" and d["score"] == 0.0 and d["laya_score"] == 0.95


@needs_classify
def test_mention_to_another_person_goes_to_classifiers_and_jev_no_decides(
        gateway_server, fake_whisper, monkeypatch):
    # Name present: classifiers are consulted; Laya low, Jev says no.
    calls = _count_classify(monkeypatch, laya_p=0.3, jev_p=0.05)
    _FakeWhisperHandler.next_text = "ask Lomin later"
    status, data = _post_ambient(gateway_server, b"RIFFdata", "node-1")
    assert data["engage"] is False
    assert data["classifier"] == "laya+jev"
    assert calls["jev"] == 1


@needs_classify
def test_laya_gets_the_addressing_rule_and_lomin_replies_in_turns(monkeypatch):
    gateway.record_reply("n8", "Sunny and 70.")
    gateway._record_ambient_transcript("n8", "thanks", "mike")
    seen = {}
    _laya_p(monkeypatch, 0.2, seen)
    _patch_jev(monkeypatch, lambda state, timeout=None: (0.1, ""))
    gateway.decide_addressed("n8", "Lomin, and tomorrow?", "mike")
    assert "talking to each other" in seen["question"]
    contents = [t["content"] for t in seen["state"]]
    assert contents == ["agent: Sunny and 70.", "mike: thanks", "mike: Lomin, and tomorrow?"]


def test_lomin_reply_stays_in_jev_state_after_followup_window():
    gateway.record_reply("n7", "Sunny and 70.")
    old = gateway.AMBIENT_FOLLOWUP_WINDOW_SEC + 5
    with gateway._ambient_lock:
        ts, reply = gateway._ambient_last_reply["n7"]
        gateway._ambient_last_reply["n7"] = (ts - old, reply)
        gateway._ambient_context["n7"][0] = (ts - old, "agent", reply)
    assert not gateway._conversation_open("n7")
    state = gateway._build_ambient_state("n7", "Lomin, thanks", "mike")
    assert "- agent: Sunny and 70." in state


# The name regexes are built from env, and with the names the gateway has always
# used they must equal the literal patterns that shipped before the plugin split.
def test_name_regexes_equal_todays_patterns(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    gw_dir = Path(__file__).resolve().parents[1]
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(gw_dir),
        "MVG_LOG_DIR": str(tmp_path),
        "MVG_HOME": str(tmp_path),
        "MVG_AGENT_NAMES": "Lomin,Lommin,Lowmin,Lemon,Loaming",
        "MVG_WAKE_NAMES": "Lomin,Lommin,Lowmin",
    }
    code = (
        "import json, gateway as g\n"
        "print(json.dumps([g._AGENT_NAME_RE.pattern, g._AGENT_NAME_RE.flags & 2 != 0,\n"
        "                  g.WAKE_PHRASE_RE.pattern, g.WAKE_PHRASE_RE.flags & 2 != 0]))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=gw_dir, env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    name_pat, name_ci, wake_pat, wake_ci = json.loads(r.stdout.strip().splitlines()[-1])
    assert name_pat == r"\b(?:Lomin|Lommin|Lowmin|Lemon|Loaming)\b"
    assert wake_pat == r"^[\W_]*(?:hey|hi|hello|okay|ok|yo)[,.!]?\s+(?:lomin|lommin|lowmin)\b"
    assert name_ci and wake_ci
