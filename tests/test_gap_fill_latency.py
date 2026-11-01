#!/usr/bin/env python3
"""Gap-fill slots and the per-turn latency log on /ask/stream, with a fake
slow brain and a fake desktop TTS. Run: python3 -m pytest ./tests -v

Timeline with the fixture (AFTER 100, MID 300, SLOW 700, TICK 200, LONG 500 ms,
each fixture clip ~0.42 s of playback, GAP_FILL_QUIET_MS left at 800): the opener
lands near 0.15 s and plays to ~0.57 s. The mid slot needs 800 ms of quiet after
that, so it cannot fire before ~1.4 s; the slow slot needs 800 ms after the mid
clip. Tests that need the mid or slow slot therefore use a later brain delay than
the spec's 0.6 s / 1.4 s (see the PR body)."""
import base64
import http.client
import json
import sys
import time
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402
# Reuse the fake desktop TTS, gateway server and request helpers.
from test_ask_stream_pcm import (  # noqa: E402,F401
    FakeTTS, _env, fake_tts, gw, _post, _events, _last_log,
)

FILL_BYTES = 9600 * 2 + 1000  # 3 chunks per clip, last one short (even length)


def _make_clips(root: Path) -> None:
    clips = {}
    marks = {"vocal_open": ["v1", "v2"], "vocal_long": ["w1"], "handoff": ["f1"],
             "sfx_generic": ["g1", "g2"], "sfx_home": ["h1"], "sfx_lookup": ["k1"],
             "sfx_calendar_lists": ["c1"], "sfx_tick": ["t1", "t2"]}
    for n, (cat, slugs) in enumerate(marks.items(), start=1):
        (root / cat).mkdir(parents=True)
        for slug in slugs:
            with wave.open(str(root / cat / f"{slug}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(24000)
                w.writeframes(bytes([0x50 + n, 0]) * (FILL_BYTES // 2))
            clips[f"{cat}/{slug}.wav"] = {"category": cat, "text": f"Clip {slug}."}
    (root / "manifest.json").write_text(json.dumps({"clips": clips}))


@pytest.fixture(autouse=True)
def _gap(monkeypatch, tmp_path):
    root = tmp_path / "gap-fill"
    root.mkdir()
    _make_clips(root)
    monkeypatch.setattr(gateway, "GAP_FILL_DIR", root)
    monkeypatch.setattr(gateway, "GAP_FILL", True)
    monkeypatch.setattr(gateway, "GAP_FILL_AFTER_MS", 100)
    monkeypatch.setattr(gateway, "GAP_FILL_MID_MS", 300)
    monkeypatch.setattr(gateway, "GAP_FILL_SLOW_MS", 700)
    monkeypatch.setattr(gateway, "GAP_FILL_TICK_MS", 200)
    monkeypatch.setattr(gateway, "GAP_FILL_LONG_MS", 500)
    monkeypatch.setattr(gateway, "_gap_prev_turn", set())
    gateway._gap_cache.clear()
    yield
    gateway._gap_cache.clear()


def _slow_brain(monkeypatch, delay, text="Sentence 1 is here. Sentence 2 is here.", gap_between=0.0):
    def brain(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        time.sleep(delay)
        # first partial stops mid-second-sentence: the segmenter can only
        # confirm sentence 1 until more text arrives
        cut = text.index(". ") + 2 + 8
        on_partial(text[:cut])
        time.sleep(gap_between)
        on_partial(text)
        return text
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)


def _sentences(events):
    return [d for e, d in events if e == "sentence"]


def _run(gw):
    conn, resp = _post(gw)
    events = _events(resp)
    conn.close()
    return events


def _run_prompt(gw, prompt):
    conn = http.client.HTTPConnection("127.0.0.1", gw, timeout=15)
    payload = json.dumps({"prompt": prompt, "session_id": "s", "node_id": "n"})
    conn.request("POST", "/ask/stream", body=payload, headers={"Content-Type": "application/json"})
    events = _events(conn.getresponse())
    conn.close()
    return events


def _latency(tmp_path, kind="turn"):
    lines = (tmp_path / "voice-latency.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if r["kind"] == kind]


def _is_real(d):
    return d["engine"] != "gap-fill"


def test_filler_emitted_once_before_first_real_audio(gw, fake_tts, monkeypatch, tmp_path):
    _slow_brain(monkeypatch, 0.6)
    events = _run(gw)
    sents = _sentences(events)
    first_real = next(i for i, d in enumerate(sents) if _is_real(d))
    fillers = [d for d in sents[:first_real]]
    assert fillers and all(d["engine"] == "gap-fill" for d in fillers)
    # the opener is vocal and carries no caption
    assert all(d["text"] == "" for d in fillers)
    assert len(fillers) == 3  # one vocal clip, three chunks
    # whole clip before any real audio, in order, real audio never inside it
    assert sum(len(base64.b64decode(d["audio_b64"])) for d in fillers) == FILL_BYTES
    assert all(_is_real(d) for d in sents[first_real:])
    assert events[-1][0] == "done"
    rec = _last_log(tmp_path)
    assert len(rec["gap_fill"]) == 1 and rec["gap_fill"][0].startswith("vocal_open/")


def test_opener_is_vocal_with_no_caption(gw, fake_tts, monkeypatch, tmp_path):
    # Brain delay 0.4 s (spec: 0.25 s): the opener fires at ~0.15 s, and 0.25 s
    # leaves too little margin over the 0.15 s poll interval.
    _slow_brain(monkeypatch, 0.4)
    events = _run(gw)
    sents = _sentences(events)
    gap_events = [d for d in sents if not _is_real(d)]
    assert gap_events
    assert all(d["filler"] is True for d in gap_events)
    assert gap_events[0]["text"] == ""
    assert _last_log(tmp_path)["gap_fill"][0].startswith("vocal_open/")
    assert events[-1][0] == "done"


def test_mid_slot_plays_topical_effect(gw, fake_tts, monkeypatch, tmp_path):
    # Brain delay 1.8 s (spec: 0.6 s): the mid slot needs 800 ms of quiet after
    # the opener, so it cannot fire before ~1.4 s.
    _slow_brain(monkeypatch, 1.8)
    events = _run_prompt(gw, "turn on the kitchen lights")
    used = _last_log(tmp_path)["gap_fill"]
    assert len(used) == 2
    assert used[0].startswith("vocal_open/") and used[1] == "sfx_home/h1"
    sents = _sentences(events)
    fillers = [d for d in sents if not _is_real(d)]
    assert all(d["text"] == "" for d in fillers)  # no caption on effects
    assert events[-1][0] == "done"


def test_untopical_mid_slot_plays_generic_effect(gw, fake_tts, monkeypatch, tmp_path):
    _slow_brain(monkeypatch, 1.8)
    _run_prompt(gw, "tell me a joke about cats")
    used = _last_log(tmp_path)["gap_fill"]
    assert len(used) == 2
    assert used[0].startswith("vocal_open/")
    assert used[1].startswith("sfx_generic/")


def test_slow_turn_plays_long_vocal_then_ticks(gw, fake_tts, monkeypatch, tmp_path):
    # Brain delay 4.2 s (spec: 1.4 s): the long vocal needs 800 ms of quiet
    # after the mid effect, so it fires at ~2.8 s.
    _slow_brain(monkeypatch, 4.2)
    _run_prompt(gw, "tell me a joke about cats")
    used = _last_log(tmp_path)["gap_fill"]
    assert used[0].startswith("vocal_open/")
    assert used[1].startswith("sfx_generic/")
    assert used[2] == "vocal_long/w1"
    ticks = [u for u in used[3:] if u.startswith("sfx_tick/")]
    assert 1 <= len(ticks) <= gateway.GAP_FILL_MAX_TICKS
    assert len(used) == 3 + len(ticks)


def test_no_vocal_after_real_audio(gw, fake_tts, monkeypatch, tmp_path):
    # Real audio plays, then the brain stalls for well over GAP_FILL_LONG_MS:
    # only machine ticks may fill the gap.
    monkeypatch.setattr(FakeTTS, "sizes", (6000,))  # 0.125 s per sentence: playback ends fast
    _slow_brain(monkeypatch, 0.0, gap_between=1.6,
                text="Sentence 1 is here. Sentence 2 is here. Sentence 3 is here.")
    events = _run(gw)
    sents = _sentences(events)
    kinds = ["real" if _is_real(d) else "fill" for d in sents]
    assert kinds[0] == "real"
    assert "fill" in kinds
    assert [d["text"] for d in sents if d["text"] and _is_real(d)] == [
        "Sentence 1 is here.", "Sentence 2 is here.", "Sentence 3 is here."]
    used = _last_log(tmp_path)["gap_fill"]
    assert used and all(u.startswith("sfx_tick/") for u in used)
    fill = [d for d in sents if not _is_real(d)]
    assert sum(len(base64.b64decode(d["audio_b64"])) for d in fill) == FILL_BYTES * len(used)


def test_ticks_capped_at_max_and_not_interleaved(gw, fake_tts, monkeypatch, tmp_path):
    # 3 s of brain silence after real audio: ticks fire every LONG_MS, but the
    # per-turn cap holds them to GAP_FILL_MAX_TICKS (2 here).
    monkeypatch.setattr(FakeTTS, "sizes", (6000,))
    monkeypatch.setattr(gateway, "GAP_FILL_MAX_TICKS", 2)
    _slow_brain(monkeypatch, 0.0, gap_between=3.0,
                text="Sentence 1 is here. Sentence 2 is here.")
    events = _run(gw)
    sents = _sentences(events)
    used = _last_log(tmp_path)["gap_fill"]
    assert len(used) == 2 and all(u.startswith("sfx_tick/") for u in used)
    kinds = ["real" if _is_real(d) else "fill" for d in sents]
    runs = [k for i, k in enumerate(kinds) if i == 0 or k != kinds[i - 1]]
    assert runs[0] == "real" and "fill" in runs
    assert [d["text"] for d in sents if d["text"] and _is_real(d)] == [
        "Sentence 1 is here.", "Sentence 2 is here."]
    assert events[-1][0] == "done"


def test_handoff_topic_keeps_caption(gw, fake_tts, monkeypatch, tmp_path):
    # Brain delay 1.8 s (spec: 0.6 s): the handoff clip is a mid-slot effect.
    _slow_brain(monkeypatch, 1.8)
    events = _run_prompt(gw, "put that in the build queue")
    used = _last_log(tmp_path)["gap_fill"]
    assert used[0].startswith("vocal_open/") and used[1] == "handoff/f1"
    captioned = [d["text"] for d in _sentences(events) if d["text"] and not _is_real(d)]
    assert captioned == ["Clip f1."]
    assert events[-1][0] == "done"


def test_missing_vocal_category_is_silent(gw, fake_tts, monkeypatch, tmp_path):
    clips = {k: v for k, v in gateway.load_gap_clips().items() if k != "vocal_open"}
    monkeypatch.setattr(gateway, "load_gap_clips", lambda directory=None: clips)
    _slow_brain(monkeypatch, 0.6)
    events = _run(gw)
    assert events[-1][0] == "done"
    assert _last_log(tmp_path)["gap_fill"] == []
    assert all(_is_real(d) for d in _sentences(events))


def test_no_filler_when_brain_is_fast(gw, fake_tts, monkeypatch, tmp_path):
    _slow_brain(monkeypatch, 0.0)
    sents = _sentences(_run(gw))
    assert sents and all(_is_real(d) for d in sents)
    assert _last_log(tmp_path)["gap_fill"] == []
    assert _latency(tmp_path)[-1]["gap_fill_first_ms"] is None


def test_pcm_format_throughout_and_turn_event_first(gw, fake_tts, monkeypatch):
    _slow_brain(monkeypatch, 0.6)
    events = _run(gw)
    assert events[0][0] == "turn"
    assert len(events[0][1]["turn_id"]) == 36
    sents = _sentences(events)
    assert sents and all(
        d["format"] == "pcm" and d["sample_rate"] == 24000 and d["channels"] == 1
        and len(base64.b64decode(d["audio_b64"])) % 2 == 0 for d in sents)


def test_consecutive_turns_do_not_repeat_the_clip(gw, fake_tts, monkeypatch, tmp_path):
    _slow_brain(monkeypatch, 0.5)
    seen = []
    for _ in range(6):
        _run(gw)
        seen.append(_last_log(tmp_path)["gap_fill"][0])
    assert all(a != b for a, b in zip(seen, seen[1:]))


def test_no_repeat_across_turns_with_opener_and_effect(gw, fake_tts, monkeypatch, tmp_path):
    # Each turn plays a vocal opener AND a generic effect; the next turn must
    # not reuse either (2 vocal_open and 2 sfx_generic clips exist).
    _slow_brain(monkeypatch, 1.8)
    prev = None
    for _ in range(4):
        _run_prompt(gw, "tell me a joke about cats")
        used = _last_log(tmp_path)["gap_fill"]
        assert len(used) == 2 and used[0].startswith("vocal_open/") and used[1].startswith("sfx_generic/")
        if prev is not None:
            assert not (set(used) & set(prev)), (used, prev)
        prev = used


def test_gap_fill_flag_off(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "GAP_FILL", False)
    _slow_brain(monkeypatch, 0.6)
    sents = _sentences(_run(gw))
    assert sents and all(_is_real(d) for d in sents)
    assert _last_log(tmp_path)["gap_fill"] == []


def test_missing_clip_dir_is_harmless(gw, fake_tts, monkeypatch, tmp_path):
    monkeypatch.setattr(gateway, "GAP_FILL_DIR", tmp_path / "nope")
    _slow_brain(monkeypatch, 0.6)
    events = _run(gw)
    assert events[-1][0] == "done"
    assert all(_is_real(d) for d in _sentences(events))


def test_latency_record_fields(gw, fake_tts, monkeypatch, tmp_path):
    _slow_brain(monkeypatch, 0.5)
    events = _run(gw)
    turn_id = events[0][1]["turn_id"]
    rec = _latency(tmp_path)[-1]
    for key in ("ts", "node_id", "session_id", "turn_id", "request_start",
                "brain_first_text_ms", "first_sentence_ms", "gap_fill_first_ms",
                "tts_first_ms", "first_audio_ms", "last_audio_ms", "total_ms",
                "sentences", "reply_chars", "audio_s", "tts_engine", "target", "outcome"):
        assert key in rec, key
    assert rec["turn_id"] == turn_id
    assert rec["node_id"] == "n" and rec["session_id"] == "s"
    assert rec["outcome"] == "ok" and rec["sentences"] == 2
    assert rec["reply_chars"] == len("Sentence 1 is here. Sentence 2 is here.")
    assert rec["tts_engine"] == "chatterbox-stream" and rec["target"] == "agent"
    assert rec["audio_s"] == pytest.approx(2 * 15000 / 48000, abs=0.01)
    assert rec["brain_first_text_ms"] >= 450
    assert rec["gap_fill_first_ms"] is not None and rec["gap_fill_first_ms"] < rec["first_audio_ms"]
    assert rec["first_sentence_ms"] >= rec["brain_first_text_ms"]
    assert rec["first_audio_ms"] >= rec["first_sentence_ms"]
    assert rec["last_audio_ms"] >= rec["first_audio_ms"]
    assert rec["total_ms"] >= rec["last_audio_ms"]
    assert rec["request_start"] > 1_700_000_000_000
    assert _last_log(tmp_path)["turn_id"] == turn_id


def test_stt_latency_record_and_correlation(gw, fake_tts, monkeypatch, tmp_path):
    import http.client
    monkeypatch.setattr(gateway, "transcribe", lambda wav: "hello there")
    monkeypatch.setattr(gateway, "_last_stt", {})
    monkeypatch.setattr(gateway, "_last_stt_any", {})
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x01\x00" * 16000)
    conn = http.client.HTTPConnection("127.0.0.1", gw, timeout=10)
    conn.request("POST", "/stt?node_id=n", body=buf.getvalue(), headers={"Content-Type": "audio/wav"})
    assert json.loads(conn.getresponse().read())["text"] == "hello there"
    conn.close()
    stt = _latency(tmp_path, "stt")[-1]
    assert stt["node_id"] == "n" and stt["chars"] == 11
    assert stt["audio_s"] == 1.0 and stt["stt_ms"] >= 0
    _slow_brain(monkeypatch, 0.0)
    _run(gw)
    turn = _latency(tmp_path)[-1]
    assert turn["stt_id"] == stt["stt_id"] and turn["stt_ms"] == stt["stt_ms"]


def test_turn_record_carries_brain_stages(gw, fake_tts, monkeypatch, tmp_path):
    """The worker thread's stage sink reaches the turn record as `stages`."""
    def brain(prompt, session_id, node_id, speaker, score, timeout, on_partial):
        gateway._STAGE_SINK.stages["inserted_ms"] = 1000
        gateway._STAGE_SINK.stages["claimed_pi_ms"] = 1040
        on_partial("Sentence 1 is here. Sentence 2 is here.")
        return "Sentence 1 is here. Sentence 2 is here."
    monkeypatch.setattr(gateway, "STREAM_BRAIN_CALL", brain)
    _run(gw)
    rec = _latency(tmp_path)[-1]
    assert rec["stages"] == {"inserted_ms": 1000, "claimed_pi_ms": 1040}


def test_plugin_brain_call_stream_passes_stage_sink(monkeypatch):
    seen = {}

    def fake_stream(target, prompt, timeout, on_partial, stages_out):
        seen["stages_out"] = stages_out
        return "ok"

    monkeypatch.setattr(gateway, "_plugin_brain_call_stream", fake_stream)
    sink = {}
    gateway._STAGE_SINK.stages = sink
    try:
        gateway.plugin_brain_call_stream("hi", "s", "n", "mike", 0.9, 5, lambda t: None)
    finally:
        del gateway._STAGE_SINK.stages
    assert seen["stages_out"] is sink
    # no sink on this thread -> None, old behaviour
    seen.clear()
    gateway.plugin_brain_call_stream("hi", "s", "n", "mike", 0.9, 5, lambda t: None)
    assert seen["stages_out"] is None


@pytest.mark.parametrize("prompt, expected", [
    ("tell the primary to turn on the lights", "handoff"),   # handoff beats home
    ("hand it off to the primary", "handoff"),
    ("turn on the kitchen lights", "sfx_home"),
    ("lock the front door", "sfx_home"),
    ("what's on my calendar", "sfx_calendar_lists"),    # calendar beats lookup
    ("add milk to the shopping list", "sfx_calendar_lists"),
    ("remind me to call mom", "sfx_calendar_lists"),
    ("look up the weather", "sfx_lookup"),
    ("how many ounces in a cup", "sfx_lookup"),
])
def test_effect_categories_by_topic(monkeypatch, prompt, expected):
    monkeypatch.setattr(gateway, "GAP_FILL_TOPICAL", True)
    assert gateway.effect_categories(prompt) == (expected,)


def test_effect_categories_generic_when_no_topic(monkeypatch):
    monkeypatch.setattr(gateway, "GAP_FILL_TOPICAL", True)
    assert gateway.effect_categories("tell me a joke about cats") == gateway.GAP_FILL_SFX_GENERIC
    assert gateway.effect_categories("hi") == gateway.GAP_FILL_SFX_GENERIC


def test_effect_categories_empty_or_non_string_is_generic(monkeypatch):
    monkeypatch.setattr(gateway, "GAP_FILL_TOPICAL", True)
    for prompt in ("", "   ", None, 42):
        assert gateway.effect_categories(prompt) == gateway.GAP_FILL_SFX_GENERIC


def test_effect_categories_topical_off_is_generic(monkeypatch):
    monkeypatch.setattr(gateway, "GAP_FILL_TOPICAL", False)
    assert gateway.effect_categories("turn on the kitchen lights") == gateway.GAP_FILL_SFX_GENERIC


def test_topical_category_without_clips_falls_back_to_generic(gw, fake_tts, monkeypatch, tmp_path):
    clips = {k: v for k, v in gateway.load_gap_clips().items() if k != "sfx_home"}
    monkeypatch.setattr(gateway, "load_gap_clips", lambda directory=None: clips)
    _slow_brain(monkeypatch, 1.8)
    events = _run_prompt(gw, "turn on the kitchen lights")
    used = _last_log(tmp_path)["gap_fill"]
    assert len(used) == 2 and used[0].startswith("vocal_open/")
    assert used[1].startswith("sfx_generic/")
    assert events[-1][0] == "done"
