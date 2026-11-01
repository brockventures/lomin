#!/usr/bin/env python3
"""Tests for the /tts/stream window pacing (WindowPacer) and the underrun
harness's playback simulation. No GPU, no model: pure logic with an injected
clock. Run: python3 -m pytest ./tests -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from desktop_tts import WindowPacer, parse_ramp  # noqa: E402
import stream_underrun_harness as h  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_parse_ramp_drops_junk():
    assert parse_ramp("16, 20,24") == [16, 20, 24]
    assert parse_ramp("") == []
    assert parse_ramp(None) == []
    assert parse_ramp("0,abc,-3,18,,") == [18]


def test_fixed_mode_walks_first_ramp_then_chunk():
    p = WindowPacer(12, 30, mode="fixed", ramp=[16, 20])
    assert [p.next_size() for _ in range(6)] == [12, 16, 20, 30, 30, 30]


def test_fixed_mode_without_ramp_is_the_old_flat_schedule():
    p = WindowPacer(12, 30, mode="fixed")
    assert [p.next_size() for _ in range(4)] == [12, 30, 30, 30]


def _run_sim(rate, decode_s, prebuffer_s=0.3, first=12, la=6, max_chunk=30, windows=8, **kw):
    """Simulates the serial T3 -> decode loop with the pacer in charge and
    returns (window sizes, lead in seconds just before each chunk landed)."""
    clk = Clock()
    p = WindowPacer(first, max_chunk, clock=clk, prebuffer_s=prebuffer_s,
                    t3_rate=rate, decode_s=decode_s, **kw)
    sizes, leads = [], []
    boundary = p.next_size()
    p.window_started()
    n_seen = 0
    emitted = 0
    for _ in range(windows):
        need = boundary + la - n_seen
        clk.t += need / rate + decode_s
        if p.t_first is not None:
            leads.append(p.t_first + prebuffer_s + p.audio_s - clk.t)
        p.window_done(need, decode_s, (boundary - emitted) / 25.0)
        n_seen += need
        emitted = boundary
        size = p.next_size()
        sizes.append(size)
        boundary += size
    return sizes, leads


def test_lead_mode_never_underruns_at_nominal_throughput():
    sizes, leads = _run_sim(45.0, 0.23)
    assert leads and min(leads) >= 0.0
    assert all(16 <= s <= 30 for s in sizes)


def test_lead_mode_windows_grow_as_lead_builds():
    sizes, _ = _run_sim(45.0, 0.23, windows=12)
    assert sizes[-1] >= sizes[0]
    assert sizes[-1] == 30  # reaches the cap once the lead is comfortable


def test_lead_mode_respects_min_chunk_when_slow():
    sizes, _ = _run_sim(20.0, 0.4, windows=5, min_chunk=16)
    assert min(sizes) >= 16


def test_lead_mode_adapts_to_measured_rate():
    clk = Clock()
    p = WindowPacer(12, 30, clock=clk, t3_rate=45.0, decode_s=0.23)
    p.next_size()
    p.window_started()
    clk.t += 1.0  # T3 really ran at ~23 tok/s
    p.window_done(18, 0.23, 0.48)
    assert p.t3_rate < 45.0


# --- harness playback simulation -------------------------------------------

def _ev(*pairs):
    return [(t, int(d * h.BYTES_PER_SEC)) for t, d in pairs]


def test_underruns_none_when_chunks_arrive_ahead_of_playback():
    ev = _ev((0.6, 0.5), (1.0, 0.5), (1.4, 0.5))
    assert h.underruns(ev, 0.2)[:2] == (0, 0.0)


def test_underruns_counts_each_stall_and_worst_deficit():
    # 0.5 s of audio at 0.6, playback from 0.8 until 1.3; next lands at 1.7
    # (0.4 late), plays to 2.2; third lands at 2.5 (0.3 late).
    ev = _ev((0.6, 0.5), (1.7, 0.5), (2.5, 0.5))
    n, worst, deficits = h.underruns(ev, 0.2)
    assert n == 2 and abs(worst - 0.4) < 1e-6
    assert len(deficits) == 2


def test_bigger_prebuffer_removes_the_stall():
    ev = _ev((0.6, 0.5), (1.4, 0.5))
    assert h.underruns(ev, 0.2)[0] == 1
    assert h.underruns(ev, 0.5)[0] == 0


# --- before/after on the reported scenario ----------------------------------

def _arrivals(mode, rate=37.5, decode_s=0.2, first=12, la=6, max_chunk=30, windows=10, **kw):
    """(time, audio_s) per emitted window for the serial T3 -> decode loop at
    ~1.5x realtime with a flat 0.2 s decode, as reported for Chatterbox."""
    clk = Clock()
    p = WindowPacer(first, max_chunk, mode=mode, clock=clk, t3_rate=rate,
                    decode_s=decode_s, **kw)
    events, boundary, n_seen, emitted = [], p.next_size(), 0, 0
    p.window_started()
    for _ in range(windows):
        need = boundary + la - n_seen
        clk.t += need / rate + decode_s
        audio = (boundary - emitted) / 25.0
        events.append((clk.t, int(audio * h.BYTES_PER_SEC)))
        p.window_done(need, decode_s, audio)
        n_seen, emitted = n_seen + need, boundary
        boundary += p.next_size()
    return events


def test_before_after_reported_scenario():
    old = _arrivals("fixed")  # the previous flat 12, 30, 30... schedule
    new = _arrivals("lead")
    for prebuf in (0.2, 0.3):
        assert h.underruns(old, prebuf)[0] >= 1  # the second window stalls
        assert h.underruns(new, prebuf)[0] == 0
    assert new[0][0] == old[0][0]  # first-audio latency unchanged
