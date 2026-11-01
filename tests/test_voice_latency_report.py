"""bin/voice-latency-report.py: deterministic rollup and the turn join."""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "bin" / "voice-latency-report.py"
if not SCRIPT.exists():
    # The script lives in the karakos workspace, outside this repo.
    pytest.skip("bin/voice-latency-report.py is not in this checkout", allow_module_level=True)
spec = importlib.util.spec_from_file_location("voice_latency_report", SCRIPT)
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

NOW = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)
T0 = int(NOW.timestamp() * 1000)


def _g(i, node="bench-node", start=None, **kw):
    rec = {"ts": "2026-09-30T11:00:00Z", "kind": "turn", "node_id": node, "turn_id": f"t{i}",
           "request_start": start if start is not None else T0 + i * 100000,
           "brain_first_text_ms": 4000 + i, "first_audio_ms": 5000 + i,
           "gap_fill_first_ms": None, "total_ms": 9000, "outcome": "ok"}
    rec.update(kw)
    return rec


def _lines(recs):
    return "\n".join(json.dumps(r) for r in recs)


def test_percentiles_nearest_rank():
    assert rep.pct([], 50) is None
    assert rep.pct([10], 90) == 10
    vals = list(range(1, 11))
    assert rep.pct(vals, 50) == 5 and rep.pct(vals, 90) == 9


def test_parse_since_forms():
    assert rep.parse_since("24h", NOW).day == 29
    assert rep.parse_since("90m", NOW).hour == 10
    assert rep.parse_since("2026-09-01", NOW).month == 9


def test_window_filters_old_records():
    old = _g(1, ts="2026-09-01T00:00:00Z")
    new = _g(2)
    r = rep.build_report(_lines([old, new]), "", rep.parse_since("24h", NOW))
    assert r["counts"]["gateway_turns"] == 1


def test_stage_rollup_per_stage_and_node():
    recs = [_g(i) for i in range(1, 11)] + [_g(20, node="discord-voice", first_audio_ms=9000)]
    recs.append({"ts": "2026-09-30T11:00:00Z", "kind": "stt", "node_id": "bench-node",
                 "stt_ms": 800, "audio_s": 2.0, "chars": 30})
    r = rep.build_report(_lines(recs), "", rep.parse_since("24h", NOW))
    st = r["stages"]["gateway"]["first_audio_ms"]
    assert st["n"] == 11
    assert r["by_node"]["bench-node"]["gateway"]["first_audio_ms"] == {"n": 10, "p50": 5005.0, "p90": 5009.0}
    assert r["by_node"]["discord-voice"]["gateway"]["first_audio_ms"]["p50"] == 9000.0
    assert r["stages"]["stt"]["stt_ms"]["p50"] == 800.0
    assert "gap_fill_first_ms" not in r["stages"]["gateway"]      # all None -> no row


def test_join_by_turn_id_and_by_node_time():
    g_a = _g(1, node="discord-voice", start=T0 + 1500, gap_fill_first_ms=1300, first_audio_ms=6000)
    g_b = _g(2, node="discord-voice", start=T0 + 100000 + 900)
    g_room = _g(3, node="bench-node", start=T0 + 100000 + 800)
    d_a = {"ts": "2026-09-30T11:00:00Z", "node_id": "discord-voice", "turn_id": "t1",
           "post_start": T0, "first_audio_chunk_ms": 3000, "transcript_ms": 600}
    d_b = {"ts": "2026-09-30T11:00:00Z", "node_id": "discord-voice", "turn_id": None,
           "post_start": T0 + 100000, "first_audio_chunk_ms": 7000}
    pairs = rep.join_turns([g_a, g_b, g_room], [d_a, d_b])
    assert sorted((g["turn_id"], d["turn_id"]) for g, d in pairs) == [("t1", "t1"), ("t2", None)]
    m = rep.joined_metrics(g_a, d_a)
    assert m["pre_brain_ms"] == 1500
    assert m["felt_first_sound_ms"] == 2800          # filler at 1300 beats real audio at 6000
    assert m["ear_overhead_ms"] == 200


def test_report_is_deterministic_and_renders(tmp_path):
    recs = [_g(i) for i in range(1, 6)]
    gw = tmp_path / "g.jsonl"
    gw.write_text(_lines(recs))
    args = ["--gateway-file", str(gw), "--discord-file", str(tmp_path / "none.jsonl"),
            "--source", "local", "--now", "2026-09-30T12:00:00Z"]
    import io
    import contextlib
    outs = []
    for _ in range(2):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert rep.main(args) == 0
        outs.append(buf.getvalue())
    assert outs[0] == outs[1]
    assert "first_audio_ms" in outs[0] and "NODE bench-node" in outs[0]


def test_handoff_stage_deltas_from_raw_stages():
    stages = {"inserted_ms": 1000, "claimed_pi_ms": 1030, "claim_recv_ms": 5040,
              "dispatch_start_ms": 5050, "typed_ms": 5070, "enter_sent_ms": 5600,
              "first_text_seen_ms": 6800, "partial_write_start_ms": 6805,
              "partial_stored_pi_ms": 2830, "first_partial_seen_ms": 2850}
    h = rep.handoff_metrics({"stages": stages})
    assert h["insert_to_claim_ms"] == 30
    assert h["typed_to_enter_ms"] == 530
    assert h["enter_to_first_text_ms"] == 1200
    assert h["partial_to_gateway_seen_ms"] == 20
    assert h["call_to_first_partial_ms"] == 1850
    assert rep.handoff_metrics({}) == {} and rep.handoff_metrics({"stages": {"x": 1}}) == {}
    report = rep.build_report(_lines([_g(1, stages=stages), _g(2)]), "", NOW.replace(hour=0))
    assert report["stages"]["handoff"]["insert_to_claim_ms"]["n"] == 1
    assert "hive hand-off" in rep.render(report)


def test_intent_hits_get_their_own_stage_and_leave_brain_rows_alone():
    recs = [_g(1), _g(2),
            _g(3, intent="miss", intent_match_ms=0.03),
            _g(4, intent="time", intent_match_ms=0.05, intent_handler_ms=2.0,
               brain_first_text_ms=None, first_audio_ms=900, total_ms=1200)]
    r = rep.build_report(_lines(recs), "", NOW.replace(day=1))
    assert r["counts"]["gateway_turns"] == 3 and r["counts"]["intent_turns"] == 1
    assert r["counts"]["intent_misses"] == 1
    assert r["stages"]["gateway"]["first_audio_ms"]["n"] == 3
    assert r["stages"]["intent"]["intent_match_ms"]["n"] == 1
    assert r["stages"]["intent"]["first_audio_ms"]["p50"] == 900
    assert "LAMMAS intent fast path" in rep.render(r)
