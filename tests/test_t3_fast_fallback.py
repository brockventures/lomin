#!/usr/bin/env python3
"""Fallback behaviour of the CUDA-graph T3 path (desktop_tts._t3_token_iter).

No torch, no GPU: the fast and eager token generators are replaced by fakes,
so these pin the dispatch/fallback contract only (the numerical equivalence of
the graph step is checked by bench_t3.py on the GPU)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import desktop_tts  # noqa: E402
import t3_fast  # noqa: E402


class _Model:
    pass


def _model(dec="graph"):
    m = _Model()
    m._mdt_t3_fast = dec
    return m


def test_no_decoder_uses_eager_directly(monkeypatch):
    calls = []

    def eager(model, text, cancel=None, prefix=None):
        calls.append(prefix)
        yield from [1, 2, 3]

    monkeypatch.setattr(desktop_tts, "_t3_token_iter_eager", eager)
    monkeypatch.setattr(desktop_tts, "_t3_token_iter_fast",
                        lambda *a, **k: pytest.fail("fast path must not run"))
    assert list(desktop_tts._t3_token_iter(_model(None), "hi")) == [1, 2, 3]
    assert calls == [None]


def test_fast_path_used_when_decoder_present(monkeypatch):
    monkeypatch.setattr(desktop_tts, "_t3_token_iter_fast", lambda m, d, t, c=None: iter([7, 8, 9]))
    monkeypatch.setattr(desktop_tts, "_t3_token_iter_eager",
                        lambda *a, **k: pytest.fail("eager must not run"))
    m = _model()
    assert list(desktop_tts._t3_token_iter(m, "hi")) == [7, 8, 9]
    assert m._mdt_t3_fast == "graph"


def test_mid_stream_failure_resumes_eagerly_without_repeating_tokens(monkeypatch, capsys):
    def fast(model, dec, text, cancel=None):
        yield 10
        yield 11
        raise RuntimeError("cuda graph replay failed")

    seen = {}

    def eager(model, text, cancel=None, prefix=None):
        seen["prefix"] = list(prefix)
        yield 12
        yield 13

    monkeypatch.setattr(desktop_tts, "_t3_token_iter_fast", fast)
    monkeypatch.setattr(desktop_tts, "_t3_token_iter_eager", eager)
    m = _model()
    assert list(desktop_tts._t3_token_iter(m, "hi")) == [10, 11, 12, 13]
    assert seen["prefix"] == [10, 11]  # eager is told what was already emitted
    assert m._mdt_t3_fast is None  # disabled for the rest of the process
    err = capsys.readouterr().err
    assert err.count("T3 fast path failed") == 1  # exactly one log line
    assert "falling back to eager" in err


def test_failure_before_first_token_and_later_calls_stay_eager(monkeypatch, capsys):
    def fast(model, dec, text, cancel=None):
        raise ValueError("prefill exceeds static cache")
        yield  # pragma: no cover

    prefixes = []

    def eager(model, text, cancel=None, prefix=None):
        prefixes.append(list(prefix or []))
        yield 5

    monkeypatch.setattr(desktop_tts, "_t3_token_iter_fast", fast)
    monkeypatch.setattr(desktop_tts, "_t3_token_iter_eager", eager)
    m = _model()
    assert list(desktop_tts._t3_token_iter(m, "a")) == [5]
    assert list(desktop_tts._t3_token_iter(m, "b")) == [5]
    assert prefixes == [[], []]
    assert capsys.readouterr().err.count("T3 fast path failed") == 1  # second call never tried fast


def test_consumer_abandoning_the_stream_is_not_treated_as_failure(monkeypatch):
    def fast(model, dec, text, cancel=None):
        yield from range(100)

    monkeypatch.setattr(desktop_tts, "_t3_token_iter_fast", fast)
    monkeypatch.setattr(desktop_tts, "_t3_token_iter_eager",
                        lambda *a, **k: pytest.fail("eager must not run"))
    m = _model()
    gen = desktop_tts._t3_token_iter(m, "hi")
    assert next(gen) == 0
    gen.close()  # GeneratorExit must pass straight through
    assert m._mdt_t3_fast == "graph"


def test_setup_respects_flag_off(monkeypatch):
    monkeypatch.setattr(desktop_tts, "T3_FAST", False)
    monkeypatch.setattr(t3_fast, "try_build", lambda *a, **k: pytest.fail("must not build"))
    m = _Model()
    desktop_tts._setup_t3_fast(m)
    assert m._mdt_t3_fast is None


def test_setup_falls_back_when_capture_fails(monkeypatch):
    monkeypatch.setattr(desktop_tts, "T3_FAST", True)
    monkeypatch.setattr(t3_fast, "try_build", lambda *a, **k: None)  # try_build's failure contract
    m = _Model()
    m.t3 = object()
    desktop_tts._setup_t3_fast(m)
    assert m._mdt_t3_fast is None


def test_setup_falls_back_when_warm_run_raises_unexpectedly(monkeypatch, capsys):
    monkeypatch.setattr(desktop_tts, "T3_FAST", True)
    monkeypatch.setattr(t3_fast, "try_build", lambda *a, **k: "graph")

    def boom(*a, **k):
        raise RuntimeError("warm run exploded")
        yield  # pragma: no cover

    monkeypatch.setattr(desktop_tts, "_t3_token_iter", boom)
    m = _Model()
    m.t3 = object()
    desktop_tts._setup_t3_fast(m)
    assert m._mdt_t3_fast is None
    assert "using eager" in capsys.readouterr().err


def test_try_build_returns_none_instead_of_raising(capsys):
    # No CUDA / no torch in this interpreter, or a bogus t3: either way the
    # contract is "log one line, return None".
    assert t3_fast.try_build(object(), device="cuda", max_len=64) is None
    assert "using eager" in capsys.readouterr().err
