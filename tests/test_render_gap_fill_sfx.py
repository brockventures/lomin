#!/usr/bin/env python3
"""bin/render-gap-fill-sfx.py: 15 synthesized effects, format, peaks, length and
byte-identical reruns. Run: python3 -m pytest tests/test_render_gap_fill_sfx.py"""
import importlib.util
import json
import struct
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = ROOT / "bin" / "render-gap-fill-sfx.py"
if not _SCRIPT.exists():
    # The script lives in the karakos workspace, outside this repo.
    pytest.skip("bin/render-gap-fill-sfx.py is not in this checkout", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("render_gap_fill_sfx", _SCRIPT)
sfx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sfx)

EXPECTED = 15


def _wavs(root: Path):
    return sorted(p for p in root.rglob("*.wav"))


def _read(path: Path):
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000), path
        frames = w.getnframes()
        raw = w.readframes(frames)
    return frames, struct.unpack("<%dh" % frames, raw)


def test_renders_fifteen_clips_in_spec_format(tmp_path):
    assert sfx.main(["--out", str(tmp_path)]) == 0
    files = _wavs(tmp_path)
    assert len(files) == EXPECTED
    for path in files:
        frames, _ = _read(path)
        seconds = frames / 24000
        assert 0.1 <= seconds <= 1.5, (path.name, seconds)


def test_peaks_respect_ceilings(tmp_path):
    sfx.main(["--out", str(tmp_path)])
    for path in _wavs(tmp_path):
        _, samples = _read(path)
        peak = max(abs(s) for s in samples)
        if path.parent.name == "sfx_tick":
            assert peak <= 2000, (path.name, peak)
        else:
            assert peak <= 8200, (path.name, peak)
        assert peak > 0, path.name


def test_manifest_lists_every_clip_as_synth(tmp_path):
    sfx.main(["--out", str(tmp_path)])
    clips = json.loads((tmp_path / "manifest.json").read_text())["clips"]
    assert len(clips) == EXPECTED
    for rel, meta in clips.items():
        assert (tmp_path / rel).exists(), rel
        assert meta["category"].startswith("sfx_") and meta["engine"] == "synth"
        assert meta["text"] == ""


def test_rerun_is_byte_identical(tmp_path):
    sfx.main(["--out", str(tmp_path)])
    first = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in _wavs(tmp_path)}
    assert sfx.main(["--out", str(tmp_path), "--force"]) == 0
    second = {p.relative_to(tmp_path).as_posix(): p.read_bytes() for p in _wavs(tmp_path)}
    assert first == second
