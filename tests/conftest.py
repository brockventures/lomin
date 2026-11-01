import os
import sys
from pathlib import Path

import pytest

# The name env vars the gateway reads at import. The test suite runs with the
# same names the old hard-coded list had; test_plugin.py covers the empty default.
os.environ.setdefault("MVG_AGENT_NAMES", "Amos,Ammos,Amus,Famous,Almost")
os.environ.setdefault("MVG_WAKE_NAMES", "Amos,Ammos,Amus")
os.environ.setdefault("MVG_AGENT_NAME", "Amos")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gateway  # noqa: E402


@pytest.fixture(autouse=True)
def _log_dir_in_tmp(monkeypatch, tmp_path):
    """No test may write request/latency logs into the real workspace logs/."""
    monkeypatch.setattr(gateway, "LOG_DIR", tmp_path)


@pytest.fixture(autouse=True)
def _tts_stream_off_by_default(monkeypatch):
    """Most tests cover the whole-sentence WAV behaviour of /ask/stream
    (MVG_TTS_STREAM=0). Sub-sentence PCM streaming, the production default,
    is covered in test_ask_stream_pcm.py, which turns it back on."""
    monkeypatch.setattr(gateway, "TTS_STREAM", False)
