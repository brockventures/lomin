"""ERED LOMIN P1b: what stops anyone else running the voice gateway.

Covers the name-only listen with no classifier, the swappable TTS chain, the
speaker-ID off switch, the startup status lines, and the helpers moved out of
gateway.py. Nothing here reaches a real network service."""

import sys
from io import BytesIO
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audio_utils  # noqa: E402
import gateway  # noqa: E402
import multipart  # noqa: E402
import text_utils  # noqa: E402


WAV = b"RIFF...."  # the chain only checks what a backend returns, not its bytes


def _boom(text):
    raise RuntimeError("boom")


@pytest.fixture
def fallback_log(monkeypatch):
    logged = []
    monkeypatch.setattr(
        gateway, "log_fallback",
        lambda reason, tts_engine="piper", **extra: logged.append((reason, tts_engine)),
    )
    return logged


# --- 1. listening with no classifier ----------------------------------------

def test_no_classifier_engages_on_name(monkeypatch):
    monkeypatch.setattr(gateway, "_jev_classifier", None)
    d = gateway.classify_addressed_detail("state", None, name_present=True)
    assert d["score"] == 1.0
    assert d["classifier"] == "name-only"
    d = gateway.classify_addressed_detail("state", None, name_present=False)
    assert d["score"] == 0.0
    assert d["classifier"] == "name-only"


def test_no_classifier_decide_addressed_engages_on_name(monkeypatch):
    """End to end through the pre-gate: with no classifier, a name mentioned
    mid-utterance engages (name-only), and the same utterance without a name
    does not. Uses a fresh node id so no earlier turn is open."""
    monkeypatch.setattr(gateway, "_jev_classifier", None)
    named = gateway.decide_addressed("p1b-no-classifier", "turn on the kitchen lights, Amos")
    assert named["classifier"] == "name-only"
    assert named["score"] == 1.0
    assert gateway.AMBIENT_THRESHOLD <= named["score"]
    unnamed = gateway.decide_addressed("p1b-no-classifier-2", "turn on the kitchen lights")
    assert unnamed["classifier"] == "pregate"
    assert unnamed["score"] == 0.0


# --- 2. swappable TTS chain -------------------------------------------------

def test_tts_chain_order_and_fallback(monkeypatch, fallback_log):
    monkeypatch.setattr(gateway, "TTS_CHAIN", ["openai", "piper"])
    monkeypatch.setitem(gateway.TTS_BACKENDS, "openai", _boom)
    monkeypatch.setitem(gateway.TTS_BACKENDS, "piper", lambda text: WAV)
    assert gateway.synthesize("hi") == (WAV, "piper")
    assert len(fallback_log) == 1
    reason, engine = fallback_log[0]
    assert engine == "piper"
    assert reason.startswith("openai:") and "boom" in reason


def test_tts_chain_first_wins(monkeypatch, fallback_log):
    def piper_must_not_run(text):
        raise AssertionError("piper called after http spoke")

    monkeypatch.setattr(gateway, "TTS_CHAIN", ["http", "piper"])
    monkeypatch.setitem(gateway.TTS_BACKENDS, "http", lambda text: WAV)
    monkeypatch.setitem(gateway.TTS_BACKENDS, "piper", piper_must_not_run)
    assert gateway.synthesize("hi") == (WAV, "http")
    assert fallback_log == []


def test_tts_chain_all_fail_raises(monkeypatch, fallback_log):
    monkeypatch.setattr(gateway, "TTS_CHAIN", ["openai", "piper"])
    monkeypatch.setitem(gateway.TTS_BACKENDS, "openai", _boom)
    monkeypatch.setitem(gateway.TTS_BACKENDS, "piper", _boom)
    with pytest.raises(RuntimeError) as info:
        gateway.synthesize("hi")
    assert "openai:" in str(info.value) and "piper:" in str(info.value)
    assert fallback_log == []


def test_default_chain_is_today(monkeypatch, capsys):
    import os
    if "MVG_TTS_CHAIN" not in os.environ:  # the import-time value, when run unset
        assert gateway.TTS_CHAIN == ["desktop", "piper"]
    monkeypatch.delenv("MVG_TTS_CHAIN", raising=False)
    assert gateway._tts_chain_from_env() == ["desktop", "piper"]


def test_chain_env_drops_unknown_names_and_falls_back(monkeypatch, capsys):
    monkeypatch.setenv("MVG_TTS_CHAIN", " HTTP , nosuch,openai ")
    assert gateway._tts_chain_from_env() == ["http", "openai"]
    assert "nosuch" in capsys.readouterr().err
    monkeypatch.setenv("MVG_TTS_CHAIN", "nosuch")
    assert gateway._tts_chain_from_env() == ["piper"]


def test_desktop_entry_keeps_its_engine_label(monkeypatch, fallback_log):
    monkeypatch.setattr(gateway, "TTS_CHAIN", ["desktop", "piper"])
    monkeypatch.setattr(gateway, "desktop_tts_speak", lambda text: (WAV, "chatterbox"))
    assert gateway.synthesize("hi") == (WAV, "chatterbox")
    assert fallback_log == []


def test_synthesize_stream_skips_desktop_when_not_first(monkeypatch):
    pcm_wav = gateway.pcm_to_wav(b"\x01\x00" * 2400, 24000)

    def no_network(*args, **kwargs):
        raise AssertionError("desktop stream tried although it is not first in the chain")

    monkeypatch.setattr(gateway, "TTS_CHAIN", ["piper"])
    monkeypatch.setitem(gateway.TTS_BACKENDS, "piper", lambda text: pcm_wav)
    monkeypatch.setattr(gateway, "urlopen", no_network)
    engine, chunks = gateway.synthesize_stream("hi")
    assert engine == "piper"
    assert sum(len(c) for c in chunks) == 2400 * 2


# --- 3. speaker ID off switch -----------------------------------------------

def test_speaker_id_off(monkeypatch):
    monkeypatch.setenv("MVG_SPEAKER_ID", "0")
    assert gateway.resolve_profile("lammen")["identify"] is False


# --- 4. startup lines --------------------------------------------------------

def test_startup_lines_one_per_backend_and_never_raise(monkeypatch):
    dead = "http://127.0.0.1:1/x"  # nothing listens on port 1
    monkeypatch.setattr(gateway, "WHISPER_URL", dead)
    monkeypatch.setattr(gateway, "WHISPER_HEALTH_URL", dead)
    monkeypatch.setattr(gateway, "DESKTOP_TTS_URL", dead)
    monkeypatch.setattr(gateway, "DESKTOP_EMBED_URL", dead)
    monkeypatch.setattr(gateway, "TTS_CHAIN", ["piper", "openai", "elevenlabs", "http"])
    for k in ("ELEVENLABS_API_KEY", "MVG_ELEVENLABS_VOICE_ID", "MVG_OPENAI_TTS_URL",
              "MVG_HTTP_TTS_URL", "MVG_SPEAKER_ID"):
        monkeypatch.delenv(k, raising=False)
    lines = gateway.startup_lines()
    assert len(lines) == 5
    assert lines[0] == f"stt: {dead} unreachable"
    assert lines[1].startswith("tts: chain=piper,openai,elevenlabs,http ")
    assert "openai:missing config" in lines[1]
    assert "elevenlabs:missing config" in lines[1]
    assert "http:missing config" in lines[1]
    assert lines[2] == f"speaker-id: on embed={dead} unreachable"
    assert lines[3] in ("classifier: present", "classifier: absent (name-only)")
    assert lines[4].startswith("brain: ")


def test_startup_tts_lines_report_configured(monkeypatch):
    monkeypatch.setattr(gateway, "TTS_CHAIN", ["elevenlabs", "openai", "http"])
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setenv("MVG_ELEVENLABS_VOICE_ID", "v")
    monkeypatch.setenv("MVG_OPENAI_TTS_URL", "http://127.0.0.1:1/v1/audio/speech")
    monkeypatch.delenv("MVG_HTTP_TTS_URL", raising=False)
    line = [ln for ln in gateway.startup_lines() if ln.startswith("tts:")][0]
    assert line == "tts: chain=elevenlabs,openai,http elevenlabs:configured openai:configured http:missing config"


# --- 5. helpers moved out of gateway.py -------------------------------------

def test_moved_helpers_still_on_gateway():
    names = [
        "strip_markdown",
        "parse_multipart_parts", "parse_multipart_audio",
        "pcm_to_wav", "normalize_wav",
        "AUDIO_TARGET_RATE", "AUDIO_TARGET_CHANNELS", "AUDIO_TARGET_SAMPWIDTH",
    ]
    for name in names:
        assert hasattr(gateway, name), name
    assert gateway.strip_markdown is text_utils.strip_markdown
    assert gateway.parse_multipart_audio is multipart.parse_multipart_audio
    assert gateway.normalize_wav is audio_utils.normalize_wav
    assert gateway.AUDIO_TARGET_RATE == audio_utils.AUDIO_TARGET_RATE == 24000
