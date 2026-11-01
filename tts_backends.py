"""Swappable TTS backends for the voice gateway (ERED LOMIN P1b).

Each backend is `(text: str) -> bytes`: a complete WAV, starting with RIFF.
It raises on any failure: a missing config, a transport error, a non-2xx
status, or a body that is not a WAV. gateway.py builds its chain from these
(MVG_TTS_CHAIN); see the README "Backends" section.

Stdlib only, and no import of gateway: config is read from the environment
on every call, so a backend can be swapped or reconfigured without a reload.
"""

import json
import os
import urllib.parse
import urllib.request

ELEVENLABS_TIMEOUT = 10
OPENAI_TTS_TIMEOUT = 15
HTTP_TTS_TIMEOUT = 15


def _post_wav(url: str, data: bytes, headers: dict, timeout: float) -> bytes:
    """POST `data`, return the body if it is a 2xx WAV. Raises otherwise."""
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        body = resp.read()
    if not 200 <= status < 300:
        raise RuntimeError(f"tts HTTP {status}")
    if not body or not body.startswith(b"RIFF"):
        raise RuntimeError(f"tts returned a non-WAV body ({len(body or b'')} bytes)")
    return body


def elevenlabs_speak(text: str) -> bytes:
    """ElevenLabs text-to-speech, WAV output at 24 kHz.

    Env: ELEVENLABS_API_KEY (required), MVG_ELEVENLABS_VOICE_ID (required),
    MVG_ELEVENLABS_MODEL (default eleven_multilingual_v2)."""
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    voice = os.environ.get("MVG_ELEVENLABS_VOICE_ID", "")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    if not voice:
        raise RuntimeError("MVG_ELEVENLABS_VOICE_ID not set")
    model = os.environ.get("MVG_ELEVENLABS_MODEL") or "eleven_multilingual_v2"
    url = ("https://api.elevenlabs.io/v1/text-to-speech/"
           + urllib.parse.quote(voice, safe="")
           + "?output_format=wav_24000")
    body = json.dumps({"text": text, "model_id": model}).encode("utf-8")
    headers = {
        "xi-api-key": key,
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }
    return _post_wav(url, body, headers, ELEVENLABS_TIMEOUT)


def openai_speech_speak(text: str) -> bytes:
    """Any OpenAI-compatible /v1/audio/speech endpoint. Kokoro-FastAPI serves
    one at http://localhost:8880/v1/audio/speech.

    Env: MVG_OPENAI_TTS_URL (required), MVG_OPENAI_TTS_MODEL (default kokoro),
    MVG_OPENAI_TTS_VOICE (default af_sky), MVG_OPENAI_TTS_KEY (optional; sent
    as a bearer token only when set)."""
    url = os.environ.get("MVG_OPENAI_TTS_URL", "")
    if not url:
        raise RuntimeError("MVG_OPENAI_TTS_URL not set")
    payload = {
        "model": os.environ.get("MVG_OPENAI_TTS_MODEL") or "kokoro",
        "voice": os.environ.get("MVG_OPENAI_TTS_VOICE") or "af_sky",
        "input": text,
        "response_format": "wav",
    }
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("MVG_OPENAI_TTS_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(payload).encode("utf-8")
    return _post_wav(url, body, headers, OPENAI_TTS_TIMEOUT)


def http_tts_speak(text: str) -> bytes:
    """Generic hook: POST {"text": text} as JSON to MVG_HTTP_TTS_URL; the
    response body is the WAV."""
    url = os.environ.get("MVG_HTTP_TTS_URL", "")
    if not url:
        raise RuntimeError("MVG_HTTP_TTS_URL not set")
    body = json.dumps({"text": text}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    return _post_wav(url, body, headers, HTTP_TTS_TIMEOUT)
