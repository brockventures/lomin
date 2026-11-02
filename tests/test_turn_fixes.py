#!/usr/bin/env python3
"""Direct unit tests for review findings on PR #5:
- ADPCM 512KB size cap (413 Payload Too Large) & array.array('h') decoding
- Upfront inspect.signature check in _stream_brain_turn (no TypeError retries)
- Vision payload validation (reject non-base64 text, enforce 5MB cap, type check)
- Dual env var flag support (MVG_ENABLE_VISION/MVG_VISION_ENABLED, MVG_ENABLE_ADPCM/MVG_ADPCM_ENABLED, MVG_VISION_ADPCM)
"""

from __future__ import annotations

import base64
import http.client
import json
import math
import struct
import threading

import pytest

import gateway
from vision_adpcm import (
    ADPCM_MAGIC,
    MAX_ADPCM_BYTES,
    MAX_IMAGE_B64_BYTES,
    decode_adpcm_to_wav,
    decode_ima_adpcm,
    encode_adpcm,
    extract_vision_payloads,
)


@pytest.fixture()
def server():
    srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    gateway._ambient_reset_state()
    monkeypatch.setattr(gateway, "NODES_PATH", tmp_path / "nodes.json")
    monkeypatch.setattr(gateway, "NODE_TARGETS_PATH", tmp_path / "node-targets.json")
    monkeypatch.setattr(gateway, "WATCH_TOKENS_PATH", tmp_path / "watch-tokens.json")
    monkeypatch.delenv("MVG_NODE_TARGETS", raising=False)
    monkeypatch.delenv("MVG_TRUSTED_KEY", raising=False)


def test_adpcm_upload_size_cap_413(server, monkeypatch):
    """An ADPCM upload exceeding MAX_ADPCM_BYTES (512 KiB) returns HTTP 413."""
    monkeypatch.setenv("MVG_ENABLE_ADPCM", "1")
    oversized_adpcm = ADPCM_MAGIC + b"\x00" * (MAX_ADPCM_BYTES + 64)

    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    headers = {
        "Content-Type": "audio/wav",
        "Content-Length": str(len(oversized_adpcm)),
    }
    conn.request("POST", "/voice/turn?node_id=test-node", body=oversized_adpcm, headers=headers)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()

    assert resp.status == 413
    data = json.loads(body.decode("utf-8"))
    assert data.get("error") == "payload_too_large"


def test_adpcm_array_decoding():
    """Verify decode_ima_adpcm accurately decodes via array.array('h')."""
    sr = 16000
    n = 800
    samples = [int(12000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
    pcm_in = struct.pack(f"<{n}h", *samples)
    adpcm = encode_adpcm(pcm_in, sample_rate=sr)

    wav_out = decode_adpcm_to_wav(adpcm, sample_rate=sr)
    assert len(wav_out) >= 44 + n * 2
    pcm_out = wav_out[44: 44 + n * 2]

    # Direct test of decode_ima_adpcm array decoding
    raw_pcm = decode_ima_adpcm(adpcm[16:], num_samples=n)
    assert len(raw_pcm) == n * 2

    # Verify unpacked decoded samples match within IMA ADPCM quantization loss
    out_samples = struct.unpack(f"<{n}h", pcm_out)
    diffs = [abs(a - b) for a, b in zip(samples, out_samples)]
    avg_diff = sum(diffs) / len(diffs)
    assert avg_diff < 1500  # typical IMA ADPCM error threshold for 16-bit sine


def test_stream_brain_turn_signature_inspection_no_kwargs():
    """A brain callable without **kwargs or vision args is invoked cleanly via signature filtering."""
    calls = []

    def simple_brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial):
        calls.append({"prompt": prompt, "node_id": node_id})
        on_partial("Answer from simple brain.")
        return "Answer from simple brain."

    class DummyHandler:
        def _sse(self, event, data):
            return True

    handler = DummyHandler()
    ctx = {
        "node_id": "test-node",
        "image_b64": base64.b64encode(b"img").decode("ascii"),
        "burst_images_b64": [base64.b64encode(b"f1").decode("ascii")],
    }
    opts = {"tts": "none"}

    orig_brain = gateway.STREAM_BRAIN_CALL
    try:
        gateway.STREAM_BRAIN_CALL = simple_brain
        # Call _stream_brain_turn on the dummy handler with vision enabled
        outcome, reply = gateway.GatewayHandler._stream_brain_turn(
            handler,
            t0=0.0,
            turn_id="turn-1",
            prompt="Hello brain",
            session_id="sess-1",
            node_id="test-node",
            speaker="alice",
            speaker_score=0.9,
            ctx=ctx,
            endpoint="/ask/stream",
            opts=opts,
        )
        assert reply == "Answer from simple brain."
        assert len(calls) == 1
    finally:
        gateway.STREAM_BRAIN_CALL = orig_brain


def test_stream_brain_turn_internal_type_error_never_retried():
    """An internal TypeError raised inside the brain callable must NEVER cause a second call."""
    calls = []

    def buggy_brain(prompt, session_id, node_id, speaker, speaker_score, timeout, on_partial, **kwargs):
        calls.append(len(calls) + 1)
        # Internal TypeError caused by a bug inside the brain implementation
        raise TypeError("internal bug: unsupported operand type for +: 'NoneType' and 'int'")

    class DummyHandler:
        def _sse(self, event, data):
            return True

    handler = DummyHandler()
    ctx = {"node_id": "test-node"}
    opts = {"tts": "none"}

    orig_brain = gateway.STREAM_BRAIN_CALL
    try:
        gateway.STREAM_BRAIN_CALL = buggy_brain
        outcome, reply = gateway.GatewayHandler._stream_brain_turn(
            handler,
            t0=0.0,
            turn_id="turn-err",
            prompt="Trigger error",
            session_id="sess-err",
            node_id="test-node",
            speaker="bob",
            speaker_score=0.8,
            ctx=ctx,
            endpoint="/ask/stream",
            opts=opts,
        )
        # Must only be called ONCE - no retry on TypeError!
        assert len(calls) == 1
    finally:
        gateway.STREAM_BRAIN_CALL = orig_brain


def test_vision_payload_validation_rejects_non_base64_text():
    """Non-base64 text strings must be rejected and not treated as raw bytes."""
    # Plain text strings in 'image' or 'image_b64' must return None
    img, burst = extract_vision_payloads({"image": "not a base64 encoded string!"})
    assert img is None
    assert burst == []

    img, burst = extract_vision_payloads({"image_b64": "hello world plain text"})
    assert img is None
    assert burst == []


def test_vision_payload_validation_size_caps():
    """Vision payloads exceeding 5MB must be rejected."""
    oversized = "A" * (MAX_IMAGE_B64_BYTES + 1024)
    img, burst = extract_vision_payloads({"image_b64": oversized})
    assert img is None

    oversized_bytes = b"\xff\xd8\xff" + b"\x00" * (5 * 1024 * 1024 + 1)
    img, burst = extract_vision_payloads({"image": oversized_bytes})
    assert img is None


def test_vision_payload_validation_burst_types():
    """burst_images_b64 must be a list of valid base64 strings."""
    # Non-list rejected
    assert extract_vision_payloads({"burst_images_b64": "single-string"}) == (None, [])
    assert extract_vision_payloads({"burst_images_b64": 12345}) == (None, [])

    # Filter out non-base64 elements
    valid1 = base64.b64encode(b"frame-a").decode("ascii")
    valid2 = base64.b64encode(b"frame-b").decode("ascii")
    img, burst = extract_vision_payloads({
        "burst_images_b64": [valid1, "invalid-text", valid2, None, 999]
    })
    assert img is None
    assert burst == [b"frame-a", b"frame-b"]


def test_dual_env_flag_support(monkeypatch):
    """Both MVG_ENABLE_VISION/MVG_VISION_ENABLED and MVG_ENABLE_ADPCM/MVG_ADPCM_ENABLED must work."""
    # Test MVG_VISION_ENABLED
    monkeypatch.delenv("MVG_ENABLE_VISION", raising=False)
    monkeypatch.setenv("MVG_VISION_ENABLED", "1")
    assert gateway.is_vision_enabled() is True

    # Test MVG_ADPCM_ENABLED
    monkeypatch.delenv("MVG_ENABLE_ADPCM", raising=False)
    monkeypatch.setenv("MVG_ADPCM_ENABLED", "1")
    assert gateway.is_adpcm_enabled() is True

    # Test MVG_VISION_ADPCM alias
    monkeypatch.delenv("MVG_ENABLE_VISION", raising=False)
    monkeypatch.delenv("MVG_VISION_ENABLED", raising=False)
    monkeypatch.delenv("MVG_ENABLE_ADPCM", raising=False)
    monkeypatch.delenv("MVG_ADPCM_ENABLED", raising=False)
    monkeypatch.setenv("MVG_VISION_ADPCM", "1")
    assert gateway.is_vision_enabled() is True
    assert gateway.is_adpcm_enabled() is True
