import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

GATEWAY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GATEWAY_DIR))
import gateway  # noqa: E402


# ---------------------------------------------------------------------------
# Unit tests for consume_brain_sse_lines
# ---------------------------------------------------------------------------

def test_consume_brain_sse_openai_deltas():
    partials = []
    lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
        'data: {"choices": [{"delta": {"content": " world"}}]}\n',
        'data: {"choices": [{"delta": {"content": "!"}}]}\n',
        'data: [DONE]\n',
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "Hello world!"
    assert partials == ["Hello", "Hello world", "Hello world!"]


def test_consume_brain_sse_bytes_and_comments():
    partials = []
    lines = [
        b": keepalive comment\n",
        b'data: {"choices": [{"delta": {"content": "The"}}]}\n',
        b"\n",
        b'data: {"choices": [{"delta": {"content": " answer."}}]}\n',
        b"data: [DONE]\n",
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "The answer."
    assert partials == ["The", "The answer."]


def test_consume_brain_sse_events_sentence():
    partials = []
    lines = [
        "event: sentence\n",
        'data: {"text": "First sentence."}\n',
        "\n",
        "event: sentence\n",
        'data: {"text": "Second sentence."}\n',
        "\n",
        "event: reply\n",
        'data: {"reply": "First sentence. Second sentence."}\n',
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "First sentence. Second sentence."
    assert partials == [
        "First sentence.",
        "First sentence. Second sentence.",
        "First sentence. Second sentence.",
    ]


def test_consume_brain_sse_events_delta():
    partials = []
    lines = [
        "event: delta\n",
        'data: {"text": "Alpha "}\n',
        "\n",
        "event: delta\n",
        'data: {"text": "Beta "}\n',
        "\n",
        "event: delta\n",
        'data: {"text": "Gamma"}\n',
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "Alpha Beta Gamma"
    assert partials == ["Alpha ", "Alpha Beta ", "Alpha Beta Gamma"]


def test_consume_brain_sse_sentence_with_reply_key():
    partials = []
    lines = [
        "event: sentence\n",
        'data: {"reply": "Sentence one."}\n',
        "\n",
        "event: sentence\n",
        'data: {"reply": "Sentence two."}\n',
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "Sentence one. Sentence two."
    assert partials == ["Sentence one.", "Sentence one. Sentence two."]


def test_consume_brain_sse_standalone_reply_event():
    partials = []
    lines = [
        "event: status\n",
        'data: {"status": "thinking"}\n',
        "\n",
        "event: reply\n",
        'data: {"reply": "One shot reply."}\n',
    ]
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert reply == "One shot reply."
    assert partials == ["One shot reply."]


# ---------------------------------------------------------------------------
# Integration tests for http_brain_call_stream & BRAIN_DIRECT_AUDIO
# ---------------------------------------------------------------------------

class MockBrainServer:
    def __init__(self, handler_cls):
        self.server = HTTPServer(("127.0.0.1", 0), handler_cls)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.port = self.server.server_port

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def test_http_brain_call_stream_sse_ask_stream(monkeypatch):
    seen_headers = []

    class SSEHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen_headers.append(dict(self.headers))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunks = [
                b"event: sentence\ndata: {\"text\": \"Hello there.\"}\n\n",
                b"event: sentence\ndata: {\"text\": \"How can I help?\"}\n\n",
                b"event: reply\ndata: {\"reply\": \"Hello there. How can I help?\"}\n\n",
            ]
            for c in chunks:
                self.wfile.write(c)
                self.wfile.flush()

        def log_message(self, *args):
            pass

    server = MockBrainServer(SSEHandler)
    server.start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/ask/stream")
        monkeypatch.setattr(gateway, "HTTP_BRAIN_STREAM", True)
        partials = []
        reply = gateway.http_brain_call_stream(
            "hello", "sess-sse-1", "node-1", "alice", 0.9, 5, on_partial=partials.append
        )
        assert reply == "Hello there. How can I help?"
        assert partials == [
            "Hello there.",
            "Hello there. How can I help?",
            "Hello there. How can I help?",
        ]
        assert "text/event-stream" in seen_headers[0].get("Accept", "")
    finally:
        server.stop()


def test_http_brain_call_stream_sse_openai_deltas(monkeypatch):
    class OpenAIDeltaHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                b"data: {\"choices\": [{\"delta\": {\"content\": \"Streamed \"}}]}\n\n",
                b"data: {\"choices\": [{\"delta\": {\"content\": \"response.\"}}]}\n\n",
                b"data: [DONE]\n\n",
            ]
            for c in chunks:
                self.wfile.write(c)
                self.wfile.flush()

        def log_message(self, *args):
            pass

    server = MockBrainServer(OpenAIDeltaHandler)
    server.start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/v1/chat")
        monkeypatch.setattr(gateway, "HTTP_BRAIN_STREAM", True)
        partials = []
        reply = gateway.http_brain_call_stream(
            "test prompt", "sess-openai-1", "node-1", "", None, 5, on_partial=partials.append
        )
        assert reply == "Streamed response."
        assert partials == ["Streamed ", "Streamed response."]
    finally:
        server.stop()


def test_http_brain_call_stream_json_fallback(monkeypatch):
    # When HTTP_BRAIN_STREAM is false (default) and server returns application/json
    class JSONHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps({"reply": "standard json fallback response"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = MockBrainServer(JSONHandler)
    server.start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/brain")
        monkeypatch.setattr(gateway, "HTTP_BRAIN_STREAM", False)
        partials = []
        reply = gateway.http_brain_call_stream(
            "json prompt", "sess-json-1", "node-1", "", None, 5, on_partial=partials.append
        )
        assert reply == "standard json fallback response"
        assert partials == ["standard json fallback response"]
    finally:
        server.stop()


def test_http_brain_call_stream_json_with_stream_flag(monkeypatch):
    # When HTTP_BRAIN_STREAM is true but server answers with application/json
    class JSONHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps({"reply": "json despite stream request"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = MockBrainServer(JSONHandler)
    server.start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/brain")
        monkeypatch.setattr(gateway, "HTTP_BRAIN_STREAM", True)
        partials = []
        reply = gateway.http_brain_call_stream(
            "json prompt", "sess-json-2", "node-1", "", None, 5, on_partial=partials.append
        )
        assert reply == "json despite stream request"
        assert partials == ["json despite stream request"]
    finally:
        server.stop()


def test_brain_direct_audio_gating(monkeypatch):
    seen_payloads = []

    class RecordHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            seen_payloads.append(json.loads(body.decode("utf-8")))
            resp = json.dumps({"reply": "audio acknowledged"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *args):
            pass

    server = MockBrainServer(RecordHandler)
    server.start()
    raw_audio_b64 = base64.b64encode(b"RIFFdummywavdata").decode("ascii")
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/brain")

        # 1. BRAIN_DIRECT_AUDIO = False -> audio_b64 should NOT be in payload
        monkeypatch.setattr(gateway, "BRAIN_DIRECT_AUDIO", False)
        monkeypatch.setenv("MVG_BRAIN_DIRECT_AUDIO", "0")
        gateway.http_brain_call("test", "sess-1", "node-1", "", None, 5, audio_b64=raw_audio_b64)
        assert len(seen_payloads) == 1
        assert "audio_b64" not in seen_payloads[0]

        # 2. BRAIN_DIRECT_AUDIO = True via http_brain_call -> audio_b64 IS in payload
        monkeypatch.setattr(gateway, "BRAIN_DIRECT_AUDIO", True)
        monkeypatch.setenv("MVG_BRAIN_DIRECT_AUDIO", "1")
        gateway.http_brain_call("test", "sess-2", "node-1", "", None, 5, audio_b64=raw_audio_b64)
        assert len(seen_payloads) == 2
        assert seen_payloads[1]["audio_b64"] == raw_audio_b64

        # 3. BRAIN_DIRECT_AUDIO = True via http_brain_call_stream -> audio_b64 IS in payload
        partials = []
        gateway.http_brain_call_stream(
            "test", "sess-3", "node-1", "", None, 5, on_partial=partials.append, audio_b64=raw_audio_b64
        )
        assert len(seen_payloads) == 3
        assert seen_payloads[2]["audio_b64"] == raw_audio_b64

        # 4. Context propagation via _FRAME_CTX
        gateway._FRAME_CTX.ctx = {"audio_b64": raw_audio_b64}
        gateway.http_brain_call("test", "sess-4", "node-1", "", None, 5)
        assert len(seen_payloads) == 4
        assert seen_payloads[3]["audio_b64"] == raw_audio_b64
        gateway._FRAME_CTX.ctx = None
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Tests for Bounded SSE Stream, Validation & Shared Request Builder
# ---------------------------------------------------------------------------

def test_consume_brain_sse_max_sse_chars(monkeypatch):
    monkeypatch.setattr(gateway, "MAX_SSE_CHARS", 50)
    lines = [
        'data: {"choices": [{"delta": {"content": "' + ("A" * 30) + '"}}]}\n',
        'data: {"choices": [{"delta": {"content": "' + ("B" * 30) + '"}}]}\n',
        'data: {"choices": [{"delta": {"content": "' + ("C" * 30) + '"}}]}\n',
    ]
    partials = []
    reply = gateway.consume_brain_sse_lines(lines, on_partial=partials.append)
    assert len(reply) == 50
    assert reply == ("A" * 30) + ("B" * 20)
    assert len(partials) == 2


def test_consume_brain_sse_max_stream_seconds():
    def slow_generator():
        yield 'data: {"choices": [{"delta": {"content": "First"}}]}\n'
        time.sleep(0.2)
        yield 'data: {"choices": [{"delta": {"content": "Second"}}]}\n'

    partials = []
    reply = gateway.consume_brain_sse_lines(slow_generator(), on_partial=partials.append, max_stream_seconds=0.1)
    assert reply == "First"
    assert partials == ["First"]


def test_validate_multimodal_payload_unit(monkeypatch):
    assert gateway.validate_multimodal_payload("not a dict") == (400, "request body must be a JSON object")

    # Valid empty or normal dict
    assert gateway.validate_multimodal_payload({}) == (None, None)
    valid_b64 = base64.b64encode(b"validdata").decode("ascii")
    assert gateway.validate_multimodal_payload({
        "image_b64": valid_b64,
        "burst_images_b64": [valid_b64],
        "audio_b64": valid_b64,
    }) == (None, None)

    # image_b64 checks
    assert gateway.validate_multimodal_payload({"image_b64": 12345})[0] == 400
    assert gateway.validate_multimodal_payload({"image_b64": "invalid-b64!@#$%"})[0] == 400
    monkeypatch.setattr(gateway, "MAX_IMAGE_B64_BYTES", 20)
    assert gateway.validate_multimodal_payload({"image_b64": valid_b64 * 5})[0] == 413
    monkeypatch.undo()

    # burst_images_b64 checks
    assert gateway.validate_multimodal_payload({"burst_images_b64": "not a list"})[0] == 400
    assert gateway.validate_multimodal_payload({"burst_images_b64": [123]})[0] == 400
    assert gateway.validate_multimodal_payload({"burst_images_b64": ["bad_b64@@"]})[0] == 400
    monkeypatch.setattr(gateway, "MAX_IMAGE_B64_BYTES", 20)
    assert gateway.validate_multimodal_payload({"burst_images_b64": [valid_b64 * 5]})[0] == 413
    monkeypatch.undo()

    # audio_b64 checks
    assert gateway.validate_multimodal_payload({"audio_b64": ["not a str"]})[0] == 400
    assert gateway.validate_multimodal_payload({"audio_b64": "bad_b64$$$"})[0] == 400
    monkeypatch.setattr(gateway, "MAX_AUDIO_B64_BYTES", 20)
    assert gateway.validate_multimodal_payload({"audio_b64": valid_b64 * 5})[0] == 413
    monkeypatch.undo()


def test_build_http_brain_body_shared_helper(monkeypatch):
    monkeypatch.setattr(gateway, "ENABLE_VISION", False)
    monkeypatch.setattr(gateway, "BRAIN_DIRECT_AUDIO", False)
    monkeypatch.delenv("MVG_ENABLE_VISION", raising=False)
    monkeypatch.delenv("MVG_BRAIN_DIRECT_AUDIO", raising=False)

    body_dict, body_bytes = gateway.build_http_brain_body(
        "hello", "sess-1", "node-1", "speaker1", 0.9,
        image_b64="img", burst_images_b64=["burst"], audio_b64="aud"
    )
    assert "prompt" in body_dict
    assert "session_id" in body_dict
    assert "image_b64" not in body_dict
    assert "burst_images_b64" not in body_dict
    assert "audio_b64" not in body_dict
    assert json.loads(body_bytes.decode("utf-8")) == body_dict

    # With vision enabled
    monkeypatch.setattr(gateway, "ENABLE_VISION", True)
    body_dict, _ = gateway.build_http_brain_body(
        "hello", "sess-1", "node-1", "speaker1", 0.9,
        image_b64="img", burst_images_b64=["burst"], audio_b64="aud"
    )
    assert body_dict["image_b64"] == "img"
    assert body_dict["burst_images_b64"] == ["burst"]
    assert "audio_b64" not in body_dict

    # With direct audio enabled
    monkeypatch.setattr(gateway, "BRAIN_DIRECT_AUDIO", True)
    body_dict, _ = gateway.build_http_brain_body(
        "hello", "sess-1", "node-1", "speaker1", 0.9,
        image_b64="img", burst_images_b64=["burst"], audio_b64="aud"
    )
    assert body_dict["audio_b64"] == "aud"


def test_http_brain_call_sse_gated_by_mvg_http_stream(monkeypatch):
    sent_headers = []

    class SSEHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            sent_headers.append(dict(self.headers))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"event: reply\ndata: {\"reply\": \"streamed reply\"}\n\n")

        def log_message(self, *args):
            pass

    server = MockBrainServer(SSEHandler)
    server.start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.port}/brain")

        # 1. MVG_HTTP_STREAM = 0 / False -> does NOT accept or consume SSE
        monkeypatch.setattr(gateway, "HTTP_BRAIN_STREAM", False)
        monkeypatch.setenv("MVG_HTTP_STREAM", "0")
        with pytest.raises(Exception):
            gateway.http_brain_call("test", "sess-1", "node-1", "", None, 5)
        assert "text/event-stream" not in sent_headers[0].get("Accept", "")

        # 2. MVG_HTTP_STREAM = 1 / True -> accepts and consumes SSE
        monkeypatch.setattr(gateway, "HTTP_BRAIN_STREAM", True)
        monkeypatch.setenv("MVG_HTTP_STREAM", "1")
        reply = gateway.http_brain_call("test", "sess-2", "node-1", "", None, 5)
        assert reply == "streamed reply"
        assert "text/event-stream" in sent_headers[1].get("Accept", "")
    finally:
        server.stop()


def test_gateway_ask_endpoints_multimodal_validation(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), gateway.GatewayHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    port = server.server_port
    base_url = f"http://127.0.0.1:{port}"

    try:
        # 1. Invalid JSON body to /ask
        req = Request(f"{base_url}/ask", data=b"not json", headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=5)
        assert exc_info.value.code == 400

        # 2. Invalid image_b64 (not valid base64) to /ask
        body = json.dumps({"prompt": "hi", "image_b64": "invalid!@#$"}).encode("utf-8")
        req = Request(f"{base_url}/ask", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=5)
        assert exc_info.value.code == 400

        # 3. Oversize image_b64 to /ask -> 413
        monkeypatch.setattr(gateway, "MAX_IMAGE_B64_BYTES", 10)
        body = json.dumps({"prompt": "hi", "image_b64": "AAAA" * 5}).encode("utf-8")
        req = Request(f"{base_url}/ask", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=5)
        assert exc_info.value.code == 413
        monkeypatch.undo()

        # 4. Invalid burst_images_b64 (element not base64) to /ask/stream -> 400
        body = json.dumps({"prompt": "hi", "burst_images_b64": ["bad!@#$"]}).encode("utf-8")
        req = Request(f"{base_url}/ask/stream", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=5)
        assert exc_info.value.code == 400

        # 5. Oversize audio_b64 to /ask/stream -> 413
        monkeypatch.setattr(gateway, "MAX_AUDIO_B64_BYTES", 10)
        body = json.dumps({"prompt": "hi", "audio_b64": "AAAA" * 5}).encode("utf-8")
        req = Request(f"{base_url}/ask/stream", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(HTTPError) as exc_info:
            urlopen(req, timeout=5)
        assert exc_info.value.code == 413
        monkeypatch.undo()
    finally:
        server.shutdown()
        server.server_close()
