import http.client
import json
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

GATEWAY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GATEWAY_DIR))

import gateway  # noqa: E402
import ha_fastpath  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_ha_state(monkeypatch):
    ha_fastpath.reset_circuit_breaker()
    monkeypatch.delenv("MVG_HA_INTENT_TOKEN", raising=False)
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    yield
    ha_fastpath.reset_circuit_breaker()


@pytest.fixture()
def server():
    srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


class MockHAServer:
    """Ephemeral HTTP server for testing real urllib requests to Home Assistant."""

    def __init__(self, handler_cls):
        self.server = HTTPServer(("127.0.0.1", 0), handler_cls)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.port = self.server.server_port

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------------------
# Unit tests for try_ha_conversation
# ---------------------------------------------------------------------------

def test_try_ha_conversation_action_done():
    def mock_poster(url, headers, payload, timeout_s):
        assert url == "http://198.51.100.10:8123/api/conversation/process"
        assert headers.get("Authorization") == "Bearer test-ha-secret"
        assert headers.get("Content-Type") == "application/json"
        assert payload == {"text": "turn on the kitchen light", "language": "en"}
        assert timeout_s == 2.5
        return {
            "response": {
                "speech": {
                    "plain": {
                        "speech": "Turned on the kitchen light",
                        "extra_data": None,
                    }
                },
                "card": {},
                "language": "en",
                "response_type": "action_done",
                "data": {
                    "targets": [],
                    "success": [{"name": "Kitchen Light", "type": "entity", "id": "light.kitchen"}],
                    "failed": [],
                },
            },
            "conversation_id": "test-conv-001",
        }

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "turn on the kitchen light",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        ha_token="test-ha-secret",
        http_poster=mock_poster,
    )
    assert matched is True
    assert speech == "Turned on the kitchen light"
    assert meta["reason"] == "action_done"
    assert meta["response_type"] == "action_done"
    assert meta["conversation_id"] == "test-conv-001"
    assert meta["data"]["success"][0]["id"] == "light.kitchen"


def test_try_ha_conversation_query_answer():
    def mock_poster(url, headers, payload, timeout_s):
        return {
            "response": {
                "speech": {
                    "plain": {
                        "speech": "The temperature is 72 degrees.",
                    }
                },
                "response_type": "query_answer",
                "data": {},
            },
            "conversation_id": "test-conv-002",
        }

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "what is the temperature?",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=mock_poster,
    )
    assert matched is True
    assert speech == "The temperature is 72 degrees."
    assert meta["response_type"] == "query_answer"


def test_try_ha_conversation_no_intent_matched():
    def mock_poster(url, headers, payload, timeout_s):
        return {
            "response": {
                "speech": {
                    "plain": {
                        "speech": "Sorry, I couldn't understand that",
                    }
                },
                "response_type": "error",
                "data": {
                    "code": "no_intent_matched",
                },
            },
            "conversation_id": None,
        }

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "tell me a bedtime story",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=mock_poster,
    )
    assert matched is False
    assert speech == ""
    assert meta["reason"] == "no_intent_matched"
    assert meta["code"] == "no_intent_matched"


def test_try_ha_conversation_other_error_codes():
    for code in ("intent_not_recognized", "failed_to_handle", "timer_not_supported"):
        def mock_poster(url, headers, payload, timeout_s):
            return {
                "response": {
                    "response_type": "error",
                    "data": {"code": code},
                }
            }

        matched, speech, meta = ha_fastpath.try_ha_conversation(
            "unknown command",
            enabled=True,
            ha_url="http://198.51.100.10:8123",
            http_poster=mock_poster,
        )
        assert matched is False
        assert speech == ""
        assert meta["reason"] == "no_intent_matched"
        assert meta["code"] == code


def test_try_ha_conversation_disabled_or_vision(monkeypatch):
    # 1. Disabled via parameter
    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "turn on light",
        enabled=False,
        ha_url="http://198.51.100.10:8123",
    )
    assert matched is False
    assert speech == ""
    assert meta == {"reason": "disabled_or_vision"}

    # 2. Disabled via default env (MVG_HA_ENABLED unset)
    monkeypatch.delenv("MVG_HA_ENABLED", raising=False)
    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "turn on light",
        ha_url="http://198.51.100.10:8123",
    )
    assert matched is False
    assert meta == {"reason": "disabled_or_vision"}

    # 3. Vision turn bypass: has_images=True even when enabled
    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "what do you see in front of the lamp?",
        has_images=True,
        enabled=True,
        ha_url="http://198.51.100.10:8123",
    )
    assert matched is False
    assert speech == ""
    assert meta == {"reason": "disabled_or_vision"}


def test_try_ha_conversation_empty_transcript_or_missing_url():
    # Empty transcript
    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
    )
    assert matched is False
    assert meta == {"reason": "empty_transcript"}

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "   ",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
    )
    assert matched is False
    assert meta == {"reason": "empty_transcript"}

    # Missing HA URL
    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "turn on fan",
        enabled=True,
        ha_url="",
    )
    assert matched is False
    assert meta == {"reason": "missing_ha_url"}


def test_try_ha_conversation_network_errors():
    # Connection error
    def connection_error_poster(url, headers, payload, timeout_s):
        raise urllib.error.URLError("Connection refused")

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "turn on lights",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=connection_error_poster,
    )
    assert matched is False
    assert speech == ""
    assert meta["reason"] == "network_error"
    assert "Connection refused" in meta["error"]

    # Timeout error
    def timeout_poster(url, headers, payload, timeout_s):
        raise TimeoutError("Socket timed out")

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "turn on lights",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=timeout_poster,
    )
    assert matched is False
    assert meta["reason"] == "network_error"
    assert "Socket timed out" in meta["error"]


def test_try_ha_conversation_invalid_payloads():
    # Malformed JSON
    def malformed_json_poster(url, headers, payload, timeout_s):
        return "{broken json"

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "turn on lights",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=malformed_json_poster,
    )
    assert matched is False
    assert meta["reason"] == "invalid_json"

    # Missing response object
    def missing_resp_poster(url, headers, payload, timeout_s):
        return {"other_field": 123}

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "turn on lights",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=missing_resp_poster,
    )
    assert matched is False
    assert meta["reason"] == "missing_response_object"

    # Empty speech
    def empty_speech_poster(url, headers, payload, timeout_s):
        return {"response": {"response_type": "action_done", "speech": {"plain": {"speech": ""}}}}

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "turn on lights",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=empty_speech_poster,
    )
    assert matched is False
    assert meta["reason"] == "empty_speech"


def test_try_ha_conversation_real_http_server():
    seen_requests = []

    class RealHAHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""
            seen_requests.append({
                "path": self.path,
                "headers": dict(self.headers),
                "payload": json.loads(body.decode("utf-8")),
            })
            resp = {
                "response": {
                    "speech": {"plain": {"speech": "Turned on real lamp."}},
                    "response_type": "action_done",
                    "data": {"targets": [], "success": [{"name": "Real Lamp"}]},
                },
                "conversation_id": "real-conv-999",
            }
            resp_bytes = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_bytes)))
            self.end_headers()
            self.wfile.write(resp_bytes)

        def log_message(self, *args):
            pass

    server = MockHAServer(RealHAHandler)
    server.start()
    try:
        matched, speech, meta = ha_fastpath.try_ha_conversation(
            "turn on lamp",
            enabled=True,
            ha_url=f"http://127.0.0.1:{server.port}",
            ha_token="token-abc-123",
        )
        assert matched is True
        assert speech == "Turned on real lamp."
        assert meta["response_type"] == "action_done"
        assert len(seen_requests) == 1
        req = seen_requests[0]
        assert req["path"] == "/api/conversation/process"
        assert req["headers"].get("Authorization") == "Bearer token-abc-123"
        assert req["payload"] == {"text": "turn on lamp", "language": "en"}
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Unit tests for gateway._intent wiring
# ---------------------------------------------------------------------------

def test_gateway_intent_default_disabled(monkeypatch):
    monkeypatch.setattr(gateway, "HA_ENABLED", False)
    monkeypatch.delenv("MVG_HA_ENABLED", raising=False)
    monkeypatch.setattr(gateway, "_try_intent", lambda text: None)

    res = gateway._intent("turn on living room light")
    assert res["hit"] is False
    assert res["intent"] is None
    assert res["reason"] == "miss"


def test_gateway_intent_ha_action_done_hit(monkeypatch):
    monkeypatch.setattr(gateway, "HA_ENABLED", True)
    monkeypatch.setenv("MVG_HA_ENABLED", "1")
    monkeypatch.setattr(gateway, "_try_intent", lambda text: None)

    def mock_ha_conv(transcript, has_images=False, **kwargs):
        return True, "Turned on desk lamp.", {"reason": "action_done", "response_type": "action_done"}

    monkeypatch.setattr(gateway, "try_ha_conversation", mock_ha_conv)

    res = gateway._intent("turn on desk lamp")
    assert res["hit"] is True
    assert res["intent"] == "ha_conversation"
    assert res["reply"] == "Turned on desk lamp."
    assert res["reason"] == "action_done"
    assert res["match_ms"] >= 0.0


def test_gateway_intent_ha_no_intent_miss(monkeypatch):
    monkeypatch.setattr(gateway, "HA_ENABLED", True)
    monkeypatch.setenv("MVG_HA_ENABLED", "1")
    monkeypatch.setattr(gateway, "_try_intent", lambda text: None)

    def mock_ha_conv(transcript, has_images=False, **kwargs):
        return False, "", {"reason": "no_intent_matched", "code": "no_intent_matched"}

    monkeypatch.setattr(gateway, "try_ha_conversation", mock_ha_conv)

    res = gateway._intent("write a python script")
    assert res["hit"] is False
    assert res["intent"] is None
    assert res["reason"] == "no_intent_matched"


def test_gateway_intent_vision_bypass(monkeypatch):
    monkeypatch.setattr(gateway, "HA_ENABLED", True)
    monkeypatch.setenv("MVG_HA_ENABLED", "1")
    monkeypatch.setattr(gateway, "_try_intent", lambda text: None)

    called_has_images = []

    def mock_ha_conv(transcript, has_images=False, **kwargs):
        called_has_images.append(has_images)
        return False, "", {"reason": "disabled_or_vision"}

    monkeypatch.setattr(gateway, "try_ha_conversation", mock_ha_conv)

    res = gateway._intent("what is on the desk?", has_images=True)
    assert res["hit"] is False
    assert res["reason"] == "disabled_or_vision"
    assert called_has_images == [True]


def test_gateway_intent_plugin_precedence(monkeypatch):
    # If the household plugin answers, HA is NOT invoked
    monkeypatch.setattr(gateway, "HA_ENABLED", True)
    monkeypatch.setenv("MVG_HA_ENABLED", "1")
    monkeypatch.setattr(
        gateway,
        "_try_intent",
        lambda text: {"hit": True, "intent": "custom_plugin_intent", "reply": "plugin response", "match_ms": 1.0, "handler_ms": 1.0, "reason": "plugin"},
    )

    ha_called = []
    monkeypatch.setattr(gateway, "try_ha_conversation", lambda *a, **k: ha_called.append(True))

    res = gateway._intent("custom trigger")
    assert res["hit"] is True
    assert res["intent"] == "custom_plugin_intent"
    assert res["reply"] == "plugin response"
    assert len(ha_called) == 0


def test_gateway_intent_exception_resilience(monkeypatch):
    monkeypatch.setattr(gateway, "HA_ENABLED", True)
    monkeypatch.setenv("MVG_HA_ENABLED", "1")
    monkeypatch.setattr(gateway, "_try_intent", lambda text: None)

    def raising_ha_conv(*args, **kwargs):
        raise RuntimeError("simulated crash inside HA client")

    monkeypatch.setattr(gateway, "try_ha_conversation", raising_ha_conv)

    res = gateway._intent("turn on lights")
    assert res["hit"] is False
    assert res["reason"] == "ha_error:RuntimeError"


# ---------------------------------------------------------------------------
# Unit tests for Chunk 3 Review Fixes (Amos findings)
# ---------------------------------------------------------------------------

def test_gateway_intent_plugin_hit_false_fallback_to_ha(monkeypatch):
    """When a plugin returns {'hit': False}, it is treated as a miss so Home Assistant is consulted."""
    monkeypatch.setattr(gateway, "HA_ENABLED", True)
    monkeypatch.setenv("MVG_HA_ENABLED", "1")

    # 1. Plugin answers {'hit': False} -> HA should be consulted and hit
    monkeypatch.setattr(
        gateway,
        "_try_intent",
        lambda text: {"hit": False, "intent": "plugin_intent", "reason": "plugin_declined"},
    )
    monkeypatch.setattr(
        gateway,
        "try_ha_conversation",
        lambda transcript, has_images=False, **kwargs: (
            True,
            "Turned on fan.",
            {"reason": "action_done", "response_type": "action_done"},
        ),
    )

    res = gateway._intent("turn on fan")
    assert res["hit"] is True
    assert res["intent"] == "ha_conversation"
    assert res["reply"] == "Turned on fan."
    assert res["reason"] == "action_done"

    # 2. Plugin answers {'hit': False} with HA disabled -> plugin's miss dictionary is returned
    monkeypatch.setattr(gateway, "HA_ENABLED", False)
    monkeypatch.delenv("MVG_HA_ENABLED", raising=False)
    monkeypatch.delenv("MVG_HA_FASTPATH", raising=False)

    res_disabled = gateway._intent("turn on fan")
    assert res_disabled["hit"] is False
    assert res_disabled["reason"] == "plugin_declined"


def test_try_ha_conversation_cooldown_circuit_breaker():
    """When HA times out, circuit breaker opens and short-circuits subsequent calls."""
    call_count = 0

    def timeout_poster(url, headers, payload, timeout_s=2.5):
        nonlocal call_count
        call_count += 1
        raise TimeoutError("HA timed out")

    # Call 1: times out -> returns network_error, trips circuit
    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "turn on lamp",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        cooldown_s=10.0,
        http_poster=timeout_poster,
    )
    assert matched is False
    assert meta["reason"] == "network_error"
    assert "HA timed out" in meta["error"]
    assert call_count == 1

    # Call 2: in cooldown -> immediate circuit_open without invoking poster
    matched2, speech2, meta2 = ha_fastpath.try_ha_conversation(
        "turn on lamp",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=timeout_poster,
    )
    assert matched2 is False
    assert meta2["reason"] == "circuit_open"
    assert meta2["cooldown_remaining"] > 0
    assert call_count == 1  # poster was NOT called

    # Reset circuit breaker -> poster is invoked again
    ha_fastpath.reset_circuit_breaker()
    matched3, speech3, meta3 = ha_fastpath.try_ha_conversation(
        "turn on lamp",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=timeout_poster,
    )
    assert matched3 is False
    assert meta3["reason"] == "network_error"
    assert call_count == 2

    # A successful call clears the circuit breaker
    ha_fastpath.reset_circuit_breaker()

    def success_poster(url, headers, payload, timeout_s=2.5):
        return {
            "response": {
                "speech": {"plain": {"speech": "All clear."}},
                "response_type": "action_done",
                "data": {},
            }
        }

    matched4, speech4, meta4 = ha_fastpath.try_ha_conversation(
        "status check",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=success_poster,
    )
    assert matched4 is True
    assert speech4 == "All clear."
    # Circuit should be closed
    assert ha_fastpath._circuit_open_until == 0.0


def test_try_ha_conversation_inspect_signature():
    """Poster signatures are checked upfront without relying on TypeError retries."""
    # 3-arg poster: signature (url, headers, payload)
    called_3arg = []

    def poster_3arg(url, headers, payload):
        called_3arg.append((url, headers, payload))
        return {
            "response": {
                "speech": {"plain": {"speech": "3-arg success"}},
                "response_type": "action_done",
                "data": {},
            }
        }

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "test 3-arg",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=poster_3arg,
    )
    assert matched is True
    assert speech == "3-arg success"
    assert len(called_3arg) == 1

    # 4-arg poster: signature (url, headers, payload, timeout_s)
    called_4arg = []

    def poster_4arg(url, headers, payload, timeout_s):
        called_4arg.append((url, headers, payload, timeout_s))
        return {
            "response": {
                "speech": {"plain": {"speech": "4-arg success"}},
                "response_type": "action_done",
                "data": {},
            }
        }

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "test 4-arg",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        timeout_s=4.5,
        http_poster=poster_4arg,
    )
    assert matched is True
    assert speech == "4-arg success"
    assert len(called_4arg) == 1
    assert called_4arg[0][3] == 4.5

    # Poster that internally raises TypeError (e.g. bug inside poster)
    def broken_poster(url, headers, payload, timeout_s):
        raise TypeError("internal bug inside poster")

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "test broken poster",
        enabled=True,
        ha_url="http://198.51.100.10:8123",
        http_poster=broken_poster,
    )
    assert matched is False
    assert meta["reason"] == "network_error"
    assert "internal bug inside poster" in meta["error"]


def test_try_ha_conversation_mvg_ha_fastpath_env(monkeypatch):
    """MVG_HA_FASTPATH alias is supported alongside MVG_HA_ENABLED."""
    monkeypatch.delenv("MVG_HA_ENABLED", raising=False)
    monkeypatch.setenv("MVG_HA_FASTPATH", "1")

    assert ha_fastpath.is_ha_fastpath_enabled() is True

    def mock_poster(url, headers, payload, timeout_s=2.5):
        return {
            "response": {
                "speech": {"plain": {"speech": "Fastpath alias works"}},
                "response_type": "action_done",
                "data": {},
            }
        }

    matched, speech, meta = ha_fastpath.try_ha_conversation(
        "test fastpath alias",
        ha_url="http://198.51.100.10:8123",
        http_poster=mock_poster,
    )
    assert matched is True
    assert speech == "Fastpath alias works"


def test_post_intent_auth_enforced(server, monkeypatch):
    """POST /intent requires authorization when MVG_HA_INTENT_TOKEN is configured."""
    monkeypatch.setattr(gateway, "HA_ENABLED", True)
    monkeypatch.setenv("MVG_HA_ENABLED", "1")
    monkeypatch.setenv("MVG_HA_INTENT_TOKEN", "super-secret-ha-token")
    monkeypatch.setattr(gateway, "_try_intent", lambda text: None)

    ha_invoked = []

    def mock_ha_conv(transcript, has_images=False, **kwargs):
        ha_invoked.append(transcript)
        return True, "Turned on bedroom light.", {"reason": "action_done", "response_type": "action_done"}

    monkeypatch.setattr(gateway, "try_ha_conversation", mock_ha_conv)

    port = server

    def _post_intent(body_dict, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port)
        body = json.dumps(body_dict).encode("utf-8")
        req_headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        if headers:
            req_headers.update(headers)
        conn.request("POST", "/intent", body=body, headers=req_headers)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8")

    # 1. Unauthenticated request -> 401, HA not called
    status, body = _post_intent({"text": "turn on bedroom light", "node": "bed_node"})
    assert status == 401
    assert "unauthorized" in body
    assert len(ha_invoked) == 0

    # 2. Invalid Bearer token -> 401, HA not called
    status, body = _post_intent(
        {"text": "turn on bedroom light", "node": "bed_node"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert status == 401
    assert "unauthorized" in body
    assert len(ha_invoked) == 0

    # 3. Valid Bearer token -> 200, HA executed
    status, body = _post_intent(
        {"text": "turn on bedroom light", "node": "bed_node"},
        headers={"Authorization": "Bearer super-secret-ha-token"},
    )
    assert status == 200
    assert "Turned on bedroom light." in body
    assert len(ha_invoked) == 1

    # 4. Verified speaker provided -> 200, HA executed
    ha_invoked.clear()
    status, body = _post_intent(
        {"text": "turn on bedroom light", "node": "bed_node", "speaker": "alice"},
    )
    assert status == 200
    assert "Turned on bedroom light." in body
    assert len(ha_invoked) == 1

    # 5. MVG_DEVICE_TOKENS verification
    monkeypatch.setenv("MVG_DEVICE_TOKENS", "bed_node:dev_tok_123")
    ha_invoked.clear()
    status, body = _post_intent(
        {"text": "turn on bedroom light", "node": "bed_node"},
        headers={"Authorization": "Bearer dev_tok_123"},
    )
    assert status == 200
    assert len(ha_invoked) == 1

    # 6. Unset tokens -> open / unauthenticated allowed by default
    monkeypatch.delenv("MVG_HA_INTENT_TOKEN", raising=False)
    monkeypatch.delenv("MVG_DEVICE_TOKENS", raising=False)
    ha_invoked.clear()
    status, body = _post_intent({"text": "turn on bedroom light", "node": "bed_node"})
    assert status == 200
    assert len(ha_invoked) == 1
