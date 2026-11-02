"""tests/test_side_channel.py — Unit tests for side_channel.py and gateway parity.

Tests:
1. reply_expects_reply (whitespace, quotes, Unicode emojis, joiners, question marks).
2. forward_artifacts_to_webhook (URL / code block detection, webhook dispatch, Discord cueing).
3. emit_turn_transcript (JSONL file append, HTTP sink dispatch, error resilience).
4. gateway integration (SSE expects_reply payload and side-channel wiring).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import MagicMock

import pytest

import gateway
from side_channel import (
    emit_turn_transcript,
    forward_artifacts_to_webhook,
    reply_expects_reply,
)

# RFC 5737 documentation test addresses
DOC_WEBHOOK_URL = "http://198.51.100.10:8080/webhook"
DOC_SINK_URL = "http://198.51.100.20:8080/transcript-sink"


# ============================================================================
# 1. Tests for reply_expects_reply
# ============================================================================


class TestReplyExpectsReply:
    @pytest.mark.parametrize(
        "text",
        [
            "What time is it?",
            "How are you?",
            "Can you help me with this?",
            "Is it raining outside?",
            "Really!?",  # ends in ?
            "?",
        ],
    )
    def test_basic_questions_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "What time is it? ",
            "How are you?   \t\n",
            "Are you ready?\n\n",
            "   ?   ",
        ],
    )
    def test_trailing_whitespace_returns_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            'What time is it?"',
            "What time is it?'",
            '“What time is it?”',
            "‘What time is it?’",
            "What time is it?`",
            "What time is it?»",
            "«What time is it?»",
            'What time is it?"""',
            "What time is it?'''",
        ],
    )
    def test_trailing_quotes_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "What time is it? 🤔",
            "Shall we go? 🚀",
            "Do you agree? 👍",
            "Are you there? 👋",
            "Would you like that? ✨",
            "What do you think? 💭",
            "How is that? ☕️",
        ],
    )
    def test_trailing_emojis_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Any questions? 👨‍💻",  # ZWJ sequence
            "Can we meet tomorrow? 🤷‍♀️",  # ZWJ + gender + VS16
            "Is that cool? \u200d\ufe0f",
            "Ready? 🏃‍♂️",
        ],
    )
    def test_trailing_complex_emoji_joiners_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            '“What is your name?” 🤔 ',
            'How are you?" 😊  \n',
            "Are you coming?' 🚀✨",
            '“Ready?” 👋 \t',
        ],
    )
    def test_mixed_quotes_emojis_whitespace_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "What do you think❓",
            "Which one❔",
            "Are you sure❓ ",
            "Is this right❓ 🤔",
        ],
    )
    def test_unicode_question_ornaments_return_true(self, text: str):
        assert reply_expects_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "The weather is sunny.",
            "The weather is sunny. 😊",
            "Hello there! 👋",
            "Check this out: https://example.com/search?q=test",
            "Really?!",  # ends in !
            "Done.",
            "That's fantastic!",
            'He asked "Why is the sky blue."',
        ],
    )
    def test_non_questions_return_false(self, text: str):
        assert reply_expects_reply(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "\n\t",
            None,
            123,
            [],
            {},
        ],
    )
    def test_empty_or_invalid_inputs_return_false(self, text: Any):
        assert reply_expects_reply(text) is False


# ============================================================================
# 2. Tests for forward_artifacts_to_webhook
# ============================================================================


class TestForwardArtifactsToWebhook:
    def test_no_artifacts_returns_unchanged(self):
        raw = "The capital of France is Paris."
        spoken = "The capital of France is Paris."
        mock_poster = MagicMock()

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
        )

        assert res_spoken == spoken
        assert sent is False
        mock_poster.assert_not_called()

    def test_detects_http_and_https_urls(self):
        raw = "You can read the docs at https://example.com/docs for more details."
        spoken = "You can read the docs for more details."
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
        )

        assert sent is True
        assert res_spoken == "You can read the docs for more details. I sent the link to Discord."
        mock_poster.assert_called_once()
        args, _ = mock_poster.call_args
        assert args[0] == DOC_WEBHOOK_URL
        assert args[1]["content"] == raw.strip()

    def test_detects_fenced_code_blocks(self):
        raw = "Here is the implementation:\n```python\ndef hello():\n    return 'world'\n```"
        spoken = "Here is the implementation."
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
        )

        assert sent is True
        assert res_spoken == "Here is the implementation. I sent the link to Discord."
        mock_poster.assert_called_once()
        assert mock_poster.call_args[0][0] == DOC_WEBHOOK_URL

    def test_uses_mvg_artifact_webhook_url_env_var(self, monkeypatch):
        monkeypatch.setenv("MVG_ARTIFACT_WEBHOOK_URL", DOC_WEBHOOK_URL)
        raw = "See https://github.com/example/repo"
        spoken = "See the GitHub repository."
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url="",
            http_poster=mock_poster,
        )

        assert sent is True
        assert res_spoken == "See the GitHub repository. I sent the link to Discord."
        assert mock_poster.call_args[0][0] == DOC_WEBHOOK_URL

    def test_missing_webhook_url_returns_false_and_does_not_post(self, monkeypatch):
        monkeypatch.delenv("MVG_ARTIFACT_WEBHOOK_URL", raising=False)
        raw = "Visit https://example.com"
        spoken = "Visit the website."
        mock_poster = MagicMock()

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url="",
            http_poster=mock_poster,
        )

        assert sent is False
        assert res_spoken == spoken
        mock_poster.assert_not_called()

    @pytest.mark.parametrize(
        "already_cued_spoken",
        [
            "I sent the link to Discord for you.",
            "I already sent that over to Discord.",
            "Check Discord for the full link.",
            "I posted the link in Discord.",
            "I dropped it in Discord.",
            "I sent it to discord chat.",
        ],
    )
    def test_does_not_duplicate_cue_if_already_cued(self, already_cued_spoken: str):
        raw = "Check out https://example.com"
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=already_cued_spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
        )

        assert sent is True
        # Should not append a second cue
        assert res_spoken == already_cued_spoken
        mock_poster.assert_called_once()

    def test_empty_spoken_reply_gets_cue(self):
        raw = "https://example.com"
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply="",
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
        )

        assert sent is True
        assert res_spoken == "I sent the link to Discord."

    def test_adds_period_to_spoken_reply_if_missing(self):
        raw = "Here is the site https://example.com"
        spoken = "Here is the site"
        mock_poster = MagicMock(return_value=True)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
        )

        assert sent is True
        assert res_spoken == "Here is the site. I sent the link to Discord."

    def test_poster_failure_returns_false_and_unmodified_spoken(self):
        raw = "https://example.com"
        spoken = "Here is the link."
        mock_poster = MagicMock(return_value=False)

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
        )

        assert sent is False
        assert res_spoken == spoken

    def test_poster_exception_returns_false_and_unmodified_spoken(self):
        raw = "https://example.com"
        spoken = "Here is the link."
        mock_poster = MagicMock(side_effect=ConnectionError("Host unreachable"))

        res_spoken, sent = forward_artifacts_to_webhook(
            raw_reply=raw,
            spoken_reply=spoken,
            webhook_url=DOC_WEBHOOK_URL,
            http_poster=mock_poster,
        )

        assert sent is False
        assert res_spoken == spoken

    def test_real_http_server_post(self):
        received_requests = []

        class WebhookHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received_requests.append({
                    "path": self.path,
                    "content_type": self.headers.get("Content-Type"),
                    "payload": json.loads(body.decode("utf-8")),
                })
                self.send_response(204)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), WebhookHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{server.server_port}/api/webhooks/test"
            raw = "Look at https://example.com"
            spoken = "Look at the website"

            res_spoken, sent = forward_artifacts_to_webhook(
                raw_reply=raw,
                spoken_reply=spoken,
                webhook_url=url,
                http_poster=None,  # test real urllib path
            )

            assert sent is True
            assert res_spoken == "Look at the website. I sent the link to Discord."
            assert len(received_requests) == 1
            assert received_requests[0]["payload"]["content"] == raw
            assert "application/json" in received_requests[0]["content_type"]
        finally:
            server.shutdown()
            server.server_close()


# ============================================================================
# 3. Tests for emit_turn_transcript
# ============================================================================


class TestEmitTurnTranscript:
    def test_no_sink_configured_returns_false(self, monkeypatch):
        monkeypatch.delenv("MVG_TRANSCRIPT_SINK_URL", raising=False)
        monkeypatch.delenv("MVG_TRANSCRIPT_LOG_PATH", raising=False)

        res = emit_turn_transcript({"turn_id": "turn-1", "reply": "Hello"})
        assert res is False

    def test_invalid_turn_record_returns_false(self):
        assert emit_turn_transcript({}) is False
        assert emit_turn_transcript(None) is False  # type: ignore

    def test_log_path_appends_jsonl(self, tmp_path):
        log_file = tmp_path / "transcripts.jsonl"
        record_1 = {"turn_id": "turn-1", "prompt": "Hi", "reply": "Hello!"}
        record_2 = {"turn_id": "turn-2", "prompt": "What?", "reply": "Nothing."}

        res1 = emit_turn_transcript(record_1, log_path=str(log_file))
        res2 = emit_turn_transcript(record_2, log_path=str(log_file))

        assert res1 is True
        assert res2 is True
        assert log_file.exists()

        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == record_1
        assert json.loads(lines[1]) == record_2

    def test_log_path_creates_parent_directory(self, tmp_path):
        nested_log = tmp_path / "deep" / "nested" / "dir" / "transcripts.jsonl"
        record = {"turn_id": "t-1", "reply": "ok"}

        res = emit_turn_transcript(record, log_path=str(nested_log))
        assert res is True
        assert nested_log.exists()
        assert json.loads(nested_log.read_text(encoding="utf-8").strip()) == record

    def test_log_path_env_var(self, tmp_path, monkeypatch):
        log_file = tmp_path / "env_transcripts.jsonl"
        monkeypatch.setenv("MVG_TRANSCRIPT_LOG_PATH", str(log_file))
        record = {"turn_id": "env-turn", "reply": "from env"}

        res = emit_turn_transcript(record)
        assert res is True
        assert log_file.exists()
        assert json.loads(log_file.read_text(encoding="utf-8").strip()) == record

    def test_sink_url_dispatches_http_post(self):
        mock_poster = MagicMock(return_value=True)
        record = {"turn_id": "turn-http", "prompt": "Status", "reply": "All systems nominal."}

        res = emit_turn_transcript(record, sink_url=DOC_SINK_URL, http_poster=mock_poster)

        assert res is True
        mock_poster.assert_called_once()
        args, _ = mock_poster.call_args
        assert args[0] == DOC_SINK_URL
        assert args[1] == record

    def test_sink_url_env_var(self, monkeypatch):
        monkeypatch.setenv("MVG_TRANSCRIPT_SINK_URL", DOC_SINK_URL)
        mock_poster = MagicMock(return_value=True)
        record = {"turn_id": "env-sink", "reply": "sink-env"}

        res = emit_turn_transcript(record, http_poster=mock_poster)

        assert res is True
        mock_poster.assert_called_once()
        assert mock_poster.call_args[0][0] == DOC_SINK_URL

    def test_both_log_path_and_sink_url(self, tmp_path):
        log_file = tmp_path / "both.jsonl"
        mock_poster = MagicMock(return_value=True)
        record = {"turn_id": "both-1", "reply": "both"}

        res = emit_turn_transcript(
            record,
            sink_url=DOC_SINK_URL,
            log_path=str(log_file),
            http_poster=mock_poster,
        )

        assert res is True
        assert log_file.exists()
        assert json.loads(log_file.read_text(encoding="utf-8").strip()) == record
        mock_poster.assert_called_once()

    def test_sink_failure_with_log_path_still_succeeds(self, tmp_path):
        log_file = tmp_path / "partial.jsonl"
        mock_poster = MagicMock(side_effect=Exception("HTTP connection failed"))
        record = {"turn_id": "partial-1", "reply": "ok"}

        res = emit_turn_transcript(
            record,
            sink_url=DOC_SINK_URL,
            log_path=str(log_file),
            http_poster=mock_poster,
        )

        # File write succeeded, so overall return is True
        assert res is True
        assert log_file.exists()

    def test_sink_failure_without_log_path_returns_false(self):
        mock_poster = MagicMock(return_value=False)
        record = {"turn_id": "fail-1", "reply": "fail"}

        res = emit_turn_transcript(record, sink_url=DOC_SINK_URL, http_poster=mock_poster)
        assert res is False

    def test_real_http_sink_server(self):
        received_records = []

        class SinkHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                received_records.append(json.loads(body.decode("utf-8")))
                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), SinkHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            url = f"http://127.0.0.1:{server.server_port}/transcripts"
            record = {"turn_id": "real-server-1", "prompt": "Hello", "reply": "World"}

            res = emit_turn_transcript(record, sink_url=url, http_poster=None)

            assert res is True
            assert len(received_records) == 1
            assert received_records[0] == record
        finally:
            server.shutdown()
            server.server_close()


# ============================================================================
# 4. Tests for gateway integration
# ============================================================================


class TestGatewayIntegration:
    def test_record_reply_preserves_ambient_state(self):
        gateway._ambient_reset_state()
        gateway.record_reply("node-42", "Are you ready?")

        with gateway._ambient_lock:
            assert "node-42" in gateway._ambient_last_reply
            ts, reply = gateway._ambient_last_reply["node-42"]
            assert reply == "Are you ready?"
            assert ts > 0

    def test_reply_expects_reply_integration(self):
        assert gateway.reply_expects_reply("Would you like me to continue? 🤔") is True
        assert gateway.reply_expects_reply("I have finished the job.") is False

    def test_side_channel_wiring_in_gateway(self, tmp_path, monkeypatch):
        log_file = tmp_path / "gateway_transcripts.jsonl"
        monkeypatch.setenv("MVG_TRANSCRIPT_LOG_PATH", str(log_file))

        record = {
            "kind": "turn",
            "turn_id": "gw-turn-1",
            "node_id": "test-node",
            "prompt": "What is the weather?",
            "reply": "It is sunny today.",
        }

        success = gateway.emit_turn_transcript(record)
        assert success is True
        assert log_file.exists()

        data = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert data["turn_id"] == "gw-turn-1"
        assert data["reply"] == "It is sunny today."

    def test_handle_ask_with_artifact_webhook_and_transcript_log(self, tmp_path, monkeypatch):
        received_webhooks = []

        class MockWebhook(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                received_webhooks.append(json.loads(self.rfile.read(content_length).decode("utf-8")))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        wh_server = HTTPServer(("127.0.0.1", 0), MockWebhook)
        threading.Thread(target=wh_server.serve_forever, daemon=True).start()
        wh_port = wh_server.server_port

        log_path = tmp_path / "ask_transcript.jsonl"
        monkeypatch.setenv("MVG_ARTIFACT_WEBHOOK_URL", f"http://127.0.0.1:{wh_port}/hook")
        monkeypatch.setenv("MVG_TRANSCRIPT_LOG_PATH", str(log_path))
        monkeypatch.setattr(gateway, "BRAIN_CALL", lambda *a, **k: "Here is https://example.com/doc for details.")

        srv = gateway.Server(("127.0.0.1", 0), gateway.GatewayHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]

        try:
            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            body = json.dumps({"prompt": "Documentation", "node_id": "desk"}).encode("utf-8")
            conn.request("POST", "/ask", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            assert resp.status == 200
            _ = resp.read().decode("utf-8")
            conn.close()

            # Verify artifact forwarded
            assert len(received_webhooks) == 1
            assert "https://example.com/doc" in received_webhooks[0]["content"]

            # Verify transcript logged
            assert log_path.exists()
            log_entry = json.loads(log_path.read_text(encoding="utf-8").strip())
            assert log_entry["endpoint"] == "/ask"
            assert log_entry["node_id"] == "desk"
            assert "I sent the link to Discord." in log_entry["reply"]
        finally:
            srv.shutdown()
            srv.server_close()
            wh_server.shutdown()
            wh_server.server_close()
