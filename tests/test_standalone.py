"""The gateway imports and runs with no plugin: the HTTP brain is the default,
and every default path sits under MVG_HOME."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

GATEWAY_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(GATEWAY_DIR))
import gateway  # noqa: E402


def _clean_env(**extra):
    """A subprocess environment with no MVG_* settings from the developer's shell."""
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(GATEWAY_DIR)}
    env.update(extra)
    return env


def _run_python(code, env, cwd=GATEWAY_DIR):
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=120)


def test_imports_without_plugin(tmp_path):
    env = _clean_env(MVG_LOG_DIR=str(tmp_path), MVG_HOME=str(tmp_path))
    r = _run_python("import gateway; print(gateway.BRAIN_CALL.__name__, gateway._plugin)", env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "http_brain_call None"


def test_default_paths_live_under_mvg_home(tmp_path):
    home = tmp_path / "home"
    env = _clean_env(MVG_LOG_DIR=str(tmp_path), MVG_HOME=str(home))
    code = (
        "import json, gateway as g\n"
        "print(json.dumps([str(g.NODES_PATH), str(g.NODE_TARGETS_PATH), str(g.PIPER_CONFIG_PATH),\n"
        "                  str(g.WATCH_TOKENS_PATH), str(g.SPEAKERS_PATH), g.DEFAULT_TARGET]))\n"
    )
    r = _run_python(code, env)
    assert r.returncode == 0, r.stderr
    nodes, targets, piper, tokens, speakers, default_target = json.loads(r.stdout.strip().splitlines()[-1])
    assert nodes == str(home / "nodes.json")
    assert targets == str(home / "node-targets.json")
    assert piper == str(home / "piper-tts.json")
    assert tokens == str(home / "secrets" / "gotchi-watch-tokens.json")
    assert speakers == str(home / "speakers.json")
    assert default_target == "agent"


def test_nodes_file_env_overrides_mvg_home(tmp_path):
    env = _clean_env(MVG_LOG_DIR=str(tmp_path), MVG_HOME=str(tmp_path),
                     MVG_NODES_FILE=str(tmp_path / "elsewhere.json"))
    r = _run_python("import gateway as g; print(g.NODES_PATH)", env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == str(tmp_path / "elsewhere.json")


def test_http_brain_posts_framed_prompt(monkeypatch):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen.append(json.loads(body))
            out = json.dumps({"reply": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(gateway, "MVG_BRAIN_URL", f"http://127.0.0.1:{server.server_port}/brain")
        monkeypatch.setattr(gateway, "MVG_BRAIN_TOKEN", "")
        reply = gateway.http_brain_call("what time is it", "sess-1", "lammen", "", None, 5)
    finally:
        server.shutdown()
        server.server_close()

    assert reply == "ok"
    assert len(seen) == 1
    assert seen[0]["prompt"].startswith("[VOICE via node")
    assert seen[0]["session_id"] == "sess-1"
    assert seen[0]["node_id"] == "lammen"


def test_http_brain_without_url_raises(monkeypatch):
    monkeypatch.setattr(gateway, "MVG_BRAIN_URL", "")
    with pytest.raises(RuntimeError, match="MVG_BRAIN_URL not set"):
        gateway.http_brain_call("hi", "sess-1", "lammen", "", None, 5)


def test_plugin_brain_without_plugin_raises(monkeypatch):
    monkeypatch.setattr(gateway, "_plugin_brain_call", None)
    with pytest.raises(RuntimeError, match="no plugin brain"):
        gateway.plugin_brain_call("hi", "sess-1", "lammen", "", None, 5)
