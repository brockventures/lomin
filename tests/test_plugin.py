"""The plugin loader: MVG_PLUGIN names an importable module that supplies the
household parts. Each case runs gateway.py in a subprocess with a clean env."""
import json
import os
import subprocess
import sys
from pathlib import Path

GATEWAY_DIR = Path(__file__).resolve().parents[1]

FAKE_PLUGIN = '''
def apply_defaults():
    import os
    os.environ.setdefault("FAKE_PLUGIN_DEFAULT", "applied")

def brain_call(target, prompt, timeout):
    return f"plugin:{target}:{timeout}:{prompt[:12]}"

def try_intent(text):
    return {"intent": "lights", "text": text} if "lights" in text else None
'''


def _run(code, tmp_path, **env):
    base = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(GATEWAY_DIR)]),
        "MVG_LOG_DIR": str(tmp_path / "logs"),
        "MVG_HOME": str(tmp_path / "home"),
        "MVG_NODES_FILE": str(tmp_path / "absent-nodes.json"),
    }
    base.update(env)
    return subprocess.run([sys.executable, "-c", code], cwd=GATEWAY_DIR, env=base,
                          capture_output=True, text=True, timeout=120)


def test_named_plugin_supplies_brain_defaults_and_intents(tmp_path):
    (tmp_path / "fake_lomin_plugin.py").write_text(FAKE_PLUGIN)
    code = (
        "import json, os, gateway as g\n"
        "print(json.dumps({\n"
        "  'brain': g.BRAIN_CALL.__name__, 'mode': g.MVG_BRAIN,\n"
        "  'defaults': os.environ.get('FAKE_PLUGIN_DEFAULT'),\n"
        "  'reply': g.plugin_brain_call('hi', 's', 'kitchen-speaker', '', None, 7),\n"
        "  'intent': g._try_intent('lights on'),\n"
        "  'miss': g._try_intent('hello'),\n"
        "}))\n"
    )
    r = _run(code, tmp_path, MVG_PLUGIN="fake_lomin_plugin")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["brain"] == "plugin_brain_call" and out["mode"] == "plugin"
    assert out["defaults"] == "applied"
    assert out["reply"].startswith("plugin:agent:7:")
    assert out["intent"] == {"intent": "lights", "text": "lights on"}
    assert out["miss"] is None


def test_plugin_brain_stream_falls_back_to_one_partial(tmp_path):
    (tmp_path / "fake_lomin_plugin.py").write_text(FAKE_PLUGIN)
    code = (
        "import json, gateway as g\n"
        "seen = []\n"
        "reply = g.STREAM_BRAIN_CALL('hi', 's', 'n', '', None, 7, seen.append)\n"
        "print(json.dumps({'reply': reply, 'partials': seen}))\n"
    )
    r = _run(code, tmp_path, MVG_PLUGIN="fake_lomin_plugin")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["reply"].startswith("plugin:agent:7:")
    assert out["partials"] == [out["reply"]]


def test_missing_plugin_warns_and_runs_http(tmp_path):
    r = _run("import gateway; print(gateway.BRAIN_CALL.__name__)", tmp_path,
             MVG_PLUGIN="no_such_lomin_plugin_xyz")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "http_brain_call"
    assert "plugin 'no_such_lomin_plugin_xyz' failed to load" in r.stderr


def test_no_plugin_means_no_fast_path(tmp_path):
    r = _run("import gateway; print(gateway._try_intent('lights on'))", tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "None"


def test_mvg_brain_http_overrides_a_loaded_plugin(tmp_path):
    (tmp_path / "fake_lomin_plugin.py").write_text(FAKE_PLUGIN)
    r = _run("import gateway; print(gateway.BRAIN_CALL.__name__)", tmp_path,
             MVG_PLUGIN="fake_lomin_plugin", MVG_BRAIN="http")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "http_brain_call"


def test_empty_name_defaults_are_off(tmp_path):
    code = (
        "import json, gateway as g\n"
        "print(json.dumps({'names': list(g.AGENT_NAME_VARIANTS),\n"
        "  'mentioned': g._name_mentioned('hey Amos'),\n"
        "  'wake': g.WAKE_PHRASE_RE.match('hey Amos what time') is not None,\n"
        "  'phrase': g._variants_phrase(), 'agent': g.AGENT_NAME}))\n"
    )
    r = _run(code, tmp_path)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"names": [], "mentioned": False, "wake": False,
                   "phrase": "", "agent": "the assistant"}


def test_one_name_phrase_and_no_wake_names(tmp_path):
    code = (
        "import json, gateway as g\n"
        "print(json.dumps({'phrase': g._variants_phrase(),\n"
        "  'wake': g.WAKE_PHRASE_RE.match('hey Robin, ok') is not None,\n"
        "  'mentioned': g._name_mentioned('Robin, ok')}))\n"
    )
    r = _run(code, tmp_path, MVG_AGENT_NAMES="Robin", MVG_WAKE_NAMES="")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"phrase": "Robin", "wake": False, "mentioned": True}


def test_wake_names_default_to_agent_names(tmp_path):
    code = (
        "import json, gateway as g\n"
        "print(json.dumps({'wake': g.WAKE_PHRASE_RE.match('hey Robin, what time') is not None,\n"
        "  'other': g.WAKE_PHRASE_RE.match('hey Sam, what time') is not None}))\n"
    )
    r = _run(code, tmp_path, MVG_AGENT_NAMES="Robin")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out == {"wake": True, "other": False}
