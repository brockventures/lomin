"""A site keeps its own labels, keys, log name and classifier wording (MVG_AGENT_LABEL,
MVG_CLASSIFY_KEY, MVG_LOG_NAME, plugin AMBIENT_LAYA_QUESTION / VOICE_ADDRESSED_INSTRUCTIONS)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(code, env_extra, tmp_path=None, plugin_src=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MVG_")}
    env["MVG_LOG_DIR"] = str(tmp_path or "/tmp")
    env.update(env_extra)
    pypath = [str(ROOT)]
    if plugin_src is not None:
        (tmp_path / "siteplug.py").write_text(plugin_src)
        pypath.insert(0, str(tmp_path))
        env["MVG_PLUGIN"] = "siteplug"
    env["PYTHONPATH"] = os.pathsep.join(pypath)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_defaults(tmp_path):
    out = json.loads(_run("import gateway as g, json; print(json.dumps([g.AGENT_LABEL, g.CLASSIFY_KEY, g.LOG_NAME]))",
                          {}, tmp_path))
    assert out == ["agent", "lomin.ambient", "lomin.jsonl"]


def test_env_overrides_and_reply_label(tmp_path):
    code = ("import gateway as g, json; g.record_reply('n1', 'hello there'); "
            "print(json.dumps([g.CLASSIFY_KEY, g.LOG_NAME, g._ambient_context_lines('n1'), g._conversation_open_note('n1')]))")
    key, log, lines, note = json.loads(_run(code, {"MVG_AGENT_LABEL": "lomin", "MVG_AGENT_NAME": "Lomin",
                                                   "MVG_CLASSIFY_KEY": "site.ambient", "MVG_LOG_NAME": "site.jsonl"},
                                            tmp_path))
    assert key == "site.ambient" and log == "site.jsonl"
    assert lines and lines[-1].startswith("lomin: ")
    assert note.startswith("Lomin replied on this device") and "Lomin's last reply was" in note


def test_log_name_is_the_file_written(tmp_path):
    _run("import gateway as g; g.log_request('/probe', 1.0, 'ok')", {"MVG_LOG_NAME": "site.jsonl"}, tmp_path)
    assert (tmp_path / "site.jsonl").exists() and not (tmp_path / "lomin.jsonl").exists()


def test_plugin_supplies_classifier_wording(tmp_path):
    src = 'AMBIENT_LAYA_QUESTION = "SITE QUESTION"\nVOICE_ADDRESSED_INSTRUCTIONS = "SITE INSTRUCTIONS"\n'
    out = json.loads(_run("import gateway as g, json; print(json.dumps([g.AMBIENT_LAYA_QUESTION, g.VOICE_ADDRESSED_INSTRUCTIONS]))",
                          {}, tmp_path, plugin_src=src))
    assert out == ["SITE QUESTION", "SITE INSTRUCTIONS"]
