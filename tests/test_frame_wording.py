"""MVG_OWNER_NAME and MVG_UNKNOWN_SPEAKER let a site keep its own frame wording."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNIPPET = ("import gateway as g; "
           "c = g.make_frame_ctx({'status': 'not_verified', 'id': '', 'score': 0.2}); "
           "print(g.frame_voice_prompt('hi', 'n1', c))")


def _frame(extra_env):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MVG_")}
    env.update({"MVG_LOG_DIR": "/tmp", **extra_env})
    out = subprocess.run([sys.executable, "-c", SNIPPET], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout


def test_defaults_are_neutral():
    f = _frame({})
    assert "speaker: unknown (answer for everyone present)" in f
    assert "say the owner must confirm it in text" in f


def test_site_wording_is_used():
    f = _frame({"MVG_OWNER_NAME": "Mike", "MVG_UNKNOWN_SPEAKER": "unknown (answer as the household)"})
    assert "speaker: unknown (answer as the household)" in f
    assert "say Mike must confirm it in text" in f
    assert "the owner" not in f
