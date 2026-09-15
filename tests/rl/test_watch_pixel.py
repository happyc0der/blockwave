"""The pixel viewer plays the policy in the dynamics it was trained in.

Two sinks share one session: a video, and a live window. The video's contract
is checked the same way the board-track recorder's is (frame count and rate
through ffprobe); the live loop is run under SDL's dummy driver so the event
handling and per-game tally execute without a display.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest
import torch

from blockwave_rl.agents.nets import PixelPolicy
from blockwave_rl.env.base import N_ACTIONS
from blockwave_rl.world.watch_pixel import play_live, record


@pytest.fixture
def checkpoint(tmp_path):
    """A random pixel policy with the config the viewer reads beside it."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"agent_hz": 5.0, "gravity": 4.0, "frame_stack": 4}))
    torch.manual_seed(0)
    ckpt = run / "ckpt_00001.pt"
    torch.save(PixelPolicy((4, 88, 88), N_ACTIONS).state_dict(), ckpt)
    return ckpt


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_video_has_one_frame_per_decision_plus_the_hold(checkpoint, tmp_path):
    out = tmp_path / "game.mp4"
    decisions = record(checkpoint, out, speed=2.0, profile="flat", seed=3, max_decisions=12)
    assert decisions == 12

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "json", str(out)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["r_frame_rate"] == "10/1", "5 Hz decisions at 2x speed"
    # First frame, one per decision, then the 1.5 s hold at 10 fps.
    assert int(stream["nb_read_frames"]) == 1 + 12 + 15


def test_live_loop_runs_to_its_cap_without_a_display(checkpoint, monkeypatch):
    """The window path, minus the window: same session, same per-decision step."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    # A huge speed so the real-time pacing does not make the test wait.
    decisions = play_live(checkpoint, speed=1e4, profile="flat", seed=3, max_decisions=25)
    assert decisions == 25
    assert os.environ["SDL_VIDEODRIVER"] == "dummy"
