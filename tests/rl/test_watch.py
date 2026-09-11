"""The watch tool plays the policy in the dynamics it was trained in."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest
import torch

from blockwave_rl.agents.nets import BoardPolicy
from blockwave_rl.watch import record

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_writes_a_video_of_the_requested_length(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"horizon": 2, "agent_hz": 5.0, "gravity": 4.0}))
    torch.manual_seed(0)
    ckpt = run / "ckpt_00001.pt"
    torch.save(BoardPolicy((2, 24, 10), 14, 8).state_dict(), ckpt)

    out = tmp_path / "game.mp4"
    decisions = record(ckpt, out, speed=2.0, profile="flat", seed=3, max_decisions=12)
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
