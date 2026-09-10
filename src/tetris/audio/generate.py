"""Render every sound to ``assets/sfx``.

Run once via ``tetris gen-assets``; the game loads the wavs from then on. Kept
separate from the synth so that generating assets never drags the playback path
into a build step, or the reverse.
"""

from __future__ import annotations

from pathlib import Path

from . import synth
from .bank import default_sfx_dir


def generate(directory: Path | None = None, *, music: bool = True, quiet: bool = False) -> list[Path]:
    """Write every effect (and optionally the music loops). Returns the files."""
    directory = directory or default_sfx_dir()
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, samples in synth.build_sounds().items():
        path = directory / f"{name}.wav"
        synth.write_wav(path, samples)
        written.append(path)
        if not quiet:
            seconds = len(samples) / synth.SAMPLE_RATE
            print(f"  {name:<16} {seconds * 1000:6.0f} ms")

    if music:
        for band in range(len(synth.TEMPO_BANDS)):
            path = directory / f"music_{band}.wav"
            samples = synth.build_music(band)
            synth.write_wav(path, samples)
            written.append(path)
            if not quiet:
                seconds = len(samples) / synth.SAMPLE_RATE
                bpm = synth.BASE_BPM * synth.TEMPO_BANDS[band]
                print(f"  music_{band:<10} {seconds:6.2f} s   {bpm:.0f} bpm")

    return written
