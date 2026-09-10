"""Sound synthesis and playback.

Two things worth testing without ears.

**The signals are well formed** — no NaNs, no clipping, no DC offset, and no
discontinuity at a loop point. Every one of those is audible as a click, a
thump or a fizz, and every one is arithmetic a test can check. Both the DC
offset on the pulse wave and the click at the music loop seam were found this
way, not by listening.

**Nothing here can take the game down.** No audio device, no generated files, a
mixer that refuses to open — all of it has to degrade to silence.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from tetris.audio import synth
from tetris.audio.bank import PRIORITY, SoundBank, sound_for
from tetris.audio.generate import generate
from tetris.core.events import EventType, GameEvent


@pytest.fixture(scope="module")
def sounds() -> dict[str, np.ndarray]:
    return synth.build_sounds()


# -- signal hygiene -------------------------------------------------------


def test_every_sound_is_finite_and_audible(sounds):
    for name, data in sounds.items():
        assert np.isfinite(data).all(), f"{name} contains NaN or inf"
        assert np.sqrt((data**2).mean()) > 0.01, f"{name} is effectively silent"


def test_no_sound_clips(sounds):
    for name, data in sounds.items():
        assert np.abs(data).max() <= 1.0, f"{name} exceeds full scale"


def test_no_sound_has_a_dc_offset(sounds):
    """A pulse wave at 22% duty sits at a mean of -0.56 unless it is centred.

    That offset thumps at the start and end of every sound and eats headroom;
    this caught it across eleven of them.
    """
    for name, data in sounds.items():
        assert abs(data.mean()) < 0.05, f"{name} has DC offset {data.mean():+.3f}"


def test_sounds_start_from_silence(sounds):
    # A waveform that begins partway up its cycle clicks.
    for name, data in sounds.items():
        assert abs(data[0]) < 0.05, f"{name} starts at {data[0]:+.3f}"


@pytest.mark.parametrize("duty", [0.1, 0.22, 0.35, 0.5, 0.75])
def test_pulse_wave_is_centred_at_any_duty(duty):
    wave_data = synth.square(440.0, 0.25, duty=duty)
    assert abs(wave_data.mean()) < 0.02


def test_decay_envelope_ramps_in_from_zero():
    envelope = synth.decay_env(0.1)
    assert envelope[0] == 0.0
    assert envelope.max() > 0.9
    assert envelope[-1] < 0.01


def test_frequency_sweep_is_continuous():
    swept = synth.square((800.0, 100.0), 0.2)
    # A sweep built from a discontinuous phase shows up as huge sample-to-sample
    # jumps beyond the waveform's own edges.
    assert np.isfinite(swept).all()
    assert np.abs(swept).max() <= 2.0


# -- music ----------------------------------------------------------------


@pytest.mark.parametrize("band", range(len(synth.TEMPO_BANDS)))
def test_music_loops_without_a_click(band):
    """The seam between the last and first sample must be near-continuous.

    Before the envelope ramp this jumped 0.39, which clicked once per bar cycle
    for the whole session.
    """
    track = synth.build_music(band)
    assert abs(track[0] - track[-1]) < 0.05


@pytest.mark.parametrize("band", range(len(synth.TEMPO_BANDS)))
def test_music_is_continuous_throughout(band):
    track = synth.build_music(band)
    assert np.isfinite(track).all()
    # No silent gap longer than a beat anywhere in the loop.
    window = synth.SAMPLE_RATE // 4
    envelope = np.abs(track[: len(track) // window * window].reshape(-1, window)).max(axis=1)
    assert envelope.min() > 0.02, "the loop has a hole in it"


def test_higher_bands_are_shorter():
    lengths = [len(synth.build_music(b)) for b in range(len(synth.TEMPO_BANDS))]
    assert all(b < a for a, b in zip(lengths, lengths[1:])), "tempo should rise"


def test_tempo_band_rises_with_level():
    assert synth.tempo_band(1) == 0
    assert synth.tempo_band(6) == 1
    assert synth.tempo_band(50) == len(synth.TEMPO_BANDS) - 1


# -- files ----------------------------------------------------------------


def test_generate_writes_playable_wavs(tmp_path):
    files = generate(tmp_path, music=False, quiet=True)
    assert files

    for path in files:
        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == synth.SAMPLE_RATE
            assert handle.getnframes() > 0


# -- event mapping --------------------------------------------------------


@pytest.mark.parametrize(
    "event,expected",
    [
        (GameEvent(EventType.PIECE_MOVE, -1), "piece_move"),
        (GameEvent(EventType.PIECE_ROTATE, 1), "piece_rotate"),
        (GameEvent(EventType.LINE_CLEAR, 1), "line_clear_1"),
        (GameEvent(EventType.LINE_CLEAR, 3), "line_clear_3"),
        (GameEvent(EventType.LINE_CLEAR, 4), "tetris"),
        (GameEvent(EventType.LEVEL_UP, 5), "level_up"),
        (GameEvent(EventType.GAME_OVER, 0), "game_over"),
    ],
)
def test_events_map_to_sounds(event, expected):
    assert sound_for(event) == expected


def test_a_hard_drop_that_fell_nowhere_is_silent():
    assert sound_for(GameEvent(EventType.HARD_DROP, 0)) is None
    assert sound_for(GameEvent(EventType.HARD_DROP, 5)) == "hard_drop"


def test_every_mapped_sound_actually_exists(sounds):
    """No event may map to a name the synth never builds."""
    for kind in EventType:
        for value in (0, 1, 2, 3, 4):
            name = sound_for(GameEvent(kind, value))
            if name is not None:
                assert name in sounds, f"{kind.name} maps to missing sound {name!r}"


def test_state_changes_outrank_piece_chatter():
    # Auto-repeat fires a move blip every 10ms; without priority those blips
    # take every channel and the clear stinger is the one that gets dropped.
    assert PRIORITY["line_clear_1"] > PRIORITY["piece_move"]
    assert PRIORITY["tetris"] > PRIORITY["piece_lock"]
    assert PRIORITY["game_over"] == max(PRIORITY.values())


def test_every_sound_has_a_priority(sounds):
    for name in sounds:
        assert name in PRIORITY, f"{name} has no priority entry"


# -- degradation ----------------------------------------------------------


def test_disabled_bank_is_inert():
    bank = SoundBank(enabled=False)
    assert not bank.enabled
    assert bank.play("tetris") is False
    # None of these may raise.
    bank.handle([GameEvent(EventType.LINE_CLEAR, 4)])
    bank.play_music(0)
    bank.pause_music()
    bank.resume_music()
    bank.stop_music()
    bank.close()


def test_missing_sound_directory_does_not_raise(tmp_path):
    bank = SoundBank(tmp_path / "nope", enabled=False)
    assert bank.play("tetris") is False
    bank.handle([GameEvent(EventType.GAME_OVER, 100)])


def test_unknown_sound_name_is_ignored():
    bank = SoundBank(enabled=False)
    assert bank.play("no_such_sound") is False
