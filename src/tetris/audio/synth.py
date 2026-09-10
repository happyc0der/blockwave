"""A small chiptune synthesizer, in numpy.

Every sound the game makes is generated here and written to ``assets/sfx``. No
downloads, no licence to track, and every timbre is a number someone can change.

The palette is deliberately the one a 90s arcade board had: square waves with a
duty cycle, a saw for weight, a triangle for bass, and white noise for
percussion. Nothing here is sampled.

Sounds are float arrays in [-1, 1] at :data:`SAMPLE_RATE`, mono. They only
become 16-bit PCM at :func:`write_wav`.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44_100

#: Frequency of A4, the reference all note numbers are measured from.
A4 = 440.0

Freq = float | tuple[float, float]


def note(semitones: float, base: float = A4) -> float:
    """A frequency ``semitones`` above (or below) A4."""
    return base * (2.0 ** (semitones / 12.0))


def _samples(duration: float) -> int:
    return max(1, int(duration * SAMPLE_RATE))


def _phase(freq: Freq, duration: float) -> np.ndarray:
    """Instantaneous phase, integrating frequency so sweeps stay continuous.

    A tuple sweeps geometrically rather than linearly, because pitch is
    perceived on a log scale — a linear ramp from 800 Hz to 100 Hz spends most
    of its time sounding high, then falls off a cliff.
    """
    count = _samples(duration)
    if isinstance(freq, tuple):
        start, end = freq
        curve = np.geomspace(max(1e-3, start), max(1e-3, end), count)
    else:
        curve = np.full(count, float(freq))
    return 2.0 * np.pi * np.cumsum(curve) / SAMPLE_RATE


# -- oscillators ----------------------------------------------------------


def square(freq: Freq, duration: float, duty: float = 0.5) -> np.ndarray:
    """A pulse wave, centred on zero for any duty cycle.

    A naive pulse sits at +1 for ``duty`` of each cycle and -1 for the rest, so
    its mean is ``2 * duty - 1`` — at the 22% duty the arpeggio uses that is a
    DC offset of -0.56. Offsets like that thump when a sound starts and stops,
    and they eat headroom that the actual waveform should be using. Subtracting
    the mean costs nothing and removes both problems.
    """
    cycle = (_phase(freq, duration) / (2.0 * np.pi)) % 1.0
    return np.where(cycle < duty, 1.0, -1.0) - (2.0 * duty - 1.0)


def saw(freq: Freq, duration: float) -> np.ndarray:
    cycle = (_phase(freq, duration) / (2.0 * np.pi)) % 1.0
    return 2.0 * cycle - 1.0


def triangle(freq: Freq, duration: float) -> np.ndarray:
    cycle = (_phase(freq, duration) / (2.0 * np.pi)) % 1.0
    return 4.0 * np.abs(cycle - 0.5) - 1.0


def sine(freq: Freq, duration: float) -> np.ndarray:
    return np.sin(_phase(freq, duration))


def noise(duration: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, _samples(duration))


# -- shaping --------------------------------------------------------------


def adsr(
    duration: float,
    attack: float = 0.005,
    decay: float = 0.04,
    sustain: float = 0.7,
    release: float = 0.08,
) -> np.ndarray:
    """A classic four-stage envelope, clamped to fit ``duration``."""
    count = _samples(duration)
    attack_n = min(count, _samples(attack))
    decay_n = min(count - attack_n, _samples(decay))
    release_n = min(count - attack_n - decay_n, _samples(release))
    sustain_n = max(0, count - attack_n - decay_n - release_n)

    return np.concatenate(
        [
            np.linspace(0.0, 1.0, attack_n, endpoint=False),
            np.linspace(1.0, sustain, decay_n, endpoint=False),
            np.full(sustain_n, sustain),
            np.linspace(sustain, 0.0, release_n),
        ]
    )[:count]


#: A one-millisecond ramp in front of every percussive envelope. Short enough
#: that the attack still reads as instant, long enough that the waveform starts
#: from zero instead of jumping — a truly instant attack on a square wave clicks,
#: and at the seam of a looping music track it clicks once per bar cycle.
CLICK_GUARD = 0.001


def decay_env(duration: float, power: float = 3.0) -> np.ndarray:
    """A percussive envelope: near-instant attack, exponential fall."""
    count = _samples(duration)
    envelope = (1.0 - np.linspace(0.0, 1.0, count)) ** power

    ramp_n = min(count, _samples(CLICK_GUARD))
    if ramp_n > 1:
        envelope[:ramp_n] *= np.linspace(0.0, 1.0, ramp_n)
    return envelope


def mix(*layers: np.ndarray) -> np.ndarray:
    """Sum layers of differing lengths, padding to the longest."""
    if not layers:
        return np.zeros(1)
    length = max(len(layer) for layer in layers)
    total = np.zeros(length)
    for layer in layers:
        total[: len(layer)] += layer
    return total


def sequence(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros(1)


def silence(duration: float) -> np.ndarray:
    return np.zeros(_samples(duration))


def normalize(samples: np.ndarray, peak: float = 0.85) -> np.ndarray:
    """Scale to a target peak, then soften the extremes.

    ``tanh`` rather than a hard clip: a chiptune square is already all edges,
    and clipping one adds a fizz that reads as a bug rather than as character.
    """
    highest = float(np.max(np.abs(samples))) if samples.size else 0.0
    if highest > 0.0:
        samples = samples / highest * peak
    return np.tanh(samples * 1.2) / np.tanh(1.2) * peak


def write_wav(path: Path, samples: np.ndarray) -> None:
    """Write mono 16-bit PCM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(normalize(samples), -1.0, 1.0)
    pcm = (pcm * 32_767.0).astype("<i2")

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


# -- the sound set --------------------------------------------------------
# Each builder returns a float array. Kept as small named functions so a
# designer can tweak one sound without reading the rest.


def _blip(freq: Freq, duration: float, duty: float = 0.5, power: float = 2.5) -> np.ndarray:
    return square(freq, duration, duty) * decay_env(duration, power)


def _arpeggio(
    semitones: list[float], step: float = 0.055, duty: float = 0.5, base: float = A4
) -> np.ndarray:
    return sequence(
        *(_blip(note(s, base), step, duty, power=1.8) for s in semitones)
    )


def build_sounds(seed: int = 7) -> dict[str, np.ndarray]:
    """Every sound the game plays, by name."""
    rng = np.random.default_rng(seed)

    sounds: dict[str, np.ndarray] = {}

    # --- piece handling: short, dry, and quiet enough to hear a hundred times
    sounds["piece_move"] = _blip(note(-9), 0.022, duty=0.25) * 0.45
    sounds["piece_rotate"] = _blip((note(-2), note(3)), 0.038, duty=0.35) * 0.55
    sounds["piece_lock"] = mix(
        _blip((note(-14), note(-21)), 0.070, duty=0.5) * 0.7,
        noise(0.035, rng) * decay_env(0.035, 4.0) * 0.35,
    )
    sounds["piece_hold"] = _arpeggio([0, 7], step=0.045, duty=0.3) * 0.6
    sounds["hold_denied"] = _blip(note(-22), 0.075, duty=0.5) * 0.4

    sounds["hard_drop"] = mix(
        saw((note(10), note(-14)), 0.085) * decay_env(0.085, 2.2) * 0.75,
        noise(0.055, rng) * decay_env(0.055, 3.0) * 0.45,
    )
    sounds["soft_drop"] = _blip(note(-12), 0.018, duty=0.2) * 0.30

    # --- clears: the reward, so these get to be the biggest thing in the mix
    sounds["line_clear_1"] = _arpeggio([0, 4, 7])
    sounds["line_clear_2"] = _arpeggio([0, 4, 7, 12])
    sounds["line_clear_3"] = _arpeggio([0, 4, 7, 12, 16])
    sounds["tetris"] = mix(
        _arpeggio([0, 4, 7, 12, 16, 19, 24], step=0.062, duty=0.35),
        # A rising noise sweep underneath, for weight.
        noise(0.44, rng) * np.linspace(0.0, 0.5, _samples(0.44)) ** 2 * 0.5,
        sequence(silence(0.30), triangle((note(-24), note(-12)), 0.20) * 0.5),
    )
    sounds["t_spin"] = mix(
        _arpeggio([0, 3, 7, 10], step=0.05, duty=0.2),
        # Detuned second voice: the shimmer that says "that was not an accident".
        _arpeggio([0.15, 3.15, 7.15, 10.15], step=0.05, duty=0.2) * 0.6,
    )
    sounds["perfect_clear"] = mix(
        _arpeggio([12, 16, 19, 24, 28, 31], step=0.07, duty=0.15),
        _arpeggio([0, 4, 7, 12, 16, 19], step=0.07, duty=0.5) * 0.5,
    )
    sounds["combo"] = _blip((note(4), note(16)), 0.09, duty=0.3) * 0.55

    # --- state changes
    sounds["level_up"] = mix(
        _arpeggio([0, 5, 9, 12, 17], step=0.075, duty=0.4),
        sequence(silence(0.15), triangle(note(-24), 0.28) * adsr(0.28) * 0.55),
    )
    sounds["game_over"] = mix(
        # A falling minor line: the classic "you are done" cadence.
        _arpeggio([0, -3, -7, -12], step=0.17, duty=0.45),
        saw((note(-12), note(-30)), 0.70) * decay_env(0.70, 1.4) * 0.45,
    )
    sounds["pause"] = _arpeggio([7, 0], step=0.06, duty=0.5) * 0.6

    # --- menus
    sounds["menu_move"] = _blip(note(2), 0.028, duty=0.2) * 0.5
    sounds["menu_select"] = _arpeggio([0, 7, 12], step=0.05, duty=0.35) * 0.7
    sounds["menu_back"] = _arpeggio([7, 0], step=0.05, duty=0.35) * 0.6

    return {name: normalize(data) for name, data in sounds.items()}


# -- music ----------------------------------------------------------------

#: i - VI - III - VII in A minor. The synthwave progression, and it loops back
#: on itself cleanly, which is what a background track has to do above all else.
_PROGRESSION: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (-24.0, (0.0, 3.0, 7.0)),    # Am
    (-28.0, (-4.0, 0.0, 3.0)),   # F
    (-33.0, (-9.0, -5.0, -2.0)), # C
    (-26.0, (-2.0, 2.0, 5.0)),   # G
)

BASE_BPM = 116.0

#: Tempo multipliers, indexed by level band. The music tightening as the stack
#: speeds up is most of what makes late levels feel urgent.
TEMPO_BANDS: tuple[float, ...] = (1.0, 1.12, 1.26, 1.42)


def tempo_band(level: int) -> int:
    """Which tempo variant a level should play."""
    return min(len(TEMPO_BANDS) - 1, max(0, (level - 1) // 5))


def build_music(band: int = 0, seed: int = 11) -> np.ndarray:
    """One looping bar-cycle of synthwave at the given tempo band.

    Three voices: a triangle bass on the root, a square arpeggio over the chord
    tones, and a soft saw pad. Deliberately sparse — this plays under everything
    else for the whole session, so it has to stay out of the way of the sound
    effects that actually carry information.
    """
    rng = np.random.default_rng(seed)
    bpm = BASE_BPM * TEMPO_BANDS[min(band, len(TEMPO_BANDS) - 1)]
    beat = 60.0 / bpm
    sixteenth = beat / 4.0
    bar = beat * 4.0

    bass_parts: list[np.ndarray] = []
    arp_parts: list[np.ndarray] = []
    pad_parts: list[np.ndarray] = []

    for root, chord in _PROGRESSION:
        # Bass: eighth notes on the root, with the offbeats ducked.
        for index in range(8):
            level = 0.85 if index % 2 == 0 else 0.45
            bass_parts.append(
                triangle(note(root), beat / 2.0)
                * adsr(beat / 2.0, 0.004, 0.05, 0.55, 0.06)
                * level
            )

        # Arpeggio: sixteenths walking up and back down the chord.
        pattern = [chord[0], chord[1], chord[2], chord[1]]
        for index in range(16):
            semitone = pattern[index % 4] + (12.0 if index >= 8 else 0.0)
            arp_parts.append(
                square(note(semitone), sixteenth, duty=0.22)
                * decay_env(sixteenth, 1.6)
                * 0.30
            )

        # Pad: the chord held for the whole bar, slow in, well back in the mix.
        pad_parts.append(
            mix(*(saw(note(s - 12.0), bar) for s in chord))
            * adsr(bar, attack=0.25, decay=0.2, sustain=0.5, release=0.3)
            * 0.10
        )

    track = mix(sequence(*bass_parts), sequence(*arp_parts), sequence(*pad_parts))

    # A whisper of noise, the tape hiss the era came with.
    track = track + noise(len(track) / SAMPLE_RATE, rng) * 0.008
    return normalize(track, peak=0.62)


#: i - VI - iv - V in D minor: darker and more suspended than the game loop, so
#: the menu sits somewhere else emotionally rather than sounding like the game
#: with the drums taken out.
_MENU_PROGRESSION: tuple[tuple[float, tuple[float, float, float]], ...] = (
    (-31.0, (-7.0, -3.0, 0.0)),    # Dm
    (-36.0, (-12.0, -8.0, -5.0)),  # Bb
    (-33.0, (-9.0, -5.0, -2.0)),   # Gm
    (-29.0, (-5.0, -1.0, 2.0)),    # A
)

MENU_BPM = 84.0


def build_menu_music(seed: int = 23) -> np.ndarray:
    """The attract-mode loop: slow, wide and unhurried.

    Deliberately the opposite of the game track. No driving sixteenth arpeggio —
    a held pad, a sparse bass, and a slow sine melody floating over the top, so
    the title screen feels like somewhere you can sit rather than somewhere the
    clock is already running.
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / MENU_BPM
    bar = beat * 4.0

    pad_parts: list[np.ndarray] = []
    bass_parts: list[np.ndarray] = []
    bell_parts: list[np.ndarray] = []

    #: One note per bar, tracing a slow line over the changes.
    melody = (0.0, -3.0, -5.0, -7.0)

    for index, (root, chord) in enumerate(_MENU_PROGRESSION):
        # Pad: the chord held the whole bar, detuned against itself for width.
        voices = [saw(note(s), bar) for s in chord]
        voices += [saw(note(s + 0.12), bar) * 0.7 for s in chord]
        pad_parts.append(
            mix(*voices) * adsr(bar, attack=0.5, decay=0.3, sustain=0.75, release=0.6) * 0.16
        )

        # Bass: one long root note per bar, an octave below the game track.
        bass_parts.append(
            triangle(note(root), bar) * adsr(bar, 0.05, 0.3, 0.6, 0.5) * 0.55
        )

        # Bell: a single sine, entering late in the bar and ringing over the change.
        semitone = melody[index % len(melody)]
        bell_parts.append(
            sequence(
                silence(beat * 2.0),
                mix(
                    sine(note(semitone + 12.0), beat * 2.0),
                    sine(note(semitone + 19.0), beat * 2.0) * 0.35,
                )
                * decay_env(beat * 2.0, 1.5)
                * 0.30,
            )
        )

    track = mix(sequence(*pad_parts), sequence(*bass_parts), sequence(*bell_parts))
    track = track + noise(len(track) / SAMPLE_RATE, rng) * 0.006
    return normalize(track, peak=0.55)
