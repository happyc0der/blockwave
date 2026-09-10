"""Post-process passes, all pure numpy.

Every function here operates on the *finished* frame as float32 in 0-255. That
is deliberate: a global pass cannot occlude an individual playfield cell, which
is how the legibility invariant survives contact with the CRT look.

Keeping these out of pygame means the training loop never round-trips through a
surface, and the human window and the agent's observation get bit-identical
treatment.
"""

from __future__ import annotations

import numpy as np

# Cached masks, keyed by the parameters that shape them. The frame size is
# fixed for the life of a compositor, so these are built once and reused.
_VIGNETTE_CACHE: dict[tuple[int, int, float], np.ndarray] = {}
_SCANLINE_CACHE: dict[tuple[int, float, int], np.ndarray] = {}


def _blur1d(data: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Box blur along one axis via a running sum — O(n) regardless of radius."""
    if radius < 1:
        return data
    width = 2 * radius + 1
    pad = [(0, 0)] * data.ndim
    pad[axis] = (radius + 1, radius)
    padded = np.pad(data, pad, mode="edge")
    cumulative = np.cumsum(padded, axis=axis)

    high = [slice(None)] * data.ndim
    low = [slice(None)] * data.ndim
    high[axis] = slice(width, None)
    low[axis] = slice(0, -width)
    return (cumulative[tuple(high)] - cumulative[tuple(low)]) / width


def box_blur(data: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur. Two O(n) passes rather than one O(n*r^2)."""
    return _blur1d(_blur1d(data, radius, 0), radius, 1)


#: Bloom is computed at 1/N resolution. It is a low-frequency effect by nature,
#: so blurring at full resolution is wasted work — at 720x750 it cost 10 ms of a
#: 14 ms frame, which left the default profile barely above 60 fps. Working on a
#: strided view cuts that to roughly a third for no visible difference; the
#: aliasing the stride introduces is immediately blurred away.
BLOOM_DOWNSAMPLE = 4


def bloom(
    frame: np.ndarray,
    amount: float,
    radius: int = 3,
    threshold: float = 150.0,
    downsample: int = BLOOM_DOWNSAMPLE,
) -> np.ndarray:
    """Bleed light out of the bright neon into the dark ground around it.

    Additive, so it can only ever make a cell *more* visible against the
    background — it cannot swallow one.
    """
    if amount <= 0.0:
        return frame

    step = max(1, downsample)
    if step == 1:
        return frame + box_blur(np.maximum(frame - threshold, 0.0), radius) * amount

    height, width = frame.shape[:2]
    # Threshold and blur on the small view, so both passes do 1/step^2 the work.
    small = np.maximum(frame[::step, ::step] - threshold, 0.0)
    small = box_blur(small, max(1, radius // step))

    glow = np.repeat(np.repeat(small, step, axis=0), step, axis=1)[:height, :width]
    return frame + glow * amount


def scanlines(frame: np.ndarray, intensity: float, phase: int = 0, period: int = 3) -> np.ndarray:
    """Darken every ``period``-th row, the way a CRT does."""
    if intensity <= 0.0:
        return frame
    height = frame.shape[0]
    key = (height, intensity, phase % period)
    mask = _SCANLINE_CACHE.get(key)
    if mask is None:
        rows = np.ones((height, 1, 1), dtype=np.float32)
        rows[(np.arange(height) + phase) % period == 0] = 1.0 - intensity
        mask = rows
        _SCANLINE_CACHE[key] = mask
    return frame * mask


def vignette(frame: np.ndarray, amount: float) -> np.ndarray:
    """Fall the corners off toward black.

    The playfield sits in the middle of the scene, so the darkest part of this
    mask never lands on a cell.
    """
    if amount <= 0.0:
        return frame
    height, width = frame.shape[:2]
    key = (height, width, amount)
    mask = _VIGNETTE_CACHE.get(key)
    if mask is None:
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        radius = np.sqrt(xs * xs + ys * ys) / np.sqrt(2.0)
        mask = (1.0 - amount * radius**2)[:, :, None].astype(np.float32)
        _VIGNETTE_CACHE[key] = mask
    return frame * mask


def chromatic(frame: np.ndarray, amount: float) -> np.ndarray:
    """Split the red and blue channels sideways, like a misconverged tube."""
    shift = int(round(amount))
    if shift < 1:
        return frame
    out = frame.copy()
    out[:, :, 0] = np.roll(frame[:, :, 0], -shift, axis=1)
    out[:, :, 2] = np.roll(frame[:, :, 2], shift, axis=1)
    return out


def grade(frame: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    """Scale brightness and push contrast around mid grey."""
    if brightness != 1.0:
        frame = frame * brightness
    if contrast != 1.0:
        frame = (frame - 128.0) * contrast + 128.0
    return frame


def hue_rotate(frame: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate every hue by ``degrees`` with a single 3x3 colour matrix.

    Cheaper than a round trip through HSV, and a small rotation keeps the whole
    frame inside the Miami palette's neighbourhood rather than recolouring it
    into something the game never actually looks like.
    """
    if abs(degrees) < 1e-3:
        return frame
    theta = np.radians(degrees)
    cos, sin = np.cos(theta), np.sin(theta)
    one_third = 1.0 / 3.0
    sqrt_third = np.sqrt(one_third)

    matrix = np.array(
        [
            [
                cos + (1.0 - cos) * one_third,
                one_third * (1.0 - cos) - sqrt_third * sin,
                one_third * (1.0 - cos) + sqrt_third * sin,
            ],
            [
                one_third * (1.0 - cos) + sqrt_third * sin,
                cos + one_third * (1.0 - cos),
                one_third * (1.0 - cos) - sqrt_third * sin,
            ],
            [
                one_third * (1.0 - cos) - sqrt_third * sin,
                one_third * (1.0 - cos) + sqrt_third * sin,
                cos + one_third * (1.0 - cos),
            ],
        ],
        dtype=np.float32,
    )
    return frame @ matrix.T


def add_noise(frame: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Sprinkle sensor noise over the frame."""
    if amount <= 0.0:
        return frame
    return frame + rng.normal(0.0, amount, frame.shape).astype(np.float32)


def shake_offset(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate the whole frame.

    Moving the board rather than covering it keeps every cell readable, and for
    the agent it is free translation-robustness training.
    """
    if dx == 0 and dy == 0:
        return frame
    return np.roll(np.roll(frame, dy, axis=0), dx, axis=1)
