"""Loading, mixing and playing the generated sounds.

Two things this has to get right.

**Priority.** Holding a direction fires a move blip every 10 ms of auto-repeat.
With a naive "play everything" bank those blips eat every mixer channel and the
line-clear stinger — the one sound that actually tells you something — is the
one that gets dropped. Sounds therefore carry a priority, and a low-priority
sound will not evict a higher one.

**Never being the reason the game fails.** No audio device, no generated files,
a mixer that refuses to open: all of it degrades to silence. Audio is the last
thing that should take a game down, and headless runs must not touch a device at
all.
"""

from __future__ import annotations

from pathlib import Path

from ..core.events import EventType, GameEvent

#: Higher wins. Anything that reports a state change outranks the chatter of
#: moving a piece around.
PRIORITY: dict[str, int] = {
    "piece_move": 0,
    "soft_drop": 0,
    "piece_rotate": 1,
    "piece_hold": 1,
    "hold_denied": 1,
    "piece_lock": 2,
    "hard_drop": 2,
    "menu_move": 2,
    "menu_select": 3,
    "menu_back": 3,
    "combo": 3,
    "pause": 4,
    "line_clear_1": 5,
    "line_clear_2": 5,
    "line_clear_3": 5,
    "t_spin": 6,
    "quad": 6,
    "perfect_clear": 7,
    "level_up": 7,
    "game_over": 8,
}

#: Mixer channels reserved for effects. Enough for a clear stinger to ring out
#: under a couple of piece sounds without either being cut.
EFFECT_CHANNELS = 12


def sound_for(event: GameEvent) -> str | None:
    """Map a game event to a sound name, or ``None`` for a silent event."""
    kind = event.type

    if kind is EventType.PIECE_MOVE:
        return "piece_move"
    if kind is EventType.PIECE_ROTATE:
        return "piece_rotate"
    if kind is EventType.SOFT_DROP:
        return "soft_drop"
    if kind is EventType.HARD_DROP:
        return "hard_drop" if event.value else None
    if kind is EventType.PIECE_LOCK:
        return "piece_lock"
    if kind is EventType.PIECE_HOLD:
        return "piece_hold"
    if kind is EventType.HOLD_DENIED:
        return "hold_denied"
    if kind is EventType.LINE_CLEAR:
        # A four-line clear gets its own fanfare; smaller ones scale with count.
        # Clamped rather than trusted: an out-of-range count would otherwise
        # name a sound file that does not exist and fail silently.
        if event.value >= 4:
            return "quad"
        return f"line_clear_{event.value}" if 1 <= event.value <= 3 else None
    if kind is EventType.TSPIN:
        return "t_spin"
    if kind is EventType.PERFECT_CLEAR:
        return "perfect_clear"
    if kind is EventType.COMBO:
        return "combo"
    if kind is EventType.LEVEL_UP:
        return "level_up"
    if kind is EventType.GAME_OVER:
        return "game_over"
    return None


class SoundBank:
    """Plays sounds, or silently does nothing if it cannot."""

    def __init__(self, directory: Path | None = None, *, enabled: bool = True, volume: float = 0.8) -> None:
        self.directory = directory or default_sfx_dir()
        self.volume = volume
        self.enabled = False
        self.missing: list[str] = []
        self._sounds: dict[str, object] = {}
        self._channels: list[object] = []
        self._playing: dict[int, int] = {}  # channel index -> priority
        self._music_track: str | None = None

        if enabled:
            self._start()

    # -- setup ------------------------------------------------------------

    def _start(self) -> None:
        try:
            import pygame

            pygame.mixer.init(frequency=44_100, size=-16, channels=1, buffer=512)
            pygame.mixer.set_num_channels(EFFECT_CHANNELS + 1)
            self._channels = [pygame.mixer.Channel(i) for i in range(EFFECT_CHANNELS)]
        except Exception:
            # No device, no driver, or a mixer that will not open. Stay silent.
            return

        self._load()
        self.enabled = True

    def _load(self) -> None:
        import pygame

        if not self.directory.is_dir():
            self.missing.append(str(self.directory))
            return

        for path in sorted(self.directory.glob("*.wav")):
            if path.stem.startswith("music"):
                continue
            try:
                self._sounds[path.stem] = pygame.mixer.Sound(str(path))
            except Exception:
                self.missing.append(path.stem)

    @property
    def ready(self) -> bool:
        return self.enabled and bool(self._sounds)

    # -- playback ---------------------------------------------------------

    def play(self, name: str) -> bool:
        """Play a sound by name. Returns whether it actually started."""
        if not self.enabled:
            return False
        sound = self._sounds.get(name)
        if sound is None:
            return False

        priority = PRIORITY.get(name, 0)
        channel_index = self._claim(priority)
        if channel_index is None:
            return False

        try:
            sound.set_volume(self.volume)  # type: ignore[attr-defined]
            self._channels[channel_index].play(sound)  # type: ignore[attr-defined]
        except Exception:
            return False

        self._playing[channel_index] = priority
        return True

    def _claim(self, priority: int) -> int | None:
        """Find a channel, preferring free ones and never evicting a better sound."""
        quietest_index = None
        quietest_priority = priority

        for index, channel in enumerate(self._channels):
            try:
                busy = channel.get_busy()  # type: ignore[attr-defined]
            except Exception:
                return None
            if not busy:
                self._playing.pop(index, None)
                return index

            running = self._playing.get(index, 0)
            if running < quietest_priority:
                quietest_priority = running
                quietest_index = index

        # Every channel is busy: take one only from a strictly lesser sound.
        return quietest_index

    def handle(self, events: list[GameEvent]) -> None:
        """Play whatever an engine step produced.

        Deliberately collapses duplicates within one step: a burst of auto-repeat
        can emit several move events in a single tick, and playing each one is
        just a flam.
        """
        if not self.enabled:
            return
        seen: set[str] = set()
        for event in events:
            name = sound_for(event)
            if name is not None and name not in seen:
                seen.add(name)
                self.play(name)

    # -- music ------------------------------------------------------------

    def play_music(self, track: str, volume: float = 0.35) -> None:
        """Start (or switch to) a named loop, e.g. ``"menu"`` or ``"0"``.

        A no-op when the requested track is already playing, so the caller can
        drive this from a scene change every frame without restarting the bar.
        """
        if not self.enabled or track == self._music_track:
            return
        path = self.directory / f"music_{track}.wav"
        if not path.is_file():
            return
        try:
            import pygame

            pygame.mixer.music.load(str(path))
            pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.play(-1)
            self._music_track = track
        except Exception:
            self._music_track = None

    def pause_music(self) -> None:
        """Hold the loop where it is, so resuming does not restart the bar."""
        if not self.enabled:
            return
        try:
            import pygame

            pygame.mixer.music.pause()
        except Exception:
            pass

    def resume_music(self) -> None:
        if not self.enabled:
            return
        try:
            import pygame

            pygame.mixer.music.unpause()
        except Exception:
            pass

    def stop_music(self) -> None:
        if not self.enabled:
            return
        try:
            import pygame

            pygame.mixer.music.stop()
        except Exception:
            pass
        self._music_track = None

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            import pygame

            pygame.mixer.quit()
        except Exception:
            pass
        self.enabled = False


def default_sfx_dir() -> Path:
    """``assets/sfx`` alongside the installed package."""
    return Path(__file__).resolve().parents[3] / "assets" / "sfx"
