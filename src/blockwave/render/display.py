"""The pygame window — the only module that touches a display.

Everything above this line is numpy. This just takes the finished frame and
puts it on screen, which is why everything above can stay pure numpy.
"""

from __future__ import annotations

import numpy as np
import pygame


class Display:
    def __init__(self, size: tuple[int, int], title: str = "BLOCKWAVE", scale: int = 1, vsync: bool = True) -> None:
        pygame.display.init()
        width, height = size
        self.scale = max(1, scale)
        self._window_size = (width * self.scale, height * self.scale)

        flags = pygame.SCALED if self.scale == 1 else 0
        try:
            self.screen = pygame.display.set_mode(self._window_size, flags, vsync=int(vsync))
        except pygame.error:
            # vsync is not available on every driver; the fixed-timestep loop
            # does not depend on it, so fall back rather than refusing to start.
            self.screen = pygame.display.set_mode(self._window_size, flags)

        pygame.display.set_caption(title)
        self._surface = pygame.Surface((width, height))

    def present(self, frame: np.ndarray) -> None:
        """Blit an ``(H, W, 3)`` uint8 frame and flip."""
        # pygame surfaces are (width, height), numpy frames are (rows, cols).
        pygame.surfarray.blit_array(self._surface, np.transpose(frame, (1, 0, 2)))
        if self.scale == 1:
            self.screen.blit(self._surface, (0, 0))
        else:
            pygame.transform.scale(self._surface, self._window_size, self.screen)
        pygame.display.flip()

    def close(self) -> None:
        pygame.display.quit()


def refresh_rate(default: int = 60) -> int:
    """The display's refresh rate, for pacing the render loop."""
    try:
        rate = pygame.display.get_current_refresh_rate()
    except (AttributeError, pygame.error):
        return default
    return rate if rate > 0 else default
