import time
from collections import deque

import pygame  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]

from app.display.faces import PIXEL_OFF, PIXEL_EYE, PIXEL_MOUTH, GRID_W, GRID_H

COLOR_BG = (18, 20, 32)
COLOR_EYE = (120, 200, 255)
COLOR_MOUTH = (120, 200, 255)
COLOR_OFF = (12, 12, 16)

PALETTE = {
    PIXEL_OFF: COLOR_OFF,
    PIXEL_EYE: COLOR_EYE,
    PIXEL_MOUTH: COLOR_MOUTH,
}


class Renderer:
    def __init__(self, pixel_size=12, width=None, height=None):
        self.pixel_size = pixel_size
        self.grid_w = GRID_W
        self.grid_h = GRID_H

        self.win_w = width or (self.grid_w * pixel_size)
        self.win_h = height or (self.grid_h * pixel_size)

        self.queue = deque(maxlen=2)
        self.running = False
        self._initialized = False
        self._screen = None
        self._clock = None
        self._font = None

        self.fps_times = deque(maxlen=60)
        self._show_fps = True
        self._current_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

    def init(self):
        if self._initialized:
            return
        pygame.init()
        self._screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Bastard TV Head")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 18)
        self.running = True
        self._initialized = True

    def close(self):
        self.running = False
        if self._initialized:
            pygame.quit()
            self._initialized = False

    def push(self, grid):
        self.queue.append(grid.copy())

    def toggle_fps(self):
        self._show_fps = not self._show_fps

    def tick(self):
        if not self._initialized:
            self.init()

        screen = self._screen
        clock = self._clock
        font = self._font
        assert screen is not None and clock is not None and font is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_f:
                    self._show_fps = not self._show_fps

        if self.queue:
            self._current_grid = self.queue[-1]

        screen.fill(COLOR_BG)
        self._draw_grid(screen, self._current_grid)

        if self._show_fps:
            now = time.perf_counter()
            self.fps_times.append(now)
            if len(self.fps_times) > 1:
                fps = (len(self.fps_times) - 1) / (
                    self.fps_times[-1] - self.fps_times[0]
                )
                fps_text = font.render(f"FPS: {fps:.0f}", True, (150, 150, 160))
                screen.blit(fps_text, (6, 4))

        pygame.display.flip()
        clock.tick(60)
        return self.running

    def _draw_grid(self, screen, grid):
        ps = self.pixel_size
        ox = (self.win_w - self.grid_w * ps) // 2
        oy = (self.win_h - self.grid_h * ps) // 2

        for y in range(self.grid_h):
            py = oy + y * ps
            for x in range(self.grid_w):
                px = ox + x * ps
                color = PALETTE.get(grid[y, x], COLOR_OFF)
                pygame.draw.rect(screen, color, (px, py, ps, ps))
