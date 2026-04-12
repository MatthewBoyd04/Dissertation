"""
renderer.py — Standalone grid-world renderer for the SAR simulation.

Draws the environment state onto a tkinter Canvas embedded in the UI's
"Simulation Render" tab.  No algorithm-specific knowledge; caller supplies
a 2-D grid layout and per-frame state (agent positions, discovered tiles).

Usage:
    from renderer import GridRenderer
    r = GridRenderer(canvas_widget)
    r.setup_grid(grid_2d)               # call once per episode
    r.render_frame(agent_pos, disc)     # call each step
    r.clear()                           # reset between episodes
"""

import os
import tkinter as tk

# ── Sprite / colour config ────────────────────────────────────────────────────

_SPRITES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sprites")

# Fallback solid colours used when Pillow is not available
_TILE_COLOURS = {
    0: "#d4d4b0",        # available
    1: "#3a3a3a",        # blocked
    2: "#cc3300",        # hazard/fatal
    3: "#ffd700",        # reward
    "disc": "#aac8ee",   # discovered-available overlay
}

# One colour per drone (index 0–7)
_DRONE_COLOURS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f1c40f",
    "#9b59b6", "#e67e22", "#1abc9c", "#e91e63",
]

# Sprite file names keyed by tile-type int or special string
_SPRITE_FILES = {
    0:      "Tile_Available.png",
    1:      "Tile_Blocked.png",
    2:      "Tile_Fatal.png",
    3:      "Tile_Reward.png",
    "disc": "Tile_Discovered.png",
}
for _i in range(1, 9):
    _SPRITE_FILES[f"Drone_{_i}"] = f"Drone_{_i}.png"

# ── PIL availability ──────────────────────────────────────────────────────────

try:
    from PIL import Image, ImageTk
    _PIL = True
except ImportError:
    _PIL = False


class GridRenderer:
    """Renders a grid-world episode onto a tkinter Canvas."""

    def __init__(self, canvas: tk.Canvas) -> None:
        self._canvas = canvas
        self._grid_2d: list | None = None        # static layout (int codes)
        self._grid_size: int = 0
        self._tile_size: int = 0
        self._sprite_cache: dict = {}            # key → PhotoImage (kept alive)
        self._tile_items: list = []              # [row][col] canvas item id
        self._disc_state: list = []              # [row][col] bool (internal)
        self._agent_items: dict = {}             # agent_id → canvas item id

    # ── Public API ────────────────────────────────────────────────────────────

    def setup_grid(self, grid_2d: list) -> None:
        """Draw the static grid.  Must be called before render_frame()."""
        self._grid_2d = [row[:] for row in grid_2d]
        self._grid_size = len(grid_2d)
        self._tile_size = self._compute_tile_size()
        self._load_sprites(self._tile_size)

        self._canvas.delete("all")
        self._tile_items = [[0] * self._grid_size for _ in range(self._grid_size)]
        self._disc_state = [[False] * self._grid_size for _ in range(self._grid_size)]
        self._agent_items.clear()

        for r in range(self._grid_size):
            for c in range(self._grid_size):
                self._tile_items[r][c] = self._draw_tile(r, c, grid_2d[r][c], discovered=False)

        # Draw grid border lines on top of tiles
        total_px = self._grid_size * self._tile_size
        for i in range(self._grid_size + 1):
            pos = i * self._tile_size
            self._canvas.create_line(pos, 0, pos, total_px, fill="#555555", tags="gridline")
            self._canvas.create_line(0, pos, total_px, pos, fill="#555555", tags="gridline")

    def render_frame(self, agent_positions: dict, discovered_2d: list) -> None:
        """
        Update the canvas for one step.

        agent_positions : {"Drone_1": [row, col], ...}  — absent agents are removed
        discovered_2d   : full 2-D bool/int array (1 = discovered)
        """
        if self._grid_2d is None:
            return

        # Update discovered overlay (only newly revealed cells)
        for r in range(self._grid_size):
            for c in range(self._grid_size):
                if discovered_2d[r][c] and not self._disc_state[r][c]:
                    self._disc_state[r][c] = True
                    # Only overlay available tiles; blocked/hazard/reward keep their sprite
                    if self._grid_2d[r][c] == 0:
                        self._canvas.delete(self._tile_items[r][c])
                        self._tile_items[r][c] = self._draw_tile(r, c, 0, discovered=True)

        # Remove agents no longer active
        gone = [aid for aid in self._agent_items if aid not in agent_positions]
        for aid in gone:
            self._canvas.delete(self._agent_items.pop(aid))

        # Draw / update each agent
        for agent_id, (row, col) in agent_positions.items():
            if agent_id in self._agent_items:
                self._canvas.delete(self._agent_items[agent_id])
            self._agent_items[agent_id] = self._draw_agent(agent_id, row, col)

        self._canvas.tag_raise("gridline")
        self._canvas.update_idletasks()

    def clear(self) -> None:
        """Reset the canvas and internal state."""
        self._canvas.delete("all")
        self._grid_2d = None
        self._grid_size = 0
        self._tile_items = []
        self._disc_state = []
        self._agent_items.clear()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_tile_size(self) -> int:
        """Return the largest integer tile size that fits the canvas."""
        self._canvas.update_idletasks()
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w <= 1 or h <= 1:
            w, h = 500, 500   # canvas not yet mapped — assume default
        return max(4, min(w, h) // self._grid_size)

    def _load_sprites(self, tile_size: int) -> None:
        """Populate self._sprite_cache with PhotoImage objects."""
        self._sprite_cache.clear()
        if not _PIL:
            return
        for key, fname in _SPRITE_FILES.items():
            path = os.path.join(_SPRITES_DIR, fname)
            if not os.path.exists(path):
                continue
            try:
                img = Image.open(path).resize((tile_size, tile_size), Image.NEAREST)
                self._sprite_cache[key] = ImageTk.PhotoImage(img)
            except Exception:
                pass   # silently fall back to colour for this key

    def _draw_tile(self, row: int, col: int, tile_type: int, discovered: bool) -> int:
        """Draw one tile and return its canvas item id."""
        x = col * self._tile_size
        y = row * self._tile_size
        x2 = x + self._tile_size
        y2 = y + self._tile_size

        sprite_key = "disc" if (discovered and tile_type == 0) else tile_type
        if sprite_key in self._sprite_cache:
            return self._canvas.create_image(
                x, y, anchor="nw", image=self._sprite_cache[sprite_key]
            )
        # Fallback: filled rectangle
        colour = _TILE_COLOURS.get(sprite_key, "#888888")
        return self._canvas.create_rectangle(x, y, x2, y2, fill=colour, outline="#555555")

    def _draw_agent(self, agent_id: str, row: int, col: int) -> int:
        """Draw a drone sprite (or coloured oval) and return its canvas item id."""
        x = col * self._tile_size
        y = row * self._tile_size
        x2 = x + self._tile_size
        y2 = y + self._tile_size

        try:
            idx = int(agent_id.split("_")[1])   # "Drone_3" → 3
        except (IndexError, ValueError):
            idx = 1

        sprite_key = f"Drone_{idx}"
        if sprite_key in self._sprite_cache:
            return self._canvas.create_image(
                x, y, anchor="nw", image=self._sprite_cache[sprite_key]
            )
        # Fallback: coloured oval with a thin white outline
        colour = _DRONE_COLOURS[(idx - 1) % len(_DRONE_COLOURS)]
        pad = max(1, self._tile_size // 6)
        return self._canvas.create_oval(
            x + pad, y + pad, x2 - pad, y2 - pad,
            fill=colour, outline="#ffffff", width=1,
        )
