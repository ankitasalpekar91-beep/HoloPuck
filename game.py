"""HoloPuck game logic: puck physics, mallets, scoring and the computer opponent.

Everything here works in *table coordinates* (a flat 800 x 400 rectangle).
The vision code (vision.py) is responsible for converting camera pixels into
table coordinates, so this module never touches OpenCV or the webcam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- Table geometry (pixels in the flat table canvas) -----------------------
TABLE_W = 800
TABLE_H = 400
GOAL_H = 150          # height of each goal opening
PUCK_R = 16
MALLET_R = 32

# --- Physics tuning ----------------------------------------------------------
MAX_PUCK_SPEED = 1400.0     # px / s
MAX_MALLET_SPEED = 2500.0   # px / s (limits the "kick" a fast hand can give)
DAMPING = 0.25              # exponential velocity decay per second (air/table friction)
WALL_BOUNCE = 0.95          # fraction of speed kept after hitting a wall
MALLET_BOUNCE = 0.9         # restitution of a mallet hit
SUBSTEPS = 4                # physics sub-steps per frame (prevents tunnelling)
SERVE_DELAY = 1.2           # seconds the puck waits at centre after a goal

# name, max AI speed (px/s), aiming noise (px)
DIFFICULTY = {
    1: ("Easy", 330.0, 45.0),
    2: ("Medium", 560.0, 22.0),
    3: ("Hard", 820.0, 8.0),
}


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    """Scale vector v down so that its length is at most `limit`."""
    n = float(np.linalg.norm(v))
    if n > limit > 0:
        return v * (limit / n)
    return v


@dataclass
class Puck:
    pos: np.ndarray = field(default_factory=lambda: np.array([TABLE_W / 2, TABLE_H / 2]))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = PUCK_R


@dataclass
class Mallet:
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = MALLET_R


class Game:
    """Holds the full game state and advances it with update(dt)."""

    def __init__(self, difficulty: int = 2, win_score: int = 7, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.difficulty = difficulty
        self.win_score = win_score
        self.player = Mallet(np.array([120.0, TABLE_H / 2]))
        self.ai = Mallet(np.array([TABLE_W - 120.0, TABLE_H / 2]))
        self.puck = Puck()
        self._target = self.player.pos.copy()
        self._noise = np.zeros(2)
        self._noise_timer = 0.0
        self._slow_time = 0.0
        self.reset_match()

    # ------------------------------------------------------------------ setup
    def reset_match(self) -> None:
        self.score = [0, 0]  # [player, computer]
        self.winner: str | None = None
        self.state = "serve"  # "serve" | "play" | "over"
        self.message = ""
        self.msg_timer = 0.0
        self._serve(direction=self.rng.choice([-1, 1]))

    def set_difficulty(self, level: int) -> None:
        if level in DIFFICULTY:
            self.difficulty = level

    @property
    def difficulty_name(self) -> str:
        return DIFFICULTY[self.difficulty][0]

    def _serve(self, direction: int) -> None:
        """Put the puck in the centre; it launches towards `direction` (+1 = computer)."""
        self.puck.pos = np.array([TABLE_W / 2, TABLE_H / 2])
        self.puck.vel = np.zeros(2)
        self._serve_dir = int(direction)
        self.serve_timer = SERVE_DELAY
        self._slow_time = 0.0
        if self.state != "over":
            self.state = "serve"

    def _launch(self) -> None:
        angle = self.rng.uniform(-0.5, 0.5)
        speed = 320.0
        self.puck.vel = np.array([math.cos(angle), math.sin(angle)]) * speed * self._serve_dir
        self.state = "play"

    # ------------------------------------------------------------------ input
    def set_player_target(self, x: float, y: float) -> None:
        """Where the player wants the mallet (table coordinates)."""
        self._target = np.array([x, y], dtype=float)

    def _clamp_player(self, p: np.ndarray) -> np.ndarray:
        r = self.player.radius
        return np.array([
            min(max(p[0], r), TABLE_W / 2 - r),
            min(max(p[1], r), TABLE_H - r),
        ])

    def _clamp_ai(self, p: np.ndarray) -> np.ndarray:
        r = self.ai.radius
        return np.array([
            min(max(p[0], TABLE_W / 2 + r), TABLE_W - r),
            min(max(p[1], r), TABLE_H - r),
        ])

    # --------------------------------------------------------------------- AI
    def _predict_y_at(self, x_target: float) -> float:
        """Where the puck will be (y) when it reaches x_target, folding wall bounces."""
        p, v = self.puck.pos, self.puck.vel
        if abs(v[0]) < 1e-6:
            return float(p[1])
        t = (x_target - p[0]) / v[0]
        if t < 0:
            return float(p[1])
        y = p[1] + v[1] * t
        lo, hi = PUCK_R, TABLE_H - PUCK_R
        span = hi - lo
        y = (y - lo) % (2 * span)
        if y > span:
            y = 2 * span - y
        return float(y + lo)

    def _ai_next(self, dt: float) -> np.ndarray:
        _, speed, noise = DIFFICULTY[self.difficulty]
        p, m = self.puck, self.ai

        self._noise_timer -= dt
        if self._noise_timer <= 0:
            self._noise = self.rng.normal(0.0, noise, 2)
            self._noise_timer = 0.35

        home_x = TABLE_W - 70.0
        centre_y = TABLE_H / 2

        if self.state != "play":
            target = np.array([home_x, centre_y])
        elif p.pos[0] > TABLE_W / 2:
            if p.pos[0] > m.pos[0] + 5:
                # puck got behind the mallet: fall back and cover the goal
                target = np.array([home_x, p.pos[1]])
            else:
                # puck is in front of us: strike it
                target = p.pos + self._noise
        else:
            # puck is on the player's side: defend
            if p.vel[0] > 250 and self.difficulty >= 2:
                y = self._predict_y_at(home_x)
            else:
                y = centre_y + (p.pos[1] - centre_y) * 0.6
            target = np.array([home_x - 20.0, y]) + self._noise

        delta = target - m.pos
        dist = float(np.linalg.norm(delta))
        max_move = speed * dt
        if dist > max_move > 0:
            delta *= max_move / dist
        return self._clamp_ai(m.pos + delta)

    # ---------------------------------------------------------------- physics
    def _step_puck(self, h: float) -> None:
        pk = self.puck
        pk.pos = pk.pos + pk.vel * h
        pk.vel = pk.vel * math.exp(-DAMPING * h)

        # mallet collisions
        for m in (self.player, self.ai):
            d = pk.pos - m.pos
            dist = float(np.linalg.norm(d))
            min_d = pk.radius + m.radius
            if dist < min_d:
                n = d / dist if dist > 1e-6 else np.array([1.0, 0.0]) * (1 if m is self.player else -1)
                pk.pos = m.pos + n * min_d  # push puck out of the mallet
                vn = float(np.dot(pk.vel - m.vel, n))
                if vn < 0:  # approaching: reflect relative velocity (mallet is "infinitely heavy")
                    pk.vel = pk.vel - (1 + MALLET_BOUNCE) * vn * n
        pk.vel = _clip_norm(pk.vel, MAX_PUCK_SPEED)

        # walls
        r = pk.radius
        if pk.pos[1] < r:
            pk.pos[1] = r
            pk.vel[1] = abs(pk.vel[1]) * WALL_BOUNCE
        elif pk.pos[1] > TABLE_H - r:
            pk.pos[1] = TABLE_H - r
            pk.vel[1] = -abs(pk.vel[1]) * WALL_BOUNCE

        in_gap = abs(pk.pos[1] - TABLE_H / 2) < GOAL_H / 2 - 4
        if pk.pos[0] < r and not in_gap:
            pk.pos[0] = r
            pk.vel[0] = abs(pk.vel[0]) * WALL_BOUNCE
        elif pk.pos[0] > TABLE_W - r and not in_gap:
            pk.pos[0] = TABLE_W - r
            pk.vel[0] = -abs(pk.vel[0]) * WALL_BOUNCE

    def _check_goal(self) -> None:
        x = self.puck.pos[0]
        if x < -PUCK_R:
            self._score(0)  # computer scores
        elif x > TABLE_W + PUCK_R:
            self._score(1)  # player scores

    def _score(self, who: int) -> None:
        """who: 1 = player scored, 0 = computer scored."""
        idx = 0 if who == 1 else 1
        self.score[idx] += 1
        if self.score[idx] >= self.win_score:
            self.state = "over"
            self.winner = "player" if idx == 0 else "computer"
            self.message = "YOU WIN!  (press R)" if idx == 0 else "COMPUTER WINS  (press R)"
            self.msg_timer = 1e9
            self._serve(direction=1)
            return
        self.message = "GOAL!" if idx == 0 else "Computer scores"
        self.msg_timer = SERVE_DELAY
        # serve towards the side that just conceded
        self._serve(direction=1 if idx == 0 else -1)

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        dt = max(1e-3, min(dt, 0.05))
        self.msg_timer = max(0.0, self.msg_timer - dt)
        if self.msg_timer == 0.0 and self.state != "over":
            self.message = ""

        p0 = self.player.pos.copy()
        p1 = self._clamp_player(self._target)
        a0 = self.ai.pos.copy()
        a1 = self._ai_next(dt) if self.state != "over" else a0

        self.player.vel = _clip_norm((p1 - p0) / dt, MAX_MALLET_SPEED)
        self.ai.vel = _clip_norm((a1 - a0) / dt, MAX_MALLET_SPEED)

        if self.state != "play":
            self.player.pos, self.ai.pos = p1, a1
            if self.state == "serve":
                self.serve_timer -= dt
                if self.serve_timer <= 0:
                    self._launch()
            return

        h = dt / SUBSTEPS
        for i in range(1, SUBSTEPS + 1):
            t = i / SUBSTEPS
            self.player.pos = p0 + (p1 - p0) * t
            self.ai.pos = a0 + (a1 - a0) * t
            self._step_puck(h)
            self._check_goal()
            if self.state != "play":
                return

        # re-serve if the puck has been (almost) stationary for too long
        if float(np.linalg.norm(self.puck.vel)) < 12.0:
            self._slow_time += dt
            if self._slow_time > 4.0:
                self._serve(direction=int(self.rng.choice([-1, 1])))
        else:
            self._slow_time = 0.0
