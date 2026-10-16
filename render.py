"""Drawing code: the flat table canvas (later warped into AR) and the on-screen HUD."""

from __future__ import annotations

import cv2
import numpy as np

from game import GOAL_H, TABLE_H, TABLE_W, Game

# BGR colours
C_TABLE = (70, 45, 20)
C_LINE = (230, 220, 200)
C_PLAYER = (255, 150, 40)
C_AI = (70, 70, 245)
C_PUCK = (40, 220, 255)
C_GOAL_L = (255, 200, 80)
C_GOAL_R = (90, 90, 255)
AA = cv2.LINE_AA


def _i(p) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _mallet(img: np.ndarray, pos, radius: float, colour) -> None:
    c = _i(pos)
    r = int(radius)
    cv2.circle(img, (c[0] + 3, c[1] + 4), r, (20, 15, 10), -1, AA)      # drop shadow
    cv2.circle(img, c, r, colour, -1, AA)
    cv2.circle(img, c, r, (255, 255, 255), 2, AA)
    cv2.circle(img, c, int(r * 0.55), tuple(int(v * 0.65) for v in colour), -1, AA)
    cv2.circle(img, (c[0] - r // 4, c[1] - r // 4), max(3, r // 6), (255, 255, 255), -1, AA)


def draw_table(game: Game) -> np.ndarray:
    """Render the flat table (TABLE_W x TABLE_H) with puck and mallets."""
    img = np.full((TABLE_H, TABLE_W, 3), C_TABLE, np.uint8)

    # rink border, centre line and centre circle
    cv2.rectangle(img, (0, 0), (TABLE_W - 1, TABLE_H - 1), C_LINE, 6)
    cv2.line(img, (TABLE_W // 2, 0), (TABLE_W // 2, TABLE_H), C_LINE, 3, AA)
    cv2.circle(img, (TABLE_W // 2, TABLE_H // 2), 60, C_LINE, 3, AA)
    cv2.circle(img, (TABLE_W // 2, TABLE_H // 2), 6, C_LINE, -1, AA)

    # goals
    y0, y1 = TABLE_H // 2 - GOAL_H // 2, TABLE_H // 2 + GOAL_H // 2
    cv2.line(img, (2, y0), (2, y1), C_GOAL_L, 10, AA)
    cv2.line(img, (TABLE_W - 3, y0), (TABLE_W - 3, y1), C_GOAL_R, 10, AA)

    # mallets and puck
    _mallet(img, game.player.pos, game.player.radius, C_PLAYER)
    _mallet(img, game.ai.pos, game.ai.radius, C_AI)
    pc = _i(game.puck.pos)
    cv2.circle(img, (pc[0] + 2, pc[1] + 3), int(game.puck.radius), (20, 15, 10), -1, AA)
    cv2.circle(img, pc, int(game.puck.radius), C_PUCK, -1, AA)
    cv2.circle(img, pc, int(game.puck.radius), (255, 255, 255), 2, AA)
    return img


def _text(img, s, org, scale=0.7, colour=(255, 255, 255), thick=2) -> None:
    off = 2 if thick > 1 else 1  # thin text gets a tighter shadow so it stays crisp
    cv2.putText(img, s, (org[0] + off, org[1] + off), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + off, AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, AA)


def _centered(img, s, y, scale, colour, thick=2) -> None:
    (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    _text(img, s, ((img.shape[1] - w) // 2, y), scale, colour, thick)


def draw_hud(frame: np.ndarray, game: Game, fps: float, hand_found: bool, mode: str,
             perspective: bool) -> None:
    """Scores, messages and key hints, drawn straight onto the camera frame."""
    _centered(frame, f"YOU {game.score[0]}  -  {game.score[1]} CPU", 36, 1.0, (255, 255, 255), 2)
    _text(frame, f"{game.difficulty_name}  |  first to {game.win_score}", (12, 28), 0.55)
    _text(frame, f"{fps:4.0f} FPS", (frame.shape[1] - 110, 28), 0.55)

    if game.message:
        _centered(frame, game.message, frame.shape[0] // 2 - 120, 1.2, (0, 230, 255), 3)
    if mode == "hand" and not hand_found:
        _centered(frame, "Show your hand to the camera", frame.shape[0] - 60, 0.8, (80, 80, 255), 2)

    hint = f"1/2/3 level  P perspective({'on' if perspective else 'off'})  H landmarks  R restart  Q quit"
    _text(frame, hint, (12, frame.shape[0] - 14), 0.5, (230, 230, 230), 1)
