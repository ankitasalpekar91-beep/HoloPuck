"""HoloPuck - AR air hockey controlled by your index finger.

Run:  python main.py
Keys: 1/2/3 difficulty, P toggle perspective, H toggle hand landmarks,
      M toggle the segmentation mask inset (--classic), R restart match, Q or Esc quit.
"""

from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np

from game import DIFFICULTY, TABLE_H, TABLE_W, Game
from render import draw_hud, draw_table
from vision import DEFAULT_MODEL_PATH, HandTracker, Smoother, SkinTracker, TableView

FRAME_W, FRAME_H = 960, 540
WINDOW = "HoloPuck - AR Air Hockey"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HoloPuck: AR air hockey with hand tracking")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--difficulty", type=int, choices=sorted(DIFFICULTY), default=2)
    ap.add_argument("--win-score", type=int, default=7, help="goals needed to win")
    ap.add_argument("--flat", action="store_true", help="start with the perspective view off")
    ap.add_argument("--mouse", action="store_true",
                    help="control the mallet with the mouse instead of your hand")
    ap.add_argument("--no-camera", action="store_true",
                    help="use a plain background instead of the webcam (implies --mouse)")
    ap.add_argument("--classic", action="store_true",
                    help="track the hand with classical image processing (skin colour) instead of MediaPipe")
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH),
                    help="path to hand_landmarker.task (downloaded automatically if missing)")
    ap.add_argument("--snapshot", metavar="PNG",
                    help="headless test: simulate a few seconds, save one frame to PNG and exit")
    return ap.parse_args()


def synthetic_background() -> np.ndarray:
    """A dark gradient with a grid, used when there is no webcam."""
    y = np.linspace(40, 90, FRAME_H, dtype=np.uint8)[:, None, None]
    bg = np.repeat(np.repeat(y, FRAME_W, axis=1), 3, axis=2).copy()
    for x in range(0, FRAME_W, 60):
        cv2.line(bg, (x, 0), (x, FRAME_H), (70, 70, 70), 1)
    for yy in range(0, FRAME_H, 60):
        cv2.line(bg, (0, yy), (FRAME_W, yy), (70, 70, 70), 1)
    return bg


def run_snapshot(args: argparse.Namespace) -> None:
    """Headless smoke test: a scripted player for ~6 simulated seconds, then save a frame."""
    game = Game(difficulty=args.difficulty, win_score=args.win_score, seed=1)
    view = TableView(FRAME_W, FRAME_H, perspective=not args.flat)
    bg = synthetic_background()
    dt = 1 / 30
    frame = bg
    for k in range(180):
        t = k * dt
        # scripted "hand": sweeps around the player's half, chasing the puck when it is close
        px, py = game.puck.pos
        if px < TABLE_W / 2:
            game.set_player_target(px - 25, py)
        else:
            game.set_player_target(110 + 60 * math.sin(t * 2), TABLE_H / 2 + 100 * math.sin(t * 3))
        game.update(dt)
        frame = view.overlay(bg, draw_table(game))
        draw_hud(frame, game, 30.0, True, "mouse", view.perspective)
    cv2.imwrite(args.snapshot, frame)
    print(f"saved {args.snapshot}  score={game.score}  state={game.state}")


def main() -> None:
    args = parse_args()
    if args.snapshot:
        run_snapshot(args)
        return

    cap = None
    if not args.no_camera:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Could not open camera {args.camera}; falling back to mouse + plain background.")
            cap = None
    use_mouse = args.mouse or cap is None
    tracker = None
    if not use_mouse:
        if args.classic:
            tracker = SkinTracker()
        else:
            try:
                tracker = HandTracker(args.model)
            except SystemExit as exc:  # model missing / no internet: keep the game playable
                print(f"{exc}\nFalling back to the classical skin-colour tracker (--classic).")
                tracker = SkinTracker()
    mode = "mouse" if use_mouse else "hand"

    game = Game(difficulty=args.difficulty, win_score=args.win_score)
    view = TableView(FRAME_W, FRAME_H, perspective=not args.flat)
    smoother = Smoother(alpha=0.55)
    background = synthetic_background()
    mouse_pos = [FRAME_W * 0.3, FRAME_H * 0.6]
    show_landmarks = True
    show_mask = True  # only visible with --classic

    def on_mouse(event, x, y, flags, param):
        mouse_pos[0], mouse_pos[1] = x, y

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    last = time.perf_counter()
    fps = 30.0
    while True:
        if cap is not None:
            ok, raw = cap.read()
            if not ok:
                print("Camera frame could not be read; exiting.")
                break
            raw = cv2.flip(raw, 1)  # mirror, so moving your hand right moves the mallet right
            frame = cv2.resize(raw, (FRAME_W, FRAME_H))
        else:
            frame = background.copy()

        # ---- input: fingertip (or mouse) in camera pixels -> table coordinates
        fingertip = None
        if tracker is not None:
            fingertip = tracker.process(frame)
        elif use_mouse:
            fingertip = (mouse_pos[0], mouse_pos[1])
        if fingertip is not None:
            tx, ty = view.frame_to_table(*fingertip)
            sx, sy = smoother.update(tx, ty)
            game.set_player_target(sx, sy)

        # ---- simulate + draw
        now = time.perf_counter()
        dt = now - last
        last = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-3))
        game.update(dt)

        frame = view.overlay(frame, draw_table(game))
        if tracker is not None and show_landmarks:
            tracker.draw(frame)
        if fingertip is not None:
            cv2.circle(frame, (int(fingertip[0]), int(fingertip[1])), 9, (0, 255, 0), 2, cv2.LINE_AA)
        draw_hud(frame, game, fps, fingertip is not None, mode, view.perspective)
        mask = getattr(tracker, "mask", None)
        if show_mask and mask is not None:  # picture-in-picture of the segmentation result
            inset = cv2.cvtColor(cv2.resize(mask, (240, 135)), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(inset, (0, 0), (239, 134), (255, 255, 255), 1)
            frame[40:175, FRAME_W - 250:FRAME_W - 10] = inset
        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            game.reset_match()
        elif key in (ord("1"), ord("2"), ord("3")):
            game.set_difficulty(int(chr(key)))
        elif key == ord("p"):
            view.set_perspective(not view.perspective)
        elif key == ord("h"):
            show_landmarks = not show_landmarks
        elif key == ord("m"):
            show_mask = not show_mask

    if cap is not None:
        cap.release()
    if tracker is not None:
        tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
