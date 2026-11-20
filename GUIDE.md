# HoloPuck: Complete Guide

This guide explains **how HoloPuck works, file by file and frame by frame**, so that a student, a lecturer or a developer can understand, run, tune and extend it.

- **Student:** read §1-§4, then §8 (how to run) and the exercises in [`learning.md`](learning.md).
- **Lecturer:** §3 (pipeline), §5 (vision) and §9 (testing) are the teachable core; [`explanation.md`](explanation.md) maps them to the Image Processing syllabus.
- **Developer:** §4, §6, §7, §10 and §11.

Contents: 
1. What it is ·
2. Big picture ·
3. The frame pipeline ·
4. Coordinates ·
5. `vision.py` ·
6. `game.py` ·
7. `render.py` and `main.py` ·
8. Running and options ·
9. Testing ·
10. Tuning ·
11. Extending ·
12. Known limitations ·
13. Troubleshooting ·
14. Glossary

---

## 1. What it is

HoloPuck is an air-hockey game controlled by **your index finger in front of a webcam**. A virtual rink is drawn *on top of the live camera image* (that is the "augmented reality"). The blue mallet follows your fingertip; a red mallet is controlled by the computer. First to 7 goals wins (configurable).

It is a teaching project: the game is simple, but turning *a camera picture into a game control* needs a complete image-processing pipeline.

---

## 2. Big picture

```text
 webcam frame
     │  main.py: flip + resize
     ▼
 tracker.process(frame)            vision.py   (HandTracker or SkinTracker)  → fingertip in pixels
     │
     ▼
 view.frame_to_table(x, y)         vision.py   (inverse perspective)         → table coordinates
     │
     ▼
 smoother.update(x, y)             vision.py   (moving average)
     │
     ▼
 game.set_player_target(x, y)      game.py
 game.update(dt)                   game.py     (physics, AI, scoring)
     │
     ▼
 draw_table(game)                  render.py   (flat 800 x 400 picture)
     │
     ▼
 view.overlay(frame, table)        vision.py   (warp + alpha-blend onto the camera frame)
     │
     ▼
 HUD, landmarks, window, keys      render.py / main.py
```

| File | Responsibility | Knows about the camera? |
|---|---|---|
| `main.py` | The loop, options, keyboard, wiring | yes |
| `vision.py` | Perspective maths, hand trackers, smoothing, overlay | yes |
| `game.py` | Puck, mallets, collisions, scoring, computer opponent | **no** |
| `render.py` | Drawing the flat table and the on-screen text | no |

The key design decision: **`game.py` knows nothing about cameras or pixels.** It lives in a flat 800 × 400 "table" world. Everything that touches the real world is in `vision.py`. This makes the game logic testable with no hardware (§9).

---

## 3. The frame pipeline

Every loop iteration (one video frame) does this, in `main.py`:

| # | Step | Code | Topic |
|---|---|---|---|
| 1 | Read a frame, **mirror** it, resize to 960 × 540 | `cap.read()`, `cv2.flip(raw, 1)`, `cv2.resize` | geometric operations |
| 2 | Find the **fingertip** in camera pixels | `tracker.process(frame)` | detection / segmentation |
| 3 | Convert camera pixels → **table coordinates** | `view.frame_to_table(x, y)` | inverse perspective mapping |
| 4 | **Smooth** the position | `smoother.update(tx, ty)` | temporal filtering |
| 5 | Tell the game where the mallet should go | `game.set_player_target(sx, sy)` | input |
| 6 | **Advance the game** by the elapsed time | `game.update(dt)` | physics, AI, scoring |
| 7 | **Draw the flat table**, then **warp and blend** it over the frame | `draw_table(game)`, `view.overlay(...)` | rendering, perspective, blending |
| 8 | Draw landmarks, fingertip marker, HUD, mask inset | `tracker.draw`, `draw_hud` | visual feedback |
| 9 | Show the window and read the keyboard | `cv2.imshow`, `cv2.waitKey(1)` | UI |

Why mirror (step 1)? A raw webcam image shows you as others see you, so moving your hand right moves it *left* on screen. Flipping makes it feel like a mirror.

If no fingertip is found, step 5 is skipped: **the mallet stays where it was** until the hand returns.

---

## 4. Coordinate systems

Two coordinate systems are used. Mixing them up is the classic bug.

| System | Size | Origin | Used by |
|---|---|---|---|
| **Table** | 800 × 400 | top-left, x → right, y ↓ down | `game.py`, `render.py` |
| **Frame (camera pixels)** | 960 × 540 | top-left, x → right, y ↓ down | `vision.py`, `main.py` |

The bridge between them is one 3 × 3 matrix, `M` (table → frame), and its inverse `M_inv` (frame → table), both built in `TableView.set_perspective` from four corner pairs.

Table geometry (constants at the top of `game.py`):

```text
 (0,0)                        x = 400                       (800,0)
   ┌──────────────────────────────┬──────────────────────────────┐
   │                              │                              │
 ┌─┘  goal opening 150 px         │                              └─┐
 │  (centred on y = 200)          ●                                │
 └─┐                              │                              ┌─┘
   │  player half (x < 400)       │   computer half (x > 400)    │
   └──────────────────────────────┴──────────────────────────────┘
 (0,400)                                                   (800,400)
```

- Puck radius 16, mallet radius 32.
- The player's mallet is confined to the left half, the computer's to the right half.
- The goal opening is drawn 150 px tall (y 125-275). For scoring, the puck passes the wall when its centre is within 71 px of the centre line (`GOAL_H/2 - 4`), so the effective opening is slightly narrower than the drawing, by design, to feel fair near the posts.

---

## 5. `vision.py`: seeing and mapping

### 5.1 `TableView`: perspective transform and AR overlay

A flat rectangle viewed at an angle becomes a trapezoid. A **homography** is the 3 × 3 matrix that does this to every point (details in [`learning.md`](learning.md) §4).

- Four pairs of corners define it: table `(0,0) (800,0) (800,400) (0,400)` ↔ four points in the frame, given as fractions of the frame size:
  - perspective view: `(0.20,0.18) (0.80,0.18) (0.94,0.93) (0.06,0.93)` (narrow at the top, wide at the bottom);
  - flat view: `(0.10,0.15) (0.90,0.15) (0.90,0.95) (0.10,0.95)`.
- `frame_to_table(x, y)` and `table_to_frame(x, y)` convert single points with `cv2.perspectiveTransform`.
- `overlay(frame, table_canvas)` warps the drawn table into the frame (`cv2.warpPerspective`) and **alpha-blends** it: inside the table `out = 0.2·camera + 0.8·table`, outside `out = camera`. The blend mask is precomputed once per view.
- Pressing `P` calls `set_perspective`, which rebuilds both matrices and the mask.

Because `frame_to_table` uses the exact inverse of the matrix used to draw, **your fingertip and the drawn mallet always line up.** A unit test checks the round trip.

### 5.2 `Smoother`: exponential moving average

```text
s_new = α · measurement + (1 − α) · s_old        (α = 0.55)
```

A larger α reacts faster but passes more jitter; a smaller α is steadier but laggy. The first measurement initialises the filter.

### 5.3 `HandTracker`: MediaPipe (machine learning)

Wraps MediaPipe's `HandLandmarker` (Tasks API):

- converts the frame **BGR → RGB** (OpenCV uses BGR, MediaPipe expects RGB);
- runs in `VIDEO` mode, which tracks the hand between frames (faster and steadier than detecting from scratch every frame);
- one hand; confidences: detection 0.6, presence 0.5, tracking 0.5;
- timestamps passed to MediaPipe must strictly increase, so a millisecond clock is forced to increase by at least 1 each call;
- returns landmark **#8** (index fingertip) scaled from the 0-1 range to pixels; `None` if no hand;
- `draw()` draws the 21-point skeleton (fingertip in red).

The model file (`hand_landmarker.task`, ~8 MB) is downloaded once by `ensure_model`. On failure it raises `SystemExit` with a message that includes the URL and manual-download instructions; `main.py` catches that and falls back to `SkinTracker`.

### 5.4 `SkinTracker`: classical image processing (no ML)

Same interface as `HandTracker` (`process`, `draw`, `close`), so `main.py` can use either.

| Step | Operation | Why |
|---|---|---|
| 1 | Gaussian blur 7 × 7 | suppress sensor noise |
| 2 | BGR → **YCrCb** | separates brightness (Y) from colour (Cr, Cb); skin forms a tight cluster in Cr/Cb |
| 3 | `inRange`: Cr 133-173, Cb 77-127 (Y unrestricted) | **thresholding** → binary mask (white = skin) |
| 4 | Blank the top 20 % of the frame | ignore ceiling and often the face |
| 5 | Morphological **open** (5 × 5 ellipse, 2 iterations) then **close** (3 iterations) | remove specks, fill small holes |
| 6 | Largest contour; reject if smaller than 0.4 % of the frame | pick the hand, ignore noise |
| 7 | **Topmost point** of the contour; average x over points within 6 px of the top | fingertip as an extreme point, steadied |

`draw()` outlines the hand (green), its convex hull (orange) and the fingertip (red). With `--classic`, pressing `M` shows the binary mask in the top-right corner, which is the clearest thing to show in a demo.

**Weak spots:** skin-coloured background objects, faces in the lower part of the frame, strong coloured light, and a finger that does not point upward.

---

## 6. `game.py`: the game world

Everything is in table coordinates. `Game` holds the state and advances it with `update(dt)`.

### 6.1 State machine

```text
        reset_match()                 serve timer (1.2 s) ends
 ─────────────► SERVE ─────────────────────────────────────► PLAY
                  ▲                                            │ goal
                  │  goal, nobody has won yet                  ▼
                  └────────────────────────────────────  _score(): next serve
                                                               │ a score reaches win_score
                                                               ▼
                                                             OVER   (press R → reset_match)
```

- **serve:** puck sits at the centre for `SERVE_DELAY` = 1.2 s (mallets can still move, but nothing collides), then `_launch()` fires it at 320 px/s at a random angle within ±0.5 rad, towards the side set by `_serve_dir`.
- **After a goal** the puck is served *towards the side that conceded*.
- **over:** the winner and a message are set; the AI stops moving.
- A stuck puck (speed below 12 px/s for more than 4 s) is re-served in a random direction.

### 6.2 One update step

1. Clamp `dt` to [0.001, 0.05] s. On a slow computer the game runs in slow motion rather than taking huge, unstable steps.
2. Work out where the player's mallet *wants* to be (`_clamp_player(target)`: left half only) and where the AI wants to be (`_ai_next`).
3. Mallet **velocity** = displacement / `dt`, capped at 2,500 px/s. This is how a fast hand gives the puck a harder hit.
4. If the state is `play`, split `dt` into **4 sub-steps**. In each, interpolate both mallets to their sub-step position, then move the puck and check collisions and goals.

**Why sub-steps?** A hit happens when the puck and mallet centres are closer than 48 px (32 + 16). To jump clean *through* a mallet, the two would have to move from 48 px on one side to 48 px on the other, a relative move of about 96 px, inside a single physics step. At 30 FPS a puck at its top speed (1,400 px/s) covers about 47 px per frame, and a fast mallet (up to 2,500 px/s) about 83 px; moving towards each other they close about 130 px per frame, which is enough to skip over the collision zone ("tunnelling"). Four sub-steps cut that to about 33 px per step. A unit test checks that a fast puck does not tunnel.

### 6.3 Puck physics (`_step_puck`)

- **Move:** `pos += vel·h`, then friction `vel *= exp(−0.25·h)` (exponential decay, independent of frame rate).
- **Mallet collision:** if the centres are closer than `puck_r + mallet_r`, push the puck out along the line of centres, then reflect the velocity *relative to the mallet* with restitution 0.9:
  `vel ← vel − (1 + 0.9)·(vel_rel · n)·n`, only when the puck is approaching (`vel_rel · n < 0`). The mallet is treated as **infinitely heavy**: it is never pushed by the puck.
- **Speed limit:** puck speed is capped at 1,400 px/s.
- **Walls:** top and bottom reflect with 0.95 energy kept. Left and right walls reflect the same way **except inside the goal opening**, where the puck is free to leave.
- **Goal:** the puck centre going beyond `x < −16` scores for the computer; beyond `x > 816` scores for the player.

### 6.4 The computer opponent (`_ai_next`)

Difficulty sets the AI's maximum speed and how noisy its aim is:

| Level | Name | Max speed (px/s) | Aim noise σ (px) | Predicts bounces? |
|---|---|---|---|---|
| 1 | Easy | 330 | 45 | no |
| 2 | Medium | 560 | 22 | yes |
| 3 | Hard | 820 | 8 | yes |

The noise is re-drawn (Gaussian) every 0.35 s, so the AI "wobbles" like a person. Its behaviour:

| Situation | Target |
|---|---|
| Not playing (serve / over) | its home position `(730, 200)` |
| Puck on the AI's half and **behind** the mallet | fall back to `(730, puck_y)` to cover the goal |
| Puck on the AI's half, in front | **strike**: aim at the puck's position (plus noise) |
| Puck on the player's half, coming fast (`vx > 250`), Medium/Hard | **predict** the y where it will arrive at home x, folding wall bounces, and wait there (at x = 710, just in front of home) |
| Puck on the player's half otherwise | shadow the puck's height, scaled by 0.6 towards the centre (also at x = 710) |

The strike and defend targets (not the fall-back target) have the current aim noise added. Each frame the AI moves towards its target but no faster than its speed limit. **Wall-bounce prediction** unfolds the puck's path like a triangle wave: `y = (y − lo) mod 2·span`, mirrored back if it exceeds `span`.

> The opponent is hand-written logic (a heuristic), **not** machine learning.

---

## 7. `render.py` and `main.py`

### `render.py`

- `draw_table(game)` paints an 800 × 400 image: table colour, border, centre line and circle, two goal markers, both mallets (with shadow, rim and highlight) and the puck. It returns a flat image; `TableView.overlay` later warps it.
- `draw_hud(...)` writes onto the *camera frame*: score (`YOU a - b CPU`), difficulty and win score, FPS, goal messages, "Show your hand to the camera" when no hand is found, and the key hints. Text has a dark offset shadow so it is readable on any background.

### `main.py`

- Parses the options (§8), opens the camera (or falls back), picks a tracker, then runs the loop of §3.
- **Mouse mode:** the mouse position (in frame pixels) goes through the *same* `frame_to_table` and smoother as a fingertip, so you control the mallet by moving the mouse **over the drawn table**.
- Frame rate shown is a smoothed estimate (`0.9·old + 0.1·instant`).
- `run_snapshot` is the headless smoke test: a scripted "player" chases the puck for 180 simulated frames (6 s at 30 FPS) on a synthetic background, saves one frame to PNG and prints the score and state.

---

## 8. Running and options

```bash
pip install -r requirements.txt
python main.py                 # MediaPipe hand tracking
python main.py --classic       # classical skin tracking
python main.py --mouse         # mouse control, webcam shown as background
python main.py --no-camera     # mouse control, plain background
python main.py --snapshot out.png   # headless test
```

All options and keys are listed in the [README](README.md). Fallback behaviour:

- camera cannot be opened → mouse control on a plain background;
- MediaPipe model unavailable or MediaPipe not installed → classical tracker.

---

## 9. Testing

```bash
python -m unittest discover -s tests -v
```

`tests/test_holopuck.py` contains **16 tests** and needs no camera:

| Group | Tests |
|---|---|
| **Physics** | top-wall bounce; side-wall bounce outside the goal; fast puck does not tunnel through a mallet; mallet hit sends the puck away |
| **Rules** | goal for the player; goal for the computer; match ends at the win score and resets; player mallet cannot cross the centre line; goal opening is smaller than the table |
| **AI / stability** | a 60-second scripted rally on Hard: nothing crashes, the puck stays on the table, goals happen |
| **Vision** | homography round trip (perspective and flat); perspective table narrower at the top; overlay keeps the frame shape and only changes pixels inside the table; smoother averages correctly |
| **Classical tracker** | finds the fingertip of a synthetic hand within a few pixels; returns `None` with no hand |

**What is not tested yet:** the MediaPipe tracker on a real hand and the whole system on a real webcam in real lighting. The MediaPipe code path was checked for API compatibility, but the model could not be downloaded in the environment used for development. Please try it once on a laptop with a camera and internet.

---

## 10. Tuning

If tracking or feel is off on your setup, these are the dials:

| Want to change | Where | Notes |
|---|---|---|
| Mallet responsiveness vs steadiness | `Smoother(alpha=0.55)` in `main.py` | up = quicker, jittery; down = steady, laggy |
| Skin colour range | `SkinTracker.SKIN_LO / SKIN_HI` in `vision.py` | adjust Cr/Cb for your lighting |
| Ignore more/less of the top of the image | `roi_top` (default 0.20) | raise if your face is being detected |
| Minimum hand size | `min_area_frac` (default 0.004) | |
| Where the table sits in the frame | `QUAD_PERSPECTIVE`, `QUAD_FLAT` | fractions of frame width/height |
| How opaque the table is | `OVERLAY_ALPHA` (default 0.80) | |
| Game feel | `DAMPING`, `WALL_BOUNCE`, `MALLET_BOUNCE`, `MAX_PUCK_SPEED`, `MAX_MALLET_SPEED` in `game.py` | |
| Difficulty | `DIFFICULTY` table in `game.py` | name, speed, noise |
| MediaPipe sensitivity | `HandTracker(detection_conf, tracking_conf)` | |

---

## 11. Extending

- **Detect a real table or marker.** Replace the fixed trapezoid by finding four corners (for example ArUco markers with `cv2.aruco`) and pass them to `getPerspectiveTransform`. Only `TableView` needs to change.
- **Two players.** Set MediaPipe `num_hands=2`, make the second mallet follow the second fingertip, and replace `_ai_next` for that side.
- **Auto-calibrate skin colour.** Ask the player to put a hand in a box, read the Cr/Cb values inside it, and set the threshold range from their spread.
- **Sound and effects.** Hook into `_score` and the mallet-collision branch of `_step_puck`.
- **Handle a lost hand better.** Currently the mallet freezes at its last position; you could reset the smoother or slide the mallet back to a home position.

---

## 12. Known limitations

- The table is **fixed** in the camera frame; it is not anchored to anything in the room.
- The game is 2-D; a "hand" only moves the mallet in the table plane.
- The classical tracker depends on lighting and background.
- The MediaPipe path and real-camera tuning are untested (see §9).
- The opponent never learns; it is the same heuristic every game.
- The mallet is infinitely heavy: it never recoils from hitting the puck.

---

## 13. Troubleshooting

See the table in the [README](README.md#-troubleshooting). Two extra notes:

- **"Everything is mirrored"**: that is intentional (§3, step 1).
- **Game feels slow on an old computer**: `dt` is capped at 0.05 s, so below about 20 FPS the game runs in slow motion rather than behaving erratically. Lower `FRAME_W`/`FRAME_H` in `main.py` for speed.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **AR (augmented reality)** | Virtual content aligned with, and drawn over, a live view of the real world |
| **Homography / perspective transform** | 3 × 3 matrix mapping one plane to another, such as a flat table to a trapezoid |
| **Inverse mapping** | Going the other way (camera pixel → table coordinate) with the inverse matrix |
| **Alpha blending** | Mixing two images with a weight: `out = (1 − α)·A + α·B` |
| **Landmark** | A named point on the hand (21 per hand; #8 is the index fingertip) |
| **Segmentation** | Splitting an image into regions, here "skin" and "not skin" |
| **Morphology** | Shape-based mask clean-up: erosion, dilation, opening, closing |
| **Contour** | The outline of a connected white region in a mask |
| **EMA** | Exponential moving average, a simple low-pass filter over time |
| **Tunnelling** | A fast object passing through another between two physics steps |
| **Restitution** | How much speed survives a bounce (0.9 = 90 %) |
| **Sub-stepping** | Splitting one frame's time step into smaller steps for stability |
| **FPS** | Frames per second |
