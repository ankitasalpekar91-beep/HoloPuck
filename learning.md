# HoloPuck: What This Project Teaches

HoloPuck turns **a camera picture into a game controller**. To do that it needs a whole chain of image-processing ideas, plus some physics and some careful software design. This file explains those ideas, with the maths and worked examples, and gives exercises to test your understanding.

- **Students:** read §1-§4 in order, then try §6 *before* reading the answers.
- **Lecturers:** §2 maps the project to course topics; §6-§7 have exercises and viva questions. [`explanation.md`](explanation.md) has the same mapping in syllabus order.
- **Developers:** §5 (game physics) and §8 (lessons from the audit) are the parts most relevant to real software work.

All numbers in the worked examples were checked by running the project's own code.

---

## 1. The one-paragraph story

A webcam gives a stream of pictures. For each picture HoloPuck finds the player's index fingertip (with a machine-learning model, or with classical image processing), converts that pixel position into a position on a virtual air-hockey table, smooths it, and lets it drive a mallet. The virtual table is drawn flat, then *warped* into the camera view and blended over the live video, so the game looks like it sits in front of you. The same 3 × 3 matrix that draws the table is inverted to read your finger, which is why the two always line up.

---

## 2. Concept map: where each topic appears

| Topic | Where it shows up | File |
|---|---|---|
| **Digital image fundamentals** | A frame is a NumPy array `(height, width, 3)` of 0-255 values | `main.py` |
| **Colour models** | BGR (OpenCV) → RGB (MediaPipe); BGR → YCrCb for skin | `vision.py` |
| **Spatial filtering** | 7 × 7 Gaussian blur | `SkinTracker.process` |
| **Segmentation by thresholding** | `inRange` on Cr/Cb gives a binary skin mask | `SkinTracker.process` |
| **Morphological processing** | Opening, then closing, with a 5 × 5 ellipse | `SkinTracker.process` |
| **Contours and shape description** | `findContours`, largest area, convex hull | `SkinTracker.process / draw` |
| **Feature (extreme-point) detection** | Fingertip = topmost point of the hand contour | `SkinTracker.process` |
| **Object detection and tracking** | MediaPipe hand landmarks tracked across frames | `HandTracker` |
| **Geometric transforms** | Flip, resize, **perspective transform (homography)** and its inverse | `main.py`, `TableView` |
| **Image arithmetic** | Alpha blending of table and camera | `TableView.overlay` |
| **Temporal filtering** | Exponential moving average | `Smoother` |
| **Drawing primitives** | Circles, lines, text | `render.py` |
| **AR concepts** | Virtual object registered to a live camera view | whole project |
| **Physics and numerical methods** | Integration, damping, collisions, sub-stepping | `game.py` |
| **Algorithms and AI basics** | State machine, rule-based opponent, trajectory prediction | `game.py` |
| **Software engineering** | Separation of concerns, fallbacks, hardware-free tests | all files |

---

## 3. Image-processing ideas, one at a time

### 3.1 A picture is an array

A 960 × 540 colour frame is a grid of 540 rows × 960 columns, each pixel holding 3 numbers (blue, green, red, each 0-255). That is 540 × 960 × 3 = **1,555,200 numbers** per frame, 30 times a second. Everything else is arithmetic on that array.

### 3.2 Colour spaces: why YCrCb for skin

In RGB the *brightness* is mixed into all three channels, so the same skin looks very different in shadow and in light. YCrCb separates them:

```text
Y  = 0.299 R + 0.587 G + 0.114 B          brightness
Cr = (R − Y) · 0.713 + 128                red-difference chroma
Cb = (B − Y) · 0.564 + 128                blue-difference chroma
```

Human skin of many tones lands in a small region of the (Cr, Cb) plane, so a simple range check works:

```text
133 ≤ Cr ≤ 173   and   77 ≤ Cb ≤ 127        (Y is ignored)
```

*Check it:* the colour used in the test, BGR `(120, 160, 215)`, converts to `(Y, Cr, Cb) = (172, 159, 99)`: Cr and Cb both fall inside the range, so it counts as skin.

### 3.3 Thresholding makes a mask

Apply the range test to every pixel: inside → 255 (white), outside → 0 (black). The result is a **binary mask**, a black-and-white picture of "where is skin". With `--classic`, press `M` to see it live.

### 3.4 Morphology cleans the mask

The mask has speckles (false positives) and holes (false negatives). Two shape-based operations fix them:

| Operation | Recipe | Effect |
|---|---|---|
| **Erosion** | shrink white regions by the structuring element | deletes thin specks |
| **Dilation** | grow white regions | fills gaps, restores size |
| **Opening** | erosion, then dilation | removes small white specks, keeps big shapes |
| **Closing** | dilation, then erosion | fills small black holes inside shapes |

HoloPuck opens (2 iterations) then closes (3 iterations), with a 5 × 5 ellipse.

### 3.5 Contours and the fingertip

`findContours` traces the outline of each white blob. The **largest** blob is assumed to be the hand (and ignored if smaller than 0.4 % of the frame). The fingertip is then the **topmost point** of that outline: an *extreme point*. To stop it flickering between neighbouring pixels, x is averaged over the contour points within 6 px of the top.

### 3.6 The ML alternative

MediaPipe's hand model replaces steps 3.2-3.5 with a trained neural network that outputs 21 labelled points per hand; point 8 is the fingertip. It handles lighting and backgrounds far better, but it is a learned model, not a classical technique. See [`mllearning.md`](mllearning.md).

### 3.7 Alpha blending

To show the table *through* the camera image, mix the two pictures pixel by pixel:

```text
out = (1 − α) · camera + α · table          α = 0.8 inside the table, 0 outside
```

Example: a camera pixel of 100 under a table pixel of 200 gives `0.2·100 + 0.8·200 = 180`.

### 3.8 The perspective transform (homography)

A flat rectangle seen at an angle becomes a trapezoid. The mapping is a 3 × 3 matrix `H` applied in *homogeneous coordinates*:

```text
[x']       [h11 h12 h13] [x]
[y']  ∝    [h21 h22 h23] [y]        then   X = x'/w',   Y = y'/w'
[w']       [h31 h32 h33] [1]
```

The final division by `w'` is what makes distant things look smaller. `H` is defined only up to scale, so it has **8 unknowns**. Each point pair gives 2 equations (one for X, one for Y), so **4 pairs give exactly 8 equations**, enough to solve for `H` (`cv2.getPerspectiveTransform`). The four points must be non-collinear (no three on a line).

HoloPuck's four pairs are the table corners `(0,0) (800,0) (800,400) (0,400)` and a trapezoid in the camera frame.

- **Forward** `H` (table → frame) draws the table into the video: `warpPerspective`.
- **Inverse** `H⁻¹` (frame → table) converts a fingertip pixel into a table position: `frame_to_table`.

Because the same matrix is used both ways, drawing and input agree. The test `test_homography_round_trip` checks that table → frame → table returns the starting point.

### 3.9 Smoothing with an exponential moving average

Detection noise makes the fingertip jitter by a few pixels. The filter

```text
s_new = α · measurement + (1 − α) · s_old          (α = 0.55)
```

is a *low-pass filter over time*. How fast does it follow a sudden jump of 100 px? Each frame it closes 55 % of the remaining gap:

| Frame | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Filter output | 55.0 | 79.75 | 90.89 | 95.90 | 98.15 |

So the mallet reaches about 91 % of a jump in 3 frames (0.1 s at 30 FPS). A smaller α would be steadier but slower; a larger α faster but twitchier. This is the classic **noise versus lag** trade-off.

---

## 4. Why mirror the image?

A raw webcam image is a view *of* you, not a mirror: moving your hand right moves it left on screen, which feels wrong. `cv2.flip(frame, 1)` mirrors it horizontally so the game behaves like a mirror.

---

## 5. Game-physics and AI ideas

### 5.1 Moving things in discrete time

A computer cannot move a puck continuously; it moves it in small time steps `dt`:

```text
position ← position + velocity · dt
```

HoloPuck updates 4 times per frame with `h = dt / 4` (sub-stepping).

### 5.2 Friction as exponential decay

```text
velocity ← velocity · exp(−0.25 · h)
```

Using `exp(−k·h)` makes the slow-down depend on *time*, not on how many frames happened. With `k = 0.25`, speed halves every `ln 2 / 0.25 ≈ 2.77 s` (ignoring bounces).

### 5.3 Bouncing off a wall

A wall flips the velocity component perpendicular to it and keeps only part of the speed:

```text
v_y ← |v_y| · 0.95     (for the top wall)
```

### 5.4 Hitting a mallet (restitution)

Let `n` be the unit vector from the mallet to the puck and `v_rel = v_puck − v_mallet`. If the puck is approaching (`v_rel · n < 0`):

```text
v_puck ← v_puck − (1 + e) · (v_rel · n) · n          e = 0.9
```

*Example.* A puck moving left at 300 px/s hits a stationary mallet on its left (`n = (1, 0)`): `v_rel·n = −300`, so the new velocity is `(−300) − 1.9·(−300) = +270`. It rebounds at 270 px/s, which is 90 % of the incoming speed. If the mallet is moving towards the puck, `v_rel` is larger and the puck leaves faster: that is how a fast hand "hits harder". The mallet itself is treated as infinitely heavy and never recoils.

### 5.5 Tunnelling and sub-stepping

If an object moves further than the collision zone in a single step, the collision is never seen. At 30 FPS a puck at top speed (1,400 px/s) moves about 47 px per frame and a fast mallet (2,500 px/s) about 83 px; heading towards each other they close about 130 px per frame, while jumping clean through needs only about 96 px. Splitting each frame into 4 sub-steps cuts that to about 33 px per step, so collisions are caught.

### 5.6 The computer opponent

The opponent is a **rule-based agent**: no learning, just conditions (puck behind me → cover the goal; puck in front → strike; puck coming fast → predict where it will arrive). Difficulty changes only two numbers: how fast it can move and how noisy its aim is.

**Predicting a bounce by "unfolding".** To find where a puck will be at the AI's side, `_predict_y_at` extends its straight-line path and then folds it back into the table. The y coordinate behaves like a triangle wave:

```text
y_unfolded = y + v_y · t                      t = (x_target − x) / v_x
y = (y_unfolded − lo) mod (2 · span);  if y > span: y = 2·span − y;  result = y + lo
        where lo = 16, hi = 384 (puck radius from the walls), span = hi − lo = 368
```

*Worked example.* The puck is at `(400, 100)` moving `(500, 600)` px/s and the AI waits at x = 730. Then `t = 330/500 = 0.66 s`, `y_unfolded = 100 + 600·0.66 = 496`, which is past the bottom wall (384). The fold gives `(496 − 16) mod 736 = 480 > 368`, so `y = 736 − 480 = 256`, and `256 + 16 = 272`. Check by hand: the puck falls 284 px to the bottom wall, bounces, and travels the remaining 112 px back up: `384 − 112 = 272`. ✓ (With `v_y = 200` there is no bounce and the answer is simply `100 + 200·0.66 = 232`.)

### 5.7 State machines

The game is always in exactly one state, `serve`, `play` or `over`, and each state has simple rules for what updates and what does not (see `GUIDE.md` §6.1). State machines are how most games, protocols and UIs keep complicated behaviour manageable.

### 5.8 Separation of concerns

`game.py` never imports OpenCV and never sees a pixel; `vision.py` never knows the rules of hockey. That is why the game can be tested without a webcam, and why swapping the tracker (MediaPipe ↔ classical) changes one line in `main.py`.

---

## 6. Exercises

Try these first; answers follow.

**E1.** How many numbers make up one 960 × 540 colour frame?

**E2.** A fingertip jumps from 0 to 100 and stays there. With α = 0.55, what does the smoother output after frames 1, 2 and 3?

**E3.** A puck at `(400, 100)` moves with velocity `(500, 200)` px/s. At what height does it reach x = 730 (assume no friction)?

**E4.** The puck's friction constant is 0.25 per second. After how long does its speed halve (ignoring collisions)?

**E5.** The goal is drawn 150 px tall, but the code lets the puck through only when its centre is within 71 px of the centre line. How tall is the effective opening, and why might the authors have chosen that?

**E6.** A fast mallet and a fast puck move towards each other at 30 FPS. Why can the puck pass through the mallet without a collision being detected, and how does the code prevent it?

**E7.** Why are 4 point pairs enough to define a homography, and what goes wrong if three of the points are on one line?

**E8.** Why does `SkinTracker` ignore the top 20 % of the frame?

**E9. (Coding)** Currently the mallet freezes when your hand leaves the camera view. Modify `main.py` so the mallet slides back to its starting position `(120, 200)` after the hand has been missing for 1 second. What state do you need to add?

**E10. (Coding)** Add an `--alpha` option that sets the smoother's α, and write a test that a larger α reaches 90 % of a step in fewer frames.

### Answers

- **E1.** 540 × 960 × 3 = **1,555,200**.
- **E2.** **55.0, 79.75, 90.89** (see the table in §3.9).
- **E3.** `t = (730 − 400)/500 = 0.66 s`; `y = 100 + 200 × 0.66 =` **232**.
- **E4.** `ln 2 / 0.25 ≈` **2.77 s**.
- **E5.** `2 × 71 =` **142 px**. The code does not say why; a plausible reason is that a puck skimming the post would visually clip it, and the slightly narrower scoring zone avoids goals that look like they hit the frame. (Good habit: when the code gives no reason, say that it is your inference.)
- **E6.** Collisions are only tested at discrete steps. The two objects together can close about 130 px per frame, more than the roughly 96 px needed to jump past the collision zone, so they might never be "inside" each other at any checked moment. The code splits each frame into **4 sub-steps** (about 33 px per step) and interpolates the mallets between frames.
- **E7.** A homography has 8 unknowns and each point pair gives 2 equations: 4 × 2 = 8. If three points are collinear, the equations become dependent (a degenerate, non-invertible mapping), so no unique solution exists.
- **E8.** To avoid detecting the player's face (and the ceiling), which are skin-coloured and would otherwise be a larger blob than the hand.
- **E9.** Add a timer for "frames or seconds since the fingertip was last seen" in the main loop. When it exceeds 1 s, call `game.set_player_target(120, 200)` each frame (and optionally `smoother.reset()`). The smoother's output then moves the mallet back. Remember to reset the timer when a fingertip is found.
- **E10.** Add `ap.add_argument("--alpha", type=float, default=0.55)` and pass it to `Smoother(alpha=args.alpha)`. For the test, feed constant 100 into two smoothers (α = 0.3 and 0.8) and count frames until each exceeds 90: the larger α needs fewer frames (α = 0.8 needs 2; α = 0.3 needs 7).

---

## 7. Questions a lecturer might ask (viva)

The longer list in [`explanation.md`](explanation.md) §8 covers the image-processing side. More:

1. **Which parts are classical image processing and which are machine learning?** Classical: blur, colour conversion, thresholding, morphology, contours, perspective transform, blending, smoothing. ML: the MediaPipe hand-landmark model. The opponent AI is neither: it is hand-written rules.
2. **Why convert coordinates with the inverse homography instead of drawing the mallet at the fingertip pixel?** The game physics runs on a flat table; the fingertip must be expressed in that same flat space. Using the inverse of the drawing matrix keeps both consistent.
3. **Why smooth the position?** Detection noise would make the mallet vibrate; the moving average is a low-pass filter that trades a little lag for stability.
4. **What does alpha blending do here?** It mixes the rendered table with the live video so the camera image remains visible through it.
5. **Why does the game still work with no webcam?** Because `game.py` is independent of the camera; `main.py` falls back to mouse input.
6. **How do you know the physics is right without playing?** Unit tests set up exact situations (puck at a given position and velocity) and check outcomes: bounces, goals, no tunnelling.
7. **What would you improve first?** Test and tune with a real webcam, then replace the fixed trapezoid by marker-based table detection.

---

## 8. Lessons from the pre-release audit

Before documentation, the project was cloned, run, and its tests executed. What it showed:

| Finding | Lesson |
|---|---|
| **16 of 16 tests pass and the headless snapshot runs** | Good engineering: hardware-free tests make the core trustworthy. |
| **The downloaded model file (`hand_landmarker.task`, ~8 MB) was not in `.gitignore`** | Anything a program downloads or generates should be ignored by version control, or it will end up in the repository. Fixed. |
| **The README was not valid Markdown** (no headings, no table separators), so it would render as plain text on GitHub | Documentation is part of the product; preview it where it will be read. Rewritten. |
| **The real-webcam path has never been run**, as `explanation.md` itself states | Passing tests do not prove the whole system works. State clearly what is and is not verified; the docs now carry a status table. |
| **The sandbox could not download the model**, so the MediaPipe path was checked only for API compatibility | Network-dependent code needs a graceful fallback; this project has one (classic tracker). |

The meta-lesson: **say plainly what you tested, and what you did not.**

---

## 9. Where to go next

- Test the MediaPipe path on a real laptop and note the lighting conditions where each tracker works.
- Replace the fixed table with one found from four ArUco markers.
- Replace the heuristic opponent with a learned one, or tune the heuristic with data (see [`mllearning.md`](mllearning.md)).
- Add two-player mode with `num_hands=2`.
