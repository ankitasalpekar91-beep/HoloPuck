# HoloPuck — Team Explanation

> **Project:** HoloPuck — AR Air Hockey  
> **Subject:** Image Processing  
> **Stack:** Python · OpenCV · MediaPipe · NumPy

This file explains **what the project is, how every part works, and which Image Processing topic each part belongs to**, so that everyone on the team can explain any part in the demo or viva.

> **Note on the syllabus mapping:** the topic names below follow the standard Image Processing textbook order (Gonzalez & Woods style). If our college syllabus uses different unit names, only the *left column* of the mapping table in Section 4 needs renaming — the code locations stay the same.

---

## 1. What is HoloPuck?

HoloPuck is an air hockey game that you play **with your finger, in front of a webcam**. There is no mouse and no controller.

- The webcam films you.
- The program finds your **index fingertip** in every video frame.
- A virtual air-hockey table is drawn **on top of the live camera image** (that is the "AR" part).
- Your fingertip moves the blue mallet. A computer opponent controls the red mallet. First to 7 goals wins.

The game itself is simple. The interesting part — and the part that matters for our subject — is the **image-processing pipeline** that turns a camera picture into a game control.

---

## 2. The pipeline (one video frame, start to finish)

```
 Webcam frame (BGR image)
        │
        ▼
 ① Mirror flip + resize to 960×540            ← geometric operations
        │
        ▼
 ② Find the fingertip (pixel x, y)            ← detection / segmentation
        │        ├─ MediaPipe mode : ML hand landmarks, landmark #8 = index tip
        │        └─ Classic mode   : blur → YCrCb → threshold → morphology
        │                            → contours → topmost point
        ▼
 ③ Camera pixel → table coordinate            ← inverse perspective transform (homography)
        │
        ▼
 ④ Smooth the position (EMA filter)           ← temporal filtering
        │
        ▼
 ⑤ Game update: mallet moves, puck physics,   ← game logic (no image processing)
    collisions, AI opponent, score
        │
        ▼
 ⑥ Draw the flat table (OpenCV drawing)       ← rendering
        │
        ▼
 ⑦ Warp table into the camera view (perspective transform)
    and alpha-blend it over the live image    ← geometric transform + image blending
        │
        ▼
 ⑧ Draw score / HUD and show the window
```

Steps ③ and ⑦ use the **same matrix `M`** — one forwards (table → camera), one inverted (camera → table). That is why the finger and the drawn mallet always line up.

---

## 3. Files in the project

| File | What it does |
|---|---|
| `main.py` | The program you run. Opens the webcam, runs the loop in Section 2, handles keys and command-line options. |
| `vision.py` | **All the image processing:** `TableView` (perspective transform + overlay), `HandTracker` (MediaPipe), `SkinTracker` (classical tracker), `Smoother` (EMA filter). |
| `game.py` | Game rules: puck physics, wall bounces, mallet collisions, goals, scoring, computer opponent. Works in flat table coordinates and never touches the camera. |
| `render.py` | Draws the flat table, mallets, puck and the on-screen text (HUD) using OpenCV drawing functions. |
| `tests/test_holopuck.py` | 16 automatic tests (physics, goals, homography, tracker, filter). |
| `requirements.txt` | Libraries to install. |
| `explanation.md` | This file. |

`vision.py` is the file to read first for the syllabus; `game.py` is ordinary programming.

---

## 4. Mapping to the Image Processing syllabus

| Syllabus topic | Where it appears in HoloPuck | Code |
|---|---|---|
| **Digital image fundamentals** — an image is a 2-D array of pixels; sampling, resolution, channels | Every webcam frame is a NumPy array of shape `(height, width, 3)`, values 0–255. We resize it to 960×540. | `main.py` → `cv2.resize` |
| **Colour models / colour image processing** — RGB, YCbCr, HSV | OpenCV gives **BGR**, MediaPipe needs **RGB** (`cvtColor`). The classic tracker converts to **YCrCb** because skin colour forms a tight cluster in the Cr/Cb channels, almost independent of brightness. | `vision.py` → `HandTracker.process`, `SkinTracker.process` |
| **Spatial filtering / smoothing** — Gaussian filter, noise removal | A 7×7 **Gaussian blur** removes sensor noise before thresholding. | `SkinTracker.process` → `cv2.GaussianBlur` |
| **Image segmentation — thresholding** | `cv2.inRange` keeps only pixels whose Cr/Cb values fall in the skin range → a **binary mask** (white = skin). | `SkinTracker.process` |
| **Morphological image processing** — erosion, dilation, opening, closing | **Opening** (erode → dilate) deletes small white speckles; **closing** (dilate → erode) fills small holes in the hand. Structuring element: 5×5 ellipse. | `SkinTracker.process` → `cv2.morphologyEx` |
| **Representation & description — contours, boundary, convex hull** | `findContours` extracts the hand outline, `contourArea` picks the largest blob (the hand, not noise), `convexHull` is drawn for visualisation. | `SkinTracker.process` / `draw` |
| **Feature / extreme-point detection** | The **fingertip = the topmost point of the hand contour** (an extreme point). We average a thin band of points near the top so the tip does not flicker. | `SkinTracker.process` |
| **Object detection & tracking in video** | MediaPipe finds 21 hand landmarks per frame and tracks them from frame to frame; landmark **#8** is the index fingertip. | `HandTracker.process` |
| **Geometric transformations** — flip, scale, **perspective (projective) transform** | Mirror flip (`cv2.flip`), resize, and the main one: a **homography** that maps the flat 800×400 table onto a trapezoid in the camera view so it looks 3-D. | `vision.py` → `TableView`; `main.py` |
| **Coordinate mapping / inverse mapping** | The fingertip is in *camera pixels*; the game needs *table coordinates*. We apply the **inverse** homography to every fingertip position. | `TableView.frame_to_table` |
| **Image arithmetic / blending** | The virtual table is merged with the live image using **alpha blending**: `out = (1 − α)·frame + α·table`, α = 0.8 inside the table, 0 outside. | `TableView.overlay` |
| **Temporal filtering** (noise reduction over time) | Hand-tracking positions jitter by a few pixels. An **exponential moving average** `s = α·new + (1−α)·s` (α = 0.55) steadies the mallet. | `vision.py` → `Smoother` |
| **Drawing / rendering primitives** | Table, circles, lines, anti-aliased shapes and text are drawn with `cv2.circle`, `cv2.line`, `cv2.rectangle`, `cv2.putText`. | `render.py` |
| **Real-time image processing, AR basics** (AR / VR / MR concepts) | The whole project: live camera + registered virtual object = augmented reality. "Registered" means the virtual table is geometrically aligned with the camera view. | `main.py` |

> **Be honest in the viva:** MediaPipe's landmark detector is a *machine-learning (CNN-based)* model, not a classical technique. That is exactly why we also built the **classic mode** (`--classic`): it uses only syllabus techniques (blur, colour threshold, morphology, contours) and shows the same idea without a trained model.

---

## 5. The key ideas explained simply

### 5.1 Skin segmentation (classic mode)

1. **Blur** the frame slightly so single noisy pixels do not matter.
2. Convert **BGR → YCrCb**. `Y` is brightness; `Cr` and `Cb` are colour. In OpenCV: `Y = 0.299R + 0.587G + 0.114B`, `Cr = (R − Y)·0.713 + 128`, `Cb = (B − Y)·0.564 + 128`.
3. **Threshold**: keep pixels with `133 ≤ Cr ≤ 173` and `77 ≤ Cb ≤ 127`. Brightness (`Y`) is ignored, so shadows matter less than they would in RGB.
4. **Morphology** cleans the mask: opening removes dots, closing fills gaps.
5. **Largest contour** = the hand. If it is smaller than 0.4 % of the frame, we say "no hand".
6. **Topmost point** of that contour = fingertip.

Press **M** while playing with `--classic` to see the binary mask live in the top-right corner — this is the best thing to show in a demo.

**Weak spots (say them before the examiner does):** a face or any skin-coloured object in the lower 80 % of the frame confuses it; very bright or coloured lighting shifts the skin range; the finger must point upwards.

### 5.2 The perspective transform (homography)

A flat rectangle seen at an angle becomes a trapezoid. A **homography** is a 3×3 matrix `H` that does this for every point:

```
[x']       [x]
[y'] ∝ H · [y]       then divide by the third component (w) to get pixels
[w ]       [1]
```

- We know **4 point pairs**: the 4 table corners `(0,0) (800,0) (800,400) (0,400)` ↔ 4 trapezoid corners in the frame. Four pairs give 8 equations for the 8 unknowns of `H`, solved by `cv2.getPerspectiveTransform`.
- `cv2.warpPerspective(table, M, frame_size)` draws the table into the camera view.
- `M⁻¹` goes the other way: fingertip pixel → table coordinate. A unit test (`test_homography_round_trip`) checks that going there and back returns the same point.
- Press **P** to switch between the perspective view and a flat view.

### 5.3 Smoothing

Raw fingertip positions jump around by a few pixels each frame, so the mallet would shake. The EMA filter blends each new position with the previous smoothed one. A **larger α** means a faster response but more jitter; a **smaller α** is steadier but laggy. We use α = 0.55 as a compromise.

### 5.4 Why mirror the image?

A webcam image is like looking *at* yourself, not in a mirror: moving your hand right makes it go left on screen. `cv2.flip(frame, 1)` mirrors it so the game feels natural.

---

## 6. How to run it

```bash
git clone https://github.com/ankitasalpekar91-beep/HoloPuck.git
cd HoloPuck
pip install -r requirements.txt

python main.py              # play with your hand (MediaPipe)
python main.py --classic    # play with the classical skin-colour tracker
python main.py --mouse      # no camera tracking: mouse controls the mallet
python main.py --no-camera  # mouse + plain background
```

Other options: `--difficulty 1|2|3`, `--win-score N`, `--flat`, `--camera INDEX`.

**Keys:** `1 2 3` difficulty · `P` perspective on/off · `H` hand landmarks on/off · `M` mask inset on/off (`--classic`) · `R` restart · `Q` / `Esc` quit.

**First run with MediaPipe:** the hand model file (about 8 MB, `hand_landmarker.task`) is downloaded automatically once, so you need internet that first time. If the download fails, the game switches to the classic tracker by itself.

**Tips for tracking:** good light from the front, a plain background, one hand, index finger pointing up.

**Run the tests:** `python -m unittest discover -s tests -v`

**Headless check (no camera or window needed):** `python main.py --snapshot out.png` simulates a few seconds of play and saves one frame.

---

## 7. What has and has not been tested

- ✅ Automated: physics, goals, scoring, collisions, homography round-trip, overlay blending, smoothing, and the classic tracker on a synthetic hand (16 tests pass).
- ✅ The game window starts and runs without errors in a virtual display.
- ⚠️ **Not yet tested with a real webcam and a real hand.** The tracker thresholds (skin range, α, table position in the frame) may need small tuning on your own camera and lighting. The constants are at the top of `vision.py`.
- ⚠️ MediaPipe's hand model could not be downloaded in the environment where the code was written, so the MediaPipe path was checked for API compatibility but not run on real hand images. Please test it once on a laptop with internet.

---

## 8. Likely viva questions

**Q1. Which image-processing techniques did you use?**  
Colour-space conversion, Gaussian filtering, thresholding, morphological opening/closing, contour extraction and extreme-point detection (classic mode); perspective transformation, inverse mapping, flipping and resizing; alpha blending; exponential smoothing over time.

**Q2. Why YCrCb and not RGB for skin?**  
In RGB, brightness is mixed into all three channels, so the same skin looks very different in shadow. YCrCb separates brightness (Y) from colour (Cr, Cb), and skin colour stays in a small Cr/Cb region.

**Q3. What do opening and closing do?**  
Opening = erosion then dilation → removes small bright specks. Closing = dilation then erosion → fills small dark holes.

**Q4. What is a homography and how many points are needed?**  
A 3×3 projective matrix that maps one plane to another. Four non-collinear point pairs are enough (8 unknowns).

**Q5. How does the fingertip control the mallet?**  
Detect fingertip pixel → apply the inverse homography to get table coordinates → smooth → set as the mallet target.

**Q6. How is the virtual table shown on the video?**  
Draw it flat, warp it with the homography, and alpha-blend it with the camera frame.

**Q7. Why do you need smoothing?**  
Detection noise makes the position jitter. EMA is a simple low-pass filter over time.

**Q8. What are the limitations?**  
Skin-colour segmentation is lighting-dependent and confuses faces; the table position is fixed (not detected from the room); the game is 2-D.

**Q9. How is this AR and not just a game?**  
It combines a live camera view with a virtual object that is geometrically aligned to it and reacts to a real-world object (your hand).

---

## 9. Ideas for improving it

- Detect a real table or marker (ArUco / four coloured corners) and compute the homography from it instead of a fixed trapezoid.
- Add sound effects and a goal flash.
- Two-player mode using two hands (`num_hands=2`).
- Auto-calibrate the skin range from a hand placed in a box on screen.
- Replace the fixed trapezoid with sliders for the four corners.
