# HoloPuck 🏒

**AR air hockey you play with your index finger, using Python and computer vision.**

HoloPuck is a Python air-hockey game that uses a webcam and hand tracking instead of a mouse. A virtual rink is drawn on top of your live camera image, your fingertip moves the blue mallet, and a computer opponent controls the red one. First to 7 goals wins.

The project uses **OpenCV** to process live video and **MediaPipe** to detect hand landmarks. The detected fingertip position is converted into table coordinates and drives the game logic (puck movement, collisions, scoring). It explores practical **image processing and computer vision**, and the ideas behind **Augmented Reality (AR)**.

```text
 webcam ──► find fingertip ──► map to table ──► smooth ──► game physics ──► draw table
                                                                              │
                                      warp into camera view + blend ◄─────────┘
```

---

## Documentation

| You are... | Read |
|---|---|
| **Anyone** who wants to understand the project | [`GUIDE.md`](GUIDE.md): the full guide to every module |
| A **student** | [`learning.md`](learning.md): concepts, maths and exercises |
| Interested in the **ML** side | [`mllearning.md`](mllearning.md): the hand-tracking model, ML vs classical vision |
| Preparing a **demo or viva** | [`explanation.md`](explanation.md): team guide mapped to the Image Processing syllabus, with viva questions |

---

## Quick start

**Requirements:** Python 3.10 or newer, a laptop or desktop with a webcam, and a keyboard. (MediaPipe does not always publish packages for the very newest Python; 3.10-3.12 is the safest. The tests and the headless snapshot also ran fine on 3.13 with current library releases.)

```bash
git clone https://github.com/ankitasalpekar91-beep/HoloPuck.git
cd HoloPuck
pip install -r requirements.txt

python main.py              # play with your hand (MediaPipe)
python main.py --classic    # play with the classical skin-colour tracker (no ML)
python main.py --mouse      # control the mallet with the mouse
```

On the **first** MediaPipe run the hand model (`hand_landmarker.task`, about 8 MB) is downloaded automatically, so you need internet once. If the download fails, the game switches to the classic tracker by itself.

**Tracking tips:** good light from the front, a plain background, one hand, index finger pointing up.

### Controls

| Key | Action |
|---|---|
| `1` / `2` / `3` | Difficulty: Easy / Medium / Hard |
| `P` | Toggle perspective view |
| `H` | Toggle hand landmarks |
| `M` | Toggle the segmentation-mask inset (only with `--classic`) |
| `R` | Restart the match |
| `Q` or `Esc` | Quit |

### Command-line options

| Option | Meaning |
|---|---|
| `--camera N` | Webcam index (default `0`) |
| `--difficulty 1\|2\|3` | Starting difficulty (default `2`) |
| `--win-score N` | Goals needed to win (default `7`) |
| `--flat` | Start with the perspective view off |
| `--mouse` | Mouse controls the mallet; the webcam image is still shown behind the table |
| `--no-camera` | Plain background, mouse control (implies `--mouse`) |
| `--classic` | Track the hand with classical image processing instead of MediaPipe |
| `--model PATH` | Path to `hand_landmarker.task` (downloaded if missing) |
| `--snapshot out.png` | Headless test: simulate a few seconds, save one frame, exit |

If no webcam can be opened, HoloPuck falls back to mouse control on a plain background.

---

## Features

- **Real-time hand tracking**: the index fingertip is the control point for the mallet.
- **Two trackers**: MediaPipe (machine-learning model) and a classical pipeline (blur → YCrCb skin threshold → morphology → contours → topmost point).
- **AR overlay**: the flat table is warped with a perspective transform and alpha-blended onto the live camera view.
- **Puck physics**: wall bounces, mallet collisions with restitution, friction damping, speed limits, and sub-stepping so a fast puck cannot pass through a mallet.
- **Computer opponent** with three difficulty levels (speed and aiming noise; Medium and Hard also predict wall bounces).
- **Scoring**: goals, serve after each goal, first to *N* wins, restart with `R`.
- **Visual feedback**: hand skeleton, fingertip marker and segmentation-mask inset to make the tracking visible.
- **Fallbacks**: no camera → mouse; model download fails → classic tracker.

---

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main language and game logic |
| OpenCV | Webcam access, image processing, perspective transform, drawing |
| MediaPipe | Hand landmark detection and tracking (pretrained model) |
| NumPy | Coordinates, vectors and image arrays |
| Webcam | Live video of the player's hand |

---

## Project structure

```text
HoloPuck/
├── main.py            run this: webcam loop, keys, command-line options
├── vision.py          perspective transform, hand trackers, smoothing
├── game.py            puck physics, scoring, computer opponent
├── render.py          drawing the table and the on-screen text (HUD)
├── requirements.txt
├── tests/
│   └── test_holopuck.py     16 automated tests
├── README.md          you are here
├── GUIDE.md           full guide to the code
├── learning.md        concepts, maths, exercises
├── mllearning.md      the machine-learning side
├── explanation.md     team guide + Image Processing syllabus mapping + viva questions
├── LICENSE
└── .gitignore
```

`game.py` works only in flat *table coordinates* (800 × 400) and never touches the camera; `vision.py` is responsible for converting camera pixels to table coordinates. This separation is why the game can be tested without a webcam.

---

## Image-processing concepts used

1. **Image acquisition**: each webcam frame is a NumPy array (height × width × 3).
2. **Colour-space conversion**: BGR → RGB for MediaPipe; BGR → YCrCb for skin segmentation.
3. **Spatial filtering**: Gaussian blur to reduce sensor noise.
4. **Segmentation by thresholding**: keep pixels whose Cr/Cb values fall in a skin range.
5. **Morphology**: opening removes specks, closing fills holes.
6. **Contours and extreme points**: the hand outline, and its topmost point as the fingertip.
7. **Hand landmark detection and tracking** (MediaPipe): landmark 8 is the index fingertip.
8. **Geometric transforms**: mirror flip, resize, and the perspective transform (homography) in both directions.
9. **Alpha blending**: `out = (1 − α)·camera + α·table`.
10. **Temporal smoothing**: exponential moving average to steady the mallet.

`GUIDE.md` and `learning.md` explain each of these in plain language.

---

## Testing and current status

```bash
python -m unittest discover -s tests -v        # 16 tests
python main.py --snapshot out.png              # headless smoke test, no camera or window needed
```

| Area | Status |
|---|---|
| Physics, goals, scoring, collisions, no-tunnelling | covered by automated tests |
| Perspective transform round-trip, overlay, smoothing | covered by automated tests |
| Classical tracker on a synthetic hand | covered by automated tests |
| Game runs end to end headlessly | `--snapshot` works |
| MediaPipe API compatibility | the hand-landmarker API imports and is used correctly |
| **MediaPipe tracker on a real hand, real webcam** | **not yet tested** |
| **Tuning on a real camera and room lighting** | **not yet tested** |

The tracker thresholds (skin range, smoothing α, where the table sits in the frame) may need small adjustments for your camera and lighting. The constants are at the top of `vision.py`.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| "Could not open camera 0" | Another app is using the webcam, or use `--camera 1`. The game falls back to the mouse automatically. |
| Hand model will not download | Check your connection; or download it manually (the error message prints the URL) and save it next to `main.py` as `hand_landmarker.task`; or use `--classic` / `--mouse`. |
| `ModuleNotFoundError: cv2` / `mediapipe` | Run `pip install -r requirements.txt`. |
| Mallet jumps or shakes | Improve lighting and keep one hand in view. Lower the smoothing `alpha` in `main.py` (`Smoother(alpha=0.55)`) for steadier but laggier movement. |
| `--classic` follows my face or arm | Use a plain background and keep your face out of the lower part of the frame; the top 20 % of the image is ignored on purpose. |
| Window does not open on a server | Use `--snapshot out.png`, which needs no display. |

---

## Applications

Gesture-controlled games, touch-free interfaces, computer-vision learning, interactive teaching, AR demonstrations and hand-tracking experiments.

## Future scope

- Detect a real table or marker (for example ArUco) and compute the perspective transform from it.
- Two-player mode using two hands.
- Auto-calibration of the skin range; sliders for the four table corners.
- Sound effects and goal effects.
- Mobile AR (for example Unity or Vuforia).

## License

MIT. See [`LICENSE`](LICENSE).

---

**Project:** HoloPuck: AR Air Hockey · **Domain:** Image Processing and Computer Vision · **Language:** Python · **Tools:** OpenCV, MediaPipe, NumPy
