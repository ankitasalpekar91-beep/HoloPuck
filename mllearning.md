# HoloPuck and Machine Learning

HoloPuck sits on the boundary between **classical image processing** and **machine learning (ML)**, and it deliberately contains both so you can compare them. This file explains exactly *where* ML is used, how that component works, how it differs from the classical alternative, how to evaluate trackers fairly, and what ML you could add next.

- **Students:** §1-§3 explain the concepts; §6 has exercises.
- **Lecturers:** §3-§4 give a ready comparison of ML versus classical vision; §5 covers bias and privacy.
- **Developers:** §4 (evaluation), §7 (extension ideas).

> **Honest status:** the MediaPipe path has not yet been run on a real hand in front of a real webcam (see `README.md`). Statements below about how the model behaves describe how it is designed and used by the code; they are not measurements from this project. §4 shows how to measure it yourself.

---

## 1. Where ML is, and is not, in this project

| Component | Technique | ML? |
|---|---|---|
| Hand / fingertip tracking, default mode | MediaPipe **HandLandmarker** (`HandTracker` in `vision.py`) | **Yes**, a pretrained neural-network model |
| Hand tracking with `--classic` | Blur → YCrCb threshold → morphology → contours (`SkinTracker`) | No, classical image processing |
| Perspective mapping, blending, smoothing | Homography, alpha blending, moving average | No |
| Puck physics | Equations of motion, collisions | No |
| Computer opponent | Hand-written rules with random aiming noise (`_ai_next`) | No, a heuristic agent; it does not learn |

So there is **exactly one ML component**: the hand-landmark model. And HoloPuck only ever performs **inference** (using the model); it never trains anything.

---

## 2. ML basics you need for this project

| Term | Meaning | In HoloPuck |
|---|---|---|
| **Model** | A function with many numeric parameters, fitted to data | The file `hand_landmarker.task` |
| **Training** | Adjusting parameters so the model's outputs match labelled examples | Done by Google researchers, **not** here |
| **Inference** | Running the trained model on new input | `detect_for_video(image, timestamp)` on every frame |
| **Pretrained model** | A model someone already trained, ready to use | You download it once (~8 MB) |
| **Supervised learning** | Learning from input→answer pairs | Training images of hands labelled with landmark positions |
| **Regression** | Predicting numbers rather than categories | The model predicts landmark *coordinates* |
| **Confidence score** | The model's own estimate of how sure it is | Thresholds 0.6 / 0.5 / 0.5 in `HandTracker` |

---

## 3. How the MediaPipe hand model is used

### 3.1 What goes in and what comes out

- **Input:** one frame as an RGB image. OpenCV delivers BGR, so `HandTracker.process` converts it with `cvtColor(..., COLOR_BGR2RGB)`, a classic source of silent colour bugs.
- **Output:** for each detected hand, **21 landmarks** (wrist, thumb joints, finger joints, fingertips). Each landmark has x and y **normalised to 0-1** (fraction of image width and height) plus a relative depth. HoloPuck uses only landmark **#8**, the index fingertip, and multiplies x and y by the frame width and height to get pixels.

```text
 0 wrist      4 thumb tip     8 index tip     12 middle tip    16 ring tip     20 little tip
```

### 3.2 Two stages

MediaPipe's hand-landmarker bundles two models:

1. **Palm detection**: finds where a hand is in the whole image.
2. **Hand landmark**: looks closely at that region and predicts the 21 points.

Splitting the job this way is efficient: finding a hand in a big image is expensive, locating points inside a small crop is cheaper.

### 3.3 Why `VIDEO` mode matters

`HandTracker` runs the model in **video mode**. The model uses information from earlier frames to *track* the hand instead of searching from scratch each time, which is faster and steadier. In this mode every call needs a **timestamp that strictly increases**; the code guarantees this with `ts = max(now_ms, last_ts + 1)`.

### 3.4 The three confidence thresholds

| Parameter | Value | If you raise it |
|---|---|---|
| `min_hand_detection_confidence` | 0.6 | fewer false hands, but real hands are missed more often |
| `min_hand_presence_confidence` | 0.5 | tracking is dropped sooner |
| `min_tracking_confidence` | 0.5 | model re-detects more often (slower, steadier if tracking was poor) |

Thresholds are the ML version of the classical trade-off between false alarms and misses (precision vs recall).

### 3.5 What the model does *not* do

It does not know about the game, the table or the fingertip's meaning. It reports points on a hand; everything after that (mapping to the table, smoothing, physics) is ordinary code.

---

## 4. ML versus classical, and how to compare them fairly

### 4.1 Side by side

| Aspect | MediaPipe (ML) | Classical (`--classic`) |
|---|---|---|
| **Idea** | Learned from many labelled hand images | Hand-designed rules: skin colour range + shape |
| **Lighting / background** | Designed to cope with varied conditions | Sensitive: colour range shifts with light; any skin-coloured object is a candidate |
| **Hand pose** | Any pose; it finds the fingertip joint explicitly | Assumes one finger pointing up (fingertip = topmost point) |
| **Setup** | Needs the model file (one-time download) and the `mediapipe` package | Nothing extra |
| **Explainability** | A black box; you can see landmarks but not *why* | Every step visible; press `M` to see the mask |
| **Failure feel** | Usually either right or lost | Gradual: noisy mask, jumping to the face |
| **Teaching value** | Shows modern practice | Shows the fundamentals (syllabus techniques) |
| **Verified here** | API compatibility only | Unit-tested on a synthetic hand; real-camera testing pending |

These are properties of the two designs, not benchmark results.

### 4.2 Why ship both?

- **Reliability:** if the model cannot be downloaded, the game switches to the classical tracker automatically.
- **Learning:** the classical pipeline uses the syllabus techniques; the ML path shows what replaced them in practice.
- **Honesty:** `explanation.md` points out that MediaPipe is a learned model, not a classical technique.

### 4.3 Measuring a tracker yourself

Judge by numbers, not impressions. Record a short clip (or capture frames) of the **same still hand** in the conditions you care about, run both trackers, and compute:

| Metric | How | Meaning |
|---|---|---|
| **Detection rate** | frames with a fingertip ÷ total frames | how often it finds the hand |
| **Jitter (px)** | standard deviation of the fingertip position while the hand is still | how steady the mallet will be |
| **Position error (px)** | distance between the tracker's fingertip and one you marked by hand | accuracy (needs a ground-truth label) |
| **Latency** | time from hand movement to mallet movement | how responsive it feels |

A small helper for the first two (it works with either tracker, because both have `.process(frame)`). It was tested on the synthetic hand used by the unit tests:

```python
import numpy as np

def evaluate(tracker, frames):
    """Detection rate and jitter (pixels) of a tracker on frames showing a STILL hand."""
    tips = [tracker.process(f) for f in frames]
    found = [t for t in tips if t is not None]
    rate = len(found) / len(frames)
    if len(found) < 2:
        return rate, float("nan")
    pts = np.array(found, dtype=float)
    jitter = float(np.std(pts, axis=0).mean())      # mean of std(x) and std(y)
    return rate, jitter
```

On 20 identical synthetic frames `SkinTracker` gives `(1.0, 0.0)`; with Gaussian pixel noise (σ = 8) added it still finds the hand every time but jitters by about 1.1 px; on blank frames it gives `(0.0, nan)`. Real frames from your camera will behave differently, which is the point of measuring.

**Fair-test checklist:** same clip for both trackers; several lighting conditions; several people (different hands and skin tones); report numbers for each condition separately.

---

## 5. Bias, robustness and privacy

- **Data bias.** A model is only as good as its training data, and a fixed colour threshold is only as good as the people it was tuned on. Both can work worse for some users or some lighting. The classical range `Cr 133-173, Cb 77-127` is a widely used skin range, but the authors state that lighting can shift it; test with a variety of people before claiming it works for everyone.
- **Robustness.** Gloves, rings, sleeves, motion blur, a second hand, or a hand partly out of frame can confuse either approach.
- **Privacy.** Reading the code: camera frames are processed **on the local machine**; the only network access is the **one-time model download** in `ensure_model`. Frames are not saved or uploaded (the only image written to disk is the synthetic one from `--snapshot`). That is a good default for a camera application and worth stating in any report or demo.
- **Safety of the download.** The model comes from a fixed URL; if you work in a restricted environment, download it manually and pass `--model PATH`.

---

## 6. Exercises

**M1.** List the components of HoloPuck that are machine learning and those that are not.

**M2.** Does HoloPuck *train* a model or *use* one? Which function call does the work on each frame?

**M3.** Why does `HandTracker` use `VIDEO` mode and increasing timestamps?

**M4.** You raise `min_hand_detection_confidence` from 0.6 to 0.9. What changes for the player?

**M5.** The fingertip x positions (pixels) reported for a still hand over five frames are 100, 102, 98, 101, 99. Compute the jitter (population standard deviation). If the smoother (α = 0.55) is applied, will the jitter go up or down?

**M6.** Design a fair experiment to decide whether `--classic` is good enough for your classroom.

**M7.** Why is a camera application well suited to doing ML inference on the local machine rather than sending frames to a server?

**M8.** A classmate says, "The ML tracker is always better, so delete the classical one." Give two arguments against.

### Answers

- **M1.** ML: the MediaPipe hand-landmark model (`HandTracker`). Not ML: skin tracker, perspective transform, blending, smoothing, physics, and the computer opponent (rules).
- **M2.** It **uses** a pretrained model; it never trains. The call is `self._landmarker.detect_for_video(image, ts)`.
- **M3.** Video mode lets the model track the hand across frames (faster and steadier than detecting from scratch), and it requires timestamps that always increase, so the code forces `ts ≥ last_ts + 1`.
- **M4.** Fewer false detections (for example a face or a poster mistaken for a hand), but the real hand will be missed more often, so "Show your hand to the camera" appears more and the mallet freezes more.
- **M5.** Mean = 100; deviations 0, 2, −2, 1, −1; squares sum to 10; variance 10/5 = 2; **jitter = √2 ≈ 1.41 px**. The exponential moving average is a low-pass filter, so the output jitter goes **down** (at the cost of a little lag).
- **M6.** Record the same still-hand clip in the classroom's lighting, with several students; run `evaluate` for both trackers; compare detection rate and jitter per person and per lighting condition; judge against a threshold agreed beforehand (for example, "detection ≥ 95 % and jitter ≤ 3 px for everyone").
- **M7.** Privacy (video of people stays on the device), latency (no network round trip per frame) and it works offline.
- **M8.** (1) The classical tracker is the fallback when the model cannot be downloaded, so the game stays playable. (2) It is the part that uses and teaches the syllabus techniques, and it is fully inspectable (the mask view). Also: "better" has not been measured on real hands yet.

---

## 7. Ideas for adding more ML (not implemented)

1. **Gesture commands.** MediaPipe already gives all 21 landmarks. A pinch (thumb tip close to index tip) could restart the match: first as a simple distance rule, later as a small classifier trained on recorded landmarks.
2. **Auto-calibrated skin colour.** Ask the player to place a hand in a box, then fit a colour model (for example, a simple Gaussian in Cr/Cb or k-means clusters) to those pixels, and replace the fixed range.
3. **A learned opponent.** `game.py` is independent of the camera and advances with `update(dt)`, so it can be run millions of times quickly. That makes it a good environment for **reinforcement learning**: let an agent play against itself and learn to defend and attack, and compare it with the hand-written AI. This is a larger project; the physics is already there.
4. **Learned smoothing or prediction.** Replace the moving average with a Kalman filter or a small learned model that predicts where the fingertip will be in the next frame, to reduce lag.
5. **Learning from your own data.** Record fingertip trajectories and tracker failures, label them, and study when each tracker fails.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **Landmark** | A named point on the hand; MediaPipe predicts 21 |
| **Inference** | Running a trained model on new data |
| **Pretrained model** | A model trained by someone else and reused |
| **Confidence threshold** | The minimum certainty a detection needs to be accepted |
| **Regression** | Predicting continuous numbers (such as coordinates) |
| **Jitter** | Frame-to-frame wobble of a measurement that should be steady |
| **Ground truth** | Reference answers (for example hand-marked fingertip positions) used to score a system |
| **Reinforcement learning** | Learning to act by trial and error to maximise a reward |
| **k-means** | A clustering method that groups similar data points |
| **Kalman filter** | A classic algorithm that combines predictions and noisy measurements |
