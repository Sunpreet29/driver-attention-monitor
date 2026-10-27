# Driver Attention Monitor

Real-time **Driver Monitoring System (DMS)** prototype: estimates driver
drowsiness from a cabin-facing camera using facial-landmark geometry.
Runs live on a laptop webcam at interactive frame rates.

> It is built to demonstrate the camera -> feature -> decision data path and a
> validation methodology, not to be deployed in a vehicle.

### This project involves calibrating your camera
Please checkout this Github repository of [camera calibration](https://github.com/Sunpreet29/Camera-Calibration) and learn how to do that in a few simple steps.

## What it does

- Detects a face and its eye landmarks with MediaPipe FaceMesh.
- Computes the **Eye Aspect Ratio (EAR)** per eye and averages them.
- Flags **drowsiness** when the eyes stay closed longer than a set duration
  (a temporal counter, so single blinks don't trigger it).

## The signal: Eye Aspect Ratio

For six eye landmarks p1…p6 (two corners, two upper-lid, two lower-lid):

```
        ||p2 - p6|| + ||p3 - p5||
EAR  =  ---------------------------
               2 * ||p1 - p4||
```

Open eye -> EAR ≈ 0.25–0.30. Closed eye -> EAR ≈ 0.
Reference: Soukupová & Čech, *Real-Time Eye Blink Detection using Facial
Landmarks* (2016).

## Pipeline

```
webcam frame -> FaceMesh landmarks -> EAR (per eye, averaged)
             -> temporal drowsiness counter -> overlay + alarm
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.main               # press 'q' to quit
```

## Repository layout

| File | Role |
|---|---|
| `src/ear.py` | The EAR formula -> the core signal |
| `src/landmarks.py` | MediaPipe FaceMesh wrapper, eye-landmark extraction |
| `src/drowsiness.py` | Temporal state machine (blink vs. drowsy) |
| `src/main.py` | Webcam loop, overlay, FPS |

## Known limitations

- The drowsy threshold should be calibrated to the measured FPS 
  of the target camera.
- EAR degrades with glasses glare, extreme head pose, and poor lighting-
  exactly the image-quality conditions a real interior camera must handle.
- Tuned on a visible-light webcam; automotive interior cameras are near-infrared.

## Roadmap

- [ ] Yawn detection via Mouth Aspect Ratio (MAR)
- [ ] Head-pose estimation (gaze-away / distraction) via `solvePnP`
- [ ] Hands-in-wheel-region check via MediaPipe Hands
- [ ] Quantitative validation on a public dataset (see below) with reported metrics
- [ ] Config file for thresholds; demo GIF
