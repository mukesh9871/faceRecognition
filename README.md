# faceRecognition

Real-time face recognition and attendance logging from a webcam or video file.

- **Detection:** OpenCV [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), a small CNN detector that also returns 5 landmarks.
- **Recognition:** OpenCV [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface). Each face is aligned on its landmarks and turned into a 128-D embedding, which is matched by cosine similarity.
- **Tracking:** an IoU tracker keeps one identity per face across frames. A face is embedded only when it first appears, then re-checked periodically, rather than on every frame. The identity is confirmed by a vote over several frames, so names don't flicker.
- **Threads:** camera capture, inference and display each run on their own thread, so the window always shows the newest frame.
- **Attendance:** stored in SQLite, one row per person per day with first and last seen time. You can list it or export it to CSV.

Only `opencv-python`, `numpy` and `pyyaml` are required, and everything runs on CPU.

## Performance

Measured on an Intel Core Ultra 9 275HX with a 1280×720 clip of 3 moving faces, using `facerec run --benchmark 300`:

| Pipeline | ms / frame | FPS | Faces found |
|---|---|---|---|
| Old version (`face_recognition` / dlib HOG, every frame) | 412 | 2.4 | 2.0 of 3 |
| New, embedding every face on every frame (`--every-frame`) | 14.9 | 67 | 3 of 3 |
| **New, with tracking (default)** | **6.1** | **165** | **3 of 3** |

A webcam is limited to its own frame rate (usually 30 FPS), so the pipeline has plenty of spare capacity.

## Setup

Python 3.10–3.13 is supported.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
facerec download-models           # ~39 MB into models/, SHA256-verified
```

## Enrol people

Put one or more photos per person in `data/faces/<Name>/`:

```
data/faces/
├─ Mukesh/
│  ├─ front.jpg
│  └─ side.jpg
└─ ElonMusk/
   └─ ElonMusk.jpg
```

Several photos per person, with different angles and lighting, improve accuracy. Then run:

```bash
facerec enroll                              # embeds new/changed photos, caches the rest
facerec enroll --from-camera --name Alice   # auto-capture 5 photos of Alice from the webcam
```

Embeddings are cached in `data/gallery.npz`. Only new or changed photos are re-embedded.

## Run

```bash
facerec run                         # webcam from config.yaml (camera.source)
facerec run --camera 1              # another webcam
facerec run --source video.mp4      # a video file or stream URL
facerec run --no-attendance         # view only
```

Keys in the live window:

| Key | Action |
|---|---|
| `Esc` / `q` | Quit |
| `h` | Toggle the performance overlay |
| `r` | Reload the faces folder without restarting |

Box colours:

| Colour | Meaning |
|---|---|
| Green | Known person |
| Red | Unknown |
| Amber | Still deciding |

## Attendance

```bash
facerec list                        # today
facerec list --day 2026-11-08
facerec list --all
facerec export --csv today.csv      # also accepts --day / --all
```

A person is logged only after their identity has been confirmed, never as "Unknown". `last_seen` is updated at most every `attendance.cooldown_s` seconds.

## Benchmark

```bash
facerec run --source clip.mp4 --benchmark 300                # default (tracked) pipeline
facerec run --source clip.mp4 --benchmark 300 --every-frame  # without tracking, for comparison
```

This prints mean, p50 and p95 milliseconds for each stage (detect, track, embed, match), plus embeddings per frame and FPS.

## Configuration

All settings live in [config.yaml](config.yaml). The ones you are most likely to change:

| Key | Meaning |
|---|---|
| `camera.source` | Webcam index, or a file path / URL |
| `camera.backend` | `auto` uses DirectShow on Windows, which opens faster and lags less than MSMF |
| `detector.max_side` | Detection resolution. Lower is faster; higher finds smaller faces |
| `recognizer.threshold` | Cosine similarity needed for a match (default 0.363). Raise it if strangers get matched; lower it if known people show as Unknown |
| `tracker.vote_frames` / `tracker.confirm_votes` | How many frames are needed to confirm an identity |
| `tracker.reidentify_interval` | How often (in frames) a confirmed face is re-checked |

## Project layout

```
src/facerec/
  cli.py            command line (run / enroll / list / export / download-models)
  app.py            live view, benchmark, camera enrolment, reports
  config.py         typed config loaded from YAML
  capture.py        ThreadedCamera: background grab, newest-frame-only
  detectors/        Detector protocol + YuNet
  recognizers/      Recognizer protocol + SFace
  registry.py       backend name -> implementation (add new backends here)
  gallery.py        enrolment, embedding cache, vectorised matching
  tracker.py        IoU tracker, recognition scheduling, identity voting
  pipeline.py       RecognitionEngine (sync) + AsyncPipeline (worker thread)
  attendance.py     SQLite store + rate-limited, queued logger
  overlay.py        boxes, labels, HUD
tests/              pytest unit tests (no camera or models needed)
```

To add a backend, such as a GPU ArcFace model through onnxruntime:

1. Implement the `Detector` or `Recognizer` protocol.
2. Register it in `registry.py`.
3. Select it with `detector.name` or `recognizer.name` in the config.

## Tests

```bash
pytest
```

---
Copyright: mukesh9871@gmail.com
