# facerec review checklist

Go through the sections that the diff touches. Each rule comes with its reason, so you can judge cases the list doesn't cover.

## Contents
1. Architecture map
2. Real-time budget
3. Threading
4. Coordinates and models
5. Tracker and voting
6. Gallery and cache
7. Attendance
8. Config
9. Backends and registry
10. Capture
11. Privacy and safety
12. Tests
13. PR hygiene
14. Regressions already fixed once

---

## 1. Architecture map

```
cli.py ──> app.py ──┬─> registry.py ──> detectors/yunet.py, recognizers/sface.py
                    ├─> gallery.py      (build_gallery, Gallery.match)
                    ├─> capture.py      (ThreadedCamera: camera thread)
                    ├─> pipeline.py     (RecognitionEngine; AsyncPipeline: inference thread)
                    │       └─> tracker.py, gallery.match, attendance.AttendanceLogger
                    ├─> attendance.py   (AttendanceLogger: writer thread; AttendanceStore)
                    └─> overlay.py      (drawing, main thread only)
config.py <- config.yaml     structures.py: Detection, MatchResult, TrackView
```

The app runs on 4 threads:

| Thread | Does | Calls |
|---|---|---|
| main | display and keys | `cv2.imshow`, `cv2.waitKey` |
| camera | reads frames | `ThreadedCamera._loop` |
| inference | recognition | `AsyncPipeline._loop` → `RecognitionEngine.process` |
| attendance | database writes | `AttendanceLogger._writer` |

Baseline numbers, used by `bench_compare.py` (720p, 3 faces, CPU):

| Stage | Time |
|---|---|
| detect | ~5.5 ms |
| embed | ~3 ms per face (0.13 embeddings per frame when tracked) |
| total | ~6 ms per frame (~165 FPS capacity) |

The old code took 412 ms per frame.

## 2. Real-time budget

- **Embeddings happen only through tracker scheduling.** `recognizer.embed` should run only for tracks that `FaceTracker.needs_embedding` selects, in `RecognitionEngine.process`, or during enrolment in `gallery.py`. Any other per-frame embed puts the cost back to 3 ms per face per frame. That is exactly what made the old version slow.
- **Nothing on the inference path should block.** That covers `detect`, `track`, `embed`, `match` and the attendance hand-off. Look for:
  - `time.sleep`
  - file or network I/O
  - SQLite calls
  - `print` in tight loops
  - waiting on a lock the main thread holds for a long time

  Attendance must stay as `queue.put` calls to the writer thread.
- **Detection runs downscaled; recognition crops come from full resolution.** `YuNetDetector` resizes to `max_side` (640). Running YuNet on full-resolution frames is slower *and* misses large faces. Cropping recognition faces from the small frame hurts accuracy.
- **No per-face Python loops where numpy works.** Matching is one `Q @ G.T` followed by `np.maximum.reduceat`. A per-person or per-embedding Python loop in `Gallery.match` is a regression once the gallery grows.
- **Allocation in hot loops.** Watch for `frame.copy()`, `cv2.resize`, or `np.array` built from lists every frame. A `frame.copy()` that wasn't there before needs a reason. The one in `run_live` is required, because the camera's array is shared with the inference thread.
- **Prove it.** If a hot-path file changed, the review needs `bench_compare.py` numbers.

## 3. Threading

- **OpenCV GUI calls only on the main thread.** That means `imshow`, `waitKey`, `namedWindow` and `destroyAllWindows`. Calling them from a worker freezes or crashes the window on Windows.
- **Copy before drawing.** `ThreadedCamera.read` returns the shared array. Drawing on it corrupts the frame the inference thread is reading.
- **Model objects are not thread-safe.** `FaceDetectorYN.setInputSize` and `detect` keep state between calls, and so does SFace. Any second thread, such as gallery reload on the `r` key, must create its **own** detector and recognizer (`load_gallery(cfg)` with no instances does this). Never share the engine's instances.
- **Swap shared state by reassignment, not mutation.** `engine.gallery = new_gallery` is safe because reassigning an attribute is atomic. Changing a `Gallery`'s arrays in place while `match` runs is not.
- **Exceptions in worker threads must surface.** `AsyncPipeline` stores `error` and the main loop re-raises it. A new thread that swallows exceptions leaves the UI showing stale boxes forever.
- **Every thread must shut down.**
  - `stop()` and `close()` join with a timeout.
  - Daemon flags are set.
  - The attendance queue is drained on `close()`, so the last updates aren't lost.
- **Locks protect only short critical sections.** No inference, I/O or GUI work while holding `AsyncPipeline._lock` or `ThreadedCamera._cond`.

## 4. Coordinates and models

- **Box format:** boxes are `(x, y, w, h)` in **full-frame pixels** everywhere (`Detection.bbox`, `TrackView.bbox`). The old code mixed this up with `(top, right, bottom, left)`. Check every new unpacking.
- **YuNet row layout:** `[x, y, w, h, 5×(lx, ly), score]`, 15 values.
  - When rescaling, divide **only `[:14]`**. The score must not be scaled.
  - `Detection.raw` must stay the full-resolution 15-value row, because `SFace.alignCrop` consumes it.
- **Scale factors come from the actual resize,** never a hard-coded constant. The old code had `* 4`, which broke as soon as the scale changed.
- **Embeddings are L2-normalised float32.** `Gallery` re-normalises, but a recognizer returning unnormalised vectors makes thresholds meaningless elsewhere.
- **Thresholds belong to a backend.**
  - The 0.363 cosine threshold is SFace's.
  - A new recognizer needs its own default threshold, documented in `config.yaml`.
  - Thresholds don't carry over between backends.
- **Colour order:** OpenCV models here take **BGR**. Don't add `cvtColor(BGR2RGB)`; that was a dlib requirement.
- **Large images:** YuNet misses big faces at full resolution, so enrolment relies on the same downscale.

## 5. Tracker and voting

- **Scheduling decides cost.** Changes to `needs_embedding` decide how much work runs per frame. With the defaults (`vote_frames=5`, `reidentify_interval=30`), a steady face costs 5 embeddings plus one every 30 frames. Check new conditions for cases that fire every frame, e.g. an area-growth comparison against a value that is never updated.
- **Voting semantics:**
  - Majority over the last `vote_frames` votes.
  - Confirmed when the top name has at least `min(confirm_votes, vote_frames)` votes.
  - A confirmed identity flips only when another name reaches the quorum.
  - `None` (Unknown) is a valid vote and can be confirmed.

  Changes here need tests for flicker resistance and identity switching.
- **Association:** greedy IoU at `iou_threshold`. Watch for one detection being matched to two tracks, and for tracks never expiring (`misses > max_misses`).
- **Small faces** (`min_side < min_face_size`) are tracked but never embedded. Make sure new paths respect this, including `always_embed`.
- **Track IDs are monotonic** and never reused, because attendance events record `track_id`.

## 6. Gallery and cache

- **Cache validity:** an entry is reused only if its relative path, mtime_ns, size **and** person all match, and the file's `backend` equals `registry.backend_id(cfg)`. Anything that changes the embedding space must change `backend_id`, or stale vectors are silently mixed with new ones:
  - a new model file
  - preprocessing changes
  - a new embedding dimension
- **`np.load(..., allow_pickle=False)`** always. The cache is a file on disk, and pickle would let it execute code.
- **Rows must be sorted by label.** `Gallery.__init__` sorts by label so that `reduceat` start indices are valid. Anything that builds `Gallery` must go through `__init__`, not assign `embeddings` and `labels` directly.
- **Enrolment photos:**
  - A photo with no face is skipped with a reason, not a crash.
  - With several faces, the largest is used and a warning is logged.
  - An empty gallery returns Unknown and must not crash (the old code crashed in `np.argmin` on an empty array).
- **Read images with `gallery.read_image`** (`np.fromfile` + `imdecode`), not `cv2.imread`. `imread` fails on non-ASCII Windows paths, such as people's names.

## 7. Attendance

- **Never log Unknown.** Only `t.confirmed and t.name` tracks reach the logger.
- **Rate limit:** at most one `last_seen` update per person per `cooldown_s`, and one `arrived` event per track per confirmed identity (`Track.announced`). Logging every frame was a bug in the old code.
- **SQLite connections:** one connection per thread. The writer thread owns its own. Readers (`list`, `export`) open a separate `AttendanceStore`. Never share a connection across threads.
- **Parameterised SQL only.** No f-strings or `%` formatting in `execute`.
- **Upsert keeps the earliest `first_seen` and the latest `last_seen`.** `MIN`/`MAX` comparisons on ISO strings work only while the timestamp format is unchanged (`YYYY-MM-DD HH:MM:SS`, local time). If the format changes, ordering breaks.
- **Day boundaries use local time** (`datetime.fromtimestamp`). Tests inject a clock; keep that seam.
- **Schema changes** need `CREATE TABLE IF NOT EXISTS`-style migration of existing `data/attendance.db` files, or a documented reset. Adding a NOT NULL column without a default breaks existing databases.

## 8. Config

- **A new setting must be added everywhere:**
  - the dataclass in `config.py`
  - `config.yaml`, with a comment
  - the README config table, if users would tune it
  - the CLI override in `cli._apply_overrides`, if it's commonly changed

  `load_config` rejects unknown keys, so a setting in YAML but missing from the dataclass crashes at startup.
- **Paths go through `cfg.resolve()`.** They are relative to the config file, not the working directory.
- **No hard-coded tunables in library code:** thresholds, sizes, intervals, camera index. The old code hard-coded camera index `1`.

## 9. Backends and registry

- A new backend:
  1. implements the `Detector` or `Recognizer` protocol (`name`, `dim`, `detect` / `embed`)
  2. registers a builder in `registry.py`
  3. imports its optional dependencies **inside** the builder (e.g. onnxruntime), so the default install works without them
- **Model files** come from `model_zoo.MODELS` with a pinned SHA256. They are downloaded to `models/`, and never committed.
- **GPU code** must fall back to CPU when the CUDA execution provider isn't available, and log which provider it used.
- **`backend_id`** must change for a new recognizer (see section 6).

## 10. Capture

- **Webcams on Windows** use `CAP_DSHOW`, MJPG FOURCC and `BUFFERSIZE=1`. Changing this backend brings back multi-second opens and lag.
- **Check `cap.read()`'s return value** (`ok`). The old code crashed on `None` frames when the camera was missing.
- **Video files** play at native FPS in live mode. Benchmark mode reads synchronously, loops short clips and skips warm-up frames.
- **Sources** go through `parse_source`, because `"0"` means webcam 0, not a file.

## 11. Privacy and safety

- **Face photos and attendance are personal data.** Flag any added files of these kinds:
  - `*.db`
  - `*.npz`
  - `*.onnx`
  - `*.csv`
  - new photos under `data/faces/`

  Check `.gitignore` still covers the generated ones. New face photos need the user's explicit intent.
- **No network calls** except `model_zoo` downloads, which are checksum-verified. No telemetry, and no uploading frames or embeddings.
- **Path safety:** person names become directory names (`validate_person_name`). Any new path built from user input must be validated the same way, with no `..` and no separators.
- **No secrets in config or code.** Stream URLs with credentials belong in a local config, not in `config.yaml`.

## 12. Tests

- **New logic in these modules needs pytest coverage that runs without a camera or models:**
  - `tracker.py`
  - `gallery.py`
  - `attendance.py`
  - `pipeline.py`
  - `config.py`

  Use fakes like the existing ones: `ScriptedDetector`, `CountingRecognizer`, `RecordingAttendance`, `FakeClock`, `_FakeDetector`.
- **Performance-relevant behaviour should be asserted by counting** (e.g. `CountingRecognizer.calls`), not by timing. That keeps CI deterministic.
- **Time-dependent tests use exactly representable float steps** (0.25, 0.5). Steps of 0.1 once truncated a timestamp to the wrong second.
- **Bug fixes** come with a regression test that fails before the fix.

## 13. PR hygiene (user preferences)

- **Focused commits:** one logical change each, with tests in the same commit as the module.
- **Stacking:** large work is split into stacked PRs titled `[n/N] ...`. Each PR is based on the previous branch, and its body lists the stack and the merge order.
- **Every PR in a stack must pass its own tests at its tip.** Check this when reviewing stacked PRs.
- **No `Co-Authored-By: Claude...` trailers and no "Generated with Claude Code" footers.** `review_scan.py` flags them.
- **PR descriptions** should state benchmark numbers when hot-path code changed, and should list anything not verified on a real webcam.

## 14. Regressions already fixed once

Watch for these coming back:

| Old bug | Where the fix lives |
|---|---|
| Hard-coded `* 4` box scale | `YuNetDetector.detect` scale handling |
| `face_encodings(img)[0]` on a photo with no face | `_embed_image` returns a skip reason |
| `np.argmin` on an empty gallery | `Gallery.match` early return |
| `ret` unchecked after `VideoCapture.read` | `ThreadedCamera._loop` failure handling |
| Distance shown as a "match percentage" | `TrackView.score` is cosine similarity; higher is better |
| Overlay drawn on the frame before detection | drawing happens on `canvas = frame.copy()` after inference |
| Attendance written every frame, including Unknown | `announced` and cooldown in `RecognitionEngine.process` / `AttendanceLogger` |
| Camera index hard-coded to 1 | `camera.source` config and `--camera` |
