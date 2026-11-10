---
name: facerec-review
description: Review code changes in the faceRecognition repo (src/facerec - OpenCV YuNet/SFace real-time face recognition with tracking, threaded capture and SQLite attendance). Use this whenever the user asks to review, check, audit or sanity-check a diff, branch, commit, staged changes or pull request in this repo ("review my changes", "is this PR ok to merge", "check PR #5", "did I break anything", "look over the tracker change"), and before merging any stacked PR here. It knows the repo's real-time performance budget, threading rules, coordinate conventions and attendance invariants, and can prove a change did not slow the pipeline with a before/after benchmark. Prefer it over a generic review for this repo.
---

# facerec code review

This app has to stay **real time**. Detection, tracking, embedding and display must keep up with the camera, and attendance must be correct and private. Most bugs that matter here don't show up in a quick read:
- an extra embedding per frame
- a blocking call on the inference thread
- a GUI call off the main thread
- a box scaled from the wrong frame
- a cache that isn't invalidated
- "Unknown" written to attendance

This skill reviews changes with those risks in mind and backs its claims with evidence: test runs, benchmark numbers, and exact `file:line` references.

## 1. Work out what to review

Pick the target from the request. If it's unclear, use the current branch against its base.

| Request | Diff command |
|---|---|
| A PR number | `"C:\Program Files\GitHub CLI\gh.exe" pr diff <n>` (gh is not on PATH here). Use `gh pr view <n> --json baseRefName,headRefName,title,body,commits` to get the base. These PRs are often **stacked**: review against the PR's own base branch, not `main`. |
| Branch / "my changes" | `git diff <base>...HEAD`, plus `git diff` for uncommitted work. The base is the branch this one was cut from. For the `facerec/NN-*` stack it is the previous branch in the stack. |
| A commit | `git show <sha>` |

Read every changed file **in full**, not just the hunks. Then open the callers and callees of anything whose behaviour changed. The architecture map in `references/checklist.md` shows which modules talk to each other.

## 2. Collect evidence before judging

Run these from the repo root. Use `.venv\Scripts\python.exe` when it exists. Otherwise use the active Python, which must be 3.10–3.13.

1. **Static scan.** Prints risky patterns in the *added* lines, config keys missing from `config.yaml` or the README, artefacts that shouldn't be committed, commit-message problems, and source files changed without test changes:
   ```
   python .claude/skills/facerec-review/scripts/review_scan.py --base <base> [--worktree]
   ```
   Every hit is a **lead, not a finding**. Open the line and decide whether it's really a problem in context.
2. **Tests:** `python -m pytest -q`. They need no camera and no models. Report the exact pass/fail line.
3. **Benchmark.** Run it whenever the diff touches `detectors/`, `recognizers/`, `tracker.py`, `pipeline.py`, `capture.py`, `gallery.py` (the matching path) or `registry.py`:
   ```
   python .claude/skills/facerec-review/scripts/bench_compare.py --base <base>
   ```
   It builds a synthetic 720p clip of 3 moving faces from `data/faces`, then benchmarks the base and head code on it. Each version runs in a temporary git worktree. You get mean ms per stage, embeddings per frame and FPS for both. Treat these as regressions:
   - total mean time up by more than about 15%
   - embeddings per frame rising above about 0.3 in tracked mode
   - fewer faces found per frame

   If the models are missing, run `python -m facerec download-models` first (the `facerec.exe` launcher can be blocked by Windows Smart App Control; `python -m facerec` always works). If it can't run (e.g. no enrolment photos), say so in the report instead of guessing.

Don't start the live webcam window. It needs a person in front of the camera and blocks the session. If a change can only be verified live, list it under "Not verified" with the exact command for the user to run.

## 3. Review against the checklist

Read `references/checklist.md` and go through every section that applies to the diff:
- real-time budget
- threading
- coordinates and models
- tracker and voting
- gallery and cache
- attendance
- config
- backends
- capture
- privacy and safety
- tests
- PR hygiene

The checklist explains *why* each rule exists. Use the reason to judge new cases it doesn't list.

Also look for ordinary bugs: off-by-one errors, wrong conditions, unhandled `None`, resource leaks, exceptions swallowed in worker threads.

## 4. Report

For each finding, describe a concrete failure: given this input or state, what goes wrong and what the user sees. A finding you can't connect to a failure is a suggestion, not a bug. Rank findings by severity:

- **Blocker:** wrong results, a crash, data loss or corrupt attendance, privacy leaks, or real-time broken (a benchmark regression, or a blocking call or GUI call on the wrong thread).
- **Should fix:** correctness or performance risk under realistic conditions, missing tests for new logic, config not wired through.
- **Nit:** naming, style, small clarity improvements. Keep these few.

Use this structure:

```markdown
## Review: <target> (<base>...<head>)

**Verdict:** Ready to merge | Merge after fixes | Needs rework. One sentence on why.

### Evidence
- Tests: `17 passed` (or failures with names)
- Benchmark: base 6.1 ms / 165 fps / 0.13 emb per frame -> head 6.3 ms / 160 fps / 0.13 (no regression)
- Static scan: N leads, M confirmed (listed below)

### Blockers
1. [file.py:123](src/facerec/file.py#L123) **Short title.** What is wrong. *Failure:* concrete scenario. *Fix:* specific change.

### Should fix
...

### Nits
...

### Not verified
- e.g. live webcam behaviour: run `python -m facerec run --camera 0` and check X.
```

Leave out empty sections. Make file references clickable markdown links relative to the repo root. If nothing survives verification, say so plainly and still show the evidence. A clean review backed by numbers is useful.

## Posting to GitHub (only when asked)

Post review comments to the PR only if the user asks. Use `gh pr review <n> --comment --body-file <file>`, or `--request-changes` if they ask for that.

The user doesn't want AI attribution in this repo. Don't add "Generated with Claude Code" or any Co-Authored-By line to comments or commits.

Don't fix the code during a review unless the user asks. If they do, keep each fix as its own focused commit.
