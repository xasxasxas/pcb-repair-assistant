# PCB Repair Assistant: progress

MS course project. **Final submission: Monday 30 Nov 2026.** The class demo uses the Streamlit web app; the offline Android app (same models) follows later.

- **Status:** Track A done. Tiled detection, the YOLOv8s retrain on tiles and IC marking OCR are all done, measured and live. Remaining: buffer, optional Track B, slides and rehearsal.
- **Repo:** `xasxasxas/pcb-repair-assistant` (private). The work log is [at the bottom](#work-log).
- **Live demo:** https://pcb-repair-assistant-b7yfndd6wrb2vbxtzvpszz.streamlit.app/
- **Class briefing:** [docs/BRIEFING.md](docs/BRIEFING.md) · **Track A/B explained with pictures:** [docs/TRACKS_EXPLAINED.md](docs/TRACKS_EXPLAINED.md) · **Deploy steps:** [docs/DEPLOY.md](docs/DEPLOY.md)

---

## What the app does

A faulty board arrives at the repair bench. The technician photographs it, adds a short note, and works through six steps on one screen. Each step is labelled **AI** (pretrained deep learning) or **ML** (classical models we trained ourselves).

1. **AI: read the asset tag.** OCR finds the yellow tag in the photo, crops and cleans it, and reads board family, revision and serial number.
2. **AI: find the components.** An object detector boxes every part it recognises and counts them by type.
3. **Board history.** The serial number pulls age and previous repairs from the board registry. This is a lookup, not AI.
4. **Technician note.** A short typed description, e.g. *"Green deposits on the pins, salt residue across the lower edge."*
5. **ML: diagnose.** A text classifier turns the note into one of six fault categories and shows the words that drove the decision.
6. **ML: recommend.** A Random Forest predicts the risk that the board is beyond economic repair (BER) and the expected repair hours. The job card is saved to a log and exported as CSV.

## The techniques, one by one

All scores are on held-out data the models never saw during training.

| Type | Technique | How it works | Result |
|---|---|---|---|
| AI, pretrained | **OCR on chip markings: Tesseract + parts catalogue** | Each large detected IC is cropped, cleaned up and read at four rotations, and the text is matched against a local parts list. See Track A step 3. | Readable chips identified: plain **4/36**, ours **24/36**, 0 false claims |
| AI, pretrained | **OCR: Tesseract** | Tesseract reads text with an LSTM neural network. Raw OCR on the whole photo fails, because silkscreen text and part markings drown out the tag. We first find the yellow tag by colour, crop it, stretch contrast and apply an Otsu threshold, then run OCR on the clean crop. A parser fixes common confusions such as O read as 0. | Serials read on 50 demo photos: raw **15/50**, with tag cleanup **49/50** |
| AI, fine-tuned by us | **Part detector: YOLOv8s, run tiled** | A convolutional network that predicts boxes and part types in one pass. It's fine-tuned from COCO weights on 785 tiles cut from 137 real board photos (Roboflow 100, CC BY 4.0), merged into 7 types: IC, capacitor, resistor, connector, transistor/diode, inductor, switch. The split is by board, so no board appears in both training and testing. | Whole test boards, 5 common types: mAP@50 **0.52** (was 0.41); resistors 0.50, capacitors 0.44 (see Track A) |
| ML, trained by us | **Fault classifier: TF-IDF + Logistic Regression** | TF-IDF turns a note into word and word-pair weights, high when a term is frequent in this note but rare overall. Naive Bayes, Logistic Regression and Linear SVM were compared with 5-fold cross-validation. Logistic Regression was kept because it gives probabilities and per-word explanations. | Test accuracy **95%** (120 notes), CV F1 0.977 |
| ML, trained by us | **Repair predictor: Random Forest** | Board type, age, previous repairs and predicted fault are fed to a Random Forest classifier (BER risk) and a Random Forest regressor (repair hours). Feature importance shows what drives each: age and fault type drive scrap decisions, and fault type drives repair time. | BER ROC AUC **0.82**; hours error **0.43 h** (guessing the average: 1.24 h) |
| AI, archived | **Speech-to-text: Whisper** | Switched off for the web demo, since the hosted app can't rely on a microphone. The code is kept (`PCB_VOICE=1` plus `requirements-voice.txt`). The Android app has its own on-device speech recognition (Vosk). | |

**Data:** the repair records (600) and board histories are synthetic, generated from fault-specific phrase banks with a fixed seed, so no real facility data is used. The board photos are real and openly licensed.

---

## Track A: more detailed component detection

**Problem.** Board photos are about 2,000–2,900 px wide, but the detector looks at the whole photo shrunk to 1,024 px. A small SMD resistor becomes 2–4 px wide, too small for the network to see.

**Step 1: tiled (sliced) detection. Done.**
Enlarge the photo, cut it into overlapping 1,024 px tiles, and detect parts in each tile, so small parts keep enough pixels. Boxes cut off by a tile edge are dropped, since a neighbouring tile sees the part whole. One extra pass over the whole photo catches large parts. Duplicates are merged with non-maximum suppression.

*With the old model on the Zybo demo board (1686), tiling found 160 parts against 97 for a single pass. The figure in step 2 shows the final before/after.*

Scored on the 22 held-out test boards (4,709 hand-labelled parts) at full resolution, same model in every row (`detector/eval_tiling.py` → `models/tiling_eval.json`):

| Detection mode | mAP@50 (5 common types) | Capacitor AP | Resistor AP | Recall @0.30 | Precision @0.30 | Parts / board | Time / board |
|---|---|---|---|---|---|---|---|
| Single pass (old) | 0.405 | 0.294 | 0.305 | 0.34 | 0.46 | 96 | 0.12 s |
| **Tiled, zoom ×1.5 (app default until step 2)** | **0.455** | 0.366 | **0.379** | 0.46 | 0.40 | 181 | 0.52 s |
| Tiled, zoom ×2 | 0.437 | **0.406** | 0.371 | **0.47** | 0.36 | **217** | 0.68 s |

Tiling finds about twice as many parts and raises recall from 34% to 46%, without retraining. It costs about four times the compute and some precision, because the model has never seen parts at the enlarged size. Step 2 fixes that. Inductors and switches (under 20 test parts each) are left out of the mean, since a handful of boxes swings their score.

**Step 2: retrain YOLOv8s on tiles (free Kaggle GPU). Done.**
- Tile set: `detector/prepare_tiles.py --zip` makes 785 training tiles from 137 boards, at 2,048 px long side.
- Notebook: [detector/kaggle_train_tiles.ipynb](detector/kaggle_train_tiles.ipynb). It ran on a T4 for 89 min and stopped early at epoch 112, keeping the best weights from epoch 87.
- ONNX export: done locally from `best.pt`, in a separate throwaway venv (`torch` CPU + `ultralytics==8.3.40` + `onnxscript`), so the app's own environment stays small. ONNX and PyTorch outputs agree to 0.0007. It's opset 18, which ONNX Runtime Android 1.19.2 supports.
- Installed in `models/` and `android/app/src/main/assets/models/` (same bytes; a test checks this). `parts.json` carries `"tiling": {"long_side": 2048, ...}`, so the app's zoom becomes 2048 / 1024 = 2.

Same 22 test boards, same script, both models (`models/tiling_eval.json` holds every row):

| Model and mode | mAP@50 (5 common types) | Capacitor AP | Resistor AP | Recall @0.30 | Precision @0.30 | Parts / board | Time / board |
|---|---|---|---|---|---|---|---|
| YOLOv8n, single pass | 0.405 | 0.294 | 0.305 | 0.34 | 0.46 | 96 | 0.12 s |
| YOLOv8n, tiled ×1.5 | 0.455 | 0.366 | 0.379 | 0.46 | 0.40 | 181 | 0.52 s |
| YOLOv8s (tiles), single pass | 0.358 | 0.283 | 0.306 | 0.20 | 0.53 | 87 | 0.25 s |
| YOLOv8s (tiles), tiled ×1.5 | 0.500 | 0.399 | 0.456 | 0.44 | 0.53 | 177 | 0.95 s |
| **YOLOv8s (tiles), tiled ×2 (app)** | **0.522** | 0.442 | **0.495** | **0.49** | 0.51 | 202 | 1.45 s |
| YOLOv8s (tiles), tiled ×2.5 | 0.516 | **0.451** | 0.492 | 0.46 | 0.45 | 217 | 1.71 s |

![Before: previous model, single pass. Now: retrained model, tiled](docs/tiling_before_after.jpg)

*Zybo demo board 1686: 97 parts with the old model in one pass, 203 with the retrained model tiled. The new model also stops boxing the yellow asset tag as a connector.*

What it means:
- Retraining fixed the precision cost of tiling (40% → 51%) and lifted the small parts most: resistors 0.38 → 0.50, capacitors 0.37 → 0.44.
- Zoom 2 is best, which is exactly the scale the tiles were cut at.
- The tile-trained model is worse in a single pass (0.358). It has never seen parts that small, so tiling stays on by default. Android still runs a single pass, so the tiling loop needs porting before the phone gets this gain (see Android readiness).
- Per-tile scores from the YOLO validator (in `parts.json`): mAP@50 0.473 (was 0.349 for the nano model on whole photos).
- Cost: YOLOv8s is about 3× the size (45 MB ONNX) and about 1.5 s per board on a laptop CPU. The demo boards check out the same: board_8601 has 282 parts and both Intel chips named, board_9387 has the Altera 5CEBA4F23 named, and memory is about 0.6 GB with both models loaded.

**Step 3: read IC markings. Done.**
For the 4 largest detected ICs on a board (at least 60 px in the photo), the app crops the chip and reads its part number, then looks it up in a local parts catalogue (`data/ic_catalogue.csv`, 62 parts). It shows the part, manufacturer and function, e.g. **82437FX · Intel · 430FX PCIset system controller**.

Why it's hard, and what the pipeline does (`ic_ocr.py`):
- Chip markings are tiny, low-contrast laser etching, light text on a dark package, and the chip can sit at any of four rotations.
- The crop is upscaled, contrast-stretched, sharpened and inverted to dark-on-light text.
- Tesseract reads it at 0°, 90°, 180° and 270°, with up to 3 clean-ups: the whole crop; pins trimmed plus a median filter; sparse-text mode.
- The reading Tesseract is most confident in wins, and an exact catalogue hit stops the search early.
- The catalogue match tolerates OCR look-alikes (S/5, O/0, B/8, Z/2, I/1). To avoid inventing parts from junk, fuzzy matching stays within one printed line, the raw text must contain at least half the key's digits, and all-letter keys need an exact match.

Ground truth (`tests/ic_markings.json`): the two largest detected ICs on each of the 50 demo boards, 91 crops in all, read by eye.
- **36** carry a readable part number.
- **39** are unreadable: covered by stickers or heatsinks, unmarked, or too small.
- **16** are not chips at all (detector false positives: connectors, switches, a relay).

Result (`tools/eval_ic_ocr.py` → `models/ic_ocr_eval.json`):

| Pipeline | Readable chips identified | Wrong part named | Part claimed on the 55 other crops | Time / chip |
|---|---|---|---|---|
| Plain Tesseract on the crop | 4 / 36 (11%) | 0 | 0 | 0.13 s |
| **Our pipeline** | **24 / 36 (67%)** | **0** | **0** | 2.3 s |

- The catalogue includes the parts on these boards, so this measures reading and matching, not discovering unknown parts. That is how a repair shop's parts list would be used.
- Higher-resolution photos didn't help (14/36 on the 2,000+ px originals), because the upscaling already covers resolution. The misses come from texture, glare and logos, not pixel count.
- The OCR corrected one of my hand labels: the chip I read as VT8231 is really a VT8233A.
- Good demo boards for this step: **board_8601** (Intel 82437FX + 82371FB) and **board_9387** (Altera Cyclone V). board_2485 (Xilinx Artix-7) and board_2542 (MicroZed, Xilinx Zynq) also work.

**Track B (optional, only if time allows):** visual fault detection with a golden-board anomaly model (PatchCore on the VisA PCB images).

## Android readiness

- Every model ships as ONNX or JSON, which the phone runs directly.
- Tiling reuses the same decode and NMS rule as `YoloDecoder.java`; the tiling loop itself still needs porting. Until then the phone runs the tile-trained model in a single pass, which scores below the old model (0.358 vs 0.405), so porting tiling is the first Android task.
- The phone/Python parity tests stay in CI.
- IC marking OCR: the parts catalogue is a plain CSV the app can bundle. On the phone, the OCR engine would be ML Kit instead of Tesseract, with the same crop, rotation and matching logic; the ground-truth crops can score it the same way.

## Plan to submission

| Day | Work | Status |
|---|---|---|
| Mon 23 Nov | Migration, baseline, voice archived, tiled detection measured and in the app | done |
| Tue 24 Nov | YOLOv8s retrain on tiles (Kaggle), then re-score on whole test boards | done |
| Wed 25 Nov | IC marking OCR | done early (Mon) |
| Thu 26 Nov | Deploy on Streamlit Community Cloud, end-to-end test | done early (Mon) |
| Fri 27 Nov | Track B, only if time allows | optional |
| Sat–Sun | Slides, demo rehearsal, final report numbers | |

## Known limitations

- Repair records are synthetic.
- The detector finds about half the labelled parts (recall 49%); the misses are mostly the smallest passives on dense boards.
- Rare part types (inductors, switches) have too few examples to score reliably.
- The web demo takes typed notes only.

---

## Work log

### 2026-11-23 (Mon)
- Migrated from the previous assistant. Read the old conversation and the repo snapshot. The snapshot zip predated the trained detector; the live repo had it.
- Created the new private repo `xasxasxas/pcb-repair-assistant` and pushed the full history (Actions were off during the push to avoid a 5 h training run).
- Local environment: `.venv`, Tesseract and ffmpeg (winget). Baseline: 39 tests pass, 2 skipped (they need espeak-ng, used in CI only).
- Archived voice input in the web app. `faster-whisper` moved to `requirements-voice.txt`, which CI still installs.
- **Track A step 1:** `PartDetector.detect_tiled`, vectorised class-wise NMS, `eval_tiling.py`. Result: mAP@50 (5 common types) 0.405 → 0.455, parts per board 96 → 181. Added the "Detailed detection (tiles)" toggle and the comparison table on the Model Evaluation tab.
- Built the tiled training set and the Kaggle notebook; `train.py` gained `--data/--device/--no-install`. The training workflow is now manual-only.
- Fix: a long-running local server kept a detector built from old code (`AttributeError: detect_tiled`). The detector cache now reloads when `detector_utils.py` changes.
- Fix: Kaggle rejected `pcb_tiles.zip` because Windows' Compress-Archive wrote `\` inside the zip paths. Rebuilt it with `/`; `prepare_tiles.py --zip` now does this.
- Tests: 42 pass; CI green.
- **Track A step 3, IC marking OCR** (while Kaggle trains): `ic_ocr.py`, a parts catalogue, 91 hand-labelled chip crops, `tools/eval_ic_ocr.py`.
  - Plain Tesseract identifies 4/36 readable chips; ours identifies 24/36, with 0 wrong parts and 0 false claims on 55 non-readable crops.
  - What it took: three image clean-ups × four rotations (16 → 20), look-alike folding (→ 26, but 3 false claims), then line-restricted, digit-checked fuzzy matching (→ 24, 0 false claims).
  - App: a "Chip markings" panel under Components, cached per photo, plus the IC list in the saved job card and a results table on the Model Evaluation tab.
  - Tests: 50 pass.
- Kaggle run failed at the first cell: the notebook's Internet was off (`Failed to resolve 'pypi.org'`), so `ultralytics` never installed. The notebook now checks for Internet. Online it installs from pip; offline it installs from a `kaggle_offline` input dataset (`data/kaggle_offline.zip`: ultralytics wheels, YOLOv8s starting weights, the AMP-check model and the plot font). ONNX export is optional there, and can be done locally from `best.pt`.
- Kaggle run 2: training started (YOLOv8s, 1024 px tiles, about 50 s per epoch on a T4, so about 1.5 h for 100 epochs). It then crashed after epoch 1 in ultralytics' Ray Tune hook (`ray.train._internal.session` has no `_get_session`): Kaggle pre-installs a newer `ray`. The notebook now switches off all ultralytics third-party integrations (Ray Tune, W&B, MLflow, Comet, ClearML, DVC, Neptune, TensorBoard, HUB) before training.
- **Deployment prep:** dropped ffmpeg from `packages.txt` (only needed for the archived voice input). A headless run of the full bench flow on 3 boards took 8–11 s per new board and about 0.5 GB of memory, within Streamlit Cloud's free tier. I tried to deploy through the browser, but it isn't signed in to share.streamlit.io, and signing in means accepting Streamlit's terms and granting GitHub access, which only the account owner should do. I stopped there and wrote the steps in `docs/DEPLOY.md`.
- **Class briefing:** `docs/BRIEFING.md`, written for the MS AI class (submit and present). It covers the problem, the system flow, the AI vs ML parts with held-out results, this week's work (tiled detection, IC marking OCR, retrain in progress), limitations, data licences, the plan, and the demo script.
- Demo fact-check while writing the briefing: board_8601 (17 years old) is BER whatever the note says, so the Repairable → BER flip has to be shown on board_1686 (31% → 71% with the corrosion note). On board_8601, tiled detection finds 273 parts against 195 for a single pass.
- **Deployed** on Streamlit Community Cloud: https://pcb-repair-assistant-b7yfndd6wrb2vbxtzvpszz.streamlit.app/ (deployed by the account owner).
  - Checked on the live site: the health endpoint returns ok.
  - Default board_1075: tag read (S/N 1075), tiled detection 368 parts, chip reading identifies a Realtek RTL8110.
  - Board history fills in; the corrosion note gives Corrosion at 99% → BER 59%, 5.6 h.
  - All Model Evaluation sections render.
- Fix after the live check: chips with no catalogue match used to show raw OCR noise (e.g. "GHALOUS46A") as if it were a part number. They now show "Not identified", and the raw text and unconfirmed reading stay in the "OCR text" expander.

### 2026-11-24 (Tue)
- The first completed Kaggle run was lost: it was an interactive draft session, and the PC turned off, which ends the session and wipes `/kaggle/working`. Re-ran it and downloaded `detector_export.zip` while the session was still live. Lesson: use Save Version → "Save & Run All (Commit)" for long runs, because its output is kept.
- Kaggle result: 112 epochs in 89 min (early stop, best epoch 87). Test tiles: mAP@50 0.473, P 0.535, R 0.469. ONNX export failed on Kaggle (`onnxscript` missing), as allowed for.
- Exported ONNX locally in `data/export_venv` (outside the repo). ONNX vs PyTorch max difference 0.0007, opset 18. Installed in web and Android. The old nano model is backed up in `data/old_model_v8n/`.
- Whole-board evaluation (`eval_tiling.py --zooms 1.5,2,2.5`): tiled ×2 is best, at mAP@50 0.522 on the 5 common types (old best 0.455), recall 0.49, precision 0.51. Single pass with the new model drops to 0.358. Full table under Track A step 2.
- App: the detector name now comes from `parts.json` (`"model": "YOLOv8s"`). The Model Evaluation table shows both models with a Model column and marks the row the app uses.
- Tests: the detector quality floor is raised (mAP@50 ≥ 0.40, resistor ≥ 0.40, capacitor ≥ 0.35). The tiling test now checks tiling helps each model, that the app's row is the best, and that its zoom matches `parts.json`. 50 pass, 2 skipped.
- New `tools/make_detection_figure.py` regenerates `docs/tiling_before_after.jpg` (old single pass 97 parts vs new tiled 203 on board_1686).
- Demo check with the new model: board_8601 has 282 parts tiled / 194 single, and 82437FX and 82371FB are named; board_9387 has 5CEBA4F23 named; board_1686 has no chip named (same as before).
- Docs updated: README detector section, BRIEFING (retraining section with results, plan, demo numbers), DEPLOY (memory, timing), Android model labels.
- Track A/B explainer with figures: `docs/TRACKS_EXPLAINED.md`. The figures in `docs/figures/` (tile grid, what the network sees per mode, results chart, chip OCR stages, Track A vs B, golden-board diagram) are made from project data by `tools/make_explainer_figures.py`. The Track B diagram is marked as an illustration: the part was erased digitally and the heat map drawn by hand.
- Track B (PatchCore on VisA pcb1–pcb4) is waiting for the go-ahead on downloading the dataset (VisA_20220922.tar, 1.93 GB, from the official Amazon bucket, CC BY 4.0) and the ResNet-18 weights (about 45 MB).
