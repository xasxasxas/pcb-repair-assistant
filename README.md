# PCB Repair Assistant

MS Artificial Intelligence course project. A bench assistant for circuit-board repair: the technician photographs a faulty board and types one line about the fault, and the app fills in the job card.

**Live demo:** https://pcb-repair-assistant-b7yfndd6wrb2vbxtzvpszz.streamlit.app/
**Class briefing:** [docs/BRIEFING.md](docs/BRIEFING.md) · **How the detector work is explained, with pictures:** [docs/TRACKS_EXPLAINED.md](docs/TRACKS_EXPLAINED.md) · **Work log:** [PROGRESS.md](PROGRESS.md)

## What it does

1. **Reads the asset tag** on the photo (board type, revision, serial number).
2. **Looks up the board's history** (age, previous repairs) by serial number.
3. **Finds the components** and counts them by type.
4. **Reads the part number on the big chips** and names them from a parts catalogue.
5. **Lists the parts** as a bill of materials.
6. **Classifies the fault** from the technician's note, and shows which words drove the answer.
7. **Recommends repair or scrap** (BER: beyond economic repair) with an estimated repair time, and shows how much each input pushed the answer.
8. **Saves the job card** to a repair log.

## The AI and ML parts

"AI" here means deep-learning models. "ML" means classical models designed, trained and evaluated in this project. Every score is measured on data the model never saw during training.

| | Technique | What it does | Result |
|---|---|---|---|
| AI, pretrained | OCR (Tesseract) on the asset tag | Reads the serial number | 49 of 50 demo photos, against 15 of 50 for raw OCR |
| AI, fine-tuned here | YOLOv8s object detector, run on tiles | Finds 7 types of component | mAP@50 0.52 on the 5 common types; recall 49%, precision 51% |
| AI, pretrained | OCR on chip markings + parts catalogue | Names the larger chips | 24 of 36 readable chips, against 4 of 36 for plain OCR; no false claims |
| ML, trained here | TF-IDF + Logistic Regression | Fault category from the note | 95% accuracy on 120 test notes |
| ML, trained here | Random Forest classifier + regressor | BER risk and repair hours, with reasons | ROC AUC 0.82; hours error 0.43 h against 1.24 h for guessing the average |

## The code: three files

| File | What is in it |
|---|---|
| [app.py](app.py) | The web app: the three screens (Repair Bench, Model Evaluation, Repair Log). |
| [pcb_ai.py](pcb_ai.py) | Everything the app does with a photo and a job: tag OCR, part detector with tiling, chip OCR, BOM, and the explanation of each Random Forest prediction. |
| [train.py](train.py) | How the models were made and scored: synthetic repair records, ML training and evaluation, training tiles for the detector, and the detector and chip-OCR evaluations. |

Also in the repo:

| | |
|---|---|
| [test_project.py](test_project.py) | 28 automated checks covering every part above, including a full run of the app. |
| [train_detector_kaggle.ipynb](train_detector_kaggle.ipynb) | The notebook that trained the detector on a free Kaggle GPU (89 minutes). |
| `data/` | Synthetic repair records, the board registry, the chip catalogue, and the hand-labelled chip crops. |
| `models/` | The trained models and their recorded scores. |
| `sample_images/` | 50 real board photos for the demo (Roboflow 100, CC BY 4.0), each with a synthetic asset tag. |

## Run it

Needs Python 3.12 and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed.

```bash
pip install -r requirements.txt
streamlit run app.py          # the web app
pytest test_project.py        # check everything works (about 2 minutes)
```

Recreate the models and scores:

```bash
python train.py data                    # synthetic repair records (fixed seed, always the same file)
python train.py ml                      # train and score the two ML models (a few seconds)
python train.py eval-chips              # score chip OCR on the 91 hand-labelled chips
python train.py tiles RF100 OUT         # cut training tiles for the detector (needs the Roboflow dataset)
python train.py eval-detector RF100     # score the detector on the 22 unseen test boards
```

`RF100` is the unzipped [Roboflow 100 printed circuit board dataset](https://universe.roboflow.com/roboflow-100/printed-circuit-board/dataset/4). The last two commands also need `pip install pyyaml`.

The web demo is hosted on Streamlit Community Cloud, which redeploys on every push to `main` (main file `app.py`, Python 3.12; Tesseract comes from `packages.txt`).

## Limitations

- The repair records and board histories are synthetic, so the ML models learn patterns that were designed in. The same code retrains unchanged on real job records.
- The detector finds about half of the labelled parts; the misses are mostly the smallest resistors and capacitors on dense boards. BOM quantities are a starting point to check.
- Chip reading works on larger chips that are in the catalogue.
- Rare part types (inductors, switches) have too few examples to score reliably.

## Earlier work kept on the `full-version` branch

An offline Android version of the app, voice input (speech-to-text), the first detector's training scripts and the scripts that made the demo photos and report figures are not part of this simplified version. They are preserved on the `full-version` branch.

Data and licence credits are in [ATTRIBUTION.md](ATTRIBUTION.md).
