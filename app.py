"""
PCB Repair Assistant - Streamlit app

AI (deep learning)                       ML (classical, trained by us on repair records)
  1. OCR           - Tesseract              4. Fault classifier - TF-IDF + Logistic Regression
  2. Speech        - Whisper (archived)     5. Repair predictor - Random Forest
  3. Part detector - YOLOv8n (fine-tuned by us on open PCB photos)

Run locally:  streamlit run app.py

Voice input is archived for the web demo (no microphone in the classroom setup).
The code is kept: set PCB_VOICE=1 and `pip install -r requirements-voice.txt` to
turn it back on. The Android app keeps its own offline speech-to-text (Vosk).
"""
import json
import os
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from detector_utils import PartDetector
from ocr_utils import find_label, parse_label, preprocess, read_text

VOICE_ENABLED = os.environ.get("PCB_VOICE", "0") == "1"

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
SAMPLES = ROOT / "sample_images"

BLUE = "#2a78d6"
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
FAULT_NAMES = {
    "short_circuit": "Short circuit", "open_circuit": "Open circuit",
    "component_failure": "Component failure", "corrosion": "Corrosion",
    "connector_fault": "Connector fault", "power_supply": "Power supply",
}
FEATURE_NAMES = {"age_years": "Board age", "fault_category": "Fault category",
                 "previous_repairs": "Previous repairs", "board_type": "Board type"}
EXAMPLE_NOTES = [
    "No output on the 5V rail, visible corrosion near the connector, regulator U3 is hot.",
    "Dead short on the 3.3V rail, capacitor C12 shorted to ground, will replace and retest.",
    "Connector J2 pins bent and loose contact, cable intermittent at the plug.",
    "Green deposits on the pins, salt residue across the lower edge, moisture ingress.",
]

st.set_page_config(page_title="PCB Repair Assistant", page_icon="🔧", layout="wide")


# ---------------------------------------------------------------- loading
def _load_models():
    return (joblib.load(MODELS / "fault_classifier.joblib"),
            joblib.load(MODELS / "outcome_model.joblib"),
            joblib.load(MODELS / "hours_model.joblib"))


@st.cache_resource(show_spinner="Preparing ML models...")
def load_ml():
    """Load the trained models. If they are missing or were saved with a
    different scikit-learn version, retrain from the CSV (about 5 seconds)."""
    import sklearn
    try:
        saved = json.loads((MODELS / "metrics.json").read_text())["sklearn_version"]
        if saved == sklearn.__version__:
            return _load_models()
    except Exception:
        pass
    import train_models
    train_models.main()
    return _load_models()


@st.cache_resource(show_spinner="Loading Whisper speech model (first run only)...")
def load_whisper():
    from speech_utils import load_model
    return load_model()


@st.cache_resource(show_spinner="Loading part detector...")
def _load_detector(model_version):
    return PartDetector() if model_version else None


def load_detector():
    """Cached per model file AND detector code, so a newly trained model or an
    edited detector_utils.py is picked up without a restart (a cached object
    built from old code would lack new methods)."""
    import detector_utils
    if not PartDetector.available():
        return _load_detector(0)
    code = Path(detector_utils.__file__).stat().st_mtime_ns
    return _load_detector((detector_utils.MODEL.stat().st_mtime_ns, code, id(PartDetector)))


def tile_zoom(detector):
    """Zoom for tiled detection: the scale the model was trained at if it was
    trained on tiles (parts.json "tiling"), else 1.5 (best on the test boards
    for the whole-board model, see models/tiling_eval.json)."""
    t = detector.meta.get("tiling")
    return t["long_side"] / detector.size if t else 1.5


@st.cache_data(max_entries=32, show_spinner=False)
def _detect_cached(key, tiled, _detector, _img):
    if tiled:
        return _detector.detect_tiled(_img, zoom=tile_zoom(_detector))
    return _detector.detect(_img)


def image_key(img):
    import hashlib
    return hashlib.md5(img.convert("RGB").tobytes()).hexdigest()


def run_detection(detector, img, tiled):
    return _detect_cached(image_key(img) + str(id(detector.session)), tiled, detector, img)


@st.cache_resource
def load_ic_catalogue():
    from ic_ocr import load_catalogue
    return load_catalogue()


@st.cache_data(max_entries=32, show_spinner=False)
def read_chip_markings(key, _img, _parts):
    from ic_ocr import read_ics
    return read_ics(_img, _parts, max_ics=4, catalogue=load_ic_catalogue())


@st.cache_data
def load_registry():
    return pd.read_csv(ROOT / "data" / "board_registry.csv", dtype={"serial": str})


fault_clf, outcome_clf, hours_reg = load_ml()
metrics = json.loads((MODELS / "metrics.json").read_text())
registry = load_registry()
KNOWN_BOARDS = sorted(registry["board_type"].unique())

if "log" not in st.session_state:
    st.session_state.log = []


def badge(kind, text):
    colour = "#1c5cab" if kind == "AI" else "#8a3c12"
    bg = "#e6f0fc" if kind == "AI" else "#fdeee6"
    return (f"<span style='background:{bg};color:{colour};border-radius:6px;"
            f"padding:2px 8px;font-size:0.8rem;font-weight:600'>{kind}</span> "
            f"<span style='font-weight:600'>{text}</span>")


def explain_note(note, predicted):
    """Words in this note that pushed the classifier towards its answer."""
    tfidf = fault_clf.named_steps["tfidf"]
    lr = fault_clf.named_steps["clf"]
    vec = tfidf.transform([note])
    idx = list(lr.classes_).index(predicted)
    contrib = vec.multiply(lr.coef_[idx]).toarray().ravel()
    vocab = tfidf.get_feature_names_out()
    top = np.argsort(contrib)[::-1][:6]
    return [(vocab[i], float(contrib[i])) for i in top if contrib[i] > 0]


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔧 PCB Repair Assistant")
    st.caption("Hands-free job cards for the repair bench")
    st.markdown("**What the app demonstrates**")
    st.markdown(badge("AI", "OCR (Tesseract)") + "<br><small>Reads the board label and chip markings</small>",
                unsafe_allow_html=True)
    if VOICE_ENABLED:
        st.markdown(badge("AI", "Speech-to-text (Whisper)") + "<br><small>Turns the voice note into text</small>",
                    unsafe_allow_html=True)
    st.markdown(badge("AI", "Part detector (YOLOv8n)") + "<br><small>Finds and classifies components</small>",
                unsafe_allow_html=True)
    st.markdown(badge("ML", "Fault classifier") + "<br><small>TF-IDF + Logistic Regression</small>",
                unsafe_allow_html=True)
    st.markdown(badge("ML", "Repair predictor") + "<br><small>Random Forest (BER risk + hours)</small>",
                unsafe_allow_html=True)
    st.divider()
    st.caption("AI = deep-learning models (OCR and speech used as-is, detector fine-tuned by us).  \n"
               f"ML = classical models we trained on {metrics['n_records']} synthetic repair records.")

bench, evaluation, logtab = st.tabs(["🛠️ Repair Bench", "📊 Model Evaluation", "📋 Repair Log"])

# ================================================================ BENCH
with bench:
    # ---------- Step 1: OCR
    st.markdown("### 1 · Photograph the board")
    st.markdown(badge("AI", "OCR reads the board label"), unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        src = st.radio("Image source", ["Sample photo", "Upload", "Camera"],
                       horizontal=True, label_visibility="collapsed")
        img = None
        if src == "Sample photo":
            samples = sorted(SAMPLES.glob("*.jpg"))
            src_names = dict(zip(registry["image"], registry["source"]))
            pick = st.selectbox(f"Demo board ({len(samples)} real PCB photos)", samples, key="sample",
                                format_func=lambda p: f"{p.stem} · {src_names.get(p.name, '')}")
            img = Image.open(pick)
        elif src == "Upload":
            f = st.file_uploader("Board photo", type=["jpg", "jpeg", "png"])
            img = Image.open(f) if f else None
        else:
            f = st.camera_input("Take a photo of the label")
            img = Image.open(f) if f else None
        if img is not None:
            st.image(img, caption="Board photo" + (" · Roboflow 100 PCB dataset, CC BY 4.0"
                                                   if src == "Sample photo" else ""), width="stretch")

    ocr = {"board_type": None, "revision": None, "serial": None}
    with c2:
        if img is not None:
            clean = st.toggle("Auto-crop label + image cleanup", value=True,
                              help="Turn off to show the class how raw OCR performs.")
            with st.spinner("Reading label..."):
                raw_text = read_text(img, clean=clean)
            ocr = parse_label(raw_text, KNOWN_BOARDS)
            if clean:
                st.image(preprocess(find_label(img)), caption="What the OCR sees",
                         width="stretch")
            st.text_area("Raw OCR text", raw_text.strip() or "(nothing read)", height=110)
            ok = ocr["serial"] is not None
            st.markdown(("✅ " if ok else "⚠️ ") +
                        f"**Board:** {ocr['board_type'] or '?'} &nbsp; "
                        f"**Rev:** {ocr['revision'] or '?'} &nbsp; "
                        f"**S/N:** {ocr['serial'] or 'not found'}")

    # ---------- Step 1b: part detection
    detector = load_detector()
    parts = []
    if img is not None:
        st.markdown("#### Components on the board")
        st.markdown(badge("AI", "Part detector (YOLOv8n, fine-tuned on open PCB photos)"), unsafe_allow_html=True)
        if detector is None:
            st.caption("Part detector model not installed (models/parts.onnx).")
        else:
            tiled = st.toggle("Detailed detection (tiles)", value=True, key="tiled",
                              help="Cuts the photo into overlapping tiles so small resistors and capacitors "
                                   "keep enough pixels to be found. Turn off to compare with a single pass.")
            with st.spinner("Detecting components..."):
                parts = run_detection(detector, img, tiled)
            drawn, colours = detector.draw(img, parts)
            d1, d2 = st.columns([1.6, 1])
            d1.image(drawn, caption=f"{len(parts)} components detected "
                                    f"({'tiled' if tiled else 'single pass'})", width="stretch")
            with d2:
                counts = pd.Series([p["name"] for p in parts], dtype=str).value_counts()
                for name, n in counts.items():
                    st.markdown(f"<span style='color:{colours[name]}'>■</span> {name} × {n}",
                                unsafe_allow_html=True)
                if not parts:
                    st.caption("No components above the confidence threshold.")

            # ---------- Step 1c: chip markings (OCR on each detected IC)
            st.markdown("#### Chip markings")
            st.markdown(badge("AI", "OCR reads the part number on each large IC, then looks it up "
                                    "in the parts catalogue"), unsafe_allow_html=True)
            if st.toggle("Read IC markings", value=True, key="read_ics",
                         help="Crops the 4 largest detected ICs, tries all four rotations and three "
                              "image clean-ups, and matches the text against data/ic_catalogue.csv."):
                with st.spinner("Reading chip markings (a few seconds per chip)..."):
                    chips = read_chip_markings(image_key(img) + str(tiled), img, parts)
                st.session_state.chips = chips
                if not chips:
                    st.caption("No ICs large enough to read (needs about 60 px or more in the photo).")
                cols = st.columns(4)
                for col, chip in zip(cols, chips):
                    with col:
                        x1, y1, x2, y2 = (round(v) for v in chip["box"])
                        st.image(img.convert("RGB").crop((x1, y1, x2, y2)), width="stretch")
                        if chip["match"]:
                            m = chip["match"]
                            st.markdown(f"**{m['key']}**  \n{m['manufacturer']}  \n"
                                        f"<small>{m['description']}</small>", unsafe_allow_html=True)
                        elif chip["part"]:
                            st.markdown(f"**{chip['part']}**  \n<small>not in catalogue</small>",
                                        unsafe_allow_html=True)
                        else:
                            st.markdown("<small>marking not readable</small>", unsafe_allow_html=True)
                        with st.expander("OCR text"):
                            st.text(chip["text"].strip() or "(nothing read)")
                            st.caption(f"rotation {chip['rotation']}°")
            else:
                st.session_state.chips = []

    # ---------- Step 2: board history lookup
    st.markdown("### 2 · Board history")
    serials = registry["serial"].tolist()
    default = serials.index(ocr["serial"]) if ocr["serial"] in serials else None
    serial = st.selectbox("Serial number (auto-filled by OCR, or pick manually)",
                          serials, index=default, placeholder="Select serial")
    board = registry.set_index("serial").loc[serial] if serial else None
    if board is not None:
        m1, m2, m3 = st.columns([2, 1, 1])
        m1.metric(f"Board type (rev {board['revision']})", board["board_type"])
        m2.metric("Age", f"{board['age_years']} yrs")
        m3.metric("Previous repairs", int(board["previous_repairs"]))
        st.caption(f"Last recorded fault: {FAULT_NAMES[board['last_fault']]}")

    # ---------- Step 3: technician note (voice input archived for the web demo)
    if VOICE_ENABLED:
        st.markdown("### 3 · Voice note")
        st.markdown(badge("AI", "Speech-to-text transcribes the technician"), unsafe_allow_html=True)
        mode = st.radio("Note input", ["Record voice", "Upload audio", "Type / example"],
                        horizontal=True, label_visibility="collapsed")
    else:
        st.markdown("### 3 · Technician note")
        st.caption("Type the fault description, or pick an example note.")
        mode = "Type / example"
    note = ""
    if mode in ("Record voice", "Upload audio"):
        from speech_utils import transcribe
        audio = (st.audio_input("Press the mic and describe the fault")
                 if mode == "Record voice"
                 else st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "ogg"]))
        if audio is not None:
            key = f"tx_{hash(audio.getvalue())}"
            if key not in st.session_state:
                with st.spinner("Transcribing..."):
                    st.session_state[key] = transcribe(
                        load_whisper(), audio.getvalue())
            note = st.text_area("Transcript (editable)", st.session_state[key])
    else:
        ex = st.selectbox("Example note", ["(type my own)"] + EXAMPLE_NOTES)
        note = st.text_area("Technician note", "" if ex == "(type my own)" else ex)

    # ---------- Step 4 + 5: ML predictions
    if note.strip() and board is not None:
        st.markdown("### 4 · Fault diagnosis")
        st.markdown(badge("ML", "Fault classifier (TF-IDF + Logistic Regression)"),
                    unsafe_allow_html=True)
        proba = fault_clf.predict_proba([note])[0]
        classes = fault_clf.classes_
        fault = classes[int(np.argmax(proba))]
        f1, f2 = st.columns([1, 1.3])
        with f1:
            st.metric("Predicted fault", FAULT_NAMES[fault], f"{proba.max():.0%} confidence",
                      delta_color="off")
            words = explain_note(note, fault)
            if words:
                st.markdown("**Words that drove this decision:** " +
                            " ".join(f"`{w}`" for w, _ in words))
        with f2:
            pdf = pd.DataFrame({"Fault": [FAULT_NAMES[c] for c in classes],
                                "Probability": proba}).sort_values("Probability", ascending=False)
            chart = alt.Chart(pdf).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
                x=alt.X("Probability:Q", axis=alt.Axis(format="%", grid=False), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Fault:N", sort="-x", title=None, scale=alt.Scale(paddingInner=0.35)),
                tooltip=["Fault", alt.Tooltip("Probability:Q", format=".1%")],
            ).properties(height=alt.Step(30))
            st.altair_chart(chart, width="stretch")

        st.markdown("### 5 · Repair recommendation")
        st.markdown(badge("ML", "Random Forest: BER risk + estimated hours"), unsafe_allow_html=True)
        X = pd.DataFrame([{"board_type": board["board_type"], "fault_category": fault,
                           "age_years": int(board["age_years"]),
                           "previous_repairs": int(board["previous_repairs"])}])
        ber_p = float(outcome_clf.predict_proba(X)[0, 1])
        hours = float(hours_reg.predict(X)[0])
        decision = "BER (beyond economic repair)" if ber_p >= 0.5 else "Repairable"
        r1, r2, r3 = st.columns(3)
        r1.metric("Recommendation", ("⛔ " if ber_p >= 0.5 else "✅ ") + decision.split(" (")[0])
        r2.metric("BER risk", f"{ber_p:.0%}")
        r3.metric("Estimated repair time", f"{hours:.1f} h")
        st.caption(f"Inputs used: {board['board_type']} · {FAULT_NAMES[fault]} · "
                   f"{int(board['age_years'])} yrs old · {int(board['previous_repairs'])} previous repairs")

        st.markdown("### 6 · Save job card")
        if st.button("💾 Save to repair log", type="primary"):
            st.session_state.log.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "serial": serial, "board_type": board["board_type"],
                "note": note, "fault": FAULT_NAMES[fault],
                "confidence": round(float(proba.max()), 2),
                "recommendation": decision, "ber_risk": round(ber_p, 2),
                "est_hours": round(hours, 1),
                "parts": "; ".join(f"{k} x{v}" for k, v in
                                   pd.Series([p["name"] for p in parts], dtype=str).value_counts().items()),
                "ics": "; ".join(c["match"]["key"] for c in st.session_state.get("chips", []) if c["match"]),
            })
            st.success(f"Saved job card for S/N {serial}. See the Repair Log tab.")
    elif board is None:
        st.info("Select or scan a board to continue.")
    else:
        st.info("Add a technician note to get the diagnosis.")

# ================================================================ EVALUATION
with evaluation:
    fc = metrics["fault_classifier"]
    st.markdown("## ML model 1 · Fault classifier")
    st.markdown("Text classification: technician note → fault category. "
                "TF-IDF (word + bigram) features, three algorithms compared with 5-fold "
                f"cross-validation, {fc['test_size']} held-out test notes.")
    comp = pd.DataFrame(fc["comparison"])
    e1, e2 = st.columns([1.25, 1])
    with e1:
        st.dataframe(comp.rename(columns={
            "model": "Model", "cv_f1_mean": "CV F1", "cv_f1_std": "± std",
            "test_accuracy": "Test acc.", "test_f1_macro": "Test F1"}),
            hide_index=True, width="stretch")
        st.caption(f"Chosen: **{fc['chosen']}** — best scores, plus it gives probabilities "
                   "and per-word explanations.")
        kw = pd.DataFrame({"Fault": [FAULT_NAMES[k] for k in fc["top_keywords"]],
                           "Top words": [", ".join(v[:5]) for v in fc["top_keywords"].values()]})
        st.markdown("**Top words learned per fault**")
        st.dataframe(kw, hide_index=True, width="stretch")
    with e2:
        labels = [FAULT_NAMES[l] for l in fc["labels"]]
        cm = np.array(fc["confusion_matrix"])
        cmdf = pd.DataFrame([{"Actual": labels[i], "Predicted": labels[j], "Count": int(cm[i, j])}
                             for i in range(len(labels)) for j in range(len(labels))])
        base = alt.Chart(cmdf).encode(
            x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=-40, labelOverlap=False)),
            y=alt.Y("Actual:N", sort=labels))
        heat = base.mark_rect(stroke="#fcfcfb", strokeWidth=2).encode(
            color=alt.Color("Count:Q", scale=alt.Scale(range=BLUE_RAMP), legend=None),
            tooltip=["Actual", "Predicted", "Count"])
        text = base.mark_text(fontSize=12).encode(
            text="Count:Q",
            color=alt.condition(alt.datum.Count > cm.max() / 2, alt.value("white"), alt.value("#0b0b0b")))
        st.markdown("**Confusion matrix (test set)**")
        st.altair_chart((heat + text).properties(height=330), width="stretch")

    st.divider()
    oc, hr = metrics["outcome"], metrics["hours"]
    st.markdown("## ML model 2 · Repair outcome predictor")
    st.markdown("Tabular ML: board type, age, previous repairs and fault category → "
                "(a) Repairable vs BER with a Random Forest classifier, "
                "(b) repair hours with a Random Forest regressor.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("BER accuracy", f"{oc['test_accuracy']:.0%}",
              f"{oc['test_accuracy'] - oc['baseline_accuracy']:+.0%} vs always-'Repairable'")
    k2.metric("BER ROC AUC", f"{oc['test_roc_auc']:.2f}", f"CV {oc['cv_roc_auc_mean']:.2f} ± {oc['cv_roc_auc_std']:.2f}",
              delta_color="off")
    k3.metric("Hours error (MAE)", f"{hr['test_mae']:.2f} h",
              f"{hr['test_mae'] - hr['baseline_mae']:+.2f} h vs average guess", delta_color="inverse")
    k4.metric("Hours R²", f"{hr['test_r2']:.2f}")

    def importance_chart(imp, title):
        df = pd.DataFrame({"Feature": [FEATURE_NAMES[k] for k in imp],
                           "Importance": list(imp.values())})
        return alt.Chart(df, title=title).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("Importance:Q", axis=alt.Axis(format="%", grid=False)),
            y=alt.Y("Feature:N", sort="-x", title=None, scale=alt.Scale(paddingInner=0.35)),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")],
        ).properties(height=alt.Step(34))

    i1, i2 = st.columns(2)
    i1.altair_chart(importance_chart(oc["feature_importance"], "What drives BER decisions"),
                    width="stretch")
    i2.altair_chart(importance_chart(hr["feature_importance"], "What drives repair time"),
                    width="stretch")
    st.caption(f"Data: {metrics['n_records']} synthetic repair records "
               f"({metrics['ber_rate']:.0%} BER), 80/20 stratified split. "
               "Synthetic data modelled on general repair patterns; no real facility data used.")

    st.divider()
    st.markdown("## AI model · Part detector")
    pj = MODELS / "parts.json"
    if pj.exists():
        pm = json.loads(pj.read_text())
        st.markdown("Object detection: YOLOv8n (COCO-pretrained) fine-tuned on real PCB photos, "
                    f"input {pm['imgsz']} px. Scores below are on test boards never used in training.")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("mAP@50", f"{pm['metrics']['map50']:.2f}")
        q2.metric("mAP@50-95", f"{pm['metrics']['map50_95']:.2f}")
        q3.metric("Precision", f"{pm['metrics']['precision']:.2f}")
        q4.metric("Recall", f"{pm['metrics']['recall']:.2f}")
        ap = pd.DataFrame({"Part type": list(pm["per_class_ap50"]),
                           "AP@50": list(pm["per_class_ap50"].values())})
        st.altair_chart(alt.Chart(ap, title="Detection quality per part type").mark_bar(
            color=BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("AP@50:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(grid=False)),
            y=alt.Y("Part type:N", sort="-x", title=None, scale=alt.Scale(paddingInner=0.35)),
            tooltip=["Part type", alt.Tooltip("AP@50:Q", format=".2f")],
        ).properties(height=alt.Step(30)), width="stretch")
        st.caption(pm["dataset"] + f" Trained for {pm['training']['minutes']:.0f} min"
                   + (" on a 4-core CPU." if pm["training"].get("device", "cpu") == "cpu" else " on a GPU."))
        te = MODELS / "tiling_eval.json"
        if te.exists():
            tv = json.loads(te.read_text())
            st.markdown("**Single pass vs tiled detection** "
                        f"({tv['test_boards']} unseen test boards, {tv['labelled_parts']:,} labelled parts, "
                        "full-resolution photos)")
            rows = [{"Mode": "Single pass (whole photo → 1024 px)" if r["mode"] == "single"
                     else f"Tiled, zoom ×{r['zoom']:g} + whole-photo pass",
                     "mAP@50 (5 common types)": r["map50_common"],
                     "AP@50 weighted by part count": r["ap50_part_weighted"],
                     "Capacitor AP": r["per_class_ap50"]["Capacitor"],
                     "Resistor AP": r["per_class_ap50"]["Resistor"],
                     "Recall @0.30": r["recall@0.30"], "Precision @0.30": r["precision@0.30"],
                     "Parts found / board": r["parts_shown_per_board"],
                     "Seconds / board": r["seconds_per_board"]} for r in tv["results"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Same model in every row; only how the photo is fed to it changes. Tiling keeps small "
                       "parts at a usable pixel size, so recall on resistors and capacitors rises, at the cost "
                       "of more passes and somewhat lower precision. Measured by detector/eval_tiling.py. "
                       "Inductor and Switch (under 20 test parts each) are left out of the 5-type mean.")
    else:
        st.caption("Part detector not trained yet.")

    st.divider()
    st.markdown("## AI model · OCR on the 50 demo photos")
    st.markdown("Asset tag read from real board photos with Tesseract. "
                "Locating and cleaning up the tag first is what makes OCR usable:")
    st.dataframe(pd.DataFrame({
        "Pipeline": ["Raw OCR on the whole photo", "Find tag → crop → contrast + Otsu threshold → OCR"],
        "Serial numbers read correctly": ["15 / 50 (30%)", "49 / 50 (98%)"],
    }), hide_index=True, width="stretch")
    st.caption("The one miss is a yellow tag on a yellow motherboard: the tag blends into the board. "
               "Measured by tests/test_ocr.py.")

    ice = MODELS / "ic_ocr_eval.json"
    if ice.exists():
        st.divider()
        st.markdown("## AI model · OCR on chip markings")
        ev = json.loads(ice.read_text())
        p, b = ev["pipeline"], ev["baseline_plain_tesseract"]
        st.markdown(f"Part numbers on the largest detected ICs of the 50 demo boards "
                    f"({p['readable_chips'] + p['other_crops']} chip crops, read by eye as ground truth). "
                    f"{p['readable_chips']} carry a marking a person can read; the other {p['other_crops']} are "
                    "covered by stickers or heatsinks, unmarked, too small, or not a chip at all "
                    "(detector false positives).")
        n = p["readable_chips"]
        st.dataframe(pd.DataFrame({
            "Pipeline": ["Plain Tesseract on the chip crop",
                         "Upscale + contrast + 4 rotations × 3 clean-ups + catalogue match"],
            "Readable chips identified": [f"{b['part_identified']} / {n} ({b['part_identified'] / n:.0%})",
                                          f"{p['part_identified']} / {n} ({p['part_identified'] / n:.0%})"],
            "Wrong part named": [b["wrong_part_on_readable"], p["wrong_part_on_readable"]],
            "Part claimed on unreadable / non-chip crops":
                [f"{b['false_identifications']} / {b['other_crops']}",
                 f"{p['false_identifications']} / {p['other_crops']}"],
            "Seconds per chip": [b["seconds_per_chip"], p["seconds_per_chip"]],
        }), hide_index=True, width="stretch")
        st.caption("Chip markings are tiny, low-contrast laser etching, often rotated. The catalogue "
                   "(data/ic_catalogue.csv) is a local parts list like a repair shop keeps; it includes the "
                   "parts on these boards, so this measures reading and matching, not discovering unknown "
                   "parts. Fuzzy matching tolerates OCR look-alikes (S/5, O/0, B/8) but stays within one "
                   "printed line. Measured by tools/eval_ic_ocr.py.")

# ================================================================ LOG
with logtab:
    st.markdown("## Repair log")
    if st.session_state.log:
        log = pd.DataFrame(st.session_state.log)
        st.dataframe(log, hide_index=True, width="stretch")
        st.download_button("⬇️ Download CSV", log.to_csv(index=False), "repair_log.csv", "text/csv")
    else:
        st.info("No job cards saved yet. Complete a job on the Repair Bench tab.")
