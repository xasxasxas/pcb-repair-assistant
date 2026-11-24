"""
PCB Repair Assistant - the web app (Streamlit).      Run:  streamlit run app.py

A technician photographs a faulty board and types one line about the fault.
The app reads the asset tag, finds the components, names the big chips, lists
the parts, classifies the fault, and recommends repair or scrap, with reasons.

  AI (deep learning)                        ML (classical, trained by us)
    OCR on the asset tag    - Tesseract       Fault classifier - TF-IDF + Logistic Regression
    Part detector           - YOLOv8s         Repair predictor - Random Forest
    OCR on chip markings    - Tesseract

The AI and ML logic is in pcb_ai.py. The models are made and scored in train.py.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import pcb_ai

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
BLUE, RED, GREEN = "#2a78d6", "#d64541", "#2e9e5b"
FAULT_NAMES = {"short_circuit": "Short circuit", "open_circuit": "Open circuit",
               "component_failure": "Component failure", "corrosion": "Corrosion",
               "connector_fault": "Connector fault", "power_supply": "Power supply"}
FEATURE_NAMES = {"age_years": "Board age", "fault_category": "Fault category",
                 "previous_repairs": "Previous repairs", "board_type": "Board type"}
EXAMPLE_NOTES = [
    "No output on the 5V rail, visible corrosion near the connector, regulator U3 is hot.",
    "Dead short on the 3.3V rail, capacitor C12 shorted to ground, will replace and retest.",
    "Connector J2 pins bent and loose contact, cable intermittent at the plug.",
    "Green deposits on the pins, salt residue across the lower edge, moisture ingress.",
]

st.set_page_config(page_title="PCB Repair Assistant", page_icon="🔧", layout="wide")


# ---------------------------------------------------------------- loading (cached)
@st.cache_resource(show_spinner="Preparing ML models...")
def load_ml():
    """The three trained ML models. If they were saved with a different
    scikit-learn version, retrain them from the CSV (about 5 seconds)."""
    import sklearn
    try:
        if json.loads((MODELS / "metrics.json").read_text())["sklearn_version"] != sklearn.__version__:
            raise ValueError
    except Exception:
        import train
        train.train_ml()
    return [joblib.load(MODELS / f"{name}.joblib")
            for name in ("fault_classifier", "outcome_model", "hours_model")]


@st.cache_resource(show_spinner="Loading part detector...")
def load_detector():
    return pcb_ai.PartDetector()


@st.cache_resource
def load_catalogue():
    return pcb_ai.load_catalogue()


@st.cache_data(max_entries=32, show_spinner=False)
def detect_parts(key, tiled, _img):
    det = load_detector()
    return det.detect_tiled(_img) if tiled else det.detect(_img)


@st.cache_data(max_entries=32, show_spinner=False)
def read_chips(key, _img, _parts):
    return pcb_ai.read_ics(_img, _parts, load_catalogue())


fault_clf, outcome_clf, hours_reg = load_ml()
metrics = json.loads((MODELS / "metrics.json").read_text())
registry = pd.read_csv(ROOT / "data" / "board_registry.csv", dtype={"serial": str})
KNOWN_BOARDS = sorted(registry["board_type"].unique())
if "log" not in st.session_state:
    st.session_state.log = []


# ---------------------------------------------------------------- small helpers
def badge(kind, text):
    """Coloured AI / ML label in front of each step."""
    colour, bg = ("#1c5cab", "#e6f0fc") if kind == "AI" else ("#8a3c12", "#fdeee6")
    return (f"<span style='background:{bg};color:{colour};border-radius:6px;padding:2px 8px;"
            f"font-size:0.8rem;font-weight:600'>{kind}</span> <span style='font-weight:600'>{text}</span>")


def explain_note(note, predicted):
    """Words in this note that pushed the fault classifier towards its answer
    (each word's TF-IDF weight times the model's coefficient for that fault)."""
    tfidf, lr = fault_clf.named_steps["tfidf"], fault_clf.named_steps["clf"]
    contrib = tfidf.transform([note]).multiply(lr.coef_[list(lr.classes_).index(predicted)]).toarray().ravel()
    vocab = tfidf.get_feature_names_out()
    return [vocab[i] for i in np.argsort(contrib)[::-1][:6] if contrib[i] > 0]


def why_chart(parts, labels, scale, unit, title):
    """One bar per input: how far it pushed this prediction up (red) or down (green)."""
    df = pd.DataFrame({"Input": [labels[k] for k in parts], "Effect": [v * scale for v in parts.values()]})
    df["Direction"] = np.where(df["Effect"] >= 0, "raises", "lowers")
    digits = 0 if unit == "pts" else 1
    df["Text"] = [f"{round(v, digits):+.{digits}f} {unit}" if round(v, digits) else f"0 {unit}" for v in df["Effect"]]
    order = list(df.assign(size=df["Effect"].abs()).sort_values("size", ascending=False)["Input"])
    lim = max(df["Effect"].abs().max() * 1.45, 1e-6)
    base = alt.Chart(df, title=title).encode(
        y=alt.Y("Input:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Effect:Q", title=None, scale=alt.Scale(domain=[-lim, lim]), axis=alt.Axis(grid=False)),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=["raises", "lowers"], range=[RED, GREEN]), legend=None),
        tooltip=["Input", "Text"])
    up = base.transform_filter("datum.Effect >= 0").mark_text(align="left", dx=4).encode(x="Effect:Q", text="Text:N")
    down = base.transform_filter("datum.Effect < 0").mark_text(align="right", dx=-4).encode(x="Effect:Q", text="Text:N")
    return (bars + up + down).properties(height=alt.Step(30))


def bar_chart(names, values, title, x_title, percent=False):
    df = pd.DataFrame({"Name": names, x_title: values})
    return alt.Chart(df, title=title).mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
        x=alt.X(f"{x_title}:Q", axis=alt.Axis(format="%" if percent else "", grid=False)),
        y=alt.Y("Name:N", sort="-x", title=None, scale=alt.Scale(paddingInner=0.35)),
    ).properties(height=alt.Step(32))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔧 PCB Repair Assistant")
    st.caption("Job cards for the repair bench, with less typing")
    st.markdown("**What the app demonstrates**")
    for kind, name, what in [
            ("AI", "OCR (Tesseract)", "Reads the board label and chip markings"),
            ("AI", "Part detector (YOLOv8s)", "Finds and classifies components"),
            ("ML", "Fault classifier", "TF-IDF + Logistic Regression"),
            ("ML", "Repair predictor", "Random Forest (BER risk + hours)")]:
        st.markdown(badge(kind, name) + f"<br><small>{what}</small>", unsafe_allow_html=True)
    st.divider()
    st.caption("AI = deep-learning models (OCR used as-is, detector fine-tuned by us).  \n"
               f"ML = classical models we trained on {metrics['n_records']} synthetic repair records.")

bench, evaluation, logtab = st.tabs(["🛠️ Repair Bench", "📊 Model Evaluation", "📋 Repair Log"])

# ================================================================ REPAIR BENCH
with bench:
    # ---------- Step 1: photo + tag OCR
    st.markdown("### 1 · Photograph the board")
    st.markdown(badge("AI", "OCR reads the board label"), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        img = None
        if st.radio("Image source", ["Sample photo", "Upload"], horizontal=True,
                    label_visibility="collapsed") == "Sample photo":
            samples = sorted((ROOT / "sample_images").glob("*.jpg"))
            source = dict(zip(registry["image"], registry["source"]))
            pick = st.selectbox(f"Demo board ({len(samples)} real PCB photos)", samples,
                                format_func=lambda p: f"{p.stem} · {source.get(p.name, '')}")
            img = Image.open(pick)
        else:
            f = st.file_uploader("Board photo", type=["jpg", "jpeg", "png"])
            img = Image.open(f) if f else None
        if img is not None:
            st.image(img, caption="Board photo", width="stretch")

    ocr = {"board_type": None, "revision": None, "serial": None}
    with c2:
        if img is not None:
            clean = st.toggle("Find the label and clean it up first", value=True,
                              help="Turn off to see how raw OCR on the whole photo performs.")
            with st.spinner("Reading label..."):
                raw_text = pcb_ai.read_text(img, clean=clean)
            ocr = pcb_ai.parse_label(raw_text, KNOWN_BOARDS)
            if clean:
                st.image(pcb_ai.preprocess(pcb_ai.find_label(img)), caption="What the OCR sees", width="stretch")
            st.text_area("Raw OCR text", raw_text.strip() or "(nothing read)", height=110)
            st.markdown(("✅ " if ocr["serial"] else "⚠️ ") +
                        f"**Board:** {ocr['board_type'] or '?'} &nbsp; **Rev:** {ocr['revision'] or '?'} "
                        f"&nbsp; **S/N:** {ocr['serial'] or 'not found'}")

    parts, chips = [], []
    if img is not None:
        key = hashlib.md5(img.convert("RGB").tobytes()).hexdigest()      # cache key for this photo

        # ---------- Step 1b: part detection
        st.markdown("#### Components on the board")
        st.markdown(badge("AI", "Part detector (YOLOv8s, fine-tuned on open PCB photos)"), unsafe_allow_html=True)
        tiled = st.toggle("Detailed detection (tiles)", value=True,
                          help="Cuts the photo into overlapping tiles so small resistors and capacitors keep "
                               "enough pixels to be found. Turn off to compare with a single pass.")
        with st.spinner("Detecting components..."):
            parts = detect_parts(key, tiled, img)
        drawn, colours = load_detector().draw(img, parts)
        d1, d2 = st.columns([1.6, 1])
        d1.image(drawn, caption=f"{len(parts)} components detected ({'tiled' if tiled else 'single pass'})",
                 width="stretch")
        for name, n in pd.Series([p["name"] for p in parts], dtype=str).value_counts().items():
            d2.markdown(f"<span style='color:{colours[name]}'>■</span> {name} × {n}", unsafe_allow_html=True)

        # ---------- Step 1c: chip markings
        st.markdown("#### Chip markings")
        st.markdown(badge("AI", "OCR reads the part number on the 6 largest chips, then looks it up "
                                "in the parts catalogue"), unsafe_allow_html=True)
        with st.spinner("Reading chip markings (a few seconds per chip)..."):
            chips = read_chips(key + str(tiled), img, parts)
        if not chips:
            st.caption("No chips large enough to read (needs about 60 px or more in the photo).")
        cols = [c for _ in range(0, len(chips), 3) for c in st.columns(3)]
        for col, chip in zip(cols, chips):
            with col:
                st.image(img.convert("RGB").crop(tuple(round(v) for v in chip["box"])), width="stretch")
                if chip["match"]:
                    m = chip["match"]
                    st.markdown(f"**{m['key']}**  \n{m['manufacturer']}  \n<small>{m['description']}</small>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("**Not identified**  \n<small>no catalogue part matched</small>",
                                unsafe_allow_html=True)
                with st.expander("OCR text"):
                    st.text(chip["text"].strip() or "(nothing read)")

        # ---------- Step 1d: bill of materials
        st.markdown("#### Bill of materials (from the photo)")
        st.dataframe(pcb_ai.build_bom(parts, chips), hide_index=True, width="stretch")
        st.caption("Part counts come from the detector, part numbers from chip OCR. The detector finds about "
                   "half of the smallest parts on dense boards, so treat the quantities as a starting point "
                   "to check, not a complete BOM.")

    # ---------- Step 2: board history (a plain lookup, no model)
    st.markdown("### 2 · Board history")
    serials = registry["serial"].tolist()
    serial = st.selectbox("Serial number (auto-filled by OCR, or pick manually)", serials,
                          index=serials.index(ocr["serial"]) if ocr["serial"] in serials else None,
                          placeholder="Select serial")
    board = registry.set_index("serial").loc[serial] if serial else None
    if board is not None:
        m1, m2, m3 = st.columns([2, 1, 1])
        m1.metric(f"Board type (rev {board['revision']})", board["board_type"])
        m2.metric("Age", f"{board['age_years']} yrs")
        m3.metric("Previous repairs", int(board["previous_repairs"]))

    # ---------- Step 3: technician note
    st.markdown("### 3 · Technician note")
    example = st.selectbox("Example note", ["(type my own)"] + EXAMPLE_NOTES)
    note = st.text_area("Technician note", "" if example == "(type my own)" else example)

    if not note.strip() or board is None:
        st.info("Select a board and add a technician note to get the diagnosis.")
    else:
        # ---------- Step 4: fault diagnosis
        st.markdown("### 4 · Fault diagnosis")
        st.markdown(badge("ML", "Fault classifier (TF-IDF + Logistic Regression)"), unsafe_allow_html=True)
        proba = fault_clf.predict_proba([note])[0]
        fault = fault_clf.classes_[int(np.argmax(proba))]
        f1, f2 = st.columns([1, 1.3])
        f1.metric("Predicted fault", FAULT_NAMES[fault], f"{proba.max():.0%} confidence", delta_color="off")
        f1.markdown("**Words that drove this decision:** " + " ".join(f"`{w}`" for w in explain_note(note, fault)))
        f2.altair_chart(bar_chart([FAULT_NAMES[c] for c in fault_clf.classes_], proba, "", "Probability",
                                  percent=True), width="stretch")

        # ---------- Step 5: repair recommendation, with reasons
        st.markdown("### 5 · Repair recommendation")
        st.markdown(badge("ML", "Random Forest: BER risk + estimated hours"), unsafe_allow_html=True)
        X = pd.DataFrame([{"board_type": board["board_type"], "fault_category": fault,
                           "age_years": int(board["age_years"]),
                           "previous_repairs": int(board["previous_repairs"])}])
        ber_start, ber_parts, ber_p = pcb_ai.contributions(outcome_clf, X)
        hrs_start, hrs_parts, hours = pcb_ai.contributions(hours_reg, X)
        decision = "BER (beyond economic repair)" if ber_p >= 0.5 else "Repairable"
        r1, r2, r3 = st.columns(3)
        r1.metric("Recommendation", ("⛔ " if ber_p >= 0.5 else "✅ ") + decision.split(" (")[0])
        r2.metric("BER risk", f"{ber_p:.0%}")
        r3.metric("Estimated repair time", f"{hours:.1f} h")

        st.markdown("**Why this recommendation**")
        labels = {"board_type": f"Board type: {board['board_type']}",
                  "fault_category": f"Fault: {FAULT_NAMES[fault]}",
                  "age_years": f"Board age: {int(board['age_years'])} yrs",
                  "previous_repairs": f"Previous repairs: {int(board['previous_repairs'])}"}
        w1, w2 = st.columns(2)
        w1.altair_chart(why_chart(ber_parts, labels, 100, "pts", "BER risk"), width="stretch")
        w1.caption(f"The model starts at {ber_start:.0%} and each input moves it, ending at {ber_p:.0%}.")
        w2.altair_chart(why_chart(hrs_parts, labels, 1, "h", "Repair time"), width="stretch")
        w2.caption(f"The average job takes {hrs_start:.1f} h; the inputs move it to {hours:.1f} h.")

        # ---------- Step 6: save the job card
        st.markdown("### 6 · Save job card")
        if st.button("💾 Save to repair log", type="primary"):
            st.session_state.log.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"), "serial": serial,
                "board_type": board["board_type"], "note": note, "fault": FAULT_NAMES[fault],
                "recommendation": decision, "ber_risk": round(ber_p, 2), "est_hours": round(hours, 1),
                "why": "; ".join(f"{labels[k]} {v * 100:+.0f} pts" for k, v in
                                 sorted(ber_parts.items(), key=lambda kv: -abs(kv[1]))),
                "parts_found": len(parts),
                "chips": "; ".join(c["match"]["key"] for c in chips if c["match"]),
            })
            st.success(f"Saved job card for S/N {serial}. See the Repair Log tab.")

# ================================================================ MODEL EVALUATION
with evaluation:
    st.caption("Every score on this page comes from data the model never saw during training.")

    # ---------- ML model 1
    fc = metrics["fault_classifier"]
    st.markdown("## ML model 1 · Fault classifier")
    st.markdown("Technician note → fault category. TF-IDF features (words and word pairs); three algorithms "
                f"compared with 5-fold cross-validation, then scored on {fc['test_size']} held-out notes.")
    e1, e2 = st.columns([1.25, 1])
    with e1:
        st.dataframe(pd.DataFrame(fc["comparison"]).rename(columns={
            "model": "Model", "cv_f1_mean": "CV F1", "cv_f1_std": "± std",
            "test_accuracy": "Test accuracy", "test_f1_macro": "Test F1"}), hide_index=True, width="stretch")
        st.caption(f"Chosen: **{fc['chosen']}**. The three score about the same; this one also gives "
                   "probabilities and per-word explanations.")
        st.markdown("**Top words learned per fault**")
        st.dataframe(pd.DataFrame({"Fault": [FAULT_NAMES[k] for k in fc["top_keywords"]],
                                   "Top words": [", ".join(v[:5]) for v in fc["top_keywords"].values()]}),
                     hide_index=True, width="stretch")
    with e2:
        names = [FAULT_NAMES[l] for l in fc["labels"]]
        cm = np.array(fc["confusion_matrix"])
        cmdf = pd.DataFrame([{"Actual": names[i], "Predicted": names[j], "Count": int(cm[i, j])}
                             for i in range(len(names)) for j in range(len(names))])
        base = alt.Chart(cmdf).encode(x=alt.X("Predicted:N", sort=names, axis=alt.Axis(labelAngle=-40)),
                                      y=alt.Y("Actual:N", sort=names))
        heat = base.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("Count:Q", scale=alt.Scale(range=["#cde2fb", "#0d366b"]), legend=None))
        text = base.mark_text(fontSize=12).encode(
            text="Count:Q", color=alt.condition(alt.datum.Count > cm.max() / 2, alt.value("white"), alt.value("black")))
        st.markdown("**Confusion matrix (test notes)**")
        st.altair_chart((heat + text).properties(height=330), width="stretch")

    # ---------- ML model 2
    st.divider()
    oc, hr = metrics["outcome"], metrics["hours"]
    st.markdown("## ML model 2 · Repair predictor")
    st.markdown("Board type, age, previous repairs and fault → (a) Repairable or BER with a Random Forest "
                "classifier, (b) repair hours with a Random Forest regressor.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("BER accuracy", f"{oc['test_accuracy']:.0%}",
              f"{oc['test_accuracy'] - oc['baseline_accuracy']:+.0%} vs always 'Repairable'")
    k2.metric("BER ROC AUC", f"{oc['test_roc_auc']:.2f}")
    k3.metric("Hours error (MAE)", f"{hr['test_mae']:.2f} h",
              f"{hr['test_mae'] - hr['baseline_mae']:+.2f} h vs guessing the average", delta_color="inverse")
    k4.metric("Hours R²", f"{hr['test_r2']:.2f}")
    i1, i2 = st.columns(2)
    for col, imp, title in [(i1, oc["feature_importance"], "What drives BER decisions"),
                            (i2, hr["feature_importance"], "What drives repair time")]:
        col.altair_chart(bar_chart([FEATURE_NAMES[k] for k in imp], list(imp.values()), title, "Importance",
                                   percent=True), width="stretch")
    st.caption(f"Data: {metrics['n_records']} synthetic repair records ({metrics['ber_rate']:.0%} BER), "
               "80/20 split. No real workshop data is used.")

    # ---------- AI: part detector
    st.divider()
    tv = json.loads((MODELS / "tiling_eval.json").read_text())
    st.markdown("## AI model · Part detector")
    st.markdown("YOLOv8 object detector, pretrained on everyday photos (COCO) and fine-tuned by us on real PCB "
                f"photos. Scored on {tv['test_boards']} unseen test boards with {tv['labelled_parts']:,} "
                "hand-labelled parts.")
    st.dataframe(pd.DataFrame([{
        "Model": r["model"],
        "Mode": ("Single pass" if r["mode"] == "single" else f"Tiled, zoom ×{r['zoom']:g}")
                + ("  ← used in the app" if r.get("in_app") else ""),
        "mAP@50 (5 common types)": r["map50_common"],
        "Resistor AP": r["per_class_ap50"]["Resistor"], "Capacitor AP": r["per_class_ap50"]["Capacitor"],
        "Recall": r["recall@0.30"], "Precision": r["precision@0.30"],
        "Parts found / board": r["parts_shown_per_board"], "Seconds / board": r["seconds_per_board"],
    } for r in tv["results"]]), hide_index=True, width="stretch")
    st.caption("Tiling keeps small parts at a usable pixel size. The first model was trained on whole photos, "
               "so tiling raised its recall but cost precision. Retraining on tiles fixed that; the retrained "
               "model is meant to be run tiled. Inductors and switches (under 20 test parts each) are left out "
               "of the 5-type mean. Measured by `python train.py eval-detector`.")

    # ---------- AI: OCR
    st.divider()
    st.markdown("## AI model · OCR on the asset tag")
    st.dataframe(pd.DataFrame({
        "Pipeline": ["Raw OCR on the whole photo", "Find tag → crop → contrast + threshold → OCR"],
        "Serial numbers read correctly (50 demo photos)": ["15 / 50 (30%)", "49 / 50 (98%)"],
    }), hide_index=True, width="stretch")
    st.caption("The one miss is a yellow tag on a yellow motherboard. Measured by test_project.py.")

    st.divider()
    ev = json.loads((MODELS / "ic_ocr_eval.json").read_text())
    p, b = ev["pipeline"], ev["baseline_plain_tesseract"]
    st.markdown("## AI model · OCR on chip markings")
    st.markdown(f"{p['readable_chips'] + p['other_crops']} chip crops from the demo boards, read by eye as ground "
                f"truth. {p['readable_chips']} carry a part number a person can read. The other {p['other_crops']} "
                "(stickers, heatsinks, unmarked, or not chips) test whether the reader invents a part.")
    st.dataframe(pd.DataFrame({
        "Pipeline": ["Plain Tesseract on the chip crop", "Clean-up + 4 rotations + catalogue match"],
        "Readable chips identified": [f"{x['part_identified']} / {x['readable_chips']}" for x in (b, p)],
        "Wrong part named": [x["wrong_part_on_readable"] for x in (b, p)],
        "Part claimed where nothing is readable": [f"{x['false_identifications']} / {x['other_crops']}" for x in (b, p)],
    }), hide_index=True, width="stretch")
    st.caption("The catalogue (data/ic_catalogue.csv) is a local parts list like a repair shop keeps, so this "
               "measures reading and matching known parts. Measured by `python train.py eval-chips`.")

# ================================================================ REPAIR LOG
with logtab:
    st.markdown("## Repair log")
    if st.session_state.log:
        st.dataframe(pd.DataFrame(st.session_state.log), hide_index=True, width="stretch")
    else:
        st.info("No job cards saved yet. Complete a job on the Repair Bench tab.")
