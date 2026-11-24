"""Per-prediction explanations add up exactly; the BOM merges detector counts with chip OCR."""
import joblib
import pandas as pd
import pytest

from bom import build_bom
from conftest import ROOT
from explain_utils import FEATURES, contributions

OUT = joblib.load(ROOT / "models" / "outcome_model.joblib")
HRS = joblib.load(ROOT / "models" / "hours_model.joblib")
JOBS = pd.DataFrame([
    {"board_type": "MCU-MAIN", "fault_category": "corrosion", "age_years": 17, "previous_repairs": 2},
    {"board_type": "RF-COMM", "fault_category": "short_circuit", "age_years": 2, "previous_repairs": 0},
    {"board_type": "PSU-CTRL", "fault_category": "power_supply", "age_years": 9, "previous_repairs": 4},
])


@pytest.mark.parametrize("i", range(len(JOBS)))
def test_contributions_add_up_to_the_prediction(i):
    X = JOBS.iloc[[i]]
    s, c, p = contributions(OUT, X)
    assert set(c) == set(FEATURES)
    assert p == pytest.approx(OUT.predict_proba(X)[0, 1], abs=1e-9)
    s, c, p = contributions(HRS, X)
    assert p == pytest.approx(HRS.predict(X)[0], abs=1e-9)


def test_explanation_matches_what_the_model_learned():
    old = JOBS.iloc[[0]].assign(age_years=18)
    new = JOBS.iloc[[0]].assign(age_years=1)
    assert contributions(OUT, old)[1]["age_years"] > contributions(OUT, new)[1]["age_years"]
    assert contributions(OUT, old)[1]["age_years"] > 0      # old boards push towards scrapping


def test_bom_lines_identified_chips_and_counts():
    parts = [{"name": n} for n in ["IC"] * 3 + ["Resistor"] * 5 + ["Capacitor"] * 2]
    chips = [{"match": {"key": "82437FX", "manufacturer": "Intel", "description": "system controller"}},
             {"match": None}]
    bom = build_bom(parts, chips)
    assert list(bom["Part type"]) == ["IC", "IC", "Capacitor", "Resistor"]
    assert bom.iloc[0]["Part number"] == "82437FX" and bom.iloc[0]["Qty"] == 1
    assert bom.iloc[1]["Part number"] == "(not identified)" and bom.iloc[1]["Qty"] == 2
    assert bom["Qty"].sum() == len(parts)
    assert list(bom["Line"]) == [1, 2, 3, 4]


def test_empty_bom():
    assert len(build_bom([], [])) == 0
