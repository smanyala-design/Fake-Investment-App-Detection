import os
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "sample_apps.csv"

FEATURES = [
    "app_age_days",
    "guaranteed_returns",
    "regulator_registered",
    "withdrawal_complaints",
    "pressure_tactics",
    "transparent_fees",
    "verified_publisher",
]
LABEL = "higher_risk"


def train_demo_model():
    """Train a simple model from the included synthetic teaching dataset."""
    data = pd.read_csv(DATA_FILE)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000))
    model.fit(data[FEATURES], data[LABEL])
    return model


model = train_demo_model()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_values = {
        "app_age_days": "180",
        "guaranteed_returns": "0",
        "regulator_registered": "0",
        "withdrawal_complaints": "0",
        "pressure_tactics": "0",
        "transparent_fees": "0",
        "verified_publisher": "0",
    }
    if request.method == "POST":
        try:
            form_values = {key: request.form.get(key, "0") for key in FEATURES}
            values = {
                "app_age_days": max(0, min(int(form_values["app_age_days"]), 50000)),
                "guaranteed_returns": int(form_values["guaranteed_returns"]),
                "regulator_registered": int(form_values["regulator_registered"]),
                "withdrawal_complaints": int(form_values["withdrawal_complaints"]),
                "pressure_tactics": int(form_values["pressure_tactics"]),
                "transparent_fees": int(form_values["transparent_fees"]),
                "verified_publisher": int(form_values["verified_publisher"]),
            }
            for key in FEATURES[1:]:
                if values[key] not in (0, 1):
                    raise ValueError("Choose yes or no for each indicator.")
            sample = pd.DataFrame([values], columns=FEATURES)
            prediction = int(model.predict(sample)[0])
            probabilities = model.predict_proba(sample)[0]
            risk_probability = float(probabilities[list(model.classes_).index(1)])
            flags = []
            if values["guaranteed_returns"]:
                flags.append("Promises of guaranteed or unusually high returns")
            if not values["regulator_registered"]:
                flags.append("Regulatory registration has not been confirmed")
            if values["withdrawal_complaints"]:
                flags.append("Reports of withdrawal problems")
            if values["pressure_tactics"]:
                flags.append("Pressure to invest quickly")
            if not values["transparent_fees"]:
                flags.append("Fees are not clearly explained")
            if not values["verified_publisher"]:
                flags.append("Publisher verification has not been confirmed")
            result = {
                "label": "Higher-risk pattern" if prediction else "Lower-risk pattern",
                "percent": round(risk_probability * 100),
                "flags": flags,
                "note": "This is a model estimate based on a tiny synthetic dataset, not a verdict about the app.",
            }
        except (ValueError, TypeError):
            result = {"error": "Please enter a valid app age and select yes/no for every indicator."}
    return render_template("index.html", result=result, form_values=form_values)


if __name__ == "__main__":
    app.run(debug=True)
