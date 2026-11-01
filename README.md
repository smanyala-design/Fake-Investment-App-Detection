# Fake Investment App Risk Detection 🔎
A beginner-friendly Flask and machine-learning demo that estimates whether an investment app's supplied details resemble higher-risk or lower-risk examples.

## Important
This project is an educational risk-screening demo, **not proof that an app is fake or legitimate**. The included dataset is small and synthetic for demonstration. Do not use the result as financial or legal advice. Verify registration and warnings with the relevant official regulator.

## Features
- Web form for entering investment-app indicators.
- A scikit-learn model trained from the included sample CSV.
- Risk estimate plus a plain-language explanation of selected indicators.
- No real investment apps or personal data are collected.

## Run locally
Requires Python 3.10+.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000.

## Dataset
`data/sample_apps.csv` is a small **synthetic teaching dataset**, not a verified list of real apps. Model output depends on these limited examples and should not be interpreted as a reliable real-world fraud detector.

## Possible improvements
Use a larger, documented, legally sourced dataset; evaluate with train/test splits and precision/recall; calibrate probability estimates; add model monitoring and accessible reporting; and link users to official regulator lookup tools.
