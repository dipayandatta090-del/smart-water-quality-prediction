# Water Quality Prediction — Machine Learning + Streamlit

A beginner-friendly end-to-end project template. It trains a Random Forest classifier from a CSV dataset, saves evaluation graphs, and provides a Streamlit web app for predictions.

## 1. Expected dataset

Place your CSV file at:

`data/water_potability.csv`

The commonly used water potability dataset has these input columns:

- `ph`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic_carbon`
- `Trihalomethanes`
- `Turbidity`

and a target column named `Potability` (`0` = not potable, `1` = potable).

If your dataset uses different names, edit `TARGET_COLUMN` in `config.py`. The training script uses all numeric columns except the target as features.

## 2. Setup (Windows / VS Code)

Open this folder in VS Code. In the terminal:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Install libraries:

```bash
pip install -r requirements.txt
```

## 3. Train the model

Put your CSV in `data/`, then run from the project root:

```bash
python train.py
```

This creates:
- `output/models/water_quality_model.joblib`
- `output/models/model_metadata.json`
- graphs in `output/figures/`
- `output/metrics.json`

## 4. Run the Streamlit app

```bash
streamlit run app.py
```

Streamlit opens the app in your browser. The model must be trained first.

## 5. Project structure

```text
Water_Quality_Prediction/
├── app.py
├── train.py
├── config.py
├── requirements.txt
├── data/
│   └── water_potability.csv       # Add your dataset here
├── notebooks/
│   └── water_quality_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── model_utils.py
│   └── visualization.py
└── output/
    ├── figures/
    ├── models/
    └── metrics.json               # Generated after training
```

## Important

This is an educational prediction project, not a certified water-safety test. Do not use its prediction alone to decide whether water is safe to drink. Real drinking-water safety requires laboratory testing against applicable standards.
