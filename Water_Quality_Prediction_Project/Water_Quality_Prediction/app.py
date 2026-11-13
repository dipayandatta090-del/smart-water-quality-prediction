import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from config import MODEL_PATH, METADATA_PATH, METRICS_PATH, FIGURES_DIR

st.set_page_config(page_title="Water Quality Prediction", page_icon="💧", layout="wide")

st.title("💧 Water Quality Prediction")
st.write(
    "Enter the water sample measurements below to get a model prediction. "
    "This is an educational machine-learning demonstration, not a laboratory test."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.warning("The trained model was not found. First add your CSV to the data folder and run: `python train.py`")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

st.subheader("Enter water sample values")
feature_values = {}

with st.form("prediction_form"):
    cols = st.columns(2)
    for index, feature in enumerate(metadata["feature_names"]):
        info = metadata["feature_info"][feature]
        minimum = info["min"]
        maximum = info["max"]
        default = info["median"]

        # Streamlit sliders need a non-zero range.
        if minimum == maximum:
            value = cols[index % 2].number_input(
                feature, value=float(default), key=feature
            )
        else:
            value = cols[index % 2].number_input(
                feature,
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(default),
                key=feature
            )
        feature_values[feature] = value

    submitted = st.form_submit_button("Predict Water Quality", type="primary")

if submitted:
    input_df = pd.DataFrame([feature_values], columns=metadata["feature_names"])
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction result")
    st.write(f"**Predicted class:** {prediction}")

    # Common public dataset convention: 0 = not potable, 1 = potable.
    if str(prediction) == "1":
        st.success("The model classifies this sample as POTABLE (class 1).")
    elif str(prediction) == "0":
        st.error("The model classifies this sample as NOT POTABLE (class 0).")
    else:
        st.info(f"Model output: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        probability_df = pd.DataFrame({
            "Class": [str(c) for c in model.classes_],
            "Probability": probabilities
        })
        st.write("Prediction probabilities")
        st.bar_chart(probability_df.set_index("Class"))

st.divider()
st.subheader("Model evaluation")
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Precision", f"{metrics['precision_weighted']:.1%}")
    c3.metric("Recall", f"{metrics['recall_weighted']:.1%}")
    c4.metric("F1-score", f"{metrics['f1_weighted']:.1%}")

    for image_name, title in [
        ("target_distribution.png", "Target distribution"),
        ("correlation_heatmap.png", "Correlation heatmap"),
        ("feature_distributions.png", "Feature distributions"),
        ("confusion_matrix.png", "Confusion matrix"),
        ("feature_importance.png", "Feature importance")
    ]:
        image_path = FIGURES_DIR / image_name
        if image_path.exists():
            st.subheader(title)
            st.image(str(image_path), use_container_width=True)

st.caption(
    "Disclaimer: A machine-learning prediction cannot certify drinking-water safety. "
    "Use accredited laboratory testing and applicable drinking-water standards."
)
