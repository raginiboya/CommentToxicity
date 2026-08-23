import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from utils import predict_bulk, predict_comment


# File paths
BASE_DIR = Path(__file__).parent


# Load trained model
model = load_model(
    BASE_DIR / "toxcity_detection_model.h5",
    compile=False
)


# Load tokenizer
with open(BASE_DIR / "tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


# Toxicity labels
labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# App title
st.title("Comment Toxicity Detection")


# Single comment prediction
comment = st.text_area("Enter a comment:")

if st.button("Predict"):
    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:
        prediction = predict_comment(
            comment,
            tokenizer,
            model
        )

        st.subheader("Prediction Results")

        for label, score in zip(labels, prediction):
            st.write(f"{label}: {score:.4f}")

        if prediction[0] > 0.5:
            st.error("⚠️ Toxic Comment Detected")
        else:
            st.success("✅ Non-Toxic Comment")


# Model performance
st.divider()

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Micro F1-Score", "0.72")
col2.metric("Weighted F1-Score", "0.69")
col3.metric("Toxic F1-Score", "0.79")


# Sample test cases
st.divider()

st.subheader("Sample Test Cases")

st.markdown(
    """
- **Non-Toxic:** Thank you for your help, I really appreciate it.
- **Toxic / Insult:** You are a stupid idiot.
- **Threat Example:** I will hurt you if you come here again.
"""
)


# Bulk prediction
st.divider()

st.subheader("Bulk Prediction")

uploaded_file = st.file_uploader(
    "Upload a CSV file for bulk toxicity prediction",
    type=["csv"]
)

if uploaded_file is not None:
    bulk_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(bulk_df.head())

    if st.button("Run Bulk Prediction"):
        if "comment_text" not in bulk_df.columns:
            st.error(
                "CSV must contain a 'comment_text' column."
            )
        else:
            with st.spinner("Running bulk predictions..."):
                bulk_predictions = predict_bulk(
                    bulk_df["comment_text"],
                    tokenizer,
                    model,
                    labels
                )

                for index, label in enumerate(labels):
                    bulk_df[f"{label}_score"] = (
                        bulk_predictions[:, index]
                    )

                    bulk_df[f"{label}_prediction"] = (
                        bulk_predictions[:, index] > 0.5
                    ).astype(int)

                st.success("Bulk Prediction Completed!")

                st.dataframe(bulk_df.head(20))

                csv = bulk_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv"
                )
