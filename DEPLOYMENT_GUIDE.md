# Deployment Guide

## Comment Toxicity Detection with Streamlit

This guide explains how to set up and run the Comment Toxicity Detection project using Streamlit.

The application uses a trained deep learning model and tokenizer to analyze user comments and predict toxicity across multiple categories.

The Streamlit application supports:

* Single-comment toxicity prediction
* Toxicity probability scores
* Model performance display
* Sample test cases
* CSV upload for bulk prediction
* Downloadable prediction results

## Installation Requirements

Before running the application, make sure the following are installed on your system:

* Python
* VS Code or another Python IDE
* pip package manager

Install the required project libraries using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes the main dependencies required for this project, such as:

* Streamlit
* TensorFlow
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## Running the Streamlit Application

1. Open the project folder in VS Code.

2. Open a new terminal.

3. Run the Streamlit application using:

```bash
streamlit run data/app.py
```

4. After the application starts, Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

5. Open this URL in a web browser to access the Comment Toxicity Detection application.

The application will automatically load the saved trained model and tokenizer before making predictions.

## Using the Application

### Single Comment Prediction

1. Enter a comment in the text input box.
2. Click the **Predict** button.
3. The application will display probability scores for:

   * Toxic
   * Severe Toxic
   * Obscene
   * Threat
   * Insult
   * Identity Hate
4. The application will also display whether the comment is detected as toxic or non-toxic.

### Bulk CSV Prediction

1. Scroll to the **Bulk Prediction** section.
2. Upload a CSV file containing a column named:

```text
comment_text
```

3. Click **Run Bulk Prediction**.
4. The application will process all comments using the trained LSTM model.
5. Prediction scores and binary predictions will be added to the results.
6. Click **Download Prediction Results** to save the output as a CSV file.

## Model Files Required

The Streamlit application requires the following trained files to make predictions:

* **toxcity_detection_model.h5** – Contains the trained LSTM deep learning model.
* **tokenizer.pkl** – Contains the tokenizer fitted on the training comments.

Both files must remain in the same folder as `app.py`:

```text
data/
├── app.py
├── toxcity_detection_model.h5
└── tokenizer.pkl
```

When the Streamlit application starts, it loads these files automatically. The tokenizer converts new comments into numerical sequences, and the trained model uses those sequences to generate toxicity predictions.
