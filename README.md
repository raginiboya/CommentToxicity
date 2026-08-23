# Comment Toxicity Detection Using Deep Learning

## Project Overview

Comment Toxicity Detection Using Deep Learning is an intelligent natural language processing system developed to identify and classify harmful, abusive, or inappropriate content in online comments. The project addresses the growing need for automated content moderation on digital platforms, where large volumes of user-generated text must be reviewed efficiently, consistently, and accurately.

The system analyzes the textual content of a comment and determines whether it contains one or more forms of toxic behavior. Unlike a single-label classification system, this project uses a multi-label classification approach, allowing one comment to belong to multiple toxicity categories simultaneously. For example, a comment may be classified as both toxic and insulting, or as obscene and severely toxic.

The model predicts the following six toxicity categories:

* **Toxic:** General harmful, abusive, or offensive language.
* **Severe Toxic:** Extremely abusive or highly aggressive language.
* **Obscene:** Profane, vulgar, or sexually explicit language.
* **Threat:** Statements that express an intention to cause harm or violence.
* **Insult:** Directly offensive, degrading, or disrespectful language toward another person or group.
* **Identity Hate:** Hateful or discriminatory language directed at an individual or group based on identity-related characteristics.

The project uses an **LSTM-based deep learning neural network** for text classification. Long Short-Term Memory networks are well suited to this task because they can process sequences of words while retaining important contextual information from earlier parts of a sentence. Before being passed to the model, comments are cleaned, tokenized, converted into numerical sequences, and padded to a consistent length.

The trained model is integrated into an interactive **Streamlit web application**. The application provides two primary prediction modes:

1. **Single-Comment Prediction:** Users can enter an individual comment and receive toxicity scores for all six categories, along with an overall toxic or non-toxic classification.
2. **Bulk CSV Prediction:** Users can upload a CSV file containing multiple comments. The application processes the comments in batches, generates predictions for each toxicity category, displays the results, and provides an option to download the completed prediction file.

This project demonstrates the complete machine learning workflow, including text preprocessing, model training, multi-label classification, model evaluation, application development, and result visualization. It is intended as an educational and practical example of how deep learning and natural language processing can be applied to support automated online content moderation.

The system is designed to assist human moderators and platform administrators by identifying potentially harmful comments efficiently. However, model predictions should be treated as decision-support information rather than absolute judgments, since language can be context-dependent and automated systems may occasionally produce false positives or false negatives.

## Problem Statement

Online platforms receive a large volume of user-generated comments, making manual moderation difficult, time-consuming, and inconsistent. Harmful content such as abusive language, insults, threats, obscene language, and identity-based hate can negatively affect users and online communities.

The objective of this project is to develop a deep learning-based system that can automatically analyze textual comments and identify multiple forms of toxicity. The system performs multi-label classification across six categories: toxic, severe toxic, obscene, threat, insult, and identity hate.

The trained model is integrated with a Streamlit application so that users can analyze individual comments as well as upload CSV files for bulk toxicity prediction.

## Dataset Information

The project uses a toxic comment classification dataset containing **159,571 training comments**. Each record consists of a unique comment identifier, the original comment text, and six binary toxicity labels.

The dataset contains the following columns:

* **id:** Unique identifier assigned to each comment.
* **comment_text:** The original text of the user comment.
* **toxic:** Indicates whether the comment is generally toxic.
* **severe_toxic:** Indicates highly abusive or severely toxic content.
* **obscene:** Indicates obscene or profane language.
* **threat:** Indicates threatening language.
* **insult:** Indicates insulting or degrading language.
* **identity_hate:** Indicates hateful language related to identity.

Each toxicity label is represented using binary values:

* **0:** The comment does not belong to that category.
* **1:** The comment belongs to that category.

Since a single comment can have more than one label with a value of 1, the problem is treated as a **multi-label text classification task**.

During exploratory data analysis, the training dataset was found to contain:

* Toxic: **15,294**
* Severe Toxic: **1,595**
* Obscene: **8,449**
* Threat: **478**
* Insult: **7,877**
* Identity Hate: **1,405**

Approximately **9.58%** of the comments were labelled toxic. The distribution also showed significant class imbalance, particularly for categories such as threat and identity hate.

## Technologies Used

The following technologies and libraries were used to develop this project:

* **Python:** Main programming language used for data preprocessing, model development, and application logic.
* **Pandas:** Used for loading, exploring, cleaning, and processing CSV datasets.
* **Matplotlib:** Used for exploratory data analysis and visualization of toxicity category distributions.
* **Scikit-learn:** Used for splitting the dataset into training and testing sets and calculating evaluation metrics such as precision, recall, and F1-score.
* **TensorFlow / Keras:** Used to build, train, evaluate, save, and load the deep learning model.
* **LSTM (Long Short-Term Memory):** Used as the main neural network architecture for learning patterns and context from comment sequences.
* **Streamlit:** Used to develop the interactive web application for single-comment and bulk CSV toxicity prediction.
* **Pickle:** Used to save and reload the trained tokenizer.
* **VS Code:** Used as the development environment for the notebook, Python application, and project files.

## Project Workflow

The project was developed using the following workflow:

1. **Dataset Loading**

   * Loaded the training dataset using Pandas.
   * Inspected the dataset shape, columns, and sample records.

2. **Exploratory Data Analysis**

   * Checked for missing values.
   * Analyzed the distribution of the six toxicity categories.
   * Identified class imbalance in the dataset.

3. **Text Preprocessing**

   * Converted comments to lowercase.
   * Removed line breaks and unnecessary special characters.
   * Created a cleaned version of each comment.

4. **Tokenization**

   * Created a tokenizer using the 20,000 most frequent words.
   * Converted cleaned comments into numerical sequences.

5. **Sequence Padding**

   * Padded all sequences to a fixed length of 100 tokens so they could be processed by the neural network.

6. **Train-Test Split**

   * Split the dataset into 80% training data and 20% testing data.

7. **Model Development**

   * Built an LSTM-based deep learning model using TensorFlow/Keras.
   * Used an Embedding layer, LSTM layer, Dropout layer, and Dense output layer.

8. **Model Training**

   * Trained the model for 2 epochs using the Adam optimizer and binary cross-entropy loss.

9. **Model Evaluation**

   * Evaluated the model using precision, recall, and F1-score.
   * Analyzed performance across all six toxicity categories.

10. **Model and Tokenizer Saving**

    * Saved the trained model as an H5 file.
    * Saved the tokenizer using Pickle.

11. **Streamlit Application**

    * Developed an interactive interface for single-comment toxicity prediction.
    * Added model performance metrics and sample test cases.
    * Added CSV upload, bulk prediction, and downloadable prediction results.

## Model Architecture

The toxicity classifier was built using a Sequential deep learning model in TensorFlow/Keras.

The architecture consists of the following layers:

1. **Embedding Layer**

   * Vocabulary size: **20,000 words**
   * Embedding dimension: **128**
   * Converts each word index into a dense numerical vector representation.

2. **LSTM Layer**

   * Contains **64 LSTM units**.
   * Learns sequential patterns and contextual information from the comments.

3. **Dropout Layer**

   * Dropout rate: **0.3**
   * Helps reduce overfitting by randomly disabling a portion of neurons during training.

4. **Dense Output Layer**

   * Contains **6 output neurons**, one for each toxicity category.
   * Uses the **Sigmoid activation function** because this is a multi-label classification problem.

### Model Compilation

* **Optimizer:** Adam
* **Loss Function:** Binary Cross-Entropy
* **Metric:** Accuracy

Binary cross-entropy was selected because each toxicity category is predicted independently as either 0 or 1.

## Model Evaluation

The trained LSTM model was evaluated on the test dataset using precision, recall, and F1-score.

### Classification Results

* **Toxic**

  * Precision: **0.88**
  * Recall: **0.71**
  * F1-score: **0.79**

* **Severe Toxic**

  * Precision: **0.71**
  * Recall: **0.11**
  * F1-score: **0.19**

* **Obscene**

  * Precision: **0.83**
  * Recall: **0.74**
  * F1-score: **0.78**

* **Threat**

  * Precision: **0.00**
  * Recall: **0.00**
  * F1-score: **0.00**

* **Insult**

  * Precision: **0.75**
  * Recall: **0.61**
  * F1-score: **0.67**

* **Identity Hate**

  * Precision: **0.00**
  * Recall: **0.00**
  * F1-score: **0.00**

### Overall Performance

* **Micro Average F1-score:** 0.72
* **Macro Average F1-score:** 0.41
* **Weighted Average F1-score:** 0.69

The model performed well on more frequent categories such as toxic, obscene, and insult. Performance was lower on rare categories such as threat and identity hate because the dataset contains significantly fewer examples for these classes.

This class imbalance is an important limitation of the current model and can be addressed in future work using techniques such as class weighting, resampling, threshold tuning, or more advanced transformer-based models.

## How to Run the Application

1. Clone or download the project repository.

2. Open the project folder in VS Code.

3. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run data/app.py
```

5. Open the local Streamlit URL displayed in the terminal, usually:

```text
http://localhost:8501
```

6. Enter a comment in the text box and click **Predict** to view the toxicity probabilities and overall result.

7. For bulk prediction, upload a CSV file containing a column named:

```text
comment_text
```

Click **Run Bulk Prediction** and download the generated prediction results using the **Download Prediction Results** button.

## Business Use Cases

The Comment Toxicity Detection system can be applied in several real-world scenarios:

1. **Social Media Platforms**

   * Automatically detect and filter toxic comments in real time.

2. **Online Forums and Communities**

   * Moderate user-generated discussions and reduce harmful interactions.

3. **Content Moderation Services**

   * Support companies that provide moderation services for online platforms.

4. **Brand Safety and Reputation Management**

   * Help brands identify inappropriate or toxic content appearing around their advertisements and sponsored content.

5. **E-Learning Platforms and Educational Websites**

   * Create safer online learning environments for students and educators.

6. **News Websites and Media Outlets**

   * Moderate user comments posted on news articles and other media content.

## Project Structure

```text
CommentToxicity/
│
├── data/
│   ├── app.py
│   ├── train.csv
│   ├── test.csv
│   ├── toxcity_detection.ipynb
│   ├── toxcity_detection_model.h5
│   └── tokenizer.pkl
│
├── requirements.txt
└── README.md
```

### File Description

* **app.py** – Streamlit web application for single and bulk toxicity prediction.
* **train.csv** – Training dataset used to develop the model.
* **test.csv** – Test dataset used for bulk prediction.
* **toxcity_detection.ipynb** – Notebook containing data exploration, preprocessing, model training, and evaluation.
* **toxcity_detection_model.h5** – Saved trained LSTM model.
* **tokenizer.pkl** – Saved tokenizer used to preprocess new comments.
* **requirements.txt** – Python libraries required to run the project.
* **README.md** – Complete project documentation.
