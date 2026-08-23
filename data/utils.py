import re

from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def predict_comment(text, tokenizer, model):
    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=100
    )

    prediction = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    return prediction


def predict_bulk(comments, tokenizer, model, labels):
    cleaned_comments = (
        comments
        .fillna("")
        .astype(str)
        .apply(clean_text)
    )

    sequences = tokenizer.texts_to_sequences(cleaned_comments)

    padded_sequences = pad_sequences(
        sequences,
        maxlen=100
    )

    predictions = model.predict(
        padded_sequences,
        batch_size=512,
        verbose=0
    )

    return predictions
