import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
import pymorphy2
from flask import Flask, request, jsonify

# Initialize Flask application
app = Flask(__name__)

# Initialize tokenizer and model
tokenizer_name = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

model_path = '/app/models_data_180k/cnn_gru_sentiment_model_train'
model = load_model(model_path)

# Text preprocessing function
def preprocess_text(text):
    morph = pymorphy2.MorphAnalyzer()
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    nltk.download('stopwords', quiet=True)
    words = text.split()
    stop_words = stopwords.words('russian')
    words = [word for word in words if word not in stop_words]
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized_words)

# Function to encode texts
def encode_texts(texts):
    encoded = tokenizer(texts, padding='max_length', truncation=True, max_length=200, return_tensors="np")
    return encoded['input_ids']

# Function to classify texts
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    encoded_text = encode_texts([preprocessed_text])
    predictions = model.predict(encoded_text)
    predicted_label = np.argmax(predictions, axis=1)[0]
    sentiment_labels = ['neutral', 'positive', 'negative']
    return sentiment_labels[predicted_label]

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.json
        text = data.get('text')
        if not text or text.strip() == '':
            raise ValueError("No text provided for prediction.")
        if re.match(r'^\W+$', text) or text.isdigit():
            raise ValueError("Input should be meaningful text, not only numbers or special characters.")
        if not re.search(r'[А-Яа-я]', text):
            raise ValueError("Input should be in Russian.")

        prediction = predict_sentiment(text)
        return jsonify({'prediction': prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)
