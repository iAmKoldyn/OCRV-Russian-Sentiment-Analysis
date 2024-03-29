import numpy as np
import re
import tensorflow as tf
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

import json

# Force TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Initialize Flask app
app = Flask(__name__)

model_path = '/app/models_data_180k/bilstm_sentiment_model_train'
model = load_model(model_path)
stop_words = stopwords.words('russian')
stemmer = SnowballStemmer('russian')

# Load tokenizer from JSON file
with open('/app/models_data_180k/bilstm_sentiment_tokenizer_train/tokenizer.json') as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)
    
# Function to preprocess text
def preprocess_text(text, stop_words, stemmer):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Function to predict sentiment
def predict_sentiment(text, model, tokenizer, stop_words, stemmer, max_length=200):
    preprocessed_text = preprocess_text(text, stop_words, stemmer)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction, axis=1)[0]
    sentiment_labels = ['negative', 'neutral', 'positive']
    return sentiment_labels[predicted_label]

# Flask route for prediction
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.json
        text = data.get('text')
        if not text or text.strip() == '':
            raise ValueError("No text provided for prediction")
        if re.match(r'^\W+$', text) or text.isdigit():
            raise ValueError("Input should be meaningful text, not only numbers or special characters.")
        if not re.search(r'[А-Яа-я]', text):
            raise ValueError("Input should be in Russian.")

        prediction = predict_sentiment(text, model, tokenizer, stop_words, stemmer)
        return jsonify({'prediction': prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True)
