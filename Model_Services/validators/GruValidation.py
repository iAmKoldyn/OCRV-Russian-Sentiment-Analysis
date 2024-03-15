import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

app = Flask(__name__)

model_path = '/app/models_data_180k/gru_sentiment_model_train'
model = tf.keras.models.load_model(model_path)

with open('/app/models_data_180k/gru_sentiment_tokenizer_train/tokenizer.json') as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)

def preprocess_text(text):
    stop_words = stopwords.words('russian')
    stemmer = SnowballStemmer('russian')
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Prediction function
def predict_sentiment(text, model, tokenizer):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    sequence = tokenizer.texts_to_sequences([processed_text])
    
    max_length = 200
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length)
    
    # Predict
    predictions = model.predict(padded_sequence)
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    sentiment_labels = ['negative', 'neutral', 'positive']
    
    return sentiment_labels[predicted_label_index]

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

        prediction = predict_sentiment(text, model, tokenizer)
        return jsonify({'prediction': prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5010, debug=True)
