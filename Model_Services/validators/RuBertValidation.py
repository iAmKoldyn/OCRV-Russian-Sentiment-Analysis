from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
import re

app = Flask(__name__)

model_path = '/app/models_data_180k/rubert_sentiment_model_train'
tokenizer_name = 'DeepPavlov/rubert-base-cased'
model = tf.keras.models.load_model(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def preprocess_and_encode_texts(texts, tokenizer):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="tf")
    return dict(encoded)

def predict_sentiment(text):
    encoded_inputs = preprocess_and_encode_texts([text], tokenizer)
    predictions = model.predict(encoded_inputs)
    logits = predictions if 'logits' not in predictions else predictions['logits']
    predicted_label_index = np.argmax(logits, axis=1)[0]
    sentiment_labels = ['neutral', 'positive', 'negative']
    return sentiment_labels[predicted_label_index]

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
        
        prediction = predict_sentiment(text)
        return jsonify({'prediction': prediction}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)