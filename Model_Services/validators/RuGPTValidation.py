from flask import Flask, request, jsonify
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config
import numpy as np
import re

app = Flask(__name__)

# Load model and tokenizer on startup
model_bin_path = "/app/models_data_180k/ru_gpt_model_train/rugpt3small_based_on_gpt2.bin"
model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
config = GPT2Config.from_pretrained(model_name, num_labels=3)
model = GPT2ForSequenceClassification(config)
model.load_state_dict(torch.load(model_bin_path, map_location=torch.device('cpu')))

def preprocess_text(text):
    text = text.lower()
    return text

def predict_sentiment(text):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    processed_text = preprocess_text(text)
    inputs = tokenizer.encode(processed_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
    
    sentiment_labels = ['Neutral', 'Positive', 'Negative']
    predicted_label_index = np.argmax(predictions.cpu().numpy())
    predicted_sentiment = sentiment_labels[predicted_label_index]
    return predicted_sentiment

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
