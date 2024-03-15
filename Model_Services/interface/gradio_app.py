import gradio as gr
import requests
import re

def send_to_validator(text, model_choice):
    if not text.strip():
        return "Please enter some text to analyze."
    if text.isdigit():
        return "Input should not be only numbers."
    if re.match(r'^\W+$', text):
        return "Input should contain more than just special characters."
    if not re.search(r'[А-Яа-я]', text):
        return "Please enter text in Russian."

    if not model_choice or model_choice not in ['Rubert Sentiment Model','RuGpt Sentiment Model', 'CnnGru Sentiment Model', 'Gru Sentiment Model', 'Bilstm Sentiment Model']:
        return "Please select a valid model."

    validators = {
        'Rubert Sentiment Model': "http://rubert_validation:5000/predict",
        'RuGpt Sentiment Model': "http://gpt_validation:5001/predict",
        'CnnGru Sentiment Model': "http://cnngru_validation:5002/predict",
        'Gru Sentiment Model': "http://gru_validation:5010/predict",
        'Bilstm Sentiment Model': "http://bilstm_validation:5005/predict",

    }
    url = validators.get(model_choice)

    response = requests.post(url, json={"text": text})
    if response.status_code == 200:
        return response.json()['prediction']
    else:
        return f"Error in prediction: {response.text}"

def main():
    text_input = gr.components.Textbox(lines=2, placeholder="Type your text here...", label="Input Text")
    model_dropdown = gr.components.Dropdown(
        choices=['Rubert Sentiment Model','RuGpt Sentiment Model', 'CnnGru Sentiment Model', 'Gru Sentiment Model', 'Bilstm Sentiment Model'],
        label="Choose Model"
    )

    iface = gr.Interface(
        fn=send_to_validator,
        inputs=[text_input, model_dropdown],
        outputs="text",
        title="Text Sentiment Analysis",
        description="Enter text to analyze sentiment and choose a model."
    )
    iface.launch(server_name='0.0.0.0', server_port=7860)

if __name__ == "__main__":
    main()
