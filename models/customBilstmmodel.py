import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Bidirectional, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import json

# Configure GPU settings if available, otherwise use CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU, using CPU.")

# Download the Russian stopwords dataset from NLTK
nltk.download('stopwords', quiet=True)

# Function to preprocess text data
# This function converts text to lowercase, removes punctuation and numbers,
# splits it into words, removes stopwords, and applies stemming.
def preprocess_text(text, stop_words, stemmer):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


# Load the dataset from the provided filepath and preprocess the text column
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    stop_words = stopwords.words('russian')
    stemmer = SnowballStemmer('russian')
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, stop_words, stemmer))
    return df

# Custom BiLSTM (Bidirectional Long Short-Term Memory) model class
# It consists of an embedding layer, a spatial dropout layer for regularization,
# a bidirectional LSTM layer, and a dense output layer with a softmax activation.
class CustomBiLSTMModel(Model):
    def __init__(self, vocab_size, embedding_dim, input_length, lstm_units, dropout_rate, num_classes):
        super(CustomBiLSTMModel, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)
        self.spatial_dropout = SpatialDropout1D(dropout_rate)
        self.bilstm = Bidirectional(LSTM(lstm_units, return_sequences=False))
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.spatial_dropout(x)
        x = self.bilstm(x)
        return self.dense(x)


def main():
    # Load and preprocess the training data
    filepath = 'output_sorted.csv'
    df = load_and_preprocess_data(filepath)

    # Encode sentiment labels to a one-hot encoded format
    encoder = LabelEncoder()
    df['sentiment_numeric'] = encoder.fit_transform(df['sentiment'])
    labels = to_categorical(df['sentiment_numeric'])

    # Initialize a tokenizer, fit it on the preprocessed texts, and convert texts to padded sequences
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['processed_text'])
    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    padded = pad_sequences(sequences, maxlen=200)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)
    
    # Initialize the BiLSTM model, compile it, and start training
    model = CustomBiLSTMModel(vocab_size=5000, embedding_dim=128, input_length=200, lstm_units=64, dropout_rate=0.2,
                              num_classes=3)

    model.build(input_shape=(None, 200))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    
    # Continue training the model with early stopping to prevent overfitting.
    # Early stopping will halt the training if the validation loss doesn't improve after a specified number of epochs.
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16, callbacks=[early_stopping], verbose=2)
    
    # After training, evaluate the model on the test set to get the final accuracy.
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test Accuracy: {accuracy}')
    
    # Use the trained model to predict sentiments on the test set.
    predictions = model.predict(X_test, verbose=2)
    print(f'Test Accuracy: {accuracy}')
    
    # Save the entire model to a HDF5 file. The '.h5' extension indicates that the model should be saved in Keras format as a single HDF5 file.
    model.save('bilstm_sentiment_model', save_format='tf')

    # Save tokenizer to JSON
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    print("Tokenizer saved to tokenizer.json")
if __name__ == "__main__":
    main()
