import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from transformers import AutoTokenizer
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Configure GPU for training (if available)
def configure_gpu():
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

# Download NLTK data for text preprocessing
def download_nltk_data():
    nltk.download('stopwords', quiet=True)

# Function to preprocess text data
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

# Function to load and preprocess the dataset
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df



# Define the custom GRU model class
class CustomGRUModel(Model):
    def __init__(self, vocab_size, embedding_dim, input_length, gru_units, dropout_rate, num_classes):
        super(CustomGRUModel, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)
        self.spatial_dropout = SpatialDropout1D(dropout_rate)
        self.bidirectional_gru = Bidirectional(GRU(gru_units, return_sequences=False))
        self.dense = Dense(num_classes, activation='softmax')
        
    # The call method defines the forward pass of the model.
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.spatial_dropout(x)
        x = self.bidirectional_gru(x)
        return self.dense(x)
        
# Helper function to encode texts using the tokenizer.
def encode_texts(tokenizer, texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=200, return_tensors="np")

# Training and evaluation function for the model.
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    # Instantiate the model with the given hyperparameters.
    model = CustomGRUModel(vocab_size=5000, embedding_dim=128, input_length=200, gru_units=64, dropout_rate=0.2, num_classes=3)
    model.build(input_shape=(None, 200))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()  # Print a summary of the model.
    # Callback for early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    # Train the model using the training data and validation data    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, callbacks=[early_stopping], verbose=2)
    # Evaluate the model's performance on the test set    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f'Test Accuracy: {accuracy}')
    # Use the trained model to predict the test set   
    predictions = model.predict(X_test, verbose=2)
    return model, y_test, predictions

def main():
    configure_gpu()  # Set up GPU configuration
    download_nltk_data() # Ensure required NLTK data is downloaded
    # Load and preprocess the training data
    filepath = 'train_2.csv'
    df = load_and_preprocess_data(filepath)

    # Initialize and configure the Keras Tokenizer
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['processed_text'])
    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    padded_sequences = pad_sequences(sequences, maxlen=200)

    # Encode labels into a binary (one-hot) representation
    encoder = LabelEncoder()
    df['sentiment_numeric'] = encoder.fit_transform(df['sentiment'])
    labels = to_categorical(df['sentiment_numeric'])

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

    # Model training
    model = CustomGRUModel(vocab_size=5000, embedding_dim=128, input_length=200, gru_units=64, dropout_rate=0.2, num_classes=3)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=2)

    # Save the model and tokenizer
    model.save('gru_sentiment_model', save_format='tf')
    
    # Save tokenizer to JSON
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    print("Tokenizer saved to tokenizer.json")

if __name__ == "__main__":
    main()
