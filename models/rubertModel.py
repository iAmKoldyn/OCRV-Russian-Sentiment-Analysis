import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, GRU, Dense, Input, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l1_l2
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, TFRobertaModel, RobertaTokenizer
from sklearn.utils.class_weight import compute_class_weight
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import optuna
import pymorphy2
import os
import datetime

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

# def configure_cpu():
#     tf.config.set_visible_devices([], 'GPU')
#     print("Configured to use CPU.")

def download_nltk_data():
    nltk.download('stopwords', quiet=True)

morph = pymorphy2.MorphAnalyzer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = text.split()
    stop_words = stopwords.words('russian')  # Load Russian stop words
    
    processed_words = []
    
    for word in words:
        if word not in stop_words:  # Remove stop words
            lemmatized_word = morph.parse(word)[0].normal_form  # Lemmatize the word
            processed_words.append(lemmatized_word)
    
    return ' '.join(processed_words)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    tqdm.pandas(desc="Processing Texts")
    df['processed_text'] = df['text'].progress_apply(preprocess_text)
    
    # Display sample data
    samples = df.sample(2)
    for index, row in samples.iterrows():
        print("Original text:", row['text'])
        print("Processed text:", row['processed_text'])
        print("---")
    
    counts = df['sentiment'].value_counts()
    min_count = counts.min()
    balanced_df = pd.DataFrame()
    for sentiment in df['sentiment'].unique():
        balanced_subset = df[df['sentiment'] == sentiment].sample(n=min_count, random_state=42)
        balanced_df = pd.concat([balanced_df, balanced_subset])
    
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    balanced_df['sentiment'] = balanced_df['sentiment'].astype(int)
    
    print("Balanced dataset counts:")
    print(balanced_df['sentiment'].value_counts())

    return balanced_df

def encode_texts(tokenizer, texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors="tf")



def prepare_dataset(encoded_texts, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(encoded_texts), labels))
    dataset = dataset.shuffle(189000).batch(32)
    return dataset

def load_rubert_model():
    model_name = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True, num_labels=3)
    return model, tokenizer

def train_and_evaluate_model(train_dataset, val_dataset, model, class_weight=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    
    # Define log directory for TensorBoard
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Ensure the TensorBoard log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # TensorBoard callback
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Early Stopping callback to halt training when validation loss stops improving
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # ReduceLROnPlateau callback to reduce learning rate when a metric has stopped improving
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-5, verbose=1)
    
    # Start training with callbacks
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        verbose=1,
        callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback],
        class_weight=class_weight  # Pass class weights here
    )
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")

    return model

def get_misclassified_texts(validation_df, predictions, tokenizer):
    predicted_labels = np.argmax(predictions, axis=-1)
    misclassified_indices = np.where(validation_df['sentiment'].values != predicted_labels)[0]

    misclassified_texts = validation_df.iloc[misclassified_indices].copy()
    
    # Safely assign the new values
    misclassified_texts.loc[:, 'predicted_label'] = predicted_labels[misclassified_indices]
    
    return misclassified_texts

def main():
    configure_gpu()
    # configure_cpu()
    download_nltk_data()
    
    # Load and preprocess data
    train_filepath = 'train_dataset.csv'
    validation_filepath = 'test_dataset.csv'
    train_df = load_and_preprocess_data(train_filepath)
    validation_df = load_and_preprocess_data(validation_filepath)
    
    # Load the RuBERT model and tokenizer
    model, tokenizer = load_rubert_model()
    
    # Tokenize and encode texts for training and validation sets
    train_encodings = tokenizer(train_df['processed_text'].tolist(), truncation=True, padding='max_length', max_length=512, return_tensors="tf")
    val_encodings = tokenizer(validation_df['processed_text'].tolist(), truncation=True, padding='max_length', max_length=512, return_tensors="tf")
    
    # Prepare datasets for training and validation
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_df['sentiment'].values)).shuffle(189000).batch(20)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), validation_df['sentiment'].values)).batch(20)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['sentiment'].values),
        y=train_df['sentiment'].values
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Train the model
    model = train_and_evaluate_model(train_dataset, val_dataset, model, class_weight=class_weight_dict)
    
    # Generate predictions from the validation dataset
    predictions = model.predict(val_dataset.map(lambda x, y: x))

    # Assuming predictions are logits directly if 'logits' attribute is not present
    predicted_logits = predictions.logits if hasattr(predictions, 'logits') else predictions

    predicted_labels = np.argmax(predicted_logits, axis=1)
    true_labels = np.concatenate([y.numpy() for x, y in val_dataset])

    # Calculate and print the classification report
    print(classification_report(true_labels, predicted_labels, target_names=['Neutral', 'Positive', 'Negative']))

    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Neutral', 'Positive', 'Negative'], yticklabels=['Neutral', 'Positive', 'Negative'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    misclassified_texts = get_misclassified_texts(validation_df, predicted_logits, tokenizer)
    print("Misclassified texts sample:")
    print(misclassified_texts.sample(min(10, len(misclassified_texts))))

    try:
        model.save('rubert_sentiment_model', save_format='tf')
        print("Model saved successfully!")
    except Exception as e:
        print(f"Model save failed: {e}")


if __name__ == "__main__":
    main()