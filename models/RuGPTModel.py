import re
import os
import torch
import pymorphy2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm.auto import tqdm
from torch.optim import AdamW
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc, NewsNERTagger, PER, NamesExtractor


segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
# Initialize the NER tagger
ner_emb = NewsEmbedding()
ner_tagger = NewsNERTagger(ner_emb)

morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        # Ensure labels are a long tensor right from the initialization
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        # Access the tensor directly without re-creating it
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def recognize_entities(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    
    for span in doc.spans:
        span.normalize(morph_vocab)
    
    # Optionally, replace entity tokens with a placeholder or remove them
    for span in doc.spans:
        if span.type == PER:
            text = text.replace(span.text, '')  # Remove or replace with a placeholder
    
    return text

def correct_spelling(tokens):
    spell = SpellChecker(language='ru')  # Initialize the spell checker with Russian language
    corrected_tokens = [spell.correction(token) for token in tokens]
    return corrected_tokens

def preprocess_text(text):
    text = text.lower()
    text = recognize_entities(text)  # Recognize and handle named entities
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    processed_tokens = []
    for token in doc.tokens:
        # Perform lemmatization
        token.lemmatize(morph_vocab)
        if token.text not in stop_words and not re.match(r'\d+', token.text):
            pos_tagged_token = f"{token.lemma}_{token.pos}" if token.pos else token.lemma
            processed_token = handle_negations(pos_tagged_token)
            processed_tokens.append(processed_token)
    
    return ' '.join(processed_tokens)

def handle_negations(token):
    # Check if token starts with negation followed by "_", indicating a POS tagged negated word
    if token.startswith("не_"):
        # Prefix the entire POS-tagged token with "не" to indicate negation
        return f"не{token}"
    return token


def load_dataset(file_path, tokenizer):
    df = pd.read_csv(file_path, sep=";")
    df['processed_text'] = df['text'].apply(preprocess_text)
    le = LabelEncoder()
    df['sentiment_encoded'] = le.fit_transform(df['sentiment'])
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['processed_text'].tolist(), df['sentiment_encoded'].tolist(), test_size=0.1, random_state=42)  # Ensure conversion to list
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    return train_dataset, val_dataset

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    tqdm.pandas(desc="Processing Texts")
    df['processed_text'] = df['text'].progress_apply(preprocess_text)
    
    # Print sample data before and after processing
    print("\nSample data before and after processing:")
    for _, row in df.sample(2).iterrows():
        print(f"Original text: {row['text']}")
        print(f"Processed text: {row['processed_text']}\n---")

    # Print sentiment counts before balancing
    print("Sentiment counts before balancing:")
    print(df['sentiment'].value_counts())

    # Balancing the dataset
    counts = df['sentiment'].value_counts()
    min_count = counts.min()
    balanced_df = pd.DataFrame()
    for sentiment in df['sentiment'].unique():
        balanced_subset = df[df['sentiment'] == sentiment].sample(n=min_count, random_state=42)
        balanced_df = pd.concat([balanced_df, balanced_subset], ignore_index=True)
    
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_df['sentiment'] = balanced_df['sentiment'].astype(int)
    
    # Print sentiment counts after balancing
    print("Sentiment counts after balancing:")
    print(balanced_df['sentiment'].value_counts())

    return balanced_df

def load_rugpt_model():
    model_name = "sberbank-ai/rugpt3small_based_on_gpt2"  # Change to a larger model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=3)
    return model, tokenizer

def train(model, train_loader, val_loader, device, optimizer, num_epochs=10, model_name='rugpt3small_based_on_gpt2'):
    model.train()
    scheduler = StepLR(optimizer, step_size=2, gamma=0.85)  # Adjust learning rate schedule
    
    early_stopping_patience = 10  # Increase patience
    best_val_accuracy = 0
    save_path = f"./models/{model_name}.bin"  # Model saving path

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'Training Loss': f'{running_loss/(progress_bar.last_print_n+1):.4f}'})

        scheduler.step()  # Adjust learning rate
        
        # Validation phase with its own progress bar
        val_running_loss = 0.0
        predictions, true_labels = [], []
        val_progress_bar = tqdm(val_loader, desc="Validating")
        model.eval()
        with torch.no_grad():
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss
                val_running_loss += val_loss.item()
                predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                val_progress_bar.set_postfix({'Validation Loss': f'{val_running_loss/(val_progress_bar.last_print_n+1):.4f}'})
                
        val_loss = val_running_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Validation Accuracy: {accuracy}")
        
        if accuracy > best_val_accuracy:  # Check if current accuracy is the best
            best_val_accuracy = accuracy  # Update best accuracy
            epochs_without_improvement = 0
            print(f"New best accuracy: {accuracy}. Saving the model...")
            
            # Ensure the model directory exists
            if not os.path.exists("./models"):
                os.makedirs("./models")
            
            torch.save(model.state_dict(), save_path)  # Save the model
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered.")
                break
    
    # Final evaluation
    model.eval()
    final_predictions, final_true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            final_predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            final_true_labels.extend(labels.cpu().numpy())
    
    print(classification_report(final_true_labels, final_predictions, target_names=['Neutral', 'Positive', 'Negative']))
    return final_predictions, final_true_labels

def get_misclassified_texts(validation_df, predictions, true_labels):
    # Ensure both are numpy arrays
    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)

    misclassified_indices = np.where(true_labels != predictions)[0]
    misclassified_texts = validation_df.iloc[misclassified_indices].copy()

    # Debugging shapes and types
    print(f"Predictions shape: {predictions.shape}, Type: {type(predictions)}")
    print(f"Misclassified Indices: {misclassified_indices.shape}, Type: {type(misclassified_indices)}")

    # Assignment of predicted labels
    misclassified_texts['predicted_label'] = predictions[misclassified_indices]
    
    return misclassified_texts

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, tokenizer = load_rugpt_model()
    model.to(device)

    # Load and preprocess your datasets
    train_df = load_and_preprocess_data('train_dataset.csv')
    validation_df = load_and_preprocess_data('test_dataset.csv')

    # Tokenize and encode texts for training and validation sets
    train_encodings = tokenizer(train_df['processed_text'].tolist(), truncation=True, padding='max_length', max_length=512, return_tensors="pt")
    val_encodings = tokenizer(validation_df['processed_text'].tolist(), truncation=True, padding='max_length', max_length=512, return_tensors="pt")

    # Creating datasets directly with pre-tokenized encodings
    train_dataset = SentimentDataset(train_encodings, train_df['sentiment'].values)
    val_dataset = SentimentDataset(val_encodings, validation_df['sentiment'].values)

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Prepare for training
    optimizer = AdamW(model.parameters(), lr=1e-5)
    final_predictions, final_true_labels = train(model, train_loader, val_loader, device, optimizer)

    # Visualizing the confusion matrix
    cm = confusion_matrix(final_true_labels, final_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Neutral', 'Positive', 'Negative'], yticklabels=['Neutral', 'Positive', 'Negative'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Analyzing misclassified texts
    misclassified_texts = get_misclassified_texts(validation_df, final_predictions, final_true_labels)
    print("Misclassified texts sample:")
    print(misclassified_texts.sample(min(10, len(misclassified_texts))))

if __name__ == "__main__":
    main()