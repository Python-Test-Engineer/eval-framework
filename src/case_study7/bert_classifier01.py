import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# Suppress TensorFlow messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Optional: disable oneDNN optimizations
warnings.filterwarnings("ignore")


class TextDataset(Dataset):
    """Custom Dataset for BERT text classification"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize and encode
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BERTClassifier:
    """BERT-based text classifier"""

    def __init__(self, model_name="bert-base-uncased", num_classes=2, max_length=512):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.model.to(self.device)

        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
        print(f"Number of classes: {num_classes}")

    def prepare_data(self, texts, labels, test_size=0.2, batch_size=16):
        """Prepare training and validation datasets"""

        # Convert labels to numeric if they're strings
        if isinstance(labels[0], str):
            unique_labels = list(set(labels))
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            labels = [label_to_id[label] for label in labels]
            self.label_to_id = label_to_id
            self.id_to_label = {idx: label for label, idx in label_to_id.items()}
        else:
            self.label_to_id = None
            self.id_to_label = None

        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = TextDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = TextDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        return train_dataset, val_dataset

    def train(self, epochs=3, learning_rate=2e-5, warmup_steps=0):
        """Train the BERT classifier"""

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Training loop
        self.model.train()
        train_losses = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_loss = 0

            progress_bar = tqdm(self.train_loader, desc=f"Training")

            for batch in progress_bar:
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Update progress bar
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(self.train_loader)
            train_losses.append(avg_loss)

            # Validation
            val_accuracy = self.evaluate()
            print(f"Average training loss: {avg_loss:.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}")

        return train_losses

    def evaluate(self):
        """Evaluate the model on validation set"""
        self.model.eval()
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(actual_labels, predictions)
        return accuracy

    def predict(self, texts):
        """Make predictions on new texts"""
        self.model.eval()
        predictions = []
        probabilities = []

        # Create dataset for prediction
        dummy_labels = [0] * len(texts)  # Dummy labels for prediction
        pred_dataset = TextDataset(texts, dummy_labels, self.tokenizer, self.max_length)
        pred_loader = DataLoader(pred_dataset, batch_size=16, shuffle=False)

        with torch.no_grad():
            for batch in pred_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Get predictions and probabilities
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                batch_probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

                predictions.extend(batch_predictions)
                probabilities.extend(batch_probabilities)

        # Convert predictions back to original labels if needed
        if self.id_to_label:
            predictions = [self.id_to_label[pred] for pred in predictions]

        return predictions, probabilities

    def get_classification_report(self):
        """Get detailed classification report"""
        self.model.eval()
        predictions = []
        actual_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())

        # Convert back to original labels if needed
        if self.id_to_label:
            target_names = [self.id_to_label[i] for i in range(self.num_classes)]
        else:
            target_names = None

        report = classification_report(
            actual_labels, predictions, target_names=target_names
        )
        return report

    def save_model(self, path):
        """Save the trained model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a saved model"""
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")


# Example usage
def example_usage():
    """Example of how to use the BERT classifier"""

    # Sample data (replace with your actual data)
    texts = [
        "This movie is absolutely fantastic! Great acting and storyline.",
        "Terrible film, waste of time. Poor acting and boring plot.",
        "I loved every minute of this movie. Highly recommended!",
        "Not worth watching. Very disappointing and poorly made.",
        "Amazing cinematography and brilliant performances by all actors.",
        "One of the worst movies I've ever seen. Completely boring.",
        "Excellent script and direction. A must-watch film.",
        "Very poor quality. Would not recommend to anyone.",
    ]

    labels = [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
    ]

    # Initialize classifier
    classifier = BERTClassifier(num_classes=2, max_length=128)

    # Prepare data
    train_dataset, val_dataset = classifier.prepare_data(
        texts, labels, test_size=0.25, batch_size=8
    )

    # Train model
    print("\n" + "=" * 50)
    print("Starting training...")
    train_losses = classifier.train(epochs=2, learning_rate=2e-5)

    # Get classification report
    print("\n" + "=" * 50)
    print("Classification Report:")
    print(classifier.get_classification_report())

    # Make predictions on new data
    new_texts = [
        "This is an outstanding movie with great characters.",
        "Boring and predictable. Not recommended.",
    ]

    predictions, probabilities = classifier.predict(new_texts)

    print("\n" + "=" * 50)
    print("Predictions on new data:")
    for i, (text, pred, prob) in enumerate(zip(new_texts, predictions, probabilities)):
        confidence = max(prob)
        print(f"Text {i+1}: {text}")
        print(f"Prediction: {pred} (Confidence: {confidence:.4f})")
        print()


if __name__ == "__main__":
    # Run the basic severity classification example
    # example_usage()

    # Uncomment the line below for a more comprehensive severity analysis
    severity_analysis_example()
