import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Create sample CSV data for politeness classification
train_data = {
    'sentence': [
        "Could you please help me with this task?",
        "Do this now!",
        "I would be grateful if you could assist me.",
        "Get me coffee.",
        "Thank you so much for your time and consideration.",
        "Fix this immediately.",
        "Would it be possible to schedule a meeting?",
        "You're wrong about everything.",
        "I appreciate your patience with my questions.",
        "That's completely stupid."
    ],
    'politeness': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = polite, 0 = impolite
}

# Create validation dataset
validation_data = {
    'sentence': [
        "Excuse me, could I have a moment of your time?",
        "Move out of my way!",
        "I hope I'm not bothering you with this request.",
        "Give me the report.",
        "Thank you for considering my proposal.",
        "This is terrible work.",
        "May I suggest an alternative approach?",
        "You have no idea what you're doing.",
        "I'm sorry to interrupt, but could you clarify something?",
        "That's the dumbest thing I've ever heard."
    ],
    'politeness': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = polite, 0 = impolite
}

# Save datasets to CSV
train_df = pd.DataFrame(train_data)
validation_df = pd.DataFrame(validation_data)

train_df.to_csv('train_politeness_data.csv', index=False)
validation_df.to_csv('validation_politeness_data.csv', index=False)

print("Training data saved to train_politeness_data.csv")
print(train_df)
print("\nValidation data saved to validation_politeness_data.csv")
print(validation_df)

class PolitenessDataset:
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sentence = str(self.df.iloc[idx]['sentence'])
        label = int(self.df.iloc[idx]['politeness'])
        
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_politeness_classifier():
    # Use a small, efficient model like DistilBERT
    model_name = "distilbert-base-uncased"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2  # binary classification: polite/impolite
    )
    
    # Create training dataset
    train_texts = [str(row['sentence']) for _, row in train_df.iterrows()]
    train_labels = [int(row['politeness']) for _, row in train_df.iterrows()]
    
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    
    # Create validation dataset
    val_texts = [str(row['sentence']) for _, row in validation_df.iterrows()]
    val_labels = [int(row['politeness']) for _, row in validation_df.iterrows()]
    
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    
    eval_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./politeness_model',
        num_train_epochs=5,  # Increased epochs since we now have proper validation
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model('./politeness_model_final')
    tokenizer.save_pretrained('./politeness_model_final')
    
    print("Training completed! Model saved to './politeness_model_final'")
    
    return trainer

def evaluate_on_validation_set():
    """Comprehensive evaluation on the validation set with detailed metrics"""
    model_path = './politeness_model_final'
    
    # Load the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load validation data
    val_sentences = validation_df['sentence'].tolist()
    val_labels = validation_df['politeness'].tolist()
    
    predictions = []
    probabilities = []
    
    print("\nDetailed Validation Set Evaluation:")
    print("=" * 80)
    
    for i, sentence in enumerate(val_sentences):
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()
        
        predictions.append(predicted_class)
        probabilities.append(probs[0].tolist())
        
        true_label = "Polite" if val_labels[i] == 1 else "Impolite"
        pred_label = "Polite" if predicted_class == 1 else "Impolite"
        correct = "✓" if predicted_class == val_labels[i] else "✗"
        
        print(f"Sample {i+1}: {correct}")
        print(f"  Sentence: '{sentence}'")
        print(f"  True Label: {true_label}")
        print(f"  Predicted: {pred_label} (Confidence: {confidence:.3f})")
        print(f"  Prob[Impolite]: {probs[0][0]:.3f}, Prob[Polite]: {probs[0][1]:.3f}")
        print()
    
    # Calculate comprehensive metrics
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        confusion_matrix, classification_report
    )
    
    accuracy = accuracy_score(val_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        val_labels, predictions, average=None, labels=[0, 1]
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        val_labels, predictions, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        val_labels, predictions, average='weighted'
    )
    
    cm = confusion_matrix(val_labels, predictions)
    
    print("=" * 80)
    print("VALIDATION SET METRICS SUMMARY")
    print("=" * 80)
    print(f"Overall Accuracy: {accuracy:.3f} ({int(accuracy*len(val_labels))}/{len(val_labels)} correct)")
    print()
    
    print("Per-Class Metrics:")
    print(f"  Impolite (Class 0):")
    print(f"    Precision: {precision[0]:.3f}")
    print(f"    Recall: {recall[0]:.3f}")
    print(f"    F1-Score: {f1[0]:.3f}")
    print(f"    Support: {support[0]}")
    print()
    print(f"  Polite (Class 1):")
    print(f"    Precision: {precision[1]:.3f}")
    print(f"    Recall: {recall[1]:.3f}")
    print(f"    F1-Score: {f1[1]:.3f}")
    print(f"    Support: {support[1]}")
    print()
    
    print("Averaged Metrics:")
    print(f"  Macro Average:")
    print(f"    Precision: {precision_macro:.3f}")
    print(f"    Recall: {recall_macro:.3f}")
    print(f"    F1-Score: {f1_macro:.3f}")
    print()
    print(f"  Weighted Average:")
    print(f"    Precision: {precision_weighted:.3f}")
    print(f"    Recall: {recall_weighted:.3f}")
    print(f"    F1-Score: {f1_weighted:.3f}")
    print()
    
    print("Confusion Matrix:")
    print("                 Predicted")
    print("               Imp.  Pol.")
    print(f"True Impolite   {cm[0][0]:3d}   {cm[0][1]:3d}")
    print(f"True Polite     {cm[1][0]:3d}   {cm[1][1]:3d}")
    print()
    
    # Error analysis
    errors = []
    for i, (true, pred) in enumerate(zip(val_labels, predictions)):
        if true != pred:
            errors.append({
                'sentence': val_sentences[i],
                'true_label': 'Polite' if true == 1 else 'Impolite',
                'pred_label': 'Polite' if pred == 1 else 'Impolite',
                'confidence': max(probabilities[i])
            })
    
    if errors:
        print("ERROR ANALYSIS:")
        print("-" * 50)
        for i, error in enumerate(errors):
            print(f"Error {i+1}:")
            print(f"  Sentence: '{error['sentence']}'")
            print(f"  True: {error['true_label']}, Predicted: {error['pred_label']}")
            print(f"  Confidence: {error['confidence']:.3f}")
            print()
    else:
        print("🎉 Perfect accuracy on validation set! No errors to analyze.")
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'predictions': predictions,
        'probabilities': probabilities
    }

if __name__ == "__main__":
    # Create and train the model
    trainer = train_politeness_classifier()
    
    # Test the model
    test_model()
