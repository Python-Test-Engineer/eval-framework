import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re
import warnings
warnings.filterwarnings('ignore')

class SelfHarmDetectionAgent:
    def __init__(self, max_words=10000, max_length=100, embedding_dim=100):
        """
        Simple CNN-BiLSTM agent for detecting self-harm intention in text
        
        Args:
            max_words: Maximum number of words in vocabulary
            max_length: Maximum sequence length
            embedding_dim: Embedding dimension
        """
        self.max_words = max_words
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def prepare_data(self, texts, labels=None):
        """Prepare text data for training/prediction"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # If tokenizer doesn't exist, create and fit it
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(processed_texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post')
        
        return X, labels
    
    def build_model(self):
        """Build CNN-BiLSTM model"""
        model = Sequential([
            # Embedding layer
            Embedding(input_dim=self.max_words, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_length),
            
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            GlobalMaxPooling1D(),
            
            # Reshape for LSTM
            tf.keras.layers.RepeatVector(1),
            
            # BiLSTM layer
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.3),
            
            # Output layer (binary classification)
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        return model
    
    def train(self, texts, labels, validation_split=0.2, epochs=10, batch_size=32):
        """
        Train the model
        
        Args:
            texts: List of text messages
            labels: List of labels (1 for self-harm, 0 for normal)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Prepare data
        X, y = self.prepare_data(texts, labels)
        y = np.array(y)
        
        # Build model if not exists
        if self.model is None:
            self.build_model()
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def predict(self, texts):
        """
        Predict self-harm intention for given texts
        
        Args:
            texts: List of text messages or single text string
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Prepare data
        X, _ = self.prepare_data(texts)
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        results = []
        for i, text in enumerate(texts):
            prob = float(predictions[i][0])
            risk_level = self._get_risk_level(prob)
            
            results.append({
                'text': text,
                'self_harm_probability': prob,
                'predicted_class': 1 if prob > 0.5 else 0,
                'risk_level': risk_level,
                'confidence': max(prob, 1 - prob)
            })
        
        return results[0] if len(results) == 1 else results
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "LOW"
        elif probability < 0.7:
            return "MODERATE"
        else:
            return "HIGH"

# Example usage and demonstration
def create_sample_data():
    """Create expanded training data with more explicit examples"""
    normal_texts = [
        "I'm having a great day today!",
        "Just finished my homework, feeling accomplished",
        "Going to meet friends later for dinner",
        "I love spending time with my family on weekends",
        "Work was productive today, got a lot done",
        "Looking forward to the weekend getaway",
        "Just watched a funny movie with my partner",
        "I'm grateful for my health and happiness",
        "Having dinner with my partner tonight at our favorite restaurant",
        "Excited about my new project at work",
        "My therapy is helping me cope better with stress",
        "Finally got promoted after months of hard work",
        "Going on vacation next week to the beach",
        "Started a new hobby learning guitar",
        "Had an amazing date last night",
        "Volunteered at the shelter and felt great",
        "My family is so supportive of my goals",
        "Feeling optimistic about my future career",
        "Made some new friends at the gym",
        "Celebrated my birthday with loved ones",
        "Got accepted into my dream university",
        "My relationship is getting stronger every day",
        "Enjoying this beautiful sunny weather",
        "Learned something new and interesting today",
        "Helped a neighbor and felt good about it"
    ]
    
    concerning_texts = [
        "I can't take this anymore, everything hurts",
        "Everything feels hopeless and dark",
        "I don't want to be here anymore at all",
        "Nobody would miss me if I was gone forever",
        "I'm thinking about ending it all tonight",
        "Life isn't worth living like this anymore",
        "I want to hurt myself badly",
        "I can't go on like this much longer",
        "Maybe everyone would be better off without me here",
        "I feel like giving up completely on everything",
        "I wrote a suicide note last night",
        "I purchased pills to overdose with",
        "I've been researching ways to kill myself",
        "Planning to jump off a bridge tomorrow",
        "I have a rope ready in my closet",
        "Going to end it all after school today",
        "Cut myself again last night, felt better",
        "Nobody cares if I live or die",
        "I'm worthless and should just disappear",
        "Tired of being a burden to everyone",
        "Death seems like the only escape now",
        "I want to slit my wrists tonight",
        "Thinking about driving into oncoming traffic",
        "I have access to my dad's gun",
        "Planning my suicide for this weekend"
    ]
    
    # Combine and create labels
    texts = normal_texts + concerning_texts
    labels = [0] * len(normal_texts) + [1] * len(concerning_texts)
    
    return texts, labels

def main():
    """Demonstration of the self-harm detection agent"""
    print("🔍 Self-Harm Detection Agent Demo")
    print("=" * 40)
    
    # Create agent with adjusted parameters for better performance
    agent = SelfHarmDetectionAgent(max_words=5000, max_length=50)
    
    # Create expanded sample data
    texts, labels = create_sample_data()
    
    print(f"Training on {len(texts)} samples...")
    
    # Train with improved parameters
    history = agent.train(texts, labels, epochs=20, batch_size=8, validation_split=0.15)
    
    print("✅ Training completed!")
    
    # Check training progress
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.3f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
    
    # Quick verification on training examples
    print("\n🔧 Model verification:")
    train_test_high = agent.predict("I want to kill myself tonight")
    train_test_low = agent.predict("I'm having a wonderful day")
    print(f"High-risk test: {train_test_high['self_harm_probability']:.3f}")
    print(f"Low-risk test: {train_test_low['self_harm_probability']:.3f}")
    
    # Test messages - more elaborate and varied length (10-75 words)
    test_messages = [
        "I'm feeling really happy today! Just got some amazing news about my job application and I can't wait to share it with my family. Life has been treating me well lately and I'm grateful for all the opportunities coming my way.",
        
        "I can't handle this overwhelming pain anymore. Every day feels like a struggle and I'm drowning in my own thoughts. The darkness seems to be consuming me and I don't know how much longer I can keep pretending that everything is okay.",
        
        "Just got promoted at work after months of hard work and dedication! My manager said I've been exceeding expectations and they're excited to see what I can accomplish in this new role. Celebrating with dinner tonight.",
        
        "Nobody seems to understand what I'm going through right now. I feel completely isolated and alone, like I'm trapped in my own mind with no way out. Sometimes I wonder if things would be easier if I just wasn't here anymore.",
        
        "Going on vacation next week to visit my sister in Colorado! We're planning to go hiking in the mountains and I've been looking forward to this trip for months. It's going to be such a refreshing break from work stress.",
        
        "I've been having these terrible thoughts lately about ending everything. The weight of my problems feels unbearable and I'm scared of what I might do to myself. I don't want to burden anyone but I'm really struggling and losing hope.",
        
        "My therapy session went really well today. We talked about coping strategies and I'm starting to feel more hopeful about managing my anxiety. It's nice to have someone who listens without judgment and helps me work through difficult emotions.",
        
        "Everything in my life is falling apart and I don't see any point in continuing like this. My relationships are broken, my career is stagnant, and I feel like a complete failure. Maybe everyone would be better off if I just disappeared forever.",
        
        "Spent the weekend with my best friends and laughed until my stomach hurt. We went to this new restaurant downtown and then watched movies all night. These are the moments that remind me how lucky I am to have wonderful people.",
        
        "I keep thinking about hurting myself as a way to deal with all this emotional pain that won't go away. Physical pain seems like it would be easier to handle than this constant mental anguish. I don't know what else to do anymore.",
        
        "Finally finished my college degree after years of hard work! My parents are so proud and I'm excited to start my career in engineering. The graduation ceremony is next weekend and my whole family is coming to celebrate with me.",
        
        "The bullying at school has gotten so bad that I dread waking up every morning. Kids make fun of everything about me and I have no friends to sit with at lunch. I've been thinking it might be better if I just ended it all.",
        
        "Started a new hobby learning to play guitar and I'm really enjoying it. Music has always been therapeutic for me and now I'm creating my own. My instructor says I have natural talent and might consider performing someday.",
        
        "I wrote a suicide note last night but then deleted it this morning. I keep going back and forth between wanting to die and being scared of actually going through with it. The pain feels endless and I don't see any other way out.",
        
        "My grandmother's health is improving after her surgery last month. The doctors say she's recovering well and should be back home soon. Our whole family has been taking turns visiting her and she's in good spirits despite everything.",
        
        "I purchased some pills online and I've been staring at them for hours, thinking about taking them all at once. Part of me wants to call someone for help but another part thinks this might finally bring me the peace I've been searching for.",
        
        "Had an amazing date last night with someone I met through mutual friends. We talked for hours about our shared interests and there's definitely chemistry there. Looking forward to seeing where this relationship might lead in the future.",
        
        "My depression has reached a point where I can barely get out of bed anymore. I've been researching different methods and locations, making plans for when I finally work up the courage to go through with ending my life once and for all.",
        
        "Volunteered at the animal shelter again this weekend and it always fills my heart with joy. The dogs and cats there are so grateful for attention and love. I'm thinking about adopting a puppy soon to keep me company at home.",
        
        "I feel like such a burden to everyone around me and I'm tired of pretending to be okay when I'm not. My family would probably be relieved if they didn't have to worry about me anymore. I've been planning how to do it without causing them too much pain."
    ]
    
    print("\n🧪 Testing predictions:")
    print("-" * 40)
    
    for msg in test_messages:
        result = agent.predict(msg)
        print(f"Message: '{msg[:50]}...'")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Probability: {result['self_harm_probability']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 40)
    
    print("\n🧪 Testing predictions:")
    print("-" * 40)
    
    for msg in test_messages:
        result = agent.predict(msg)
        print(f"Message: '{msg}'")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Probability: {result['self_harm_probability']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
