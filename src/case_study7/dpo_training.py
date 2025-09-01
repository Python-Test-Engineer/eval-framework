import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig
from sklearn.model_selection import train_test_split

def load_csv_data(csv_path="dpo.csv"):
    """Load preference data from CSV file with columns: id, prompt, chosen, rejected"""
    print(f"Loading data from {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = ['id', 'prompt', 'chosen', 'rejected']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with any NaN values in required columns
        df = df.dropna(subset=['prompt', 'chosen', 'rejected'])
        
        print(f"Loaded {len(df)} samples from CSV")
        print(f"Sample data preview:")
        print(df[['prompt', 'chosen', 'rejected']].head(2))
        
        return df
        
    except FileNotFoundError:
        print(f"CSV file '{csv_path}' not found. Creating sample CSV file...")
        create_sample_csv(csv_path)
        return pd.read_csv(csv_path)
    
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

def create_sample_csv(csv_path="dpo.csv"):
    """Create a sample CSV file with the expected format"""
    sample_data = [
        {
            "id": 1,
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris, a beautiful city known for its art, culture, and history.",
            "rejected": "France's capital is Paris, I think."
        },
        {
            "id": 2,
            "prompt": "How do you make a good cup of coffee?",
            "chosen": "To make a good cup of coffee: 1) Use fresh, quality beans 2) Grind them just before brewing 3) Use the right water temperature (195-205°F) 4) Maintain proper coffee-to-water ratio (1:15-1:17) 5) Brew for the appropriate time based on your method.",
            "rejected": "Just put some coffee in hot water and stir."
        },
        {
            "id": 3,
            "prompt": "Explain photosynthesis briefly.",
            "chosen": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This occurs primarily in chloroplasts using chlorophyll, providing energy for the plant and oxygen for other organisms.",
            "rejected": "Plants use sun to make food somehow."
        },
        {
            "id": 4,
            "prompt": "What's the best way to learn programming?",
            "chosen": "The best way to learn programming involves: 1) Start with fundamentals and a beginner-friendly language 2) Practice coding regularly with small projects 3) Read others' code and contribute to open source 4) Build increasingly complex projects 5) Join programming communities for support and feedback.",
            "rejected": "Just watch some YouTube videos and you'll figure it out."
        },
        {
            "id": 5,
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of artificial intelligence where algorithms learn patterns from data to make predictions or decisions without being explicitly programmed for each specific task.",
            "rejected": "Machine learning is when computers learn stuff automatically."
        }
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_path, index=False)
    print(f"Created sample CSV file: {csv_path}")

def prepare_dataset_from_df(df, test_size=0.2):
    """Convert DataFrame to Hugging Face Dataset format with train/test split"""
    
    # Convert to format expected by DPOTrainer
    data = []
    for _, row in df.iterrows():
        data.append({
            "prompt": str(row["prompt"]).strip(),
            "chosen": str(row["chosen"]).strip(),
            "rejected": str(row["rejected"]).strip()
        })
    
    # Split into train and validation sets
    if len(data) > 1 and test_size > 0:
        train_data, eval_data = train_test_split(data, test_size=test_size, random_state=42)
        print(f"Split data: {len(train_data)} training, {len(eval_data)} evaluation samples")
        return Dataset.from_list(train_data), Dataset.from_list(eval_data)
    else:
        print(f"Using all {len(data)} samples for training (no validation split)")
        return Dataset.from_list(data), None

def main():
    # Model and tokenizer setup
    model_name = "microsoft/DialoGPT-small"  # Using a smaller model for example
    # For production, consider: "gpt2", "microsoft/DialoGPT-medium", or other suitable base models
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset from CSV
    print("Loading and preparing dataset...")
    df = load_csv_data("dpo.csv")  # You can change the filename here
    train_dataset, eval_dataset = prepare_dataset_from_df(df, test_size=0.2)
    
    # DPO Configuration
    training_args = DPOConfig(
        output_dir="./dpo_results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        remove_unused_columns=False,
        # DPO specific parameters
        beta=0.1,  # Controls strength of preference optimization
        max_length=512,
        max_prompt_length=256,
    )
    
    # Initialize DPO Trainer
    print("Initializing DPO Trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset else train_dataset,
    )
    
    # Train the model
    print("Starting DPO training...")
    dpo_trainer.train()
    
    # Save the trained model
    print("Saving trained model...")
    dpo_trainer.save_model("./dpo_model")
    tokenizer.save_pretrained("./dpo_model")
    
    print("DPO training completed!")

def test_model():
    """Test the trained model with a sample prompt"""
    print("\nTesting trained model...")
    
    # Load the trained model
    tokenizer = AutoTokenizer.from_pretrained("./dpo_model")
    model = AutoModelForCausalLM.from_pretrained("./dpo_model")
    
    # Test prompt
    prompt = "What is machine learning?"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

# Alternative: Custom CSV processing options
def validate_csv_data(df):
    """Validate and clean the CSV data"""
    print("Validating CSV data...")
    
    # Check for empty strings
    df = df[df['prompt'].str.strip() != '']
    df = df[df['chosen'].str.strip() != '']
    df = df[df['rejected'].str.strip() != '']
    
    # Check for duplicate prompts
    duplicate_prompts = df.duplicated(subset=['prompt'], keep=False)
    if duplicate_prompts.any():
        print(f"Warning: Found {duplicate_prompts.sum()} duplicate prompts")
        df = df.drop_duplicates(subset=['prompt'], keep='first')
    
    # Check response quality (basic length check)
    df = df[df['chosen'].str.len() > 10]  # Chosen responses should be substantial
    df = df[df['rejected'].str.len() > 5]   # Rejected can be shorter but not empty
    
    print(f"After validation: {len(df)} samples remaining")
    return df

def analyze_dataset(df):
    """Print dataset statistics"""
    print("\n=== Dataset Analysis ===")
    print(f"Total samples: {len(df)}")
    print(f"Average prompt length: {df['prompt'].str.len().mean():.1f} characters")
    print(f"Average chosen response length: {df['chosen'].str.len().mean():.1f} characters")
    print(f"Average rejected response length: {df['rejected'].str.len().mean():.1f} characters")
    print(f"Unique prompts: {df['prompt'].nunique()}")
    
    # Show length distribution
    print(f"\nResponse length comparison:")
    print(f"Chosen longer than rejected: {(df['chosen'].str.len() > df['rejected'].str.len()).sum()}")
    print(f"Rejected longer than chosen: {(df['rejected'].str.len() > df['chosen'].str.len()).sum()}")
    print("="*30)

if __name__ == "__main__":
    # Install required packages first:
    # pip install torch transformers datasets trl accelerate pandas scikit-learn
    
    print("Starting DPO training with CSV data...")
    print("Required packages: pip install torch transformers datasets trl accelerate pandas scikit-learn")
    
    # Run training
    main()
    
    # Test the model
    test_model()
    
    print("\nTraining complete! Check './dpo_model' directory for saved model.")
    print("\nCSV Format Expected:")
    print("- Filename: dpo.csv")
    print("- Columns: id, prompt, chosen, rejected")
    print("- The script will create a sample CSV if none exists")
