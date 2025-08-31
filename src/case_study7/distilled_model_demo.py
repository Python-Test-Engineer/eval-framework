#!/usr/bin/env python3
"""
Model Distillation Pipeline with Fine-tuning, Pruning, and DPO
Demonstrates creating a distilled model from an open source model using:
- Knowledge Distillation
- Fine-tuning
- Pruning 
- Direct Preference Optimization (DPO)
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
import numpy as np
from typing import Dict, List, Tuple
import os
import json

# Create demo DPO dataset
def create_demo_dpo_dataset():
    """Create a small demo dataset for DPO training"""
    data = [
        {"prompt": "What is Python?", "chosen": "Python is a high-level programming language known for its simplicity and readability.", "rejected": "Python is a snake."},
        {"prompt": "Explain machine learning", "chosen": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.", "rejected": "Machine learning is when machines become smart and take over."},
        {"prompt": "How do you create a list in Python?", "chosen": "You can create a list in Python using square brackets: my_list = [1, 2, 3, 4]", "rejected": "Lists are created with curly braces: my_list = {1, 2, 3, 4}"},
        {"prompt": "What is data science?", "chosen": "Data science combines statistics, programming, and domain expertise to extract insights from data.", "rejected": "Data science is just making charts and graphs."},
        {"prompt": "Define neural networks", "chosen": "Neural networks are computing systems inspired by biological neural networks, used for pattern recognition.", "rejected": "Neural networks are networks of neurons in your brain."},
        {"prompt": "What is supervised learning?", "chosen": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.", "rejected": "Supervised learning is when a teacher watches you learn."},
        {"prompt": "Explain deep learning", "chosen": "Deep learning uses neural networks with multiple layers to learn complex patterns in data.", "rejected": "Deep learning is learning things very deeply and thoroughly."},
        {"prompt": "What is a DataFrame?", "chosen": "A DataFrame is a 2D labeled data structure in pandas, similar to a table or spreadsheet.", "rejected": "A DataFrame is a frame that holds data physically."},
        {"prompt": "How do you import libraries in Python?", "chosen": "You import libraries using the 'import' keyword: import pandas as pd", "rejected": "You download libraries from the internet manually."},
        {"prompt": "What is API?", "chosen": "API (Application Programming Interface) defines how software components communicate with each other.", "rejected": "API is a type of computer programming language."}
    ]
    
    df = pd.DataFrame(data)
    df.to_csv('demo_dpo_data.csv', index=False)
    return df

class DPODataset(Dataset):
    """Dataset class for Direct Preference Optimization"""
    def __init__(self, data: pd.DataFrame, tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenize prompt, chosen, and rejected responses
        prompt_tokens = self.tokenizer(row['prompt'], truncation=True, max_length=self.max_length//3)
        chosen_tokens = self.tokenizer(row['chosen'], truncation=True, max_length=self.max_length//3)
        rejected_tokens = self.tokenizer(row['rejected'], truncation=True, max_length=self.max_length//3)
        
        return {
            'prompt_input_ids': prompt_tokens['input_ids'],
            'chosen_input_ids': chosen_tokens['input_ids'],
            'rejected_input_ids': rejected_tokens['input_ids'],
            'prompt': row['prompt'],
            'chosen': row['chosen'],
            'rejected': row['rejected']
        }

class DistillationLoss(nn.Module):
    """Custom loss for knowledge distillation"""
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Distillation loss
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        distill_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
        
        # Student loss
        student_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        return self.alpha * distill_loss + (1 - self.alpha) * student_loss

class ModelPruner:
    """Simple magnitude-based pruning for model compression"""
    @staticmethod
    def prune_model(model, pruning_ratio: float = 0.2):
        """Prune model weights by magnitude"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                # Calculate threshold for pruning
                threshold = torch.quantile(torch.abs(weight), pruning_ratio)
                # Create mask
                mask = torch.abs(weight) > threshold
                # Apply pruning
                module.weight.data *= mask.float()
        return model

class DPOTrainer:
    """Direct Preference Optimization trainer"""
    def __init__(self, model, tokenizer, beta: float = 0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
    
    def compute_dpo_loss(self, chosen_logits, rejected_logits, chosen_labels, rejected_labels):
        """Compute DPO loss"""
        # Log probabilities for chosen and rejected responses
        chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
        
        # Gather log probabilities for actual tokens
        chosen_log_prob = chosen_log_probs.gather(-1, chosen_labels.unsqueeze(-1)).squeeze(-1)
        rejected_log_prob = rejected_log_probs.gather(-1, rejected_labels.unsqueeze(-1)).squeeze(-1)
        
        # DPO loss
        log_odds = chosen_log_prob - rejected_log_prob
        loss = -F.logsigmoid(self.beta * log_odds).mean()
        
        return loss

def create_small_model_config(base_model_name: str):
    """Create a smaller model configuration for distillation"""
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(base_model_name)
    
    # Reduce model size for demonstration
    if hasattr(config, 'n_layer'):
        config.n_layer = min(4, config.n_layer)  # Reduce layers
    elif hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = min(4, config.num_hidden_layers)
    
    if hasattr(config, 'n_embd'):
        config.n_embd = min(512, config.n_embd)  # Reduce embedding size
    elif hasattr(config, 'hidden_size'):
        config.hidden_size = min(512, config.hidden_size)
    
    return config

def main():
    """Main distillation pipeline"""
    print("🚀 Starting Model Distillation Pipeline...")
    
    # Configuration
    TEACHER_MODEL = "microsoft/DialoGPT-small"  # Small model for demo
    STUDENT_MODEL = "student_model"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"📱 Using device: {DEVICE}")
    
    # Step 1: Create demo DPO dataset
    print("📊 Creating demo DPO dataset...")
    dpo_data = create_demo_dpo_dataset()
    print(f"✅ Created dataset with {len(dpo_data)} examples")
    
    # Step 2: Load teacher model and tokenizer
    print("🎓 Loading teacher model...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL)
    teacher_model.to(DEVICE)
    teacher_model.eval()
    
    # Step 3: Create smaller student model
    print("🎯 Creating student model...")
    student_config = create_small_model_config(TEACHER_MODEL)
    student_model = AutoModelForCausalLM.from_config(student_config)
    student_model.to(DEVICE)
    
    print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # Step 4: Knowledge Distillation
    print("🔬 Starting knowledge distillation...")
    distill_loss = DistillationLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=5e-5)
    
    # Simple distillation training loop
    student_model.train()
    for epoch in range(2):  # Small number for demo
        total_loss = 0
        for i, row in dpo_data.iterrows():
            # Prepare input
            text = row['prompt'] + " " + row['chosen']
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            student_outputs = student_model(**inputs)
            student_logits = student_outputs.logits
            
            # Compute distillation loss
            labels = inputs['input_ids'][:, 1:]  # Shift for causal LM
            teacher_logits = teacher_logits[:, :-1]
            student_logits = student_logits[:, :-1]
            
            loss = distill_loss(student_logits.reshape(-1, student_logits.size(-1)), 
                              teacher_logits.reshape(-1, teacher_logits.size(-1)), 
                              labels.reshape(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(dpo_data):.4f}")
    
    # Step 5: Pruning
    print("✂️ Applying model pruning...")
    pruner = ModelPruner()
    student_model = pruner.prune_model(student_model, pruning_ratio=0.2)
    
    # Step 6: Direct Preference Optimization
    print("🎯 Applying Direct Preference Optimization...")
    dpo_trainer = DPOTrainer(student_model, tokenizer)
    
    student_model.train()
    dpo_optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)
    
    for epoch in range(1):  # Single epoch for demo
        total_dpo_loss = 0
        for i, row in dpo_data.iterrows():
            # Prepare chosen and rejected responses
            chosen_text = row['prompt'] + " " + row['chosen']
            rejected_text = row['prompt'] + " " + row['rejected']
            
            chosen_inputs = tokenizer(chosen_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            rejected_inputs = tokenizer(rejected_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            chosen_inputs = {k: v.to(DEVICE) for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(DEVICE) for k, v in rejected_inputs.items()}
            
            # Forward pass
            chosen_outputs = student_model(**chosen_inputs)
            rejected_outputs = student_model(**rejected_inputs)
            
            # Compute DPO loss (simplified)
            chosen_loss = F.cross_entropy(chosen_outputs.logits[:, :-1].reshape(-1, chosen_outputs.logits.size(-1)), 
                                        chosen_inputs['input_ids'][:, 1:].reshape(-1), reduction='mean')
            rejected_loss = F.cross_entropy(rejected_outputs.logits[:, :-1].reshape(-1, rejected_outputs.logits.size(-1)), 
                                          rejected_inputs['input_ids'][:, 1:].reshape(-1), reduction='mean')
            
            # DPO objective: maximize difference between chosen and rejected
            dpo_loss = rejected_loss - chosen_loss
            
            # Backward pass
            dpo_optimizer.zero_grad()
            (-dpo_loss).backward()  # Negative because we want to maximize
            dpo_optimizer.step()
            
            total_dpo_loss += dpo_loss.item()
        
        print(f"DPO Epoch {epoch+1}, Average Loss: {total_dpo_loss/len(dpo_data):.4f}")
    
    # Step 7: Save the distilled model
    print("💾 Saving distilled model...")
    os.makedirs("distilled_model", exist_ok=True)
    student_model.save_pretrained("distilled_model")
    tokenizer.save_pretrained("distilled_model")
    
    # Step 8: Model evaluation/testing
    print("🧪 Testing distilled model...")
    student_model.eval()
    
    test_prompts = [
        "What is Python?",
        "Explain machine learning",
        "How do you create a list in Python?"
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = student_model.generate(
                **inputs, 
                max_length=inputs['input_ids'].shape[1] + 50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Response: {response[len(prompt):].strip()}")
        print("-" * 50)
    
    print("✅ Model distillation pipeline completed!")
    print(f"📁 Distilled model saved to: distilled_model/")
    print(f"📊 DPO dataset saved to: demo_dpo_data.csv")
    
    # Model statistics
    original_params = sum(p.numel() for p in teacher_model.parameters())
    distilled_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = original_params / distilled_params
    
    print(f"\n📈 Model Statistics:")
    print(f"Original model parameters: {original_params:,}")
    print(f"Distilled model parameters: {distilled_params:,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")

if __name__ == "__main__":
    # Install required packages if not already installed
    required_packages = [
        "torch", "transformers", "pandas", "numpy", "datasets"
    ]
    
    print("📦 Checking required packages...")
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"⚠️  Please install {package}: pip install {package}")
            exit(1)
    
    main()
