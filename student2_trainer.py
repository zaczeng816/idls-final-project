import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq
)
from config import Config
from typing import Dict, List
import json
from tqdm import tqdm
import wandb
from datetime import datetime

class Student2Dataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[Dict], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        
        # Process all examples with tqdm
        self.processed_examples = []
        for ex in tqdm(examples, desc="Processing examples"):
            prompt = f"Question: {ex['question']}\n\nReasoning:\n{ex['reasoning']}\n\nRate this reasoning on a scale of 1-5 based on:\n- Clarity of explanation\n- Mathematical accuracy\n- Step-by-step breakdown\n- Logical flow\n\nScore:"
            target = f" {ex['reasoning_score']}"
            
            # Tokenize
            prompt_ids = self.tokenizer.encode(prompt)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            
            # Combine
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            
            self.processed_examples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels
            })
        
    def __len__(self):
        return len(self.processed_examples)
    
    def __getitem__(self, idx) -> Dict:
        return self.processed_examples[idx]

def load_data(file_path: str) -> List[Dict]:
    """Load data from JSON file"""
    print(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['pairs'])} examples")
    return data['pairs']

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        wandb.log({"train_loss": loss.item()})
        return (loss, outputs) if return_outputs else loss

def train_student2(config: Config):
    # Initialize wandb
    run_name = f"student2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="math-reasoning-scorer",
        name=run_name,
        config={
            "model": config.student2_model,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_train_epochs
        }
    )
    
    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.student2_model)
    model = AutoModelForCausalLM.from_pretrained(config.student2_model)
    model.config.use_cache = False
    
    # Load and process data
    train_examples = load_data(config.train_file)
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = Student2Dataset(train_examples, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir_student2,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        save_steps=500,
        logging_steps=100,
        report_to="wandb",
        fp16=True
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    # Train and save
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving model...")
    trainer.save_model(config.output_dir_student2)
    
    return trainer

if __name__ == "__main__":
    config = Config()
    config.train_file = "data/gsm8k_reasoning_labels.json"
    trainer = train_student2(config)