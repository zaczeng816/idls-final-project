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
from peft import LoraConfig, get_peft_model, TaskType
import random

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
        
        # # Get model predictions
        # predictions = outputs.logits.argmax(dim=-1)
        
        # # Get actual labels (removing -100s which are used for padding)
        # labels = inputs["labels"]
        
        # # Convert predictions and labels to text for a random batch example
        # random_idx = random.randint(0, predictions.shape[0]-1)
        
        # # Get the last token (which should be the score)
        # pred_score = self.processing_class.decode(predictions[random_idx][-1:]).strip()
        
        # # Get the actual score from labels
        # valid_labels = [l for l in labels[random_idx].tolist() if l != -100]
        # true_score = self.processing_class.decode(valid_labels).strip()
        
        # Log to wandb
        wandb.log({
            "train_loss": loss.item(),
            # "predicted_score": pred_score,
            # "true_score": true_score
        })
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
    model = AutoModelForCausalLM.from_pretrained(
        config.student2_model,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False,
        return_dict=True
    )
    
    # Enable gradient checkpointing after model creation
    model.gradient_checkpointing_enable()
    
    # Add LoRA configuration with smaller parameters
    lora_config = LoraConfig(
        r=8,  # Reduced from 16 to 8
        lora_alpha=16,  # Reduced from 32 to 16
        target_modules=[
            "gate_proj",
            "up_proj", 
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Convert to LoRA model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # This will show you the reduction in trainable parameters
    model.train()  # Set the model to training mode
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)  # Ensure parameters are in float32
    
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
        per_device_train_batch_size=2,  # Reduced batch size
        gradient_accumulation_steps=8,  # Increased gradient accumulation
        save_steps=500,
        logging_steps=100,
        report_to="wandb",
        fp16=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="adamw_torch_fused",  # Use memory-efficient optimizer
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