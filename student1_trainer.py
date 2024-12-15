import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from config import Config, ReasoningExample, ConsistencyProcessor
from typing import Dict, List
import evaluate
import numpy as np
import json
from huggingface_hub import login
import wandb
wandb.init(mode="offline")

class Student1Dataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[ReasoningExample], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx) -> Dict:
        ex = self.examples[idx]
        
        # Format for reasoning generation
        prompt = f"Question: {ex.question}\nGenerate step-by-step reasoning:"
        target = f"\n{ex.reasoning}"
        
        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        
        # Combine for causal LM
        input_ids = prompt_ids + target_ids
        
        # Create labels
        labels = [-100] * len(prompt_ids) + target_ids
        
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

def load_data(file_path: str) -> List[ReasoningExample]:
    with open(file_path) as f:
        data = [json.loads(line) for line in f]
    examples = [ReasoningExample(item) for item in data]
    
    # # Apply consistency filtering
    # processor = ConsistencyProcessor()
    # filtered = processor.filter_dataset(examples)
    # return filtered

    return examples


def train_student1(config: Config):
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.student1_model)
    model = AutoModelForCausalLM.from_pretrained(config.student1_model)

    # Add LoRA to the model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=8,                          # LoRA rank
        lora_alpha=32,                # Scaling factor
        lora_dropout=0.1,             # Dropout probability
        inference_mode=False          # Training mode
    )
    model = get_peft_model(model, lora_config)
    
    # Load and process data
    train_examples = load_data(config.train_file)
    val_examples = load_data(config.val_file)
    
    # Create datasets
    train_dataset = Student1Dataset(train_examples, tokenizer)
    val_dataset = Student1Dataset(val_examples, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir_student1,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        fp16=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        # compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    # Train and save
    trainer.train()
    trainer.save_model(config.output_dir_student1)

    # Evaluate
    eval_results = trainer.evaluate()
    print("Evaluation Results:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")
    
    return trainer

if __name__ == "__main__":
    config = Config()

    # Authenticate with the Hugging Face Hub
    token = config.token
    login(token=token)
    print("Logged in successfully!")

    # Train model
    trainer = train_student1(config)