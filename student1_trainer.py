import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from huggingface_hub import login

# Set random seed
set_seed(42)

# Login to Hugging Face
token = "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"
login(token)

class MathReasoningDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Filter valid examples
        self.data = [
            item for item in data 
            if item['predict_answer'] == item['correct_answer']
        ]
        
        print(f"Loaded {len(data)} total examples")
        print(f"Kept {len(self.data)} examples with correct predictions")
        print(f"Filtered out {len(data) - len(self.data)} incorrect examples")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = f"Question: {item['question']}\nLet's solve this step by step:"
        target_text = f"{item['reasoning']}\nThe answer is {item['predict_answer']}"
        
        full_text = f"{input_text}{target_text}"
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = encodings.input_ids.clone()
        
        return {
            "input_ids": encodings.input_ids[0],
            "attention_mask": encodings.attention_mask[0],
            "labels": labels[0]
        }

def main():
    # Model and tokenizer initialization
    model_name = "google/gemma-2b"
    print("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in 8-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=True,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,  # scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    dataset = MathReasoningDataset('datasets/flattened_gsm8k_questions.json', tokenizer)
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./gemma-math-reasoning-lora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_dir='./logs',
        logging_steps=10,
        learning_rate=2e-4,  # Higher learning rate for LoRA
        weight_decay=0.01,
        fp16=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",
        warmup_steps=100,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    model.save_pretrained("./gemma-math-reasoning-lora-final")
    tokenizer.save_pretrained("./gemma-math-reasoning-lora-final")
    
    print("Training complete!")

# Function to generate predictions
def generate_prediction(model, tokenizer, question, max_length=512):
    input_text = f"Question: {question}\nLet's solve this step by step:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")