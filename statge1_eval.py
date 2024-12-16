import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import numpy as np
import re
from tqdm import tqdm

BASE_MODEL_DIR = "google/gemma-2b"  # Base model name from HuggingFace
LORA_ADAPTER_DIR = "./gemma-math-reasoning-lora-final"

print("Loading the base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    trust_remote_code=True,
    device_map="auto"
)

# Load LoRA adapter and merge it into the base model
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
model = model.merge_and_unload()  # Merge LoRA weights into the base model

# Set to evaluation mode
model = torch.compile(model) 
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)



def extract_answer(output_text):
    """
    Extracts the first integer that appears after 'The answer is' in the model output.
    """
    match = re.search(r"The answer is\s+(\d+)", output_text)
    if match:
        return int(match.group(1))  # Extract and convert to integer
    return None

def extract_last_number(answer_text):
    """
    Extracts the last numerical value from the 'answer' text field in the dataset.
    """
    numbers = re.findall(r'\d+', answer_text)
    if numbers:
        return int(numbers[-1])  # Extract the last number as an integer
    return None


def generate_prediction(model, tokenizer, question, max_length=1024, temperature=0.7):
    """
    Generates a step-by-step reasoning and extracts the final cleaned answer.
    """
    input_text = f"Question: {question}\nLet's solve this step by step:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer
    extracted_answer = extract_answer(output_text)
    
    return output_text, extracted_answer


dataset_path = "datasets/test.jsonl"  # Path to test data
output_path = "gsm8k_evaluation.json"
results = []  # To store the results 
errors = []   # To store differences for analysis

with open(dataset_path, "r") as file:
    lines = file.readlines()[:500]
    
for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Evaluating GSM8K"):
    data = json.loads(line)
    question = data['question']
    answer_text = data['answer']
    
    # Extract correct answer as the last number in 'answer' field
    correct_answer = extract_last_number(answer_text)
    # Generate reasoning and extract predicted answer
    output_text, predict_answer = generate_prediction(model, tokenizer, question)
    
    # Extract reasoning after "Let's solve this step by step:"
    reasoning = output_text.split("Let's solve this step by step:")[-1].strip()
    
    # Compute the absolute difference (error)
    if correct_answer is not None and predict_answer is not None:
        if correct_answer == 0:
            error = abs(correct_answer - predict_answer)
        else:
            error = abs(correct_answer - predict_answer) / abs(correct_answer)
        errors.append(error)
    else:
        error = None
        errors.append(error)

        
    # Store results
    result = {
        "question": question,
        "reasoning": reasoning, 
        "correct_answer": correct_answer,
        "predict_answer": predict_answer,
        "error": error
    }
    results.append(result)
    

# Calculate Metrics
valid_results = [r for r in results if r["error"] is not None]
accuracy = sum(1 for r in valid_results if r["correct_answer"] == r["predict_answer"]) / len(valid_results) * 100
average_error = np.mean([r["error"] for r in valid_results])

# Save Results to JSON
evaluation = {
    "accuracy": accuracy,
    "average_error": average_error,
    "details": results
}

with open(output_path, "w") as json_file:
    json.dump(evaluation, json_file, indent=4)
