import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import numpy as np
import re
import time
from tqdm import tqdm
from rouge import Rouge
from typing import List, Dict, Any, Optional, Tuple

BASE_MODEL_DIR = "google/gemma-2b"
LORA_ADAPTER_DIR = "./gemma-math-reasoning-lora-final"

def setup_model():
    """Initialize model and tokenizer with LoRA weights."""
    start_time = time.time()
    print("Loading the base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        trust_remote_code=True,
        device_map="auto"
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    model = model.merge_and_unload()
    model = torch.compile(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    setup_time = time.time() - start_time
    print(f"Model setup time: {setup_time:.2f} seconds")
    
    return model, tokenizer, device, setup_time

def extract_answer(output_text: str) -> Optional[int]:
    """Extracts the first integer that appears after 'The answer is'."""
    match = re.search(r"The answer is\s+(\d+)", output_text)
    if match:
        return int(match.group(1))
    return None

def extract_last_number(answer_text: str) -> Optional[int]:
    """Extracts the last numerical value from the answer text."""
    numbers = re.findall(r'\d+', answer_text)
    if numbers:
        return int(numbers[-1])
    return None

def generate_predictions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    device: str,
    num_samples: int = 1,
    max_length: int = 1024,
    temperature: float = 0.7
) -> Tuple[List[Dict[str, Any]], float]:
    """Generates multiple step-by-step reasonings and extracts answers."""
    input_text = f"Question: {question}\nLet's solve this step by step:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    results = []
    total_inference_time = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Measure inference time
            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            reasoning = output_text.split("Let's solve this step by step:")[-1].strip()
            extracted_answer = extract_answer(output_text)
            
            results.append({
                "reasoning": reasoning,
                "predicted_answer": extracted_answer,
                "inference_time": inference_time
            })
    
    return results, total_inference_time

def calculate_rouge_scores(predictions: List[Dict[str, Any]], reference: str) -> float:
    """Calculate ROUGE-L scores between predictions and reference."""
    rouge = Rouge()
    scores = []
    
    for pred in predictions:
        try:
            score = rouge.get_scores(pred["reasoning"], reference)
            scores.append(score[0]["rouge-l"]["f"])
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            continue
    
    return sum(scores) / len(scores) if scores else 0

def calculate_pass_at_k(predictions: List[Dict[str, Any]], correct_answer: int, k: int = 3) -> bool:
    """Check if any of the top k predictions match the correct answer."""
    if len(predictions) < k:
        k = len(predictions)
        
    for pred in predictions[:k]:
        if pred["predicted_answer"] == correct_answer:
            return True
    return False

def evaluate_model(
    dataset_path: str,
    num_samples: int = 1,
    use_pass_k: bool = False,
    output_path: str = "gsm8k_evaluation.json"
) -> Dict[str, Any]:
    """
    Evaluate model on GSM8K dataset with multiple metrics.
    
    Args:
        dataset_path: Path to the test dataset
        num_samples: Number of predictions per question
        use_pass_k: Whether to calculate Pass@k metric
        output_path: Where to save the results
    """
    model, tokenizer, device, setup_time = setup_model()
    results = []
    errors = []
    pass_k_results = []
    rouge_scores = []
    total_inference_time = 0
    inference_times = []
    
    with open(dataset_path, "r") as file:
        lines = file.readlines()
    
    for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Evaluating GSM8K"):
        data = json.loads(line)
        question = data['question']
        answer_text = data['answer']
        correct_answer = extract_last_number(answer_text)
        
        # Generate multiple predictions and track time
        predictions, batch_inference_time = generate_predictions(
            model, tokenizer, question, device, num_samples=num_samples
        )
        total_inference_time += batch_inference_time
        inference_times.append(batch_inference_time)
        
        # Calculate ROUGE-L score
        rouge_score = calculate_rouge_scores(predictions, answer_text)
        rouge_scores.append(rouge_score)
        
        # Calculate error for first prediction
        first_pred = predictions[0]["predicted_answer"]
        if correct_answer is not None and first_pred is not None:
            if correct_answer == 0:
                error = abs(correct_answer - first_pred)
            else:
                error = abs(correct_answer - first_pred) / abs(correct_answer)
            errors.append(error)
        
        # Calculate Pass@k if enabled
        if use_pass_k and num_samples > 1:
            passed = calculate_pass_at_k(predictions, correct_answer)
            pass_k_results.append(passed)
        
        result = {
            "question": question,
            "correct_answer": correct_answer,
            "predictions": predictions,
            "rouge_score": rouge_score,
            "inference_time": batch_inference_time
        }
        results.append(result)
    
    # Calculate metrics
    valid_predictions = [r for r in results if r["predictions"][0]["predicted_answer"] is not None]
    accuracy = sum(1 for r in valid_predictions 
                  if r["predictions"][0]["predicted_answer"] == r["correct_answer"]) / len(valid_predictions) * 100
    
    # Calculate timing metrics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    metrics = {
        "accuracy": accuracy,
        "average_error": np.mean(errors) if errors else None,
        "average_rouge_l": np.mean(rouge_scores) if rouge_scores else None,
        "model_setup_time": setup_time,
        "total_inference_time": total_inference_time,
        "average_inference_time": avg_inference_time,
        "std_inference_time": std_inference_time,
        "inference_time_per_sample": avg_inference_time / num_samples,
        "throughput": len(lines) * num_samples / total_inference_time
    }
    
    if use_pass_k and num_samples > 1:
        metrics["pass_at_k"] = sum(pass_k_results) / len(pass_k_results) * 100
    
    # Save results
    evaluation = {
        "metrics": metrics,
        "results": results
    }
    
    with open(output_path, "w") as json_file:
        json.dump(evaluation, json_file, indent=4)
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        if "time" in metric or metric == "throughput":
            if metric == "throughput":
                print(f"{metric}: {value:.2f} samples/second")
            else:
                print(f"{metric}: {value:.2f} seconds")
        else:
            print(f"{metric}: {value:.4f}")
    
    return evaluation

if __name__ == "__main__":
    dataset_path = "datasets/test.jsonl"
    output_path = "gsm8k_evaluation_extended.json"
    
    evaluation = evaluate_model(
        dataset_path=dataset_path,
        num_samples=3,  # Generate 3 samples per question
        use_pass_k=True,  # Enable Pass@k metric
        output_path=output_path
    )