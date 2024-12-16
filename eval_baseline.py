import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from rouge import Rouge
from tqdm import tqdm
from huggingface_hub import login
import numpy as np
from dataclasses import dataclass
from pathlib import Path



@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    model_name: str
    test_file: str
    output_file: str
    num_samples: int = 1
    use_pass_k: bool = False
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9
    hf_token: Optional[str] = None

def extract_numeric_answer(answer: str) -> Optional[float]:
    """
    Extract the first numeric value from a string using regex.
    
    Args:
        answer: The answer string.
    
    Returns:
        The numeric value as a float, or None if not found.
    """
    match = re.search(r"[-+]?\d*\.?\d+", answer)
    if match:
        try:
            return float(match.group().replace(",", ""))
        except ValueError:
            return None
    return None

def generate_reasoning_for_question(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    config: EvaluationConfig
) -> List[Dict[str, Any]]:
    """
    Generate multiple reasoning outputs for a given question using a pre-trained model.
    
    Args:
        model: The pre-trained language model
        tokenizer: The tokenizer
        question: The question to generate reasoning for
        config: Evaluation configuration
        
    Returns:
        List of dictionaries containing reasoning and final answers
    """
    prompt = (
        f"Question: {question}\n"
        f"Please provide a step-by-step reasoning for solving the question. "
        f"After the reasoning, explicitly state the final answer at the end, preceded by 'Final Answer:'.\n\n"
        f"Reasoning:\n"
    )
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_length=config.max_length,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        num_return_sequences=config.num_samples,
        eos_token_id=tokenizer.encode("Final Answer:")[0]
    )
    
    results = []
    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        reasoning_and_answer = decoded[len(prompt):].strip()
        reasoning, _, answer = reasoning_and_answer.partition("Final Answer:")
        final_answer = extract_numeric_answer(answer.strip()) if answer else None
        
        results.append({
            "reasoning": reasoning.strip(),
            "final_answer": final_answer
        })
    
    return results

def calculate_accuracy(results: List[Dict[str, Any]]) -> float:
    """Calculate accuracy of predictions."""
    correct = 0
    total = 0
    
    for result in results:
        correct_answer = result.get("correct_answer")
        if correct_answer is None:
            continue
            
        predicted_answer = result["reasonings"][0].get("final_answer")
        if predicted_answer is not None and abs(predicted_answer - correct_answer) < 1e-6:
            correct += 1
        total += 1
    
    return (correct / total) * 100 if total > 0 else 0

def calculate_rouge_scores(results: List[Dict[str, Any]]) -> float:
    """Calculate ROUGE scores between generated reasoning and reference answers."""
    rouge = Rouge()
    rouge_scores = []
    
    for result in results:
        correct_reasoning = result.get("answer")
        if not correct_reasoning:
            continue
            
        reasoning = result["reasonings"][0].get("reasoning", "")
        try:
            score = rouge.get_scores(reasoning, correct_reasoning)
            rouge_scores.append(score[0]["rouge-l"]["f"])
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            continue
    
    return sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0

def calculate_error_metrics(results: List[Dict[str, Any]]) -> float:
    """Calculate average error metrics for predictions."""
    errors = []
    for result in results:
        correct_answer = result.get("correct_answer")
        predicted_answer = result["reasonings"][0].get("final_answer")
        
        if correct_answer is not None and predicted_answer is not None:
            if correct_answer == 0:
                error = abs(correct_answer - predicted_answer)
            else:
                error = abs(correct_answer - predicted_answer) / abs(correct_answer)
            errors.append(error)
    
    return sum(errors) / len(errors) if errors else None

def calculate_pass_at_k(results: List[Dict[str, Any]], k: int = 3) -> float:
    """Calculate Pass@k metric."""
    correct = 0
    total = 0
    
    for result in results:
        correct_answer = result.get("correct_answer")
        if correct_answer is None:
            continue
            
        top_k_answers = [res["final_answer"] for res in result["reasonings"][:k]]
        if any(predicted_answer is not None and abs(predicted_answer - correct_answer) < 1e-6
               for predicted_answer in top_k_answers):
            correct += 1
        total += 1
    
    return (correct / total) * 100 if total > 0 else 0

def evaluate_model(config: EvaluationConfig) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate a model on the GSM8K test set.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Tuple of metrics dictionary and detailed results
    """
    # Login to Hugging Face if token provided
    if config.hf_token:
        login(token=config.hf_token)
    
    # Load model and tokenizer
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.eval()
    
    # Load test data
    print(f"Loading test data from: {config.test_file}")
    questions = []
    with open(config.test_file, "r") as file:
        for line in file:
            item = json.loads(line)
            question = item["question"]
            answer = item["answer"]
            raw_answer = answer.split("####")[-1].strip()
            correct_answer = extract_numeric_answer(raw_answer)
            questions.append({
                "question": question,
                "correct_answer": correct_answer,
                "answer": answer
            })
    
    # Process questions
    results = []
    with tqdm(total=len(questions), desc="Evaluating") as pbar:
        for pair in questions:
            reasoning = generate_reasoning_for_question(model, tokenizer, pair["question"], config)
            results.append({
                "question": pair["question"],
                "correct_answer": pair["correct_answer"],
                "answer": pair["answer"],
                "reasonings": reasoning
            })
            pbar.update(1)
    
    # Calculate metrics
    metrics = {
        "accuracy": calculate_accuracy(results),
        "rouge_score": calculate_rouge_scores(results),
        "average_error": calculate_error_metrics(results)
    }
    
    if config.use_pass_k and config.num_samples > 1:
        metrics["pass_at_3"] = calculate_pass_at_k(results, k=min(3, config.num_samples))
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    if config.output_file:
        output_data = {
            "config": vars(config),
            "metrics": metrics,
            "results": results
        }
        output_path = Path(config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"\nResults saved to: {output_path}")
    
    return metrics, results

if __name__ == "__main__":
    # Example configuration
    config = EvaluationConfig(
        model_name="google/gemma-2b",  # Example model
        test_file="datasets/test.jsonl",
        output_file="evaluation_results.json",
        num_samples=1,
        use_pass_k=False,
        hf_token="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"  # Add your Hugging Face token here if needed
    )
    
    # Run evaluation
    metrics, results = evaluate_model(config)