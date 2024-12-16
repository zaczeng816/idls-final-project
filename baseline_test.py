import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from typing import List
from rouge import Rouge
from tqdm import tqdm
from config import Config
from huggingface_hub import login
import wandb
wandb.init(mode="offline")


def extract_numeric_answer(answer: str):
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

def generate_reasoning_for_question(model, tokenizer, question, num_samples=5, max_length=200, temperature=0.7, top_p=0.9):
    """
    Generate multiple reasoning outputs for a given question using a pre-trained model.
    Extract and clean the final answer for each reasoning.
    """
    # Refined prompt
    prompt = (
        f"Question: {question}\n"
        f"Please provide a step-by-step reasoning for solving the question. "
        f"After the reasoning, explicitly state the final answer at the end, preceded by 'Final Answer:'.\n\n"
        f"Reasoning:\n"
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_samples,
        eos_token_id=tokenizer.encode("Final Answer:")[0]  # Stop generation at 'Final Answer:'
    )

    # Extract reasoning and final answers
    results = []
    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        
        # Ensure the prompt is removed from the reasoning
        reasoning_and_answer = decoded[len(prompt):].strip()
        
        # Split reasoning and final answer
        reasoning, _, answer = reasoning_and_answer.partition("Final Answer:")

        # Extract and clean the numeric final answer
        final_answer = extract_numeric_answer(answer.strip()) if answer else None
        
        # Append results
        results.append({
            "reasoning": reasoning.strip(),
            "final_answer": final_answer
        })

    return results

def calculate_accuracy(results):
    """
    Calculate accuracy based on predicted and correct answers.
    """
    correct = 0
    total = 0

    for result in results:
        correct_answer = result.get("correct_answer")
        if correct_answer is None:
            continue

        for reasoning in result["reasonings"]:
            predicted_answer = reasoning.get("final_answer")
            if predicted_answer is not None and abs(predicted_answer - correct_answer) < 1e-6:
                correct += 1
                break  # Count the first correct prediction

        total += 1

    return (correct / total) * 100 if total > 0 else 0


def calculate_rouge_scores(results):
    """
    Calculate ROUGE scores between generated reasoning and the detailed 'answer' field in the test file.
    """
    rouge = Rouge()
    rouge_scores = []

    for result in results:
        correct_reasoning = result.get("answer")  # Use the full 'answer' text from the test file
        if not correct_reasoning:
            continue

        for reasoning_output in result["reasonings"]:
            reasoning = reasoning_output.get("reasoning", "")
            try:
                score = rouge.get_scores(reasoning, correct_reasoning)
                rouge_l_f1 = score[0]["rouge-l"]["f"]
                rouge_scores.append(rouge_l_f1)
            except Exception as e:
                print(f"Error calculating ROUGE for reasoning: {reasoning}. Error: {e}")
                continue

    return sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0


def calculate_pass_at_k(results, k=3):
    """
    Calculate Pass@k metric: If at least one of the top k predictions matches the correct answer.
    """
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


def test_gsm8k_test_set_jsonl(input_file, model_name, num_samples=5, output_file=None):
    """
    Test a pre-trained model on the GSM8K test set in JSONL format and calculate metrics.
    """
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Load GSM8K test data
    questions = []
    with open(input_file, "r") as file:
        for id, line in enumerate(file):
            # if id > 1:
            #     break
            item = json.loads(line)
            question = item["question"]
            answer = item["answer"]
            raw_answer = answer.split("####")[-1].strip()
            correct_answer = extract_numeric_answer(raw_answer)
            questions.append({"question": question, "correct_answer": correct_answer, "answer": answer})

    results = []
    with tqdm(total=len(questions), desc="Processing questions", unit="question") as pbar:
        for pair in questions:
            question = pair["question"]
            correct_answer = pair["correct_answer"]

            reasonings = generate_reasoning_for_question(model, tokenizer, question, num_samples=num_samples)
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "answer": pair["answer"],  # Detailed explanation for ROUGE
                "reasonings": reasonings
            })

            pbar.update(1)

    # Calculate metrics
    accuracy = calculate_accuracy(results)
    rouge_score = calculate_rouge_scores(results)
    pass_at_3 = calculate_pass_at_k(results, k=3)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average ROUGE-L: {rouge_score:.4f}")
    print(f"Pass@3: {pass_at_3:.2f}%")

    # Save results
    if output_file:
        with open(output_file, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Results saved to {output_file}")

    return accuracy, rouge_score, pass_at_3, results


if __name__ == "__main__":
    config = Config()

    # Authenticate with the Hugging Face Hub
    login(token=config.token)

    # Path to the GSM8K test set in JSONL format
    test_file = "test.jsonl"

    # Output file for results
    output_file = "gsm8k_test_results.json"

    # Model name or path
    model_name = config.student1_model

    # Run the evaluation
    accuracy, rouge_score, pass_at_3, results = test_gsm8k_test_set_jsonl(
        test_file, model_name, num_samples=3, output_file=output_file
    )