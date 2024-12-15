import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import json
from config import Config

def generate_multiple_reasonings(model, tokenizer, question, num_samples=5, max_length=150, temperature=0.7, top_p=0.9):
    """
    Generate multiple reasonings for a given question using a fine-tuned model.

    Args:
        model: The fine-tuned student model.
        tokenizer: Tokenizer for the model.
        question: The input question.
        num_samples: Number of reasoning outputs to generate.
        max_length: Maximum length of generated outputs.
        temperature: Sampling temperature for diversity.
        top_p: Top-p sampling (nucleus sampling) for diversity.
    Returns:
        List of generated reasonings.
    """
    prompt = f"Question: {question}\nGenerate step-by-step reasoning:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_samples
    )

    reasonings = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return reasonings

def calculate_semantic_scores(reasonings, similarity_threshold=0.85):
    """
    Calculate self-consistency scores for all generated reasonings based on semantic clustering.

    Args:
        reasonings: List of generated reasonings.
        similarity_threshold: Cosine similarity threshold for clustering.
    Returns:
        List of tuples (reasoning, score), where score = number of similar outputs / total outputs.
    """
    encoder = SentenceTransformer("all-MiniLM-L6-v2") 
    embeddings = encoder.encode(reasonings)
    similarity_matrix = cosine_similarity(embeddings, embeddings)

    scores = []
    for i, row in enumerate(similarity_matrix):
        similar_count = sum(1 for sim in row if sim > similarity_threshold)
        score = similar_count / len(reasonings)
        scores.append((reasonings[i], score))

    return scores

def calculate_predicted_answer_scores(reasonings):
    """
    Calculate self-consistency scores based on predicted answers.

    Args:
        reasonings: List of generated reasonings.
    Returns:
        List of tuples (reasoning, score), where score = proportion of reasonings with the same predicted answer.
    """
    # Extract predicted answers from reasonings
    predicted_answers = []
    for reasoning in reasonings:
        try:
            # Extract the last numeric value from the reasoning as the predicted answer
            predicted_answer = float(reasoning.split()[-1].strip('.'))
        except ValueError:
            predicted_answer = None  # Handle cases where no valid number is found
        predicted_answers.append(predicted_answer)
    
    # Count occurrences of each predicted answer
    answer_counts = Counter(predicted_answers)
    total_reasonings = len(reasonings)
    
    # Assign scores based on the frequency of predicted answers
    scores = []
    for reasoning, predicted_answer in zip(reasonings, predicted_answers):
        score = answer_counts[predicted_answer] / total_reasonings if predicted_answer is not None else 0
        scores.append((reasoning, score))
    
    return scores

def apply_self_consistency_all_scores(model, tokenizer, question_data, num_samples=5):
    """
    Apply self-consistency to generate all reasonings and calculate their scores.

    Args:
        model: The fine-tuned student model.
        tokenizer: Tokenizer for the model.
        question_data: List of question dictionaries with keys "question" and optional "correct_answer".
        num_samples: Number of samples per question.
    Returns:
        List of dictionaries with generated reasonings and their scores for each question.
    """
    results = []

    for pair in question_data:
        question = pair["question"]
        correct_answer = pair.get("correct_answer")

        reasonings = generate_multiple_reasonings(model, tokenizer, question, num_samples=num_samples)
        # scores = calculate_semantic_scores(reasonings)
        scores = calculate_predicted_answer_scores(reasonings)

        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "reasonings_with_scores": scores
        })

    return results

def save_self_consistency_results_to_json(input_file, output_file, model, tokenizer, num_samples=10):
    """
    Apply self-consistency to generate and score reasonings for questions in the dataset, and save results.

    Args:
        input_file: Path to the JSON file containing the dataset with "pairs" key.
        output_file: Path to save the results in JSON format.
        model: The fine-tuned student model.
        tokenizer: Tokenizer for the model.
        num_samples: Number of samples to generate per question.
    """
    with open(input_file, "r") as file:
        input_data = json.load(file)

    if "pairs" not in input_data:
        raise ValueError("Input file must contain a 'pairs' key with the questions.")

    question_data = input_data["pairs"]
    results = apply_self_consistency_all_scores(model, tokenizer, question_data, num_samples=num_samples)

    final_results = {"pairs": []}

    for result in results:
        for idx, (reasoning, score) in enumerate(result["reasonings_with_scores"], start=1):
            predict_answer = None
            try:
                predict_answer = float(reasoning.split()[-1].strip('.'))
            except ValueError:
                pass

            entry = {
                "question": result["question"],
                "reasoning": reasoning,
                "predict_answer": predict_answer,
                "correct_answer": result.get("correct_answer"),
                "score": score,
                "approach_info": {
                    "number": idx,
                    "type": "Generated reasoning with self-consistency score"
                }
            }
            final_results["pairs"].append(entry)

    with open(output_file, "w") as file:
        json.dump(final_results, file, indent=4)

    print(f"Results saved to {output_file}")

def generate_reasonings_only(input_file, output_file, model, tokenizer, num_samples=10):
    """
    Generate multiple reasonings for questions in the dataset and save results without scoring.

    Args:
        input_file: Path to the JSON file containing the dataset with "pairs" key.
        output_file: Path to save the results in JSON format.
        model: The fine-tuned student model.
        tokenizer: Tokenizer for the model.
        num_samples: Number of samples to generate per question.
    """
    with open(input_file, "r") as file:
        input_data = json.load(file)

    if "pairs" not in input_data:
        raise ValueError("Input file must contain a 'pairs' key with the questions.")

    question_data = input_data["pairs"]

    final_results = {"pairs": []}

    for pair in question_data:
        question = pair["question"]
        correct_answer = pair.get("correct_answer")

        reasonings = generate_multiple_reasonings(model, tokenizer, question, num_samples=num_samples)

        for idx, reasoning in enumerate(reasonings, start=1):
            predict_answer = None
            try:
                # Extract the predicted answer from the reasoning
                predict_answer = float(reasoning.split()[-1].strip('.'))
            except ValueError:
                pass

            entry = {
                "question": question,
                "reasoning": reasoning,
                "predict_answer": predict_answer,
                "correct_answer": correct_answer,
                "approach_info": {
                    "number": idx,
                    "type": "Generated reasoning without scoring"
                }
            }
            final_results["pairs"].append(entry)

    with open(output_file, "w") as file:
        json.dump(final_results, file, indent=4)

    print(f"Reasonings saved to {output_file}")


if __name__ == "__main__":
    config = Config()

    # Define paths to train and validation files
    train_file = "train.json"
    valid_file = "valid.json"

    # Define output files for results
    train_output_file = "student1_train_results.json"
    valid_output_file = "student1_valid_results.json"

    # Load fine-tuned model
    model_name = config.output_dir_student1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    # Generate and save results for training and validation sets
    # save_self_consistency_results_to_json(train_file, train_output_file, model, tokenizer, num_samples=10)
    # save_self_consistency_results_to_json(valid_file, valid_output_file, model, tokenizer, num_samples=10)

    generate_reasonings_only(train_file, train_output_file, model, tokenizer, num_samples=5)
    generate_reasonings_only(valid_file, valid_output_file, model, tokenizer, num_samples=5)