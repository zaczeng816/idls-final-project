import json
import re
import os
from rouge import Rouge
import numpy as np

def extract_last_number(text):
    """Extract the last number from a text string"""
    numbers = re.findall(r'\d+', text)
    return int(numbers[-1]) if numbers else None

def extract_answer_from_prediction(prediction):
    """Extract numerical answer from prediction"""
    if not prediction or not prediction.get('reasoning'):
        return None
    numbers = re.findall(r'\$?(\d+(?:\.\d+)?)', prediction['reasoning'])
    return float(numbers[-1]) if numbers else None

def calculate_error(pred, true):
    """
    Calculate relative error with better handling of edge cases:
    - If true is 0, return absolute difference
    - Cap the maximum error at 1000 (100000% error)
    - Handle inf and nan values
    """
    if true == 0:
        return min(abs(pred - true), 1000)  # Cap absolute difference at 1000
    try:
        error = abs(pred - true) / abs(true)
        # Cap maximum error at 1000 (100000% error)
        return min(error, 1000)
    except (ZeroDivisionError, OverflowError):
        return 1000  # Return maximum error for invalid calculations

def calculate_average_error(errors):
    """
    Calculate average error while handling inf and nan values:
    - Remove inf and nan values
    - If all values are inf/nan, return 1000 (maximum error)
    """
    valid_errors = [e for e in errors if e is not None and not np.isinf(e) and not np.isnan(e)]
    if not valid_errors:
        return 1000
    return np.mean(valid_errors)

def extract_answers_and_score(json_data):
    """
    Extract final answers from JSON data and calculate ROUGE scores and accuracy metrics.
    """
    rouge = Rouge()
    results = []
    total_accuracy = 0
    errors = []
    valid_comparisons = 0
    
    for entry in json_data['results']:
        question = entry['question']
        answer_text = entry['answer']
        
        # Extract true answer
        match = re.search(r'####\s*(\d+)', answer_text)
        if match:
            true_answer = int(match.group(1))
        else:
            true_answer = extract_last_number(answer_text)
        
        # Extract predicted answer
        predicted_answer = extract_answer_from_prediction(entry['predictions'][0]) if entry['predictions'] else None
        
        # Calculate accuracy and error if both answers are available
        if true_answer is not None and predicted_answer is not None:
            accuracy = 1 if abs(true_answer - predicted_answer) < 1e-6 else 0
            error = calculate_error(predicted_answer, true_answer)
            errors.append(error)
            total_accuracy += accuracy
            valid_comparisons += 1
        else:
            accuracy = None
            error = None
        
        # Extract reasoning and calculate ROUGE scores
        original_reasoning = answer_text.split('####')[0] if '####' in answer_text else answer_text
        predicted_reasoning = entry['predictions'][0]['reasoning'] if entry['predictions'] and entry['predictions'][0]['reasoning'] else ""
        
        try:
            if predicted_reasoning and original_reasoning:
                scores = rouge.get_scores(predicted_reasoning, original_reasoning)[0]
            else:
                scores = {
                    'rouge-1': {'f': 0.0},
                    'rouge-2': {'f': 0.0},
                    'rouge-l': {'f': 0.0}
                }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            scores = {
                'rouge-1': {'f': 0.0},
                'rouge-2': {'f': 0.0},
                'rouge-l': {'f': 0.0}
            }
            
        results.append({
            'question': question,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'accuracy': accuracy,
            'relative_error': error,
            'rouge_scores': {
                'rouge1_f': scores['rouge-1']['f'],
                'rouge2_f': scores['rouge-2']['f'],
                'rougeL_f': scores['rouge-l']['f']
            },
            'original_reasoning': original_reasoning.strip(),
            'predicted_reasoning': predicted_reasoning.strip()
        })
    
    # Calculate average metrics
    avg_metrics = {
        'accuracy': total_accuracy / valid_comparisons if valid_comparisons > 0 else 0,
        'relative_error': calculate_average_error(errors),
        'rouge1_f': sum(r['rouge_scores']['rouge1_f'] for r in results) / len(results),
        'rouge2_f': sum(r['rouge_scores']['rouge2_f'] for r in results) / len(results),
        'rougeL_f': sum(r['rouge_scores']['rougeL_f'] for r in results) / len(results)
    }
    
    return results, avg_metrics

def main():
    # Get the absolute path to the evaluation_results.json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "evaluation_results.json")
    
    try:
        # First, try to read the file and check if it's empty
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"File is empty: {file_path}")
            
            # Try to parse the JSON content
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {str(e)}")
                print(f"Content of file: {content[:200]}...")  # Print first 200 chars for debugging
                return
            
            results, avg_metrics = extract_answers_and_score(data)
            
            # Print individual results
            for item in results:
                print(f"\nQuestion: {item['question']}")
                print(f"True Answer: {item['true_answer']}")
                print(f"Predicted Answer: {item['predicted_answer']}")
                if item['accuracy'] is not None:
                    print(f"Accuracy: {item['accuracy']}")
                    print(f"Relative Error: {item['relative_error']:.4f}")
                print("\nROUGE Scores:")
                for metric, score in item['rouge_scores'].items():
                    print(f"{metric}: {score:.4f}")
                print("-" * 80)
            
            # Print average metrics
            print("\nAverage Metrics:")
            print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
            print(f"Relative Error: {avg_metrics['relative_error']:.4f}")
            print(f"ROUGE-1 F1: {avg_metrics['rouge1_f']:.4f}")
            print(f"ROUGE-2 F1: {avg_metrics['rouge2_f']:.4f}")
            print(f"ROUGE-L F1: {avg_metrics['rougeL_f']:.4f}")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Attempting to read: {file_path}")

if __name__ == "__main__":
    main()