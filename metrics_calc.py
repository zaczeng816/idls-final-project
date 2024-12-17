import json
import re
from rouge import Rouge
import numpy as np

def calculate_error(pred, true):
    """
    Calculate relative error with better handling of edge cases and None values
    """
    if pred is None or true is None:
        return None
        
    if true == 0:
        return min(abs(pred - true), 1000)
    try:
        error = abs(pred - true) / abs(true)
        return min(error, 1000)
    except (ZeroDivisionError, OverflowError, TypeError):
        return 1000

def calculate_average_error(errors):
    """Calculate average error while handling None, inf and nan values"""
    valid_errors = [e for e in errors if e is not None and not np.isinf(e) and not np.isnan(e)]
    if not valid_errors:
        return 0
    return np.mean(valid_errors)

def calculate_metrics(data):
    """Calculate all evaluation metrics including ROUGE scores"""
    rouge = Rouge()
    results = []
    total_accuracy = 0
    errors = []
    valid_comparisons = 0
    
    for entry in data['details']:
        question = entry['question']
        correct_answer = entry.get('correct_answer')
        predict_answer = entry.get('predict_answer')
        reasoning = entry.get('reasoning', '')
        
        # Calculate accuracy and error only if both answers exist
        if correct_answer is not None and predict_answer is not None:
            accuracy = 1 if abs(correct_answer - predict_answer) < 1e-6 else 0
            error = calculate_error(predict_answer, correct_answer)
            total_accuracy += accuracy
            if error is not None:
                errors.append(error)
            valid_comparisons += 1
        else:
            accuracy = None
            error = None
        
        # Calculate ROUGE scores
        if reasoning:
            reasoning_text = reasoning.split('The answer is')[0].strip()
            reference_answer = f"The answer is {correct_answer}" if correct_answer is not None else ""
            
            try:
                scores = rouge.get_scores(reasoning_text, reference_answer)[0]
            except Exception as e:
                print(f"Error calculating ROUGE scores for question: {question}")
                print(f"Error: {e}")
                scores = {
                    'rouge-1': {'f': 0.0},
                    'rouge-2': {'f': 0.0},
                    'rouge-l': {'f': 0.0}
                }
        else:
            scores = {
                'rouge-1': {'f': 0.0},
                'rouge-2': {'f': 0.0},
                'rouge-l': {'f': 0.0}
            }
        
        results.append({
            'question': question,
            'true_answer': correct_answer,
            'predicted_answer': predict_answer,
            'accuracy': accuracy,
            'relative_error': error,
            'rouge_scores': {
                'rouge1_f': scores['rouge-1']['f'],
                'rouge2_f': scores['rouge-2']['f'],
                'rougeL_f': scores['rouge-l']['f']
            }
        })
    
    # Calculate average metrics only for valid comparisons
    if valid_comparisons > 0:
        avg_accuracy = (total_accuracy / valid_comparisons) * 100
    else:
        avg_accuracy = 0
        
    avg_metrics = {
        'accuracy': avg_accuracy,
        'relative_error': calculate_average_error(errors),
        'rouge1_f': sum(r['rouge_scores']['rouge1_f'] for r in results) / max(len(results), 1),
        'rouge2_f': sum(r['rouge_scores']['rouge2_f'] for r in results) / max(len(results), 1),
        'rougeL_f': sum(r['rouge_scores']['rougeL_f'] for r in results) / max(len(results), 1)
    }
    
    return results, avg_metrics

def main():
    try:
        with open('gsm8k_evaluation.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        results, avg_metrics = calculate_metrics(data)
        
        # Print individual results
        for item in results:
            print(f"\nQuestion: {item['question']}")
            print(f"True Answer: {item['true_answer']}")
            print(f"Predicted Answer: {item['predicted_answer']}")
            if item['accuracy'] is not None:
                print(f"Accuracy: {item['accuracy']}")
            if item['relative_error'] is not None:
                print(f"Relative Error: {item['relative_error']:.4f}")
            print("\nROUGE Scores:")
            for metric, score in item['rouge_scores'].items():
                print(f"{metric}: {score:.4f}")
            print("-" * 80)
        
        # Print average metrics
        print("\nAverage Metrics:")
        print(f"Accuracy: {avg_metrics['accuracy']:.4f}%")
        print(f"Relative Error: {avg_metrics['relative_error']:.4f}")
        print(f"ROUGE-1 F1: {avg_metrics['rouge1_f']:.4f}")
        print(f"ROUGE-2 F1: {avg_metrics['rouge2_f']:.4f}")
        print(f"ROUGE-L F1: {avg_metrics['rougeL_f']:.4f}")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()