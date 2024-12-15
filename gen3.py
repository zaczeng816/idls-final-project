import json
import openai
import time
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
import multiprocessing as mp

def load_questions(file_path: str, start_idx: int, end_idx: int) -> List[Dict]:
   """Load questions from start_idx to end_idx from the GSM8K JSONL file."""
   questions = []
   with open(file_path, 'r') as f:
       for i, line in enumerate(f):
           if i >= end_idx:
               break
           if i >= start_idx:
               questions.append(json.loads(line.strip()))
   return questions

def extract_numeric_answer(reasoning: str) -> Tuple[str, float]:
   """Extract the last number from the reasoning text."""
   numbers = re.findall(r'\d+', reasoning)
   return reasoning, float(numbers[-1]) if numbers else None

def extract_correct_answer(answer_string: str) -> float:
   """Extract the numeric answer from the GSM8K answer string."""
   numbers = re.findall(r'\d+', answer_string)
   return float(numbers[-1]) if numbers else None

def get_diverse_solution(problem: str, answer_string: str, approach_num: int) -> Dict:
   templates = [
       "Solve this step by step, showing each calculation clearly and explaining your reasoning:",
       "Imagine teaching this to a student who learns best through real-world examples. Solve while relating to everyday situations:", 
       "Break this down into the smallest possible steps, explaining each tiny detail:",
       "Solve this using algebra and mathematical notation, showing your work clearly:",
       "Use estimation and rounding first, then solve exactly, comparing the approaches:"
   ]
   
   prompt = f"Approach #{approach_num + 1}: {templates[approach_num]}\n\n{problem}"
   
   try:
       response = openai.ChatCompletion.create(
           model="Qwen/Qwen2.5-72B-Instruct",
           messages=[
               {"role": "system", "content": "You are a math tutor who provides detailed, step-by-step solutions. Each solution should be unique and thorough, showing all work clearly."},
               {"role": "user", "content": prompt}
           ],
           temperature=0.7 + (approach_num * 0.02),
       )
       
       reasoning = response.choices[0].message["content"]
       cleaned_reasoning, predict_answer = extract_numeric_answer(reasoning)
       correct_answer = extract_correct_answer(answer_string)
       
       return {
           "question": problem,
           "reasoning": cleaned_reasoning,
           "predict_answer": predict_answer,
           "correct_answer": correct_answer,
           "approach_info": {
               "number": approach_num + 1,
               "type": templates[approach_num].split('.')[0]
           }
       }
   except Exception as e:
       print(f"Error generating solution for approach {approach_num + 1}: {str(e)}")
       return None

def process_questions(questions: List[Dict], num_approaches: int = 5, start_idx: int = 0, end_idx: int = 8000) -> None:
   solutions = []
   file_count = 0
   total_solutions = 0
   
   for q_idx, question in tqdm(enumerate(questions), total=len(questions), desc="Processing questions"):
       # Inner progress bar for approaches
       for approach in tqdm(range(num_approaches), total=num_approaches, desc=f"Q{q_idx + start_idx} approaches", leave=False):
           solution = get_diverse_solution(
               problem=question['question'],
               answer_string=question['answer'],
               approach_num=approach
           )
           if solution:
               solutions.append(solution)
               total_solutions += 1
               
               if len(solutions) >= 100:
                   output_file = f"gsm8k_solutions_batch_{start_idx}_{file_count}.json"
                   with open(output_file, 'w') as f:
                       json.dump({
                           "pairs": solutions,
                           "metadata": {
                               "batch_number": file_count,
                               "solutions_in_batch": len(solutions),
                               "total_solutions_generated": total_solutions,
                               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                           }
                       }, f, indent=2)
                   tqdm.write(f"Saved batch {file_count} ({len(solutions)} solutions) to {output_file}")
                   tqdm.write(f"Total solutions generated so far: {total_solutions}")
                   solutions = []
                   file_count += 1
               
           time.sleep(1)
   
   if solutions:
       output_file = f"data/gsm8k_solutions_batch_{start_idx}_{file_count}.json"
       with open(output_file, 'w') as f:
           json.dump({
               "pairs": solutions,
               "metadata": {
                   "batch_number": file_count,
                   "solutions_in_batch": len(solutions),
                   "total_solutions_generated": total_solutions,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
               }
           }, f, indent=2)
       print(f"Saved final batch {file_count} ({len(solutions)} solutions) to {output_file}")

def get_api_key():
    """
    Loads the API keys from the .env file
    """
    from dotenv import load_dotenv
    import os
    load_dotenv()
    return os.environ.get("DEEPINFRA_API_KEY")

def process_range(start_idx: int, end_idx: int, num_approaches: int = 5) -> None:
    """Function to handle a range of questions"""
    openai.api_key = get_api_key()
    openai.api_base = "https://api.deepinfra.com/v1/openai"
    
    questions = load_questions("data/train.jsonl", start_idx, end_idx)
    print(f"Process {mp.current_process().name}: Loaded questions {start_idx} to {end_idx}")
    
    process_questions(questions, num_approaches, start_idx, end_idx)

def split_range(start: int, end: int, num_processes: int) -> List[Tuple[int, int]]:
    """Split a range into approximately equal chunks"""
    chunk_size = (end - start) // num_processes
    ranges = []
    for i in range(num_processes):
        chunk_start = start + (i * chunk_size)
        chunk_end = chunk_start + chunk_size if i < num_processes - 1 else end
        ranges.append((chunk_start, chunk_end))
    return ranges

def main():
    NUM_PROCESSES = 5
    TOTAL_START = 4000
    TOTAL_END = 7400
    
    # Split the range into chunks
    ranges = split_range(TOTAL_START, TOTAL_END, NUM_PROCESSES)
    
    # Create and start processes
    processes = []
    for start, end in ranges:
        p = mp.Process(target=process_range, args=(start, end))
        processes.append(p)
        p.start()
        print(f"Started process for range {start}-{end}")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All processes completed!")

if __name__ == "__main__":
    main()