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


example_question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
example_reasonings = [ 
    "Certainly! Let's solve this problem step by step, showing all the calculations and explaining the reasoning behind each step.\n\n### Step 1: Determine the number of clips sold in April\nNatalia sold clips to 48 of her friends in April. So, the number of clips sold in April is:\n\\[ \\text{Clips sold in April} = 48 \\]\n\n### Step 2: Determine the number of clips sold in May\nAccording to the problem, Natalia sold half as many clips in May as she did in April. To find out how many clips she sold in May, we need to calculate half of the number of clips sold in April:\n\\[ \\text{Clips sold in May} = \\frac{48}{2} = 24 \\]\n\n### Step 3: Calculate the total number of clips sold in April and May\nTo find the total number of clips sold in both months, we add the number of clips sold in April to the number of clips sold in May:\n\\[ \\text{Total clips sold} = \\text{Clips sold in April} + \\text{Clips sold in May} \\]\n\\[ \\text{Total clips sold} = 48 + 24 = 72 \\]\n\n### Final Answer\nNatalia sold a total of 72 clips in April and May.",
    "Sure! Let's think about this problem using a real-world example that might be easier to understand.\n\nImagine Natalia is selling clips, which are like hair accessories, to her friends. In April, she was very successful and sold clips to 48 of her friends. That's a lot of friends!\n\nNow, let's move to May. In May, Natalia wasn't as successful as she was in April. She sold half as many clips as she did in April. To figure out how many clips she sold in May, we need to take the number of clips she sold in April and divide it by 2.\n\nSo, let's do the math:\n- Number of clips sold in April: 48\n- Number of clips sold in May: \\( \\frac{48}{2} = 24 \\)\n\nNow, to find out how many clips Natalia sold altogether in April and May, we add the number of clips sold in April to the number of clips sold in May:\n- Total clips sold = Clips sold in April + Clips sold in May\n- Total clips sold = 48 + 24 = 72\n\nSo, Natalia sold a total of 72 clips in April and May.\n\nTo make it even more relatable, imagine if you had 48 cookies and you gave half of them away in May. You would have given away 24 cookies in May. If you add the cookies you gave away in April (48) to the cookies you gave away in May (24), you would have given away a total of 72 cookies. This is just like Natalia giving away clips to her friends!",
    "Certainly! Let's break down this problem into the smallest possible steps to understand how to find the total number of clips Natalia sold in April and May.\n\n### Step 1: Identify the number of clips sold in April\n- Natalia sold 48 clips in April.\n  - This is given directly in the problem.\n\n### Step 2: Determine the number of clips sold in May\n- The problem states that Natalia sold \"half as many clips in May\" as she did in April.\n  - To find half of the number of clips sold in April, we need to divide the number of clips sold in April by 2.\n  - Calculation: \\( \\frac{48}{2} = 24 \\)\n  - So, Natalia sold 24 clips in May.\n\n### Step 3: Add the number of clips sold in April and May\n- To find the total number of clips sold in both months, we add the number of clips sold in April to the number of clips sold in May.\n  - Calculation: \\( 48 + 24 = 72 \\)\n\n### Step 4: State the final answer\n- Natalia sold a total of 72 clips in April and May.\n\nSo, the final answer is **72 clips**.",
    "To solve this problem using algebra and mathematical notation, we can follow these steps:\n\n1. **Define the variable:**\n   Let \\( C \\) represent the number of clips Natalia sold in April.\n\n2. **Given information:**\n   - Natalia sold 48 clips in April.\n   - She sold half as many clips in May as she did in April.\n\n3. **Express the number of clips sold in May:**\n   Since she sold half as many clips in May as in April, the number of clips sold in May can be represented as \\( \\frac{C}{2} \\).\n\n4. **Substitute the known value:**\n   We know from the problem that \\( C = 48 \\). Therefore, the number of clips sold in May is \\( \\frac{48}{2} \\).\n\n5. **Calculate the number of clips sold in May:**\n   \\[\n   \\frac{48}{2} = 24\n   \\]\n\n6. **Calculate the total number of clips sold in April and May:**\n   The total number of clips sold is the sum of the clips sold in April and the clips sold in May:\n   \\[\n   C + \\frac{C}{2} = 48 + 24\n   \\]\n\n7. **Perform the addition:**\n   \\[\n   48 + 24 = 72\n   \\]\n\nTherefore, Natalia sold a total of 72 clips in April and May.",
    "Let's solve this problem using both estimation and rounding first, and then by solving it exactly. We'll compare the two approaches to see how close our estimation was to the exact solution.\n\n### Step 1: Estimation and Rounding\n\n1. **Estimate the number of clips sold in April:**\n   - Natalia sold clips to 48 friends in April. This is a specific number, so we don't need to round it.\n\n2. **Estimate the number of clips sold in May:**\n   - Natalia sold half as many clips in May as she did in April.\n   - Half of 48 is 24.\n\n3. **Estimate the total number of clips sold in both months:**\n   - Total estimated clips = April clips + May clips\n   - Total estimated clips = 48 + 24 = 72\n\n### Step 2: Solve Exactly\n\n1. **Calculate the number of clips sold in April:**\n   - Natalia sold clips to 48 friends in April.\n\n2. **Calculate the number of clips sold in May:**\n   - Natalia sold half as many clips in May as she did in April.\n   - Half of 48 is 24.\n\n3. **Calculate the total number of clips sold in both months:**\n   - Total exact clips = April clips + May clips\n   - Total exact clips = 48 + 24 = 72\n\n### Step 3: Compare the Approaches\n\n- **Estimation Result:** 72 clips\n- **Exact Result:** 72 clips\n\nIn this case, the estimation and the exact solution are the same. This is because the numbers involved (48 and 24) are relatively straightforward and do not require rounding.\n\n### Final Answer\n\nNatalia sold a total of 72 clips in April and May."
]

def get_api_key():
    """
    Loads the API keys from the .env file
    """
    from dotenv import load_dotenv
    import os
    load_dotenv()
    return os.environ.get("DEEPINFRA_API_KEY")


def get_reasoning_label(problem: str, reasonings: List[str]) -> Dict:
   system_prompt = ""
   prompt = ""
   scores = []
   try:
       response = openai.ChatCompletion.create(
           model="Qwen/Qwen2.5-72B-Instruct",
           messages=[
               {"role": "system", "content": system_prompt},
               {"role": "user", "content": prompt}
           ],
           temperature=0.7,
       )
       
       reasoning = response.choices[0].message["content"]
       best_reasoning = reasonings[scores.index(max(scores))]

       return {
           "question": problem,
           "reasonings": reasonings,
           "best_reasoning": best_reasoning,
           "scores": scores,
       }
   except Exception as e:
       print(f"Error generating score for question: {problem}: {str(e)}")
       return None

if __name__ == "__main__":
    openai.api_key = get_api_key()
    openai.api_base = "https://api.deepinfra.com/v1/openai"

    print(get_reasoning_label(example_question, example_reasonings))
