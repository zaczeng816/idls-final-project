import json
import openai
import time
import re
from typing import List, Dict, Tuple
from tqdm import tqdm

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


def get_reasoning_label(problem: str, reasonings: List[str], debug = False) -> Dict:
    system_prompt = """You are an expert math teacher evaluating different solution approaches. 
    Rate each solution on a scale of 1-5 based on:
    - Clarity of explanation
    - Mathematical accuracy
    - Step-by-step breakdown
    - Logical flow
    
    Provide ONLY numerical scores in this format:
    [score1, score2, score3, ...]"""

    prompt = f"""Question: {problem}

Here are {len(reasonings)} different solution approaches. Rate each one on a scale of 1-5:

{'-' * 50}
""" + "\n\n".join(f"Solution {i+1}:\n{reasoning}" for i, reasoning in enumerate(reasonings))

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
        
        if debug:
            print("DEBUG:response: ", response)
        
        # Extract scores from response
        response_text = response.choices[0].message["content"]
        
        # Parse scores safely using regex
        try:
            # Look for list pattern [n, n, n, ...]
            import re
            score_matches = re.findall(r'\d+(?:\.\d+)?', response_text)
            scores = [float(score) for score in score_matches[:len(reasonings)]]
            
            # Validate scores
            if len(scores) != len(reasonings):
                raise ValueError(f"Expected {len(reasonings)} scores, got {len(scores)}")
            if not all(1 <= score <= 5 for score in scores):
                raise ValueError("Scores must be between 1 and 5")
                
        except Exception as e:
            print(f"Error parsing scores: {str(e)}")
            return None

        best_reasoning = reasonings[scores.index(max(scores))]

        return {
            "question": problem,
            "reasonings": reasonings,
            "best_reasoning": best_reasoning,
            "scores": scores,
            "raw_response": response_text  # Include for debugging
        }
    except Exception as e:
        print(f"Error generating score for question: {problem}: {str(e)}")
        return None

if __name__ == "__main__":
    openai.api_key = get_api_key()
    openai.api_base = "https://api.deepinfra.com/v1/openai"

    # First, calculate total number of unique questions
    def count_unique_questions(data):
        i = 0
        count = 0
        while i < len(data):
            count += 1
            j = i
            while j + 1 < len(data) and data[j+1]['question'] == data[i]['question']:
                j += 1
            i = j + 1
        return count

    # Then modify the main loop with tqdm
    with open('dataset/flattened_gsm8k_questions.json', 'r') as file:
        data = json.load(file)
        total_questions = count_unique_questions(data)
        i = 0
        data_with_labels = []
        
        # Create progress bar
        with tqdm(total=total_questions, desc="Processing questions") as pbar:
            while i < len(data):
                j = i
                while j + 1< len(data) and data[j+1]['question'] == data[i]['question']:
                    j += 1
                cur_data = data[i:j+1]

                result = get_reasoning_label(cur_data[0]['question'], [d['reasoning'] for d in cur_data], debug=True)
                if result:
                    for (i, d) in enumerate(cur_data):
                        cur_data[i]['reasoning_score'] = result['scores'][i]
                        cur_data[i]['best_reasoning'] = result['best_reasoning']
                    data_with_labels += cur_data
                i = j + 1
                
                pbar.update(1)
                if i % 1000 == 0:
                    json.dump({"pairs": data_with_labels}, open('dataset/gsm8k_reasoning_labels.json', 'w'), indent=2)
        json.dump({"pairs": data_with_labels}, open('dataset/gsm8k_reasoning_labels.json', 'w'), indent=2)

