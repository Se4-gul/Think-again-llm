import anthropic
import csv
import random
from datasets import load_dataset

client = anthropic.Anthropic(api_key="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")

def run_condition(question, condition):
    if condition == "A":
        prompt = f"Let's think step by step. Answer with Yes or No first, then explain. {question}"
    elif condition == "B":
        prompt = f"Let's think step by step. After every 2 steps, stop and ask yourself: 'Is this line of reasoning correct? Am I missing another angle?' Then continue or redirect. Answer with Yes or No first, then explain. {question}"
    elif condition == "C":
        prompt = f"Let's think step by step. At a random point mid-reasoning, pause and reflect before continuing. Answer with Yes or No first, then explain. {question}"
    elif condition == "D":
        prompt = f"Let's think step by step. Before giving your final answer, review your reasoning and check for errors. Answer with Yes or No first, then explain. {question}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def extract_answer(response):
    first_line = response.strip().split('\n')[0].lower()
    if 'yes' in first_line:
        return True
    elif 'no' in first_line:
        return False
    else:
        # fallback — scan full response
        text = response.lower()
        yes_count = text.count('yes')
        no_count = text.count('no')
        return yes_count > no_count

def score_response(question, response, correct_answer, extracted_answer):
    # Score 0 if wrong answer
    if extracted_answer != correct_answer:
        return 0
    
    # If right answer, use model to score reasoning quality
    scoring_prompt = f"""
You are evaluating the quality of a reasoning response.

Question: {question}
Correct answer: {'Yes' if correct_answer else 'No'}
Response to evaluate: {response}

The response has the correct answer. Now score the quality of reasoning:
1 = Right answer but incomplete or weak reasoning
2 = Right answer with complete, well-constructed reasoning that connects all relevant facts

Reply with only a single number: 1 or 2. No other text.
"""
    result = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": scoring_prompt}]
    )
    return int(result.content[0].text.strip()[0])

# Load StrategyQA
print("Loading dataset...")
dataset = load_dataset("ChilleD/StrategyQA", split="test")

# Sample 50 random questions
random.seed(42)
indices = random.sample(range(len(dataset)), 50)
questions = [dataset[i] for i in indices]

print(f"Running experiment on {len(questions)} questions x 4 conditions = {len(questions)*4} API calls")
print("This will also make scoring calls — total ~400 API calls\n")

results = []
for idx, item in enumerate(questions):
    question = item["question"]
    correct = item["answer"]
    
    for condition in ["A", "B", "C", "D"]:
        print(f"[{idx+1}/50] Condition {condition}: {question[:60]}...")
        response = run_condition(question, condition)
        extracted = extract_answer(response)
        score = score_response(question, response, correct, extracted)
        
        results.append({
            "question": question,
            "correct_answer": correct,
            "condition": condition,
            "extracted_answer": extracted,
            "correct": extracted == correct,
            "score": score,
            "response": response
        })

# Save results
with open("strategyqa_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "correct_answer", "condition", "extracted_answer", "correct", "score", "response"])
    writer.writeheader()
    writer.writerows(results)

# Print summary
print("\n=== RESULTS SUMMARY ===")
for condition in ["A", "B", "C", "D"]:
    condition_results = [r for r in results if r["condition"] == condition]
    accuracy = sum(r["correct"] for r in condition_results) / len(condition_results)
    avg_score = sum(r["score"] for r in condition_results) / len(condition_results)
    print(f"Condition {condition}: Accuracy={accuracy:.1%}, Avg Score={avg_score:.2f}")

print("\nDone! Full results saved to strategyqa_results.csv")