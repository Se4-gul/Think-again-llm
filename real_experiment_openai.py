from openai import OpenAI
import csv
import random
from datasets import load_dataset

client = OpenAI(api_key="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")

def run_condition(question, condition, benchmark):
    if benchmark == "arc":
        answer_instruction = "Answer with only the letter A, B, C, or D first, then explain."
    elif benchmark == "strategyqa":
        answer_instruction = "Answer with Yes or No first, then explain."
    elif benchmark == "gsm8k":
        answer_instruction = "Show your working, then write your final answer as a number on the last line prefixed with ####."

    if condition == "A":
        prompt = f"Let's think step by step. {answer_instruction} {question}"
    elif condition == "B":
        prompt = f"Let's think step by step. After every 2 steps, stop and ask yourself: 'Is this line of reasoning correct? Am I missing another angle?' Then continue or redirect. {answer_instruction} {question}"
    elif condition == "C":
        prompt = f"Let's think step by step. At a random point mid-reasoning, pause and reflect before continuing. {answer_instruction} {question}"
    elif condition == "D":
        prompt = f"Let's think step by step. Before giving your final answer, review your reasoning and check for errors. {answer_instruction} {question}"
    elif condition == "E":
        prompt = f"Let's think step by step. If at any point you notice you are choosing between two equally plausible paths, drawing on knowledge from a different domain than you started with, or relying on an assumption you haven't verified — pause, flag it explicitly, and reconsider before continuing. {answer_instruction} {question}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def extract_answer(response, benchmark):
    if benchmark == "strategyqa":
        first_line = response.strip().split('\n')[0].lower()
        if 'yes' in first_line:
            return True
        elif 'no' in first_line:
            return False
        else:
            text = response.lower()
            return text.count('yes') > text.count('no')

    elif benchmark == "arc":
        import re
        # Strip markdown bold/italic and clean first line
        first_line = response.strip().split('\n')[0]
        first_line = re.sub(r'[\*\_]+', '', first_line).strip().upper()
        for letter in ['A', 'B', 'C', 'D']:
            if first_line.startswith(letter):
                return letter
        # fallback — scan full response for first standalone letter
        matches = re.findall(r'\b([ABCD])\b', response[:200])
        return matches[0] if matches else 'A'

    elif benchmark == "gsm8k":
        if '####' in response:
            part = response.split('####')[-1].strip().replace(',', '')
            parts = part.split()
            return parts[0] if parts else '0'
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        return numbers[-1] if numbers else '0'

def score_response(question, response, correct_answer, extracted_answer, benchmark):
    if benchmark == "strategyqa":
        is_correct = extracted_answer == correct_answer
    elif benchmark == "arc":
        is_correct = extracted_answer == correct_answer
    elif benchmark == "gsm8k":
        try:
            is_correct = float(str(extracted_answer).replace(',', '')) == float(str(correct_answer).replace(',', ''))
        except:
            is_correct = str(extracted_answer).strip() == str(correct_answer).strip()

    if not is_correct:
        return 0, False

    scoring_prompt = f"""
You are evaluating the quality of a reasoning response.

Question: {question}
Correct answer: {correct_answer}
Response to evaluate: {response}

The response has the correct answer. Score the quality of reasoning:
1 = Right answer but incomplete or weak reasoning
2 = Right answer with complete, well-constructed reasoning

Reply with only a single number: 1 or 2. No other text.
"""
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=10,
        messages=[{"role": "user", "content": scoring_prompt}]
    )
    return int(result.choices[0].message.content.strip()[0]), True

print("Loading datasets...")
strategyqa = load_dataset("ChilleD/StrategyQA", split="test")
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
gsm8k = load_dataset("openai/gsm8k", "main", split="test")

random.seed(42)
n = 150
benchmarks = {
    "strategyqa": {
        "data": [strategyqa[i] for i in random.sample(range(len(strategyqa)), n)],
    },
    "arc": {
        "data": [arc[i] for i in random.sample(range(len(arc)), n)],
    },
    "gsm8k": {
        "data": [gsm8k[i] for i in random.sample(range(len(gsm8k)), n)],
    }
}

def format_arc_question(item):
    choices = item['choices']
    formatted = item['question'] + "\n"
    for label, text in zip(choices['label'], choices['text']):
        formatted += f"{label}. {text}\n"
    return formatted

def format_gsm8k_answer(answer):
    if '####' in answer:
        return answer.split('####')[-1].strip().replace(',', '')
    return answer.strip()

print(f"Running experiment: {n} questions x 3 benchmarks x 5 conditions\n")

results = []
for benchmark_name, benchmark_data in benchmarks.items():
    for idx, item in enumerate(benchmark_data["data"]):
        if benchmark_name == "arc":
            question = format_arc_question(item)
            correct = item["answerKey"]
        elif benchmark_name == "gsm8k":
            question = item["question"]
            correct = format_gsm8k_answer(item["answer"])
        else:
            question = item["question"]
            correct = item["answer"]

        for condition in ["A", "B", "C", "D", "E"]:
            print(f"[{benchmark_name}][{idx+1}/{n}] Condition {condition}: {question[:50]}...")
            response = run_condition(question, condition, benchmark_name)
            extracted = extract_answer(response, benchmark_name)
            score, is_correct = score_response(question, response, correct, extracted, benchmark_name)

            results.append({
                "benchmark": benchmark_name,
                "question": question,
                "correct_answer": correct,
                "condition": condition,
                "extracted_answer": extracted,
                "correct": is_correct,
                "score": score,
                "response": response
            })

with open("real_experiment_openai.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["benchmark", "question", "correct_answer", "condition", "extracted_answer", "correct", "score", "response"])
    writer.writeheader()
    writer.writerows(results)

print("\n=== RESULTS SUMMARY ===")
for benchmark_name in ["strategyqa", "arc", "gsm8k"]:
    print(f"\n{benchmark_name.upper()}:")
    for condition in ["A", "B", "C", "D", "E"]:
        condition_results = [r for r in results if r["condition"] == condition and r["benchmark"] == benchmark_name]
        accuracy = sum(r["correct"] for r in condition_results) / len(condition_results)
        avg_score = sum(r["score"] for r in condition_results) / len(condition_results)
        print(f"  Condition {condition}: Accuracy={accuracy:.1%}, Avg Score={avg_score:.2f}")

print("\nDone! Results saved to real_experiment_openai.csv")