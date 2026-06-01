import anthropic
import csv

client = anthropic.Anthropic(api_key="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")

def run_condition(question, condition):
    if condition == "A":
        prompt = f"Let's think step by step. {question}"
    elif condition == "B":
        prompt = f"Let's think step by step. After every 2 steps, stop and ask yourself: 'Is this line of reasoning correct? Am I missing another angle?' Then continue or redirect. {question}"
    elif condition == "C":
        prompt = f"Let's think step by step. At a random point mid-reasoning, pause and reflect before continuing. {question}"
    elif condition == "D":
        prompt = f"Let's think step by step. Before giving your final answer, review your reasoning and check for errors. {question}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def score_response(question, response, correct_answer):
    scoring_prompt = f"""
You are evaluating the quality of a reasoning response.

Question: {question}
Correct answer: {correct_answer}
Response to evaluate: {response}

Score the response on this scale:
0 = Wrong answer
1 = Right answer but incomplete or weak reasoning
2 = Right answer with complete, well-constructed reasoning

Reply with only a single number: 0, 1, or 2.
"""
    result = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": scoring_prompt}]
    )
    return int(result.content[0].text.strip()[0])

pilot_questions = [
    {
        "question": "If a store reduces a price by 20% and then increases it by 20%, is the final price the same as the original?",
        "answer": "No, the final price is 96% of the original"
    },
    {
        "question": "Would a penguin sink in the Dead Sea?",
        "answer": "No, it would float"
    },
    {
        "question": "Can a person born blind dream in images?",
        "answer": "No, they typically dream in other senses"
    }
]

results = []
for item in pilot_questions:
    for condition in ["A", "B", "C", "D"]:
        print(f"Running condition {condition} on: {item['question'][:50]}...")
        response = run_condition(item["question"], condition)
        score = score_response(item["question"], response, item["answer"])
        results.append({
            "question": item["question"],
            "condition": condition,
            "response": response,
            "score": score
        })

with open("pilot_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "condition", "response", "score"])
    writer.writeheader()
    writer.writerows(results)

print("Done! Results saved to pilot_results.csv")