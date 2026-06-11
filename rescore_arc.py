import anthropic
import csv
import os
import re

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def extract_arc_answer(response):
    text = response.strip()
    
    # Priority 1: explicit final-answer phrases, take the LAST occurrence
    # catches "Answer: D", "The correct answer is C", "answer is actually D"
    patterns = [
        r'(?:correct answer is|answer is actually|final answer[:\s]+|answer[:\s]+)\*{0,2}\s*([ABCD])\b',
    ]
    all_matches = []
    for p in patterns:
        all_matches += [(m.start(), m.group(1)) for m in re.finditer(p, text, re.IGNORECASE)]
    if all_matches:
        # take the answer that appears LATEST in the response
        return max(all_matches, key=lambda x: x[0])[1].upper()
    
    # Priority 2: last non-empty line if it's just a bolded letter
    lines = [l for l in text.split('\n') if l.strip()]
    last = re.sub(r'[\*\_#\.]+', '', lines[-1]).strip().upper()
    if last in ['A', 'B', 'C', 'D']:
        return last
    
    # Priority 3: first line if it starts with a bare letter (answer-first format)
    first = re.sub(r'[\*\_#]+', '', lines[0]).strip().upper()
    for letter in ['A', 'B', 'C', 'D']:
        if first.startswith(letter):
            return letter
    
    # Last resort: last standalone letter anywhere
    matches = re.findall(r'\b([ABCD])\b', text)
    return matches[-1] if matches else None

def score_quality(question, response, correct_answer):
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
    result = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": scoring_prompt}]
    )
    return int(result.content[0].text.strip()[0])

# Load existing results
with open("real_experiment_haiku.csv", newline='') as f:
    reader = csv.DictReader(f)
    results = list(reader)

print("Re-extracting ARC answers...")
fixed = 0
rescored = 0

for row in results:
    if row["benchmark"] != "arc":
        continue
    
    new_extracted = extract_arc_answer(row["response"])
    old_extracted = row["extracted_answer"]
    
    if new_extracted != old_extracted:
        fixed += 1
        row["extracted_answer"] = new_extracted
        new_correct = new_extracted == row["correct_answer"]
        old_correct = row["correct"] == "True"
        row["correct"] = str(new_correct)
        
        if new_correct and not old_correct:
            # newly correct — needs a quality score
            print(f"Rescoring newly correct answer ({rescored+1})...")
            row["score"] = str(score_quality(row["question"], row["response"], row["correct_answer"]))
            rescored += 1
        elif not new_correct:
            row["score"] = "0"

print(f"\nFixed {fixed} extractions, rescored {rescored} responses")

# Save corrected file
with open("real_experiment_haiku_fixed.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["benchmark", "question", "correct_answer", "condition", "extracted_answer", "correct", "score", "response"])
    writer.writeheader()
    writer.writerows(results)

# Print corrected ARC summary
print("\n=== CORRECTED ARC SUMMARY ===")
for condition in ["A", "B", "C", "D", "E"]:
    subset = [r for r in results if r["condition"] == condition and r["benchmark"] == "arc"]
    accuracy = sum(1 for r in subset if r["correct"] == "True") / len(subset)
    avg_score = sum(int(r["score"]) for r in subset) / len(subset)
    print(f"Condition {condition}: Accuracy={accuracy:.1%}, Avg Score={avg_score:.2f}")

print("\nSaved to real_experiment_haiku_fixed.csv")
