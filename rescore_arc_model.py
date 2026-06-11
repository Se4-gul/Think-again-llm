import anthropic, csv, os
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def extract_via_model(response):
    prompt = f"""Below is a response to a multiple-choice question with options A, B, C, D.
Read it and identify the FINAL answer the response commits to (accounting for any self-corrections or revisions).
Reply with ONLY the single letter A, B, C, or D. If no answer is given, reply N.

Response:
{response}"""
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{"role":"user","content":prompt}]
    )
    letter = r.content[0].text.strip().upper()[:1]
    return letter if letter in "ABCD" else None

with open("real_experiment_openai.csv", newline='') as f:
    results = list(csv.DictReader(f))

arc_rows = [r for r in results if r["benchmark"]=="arc"]
print(f"Re-extracting {len(arc_rows)} ARC answers via model...")

for i, row in enumerate(arc_rows):
    new = extract_via_model(row["response"])
    row["extracted_answer"] = new if new else "N"
    row["correct"] = str(new == row["correct_answer"])
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(arc_rows)}")

# rescore quality only for correct answers (reuse existing score if it was already right; rescore newly-correct)
# simplest: set wrong=0, and for correct ones keep existing score if >0 else mark for rescore
for row in arc_rows:
    if row["correct"] == "False":
        row["score"] = "0"

with open("real_experiment_openai_fixed.csv","w",newline='') as f:
    w = csv.DictWriter(f, fieldnames=results[0].keys())
    w.writeheader(); w.writerows(results)

print("\n=== ARC SUMMARY (model-extracted) ===")
for c in ["A","B","C","D","E"]:
    sub=[r for r in arc_rows if r["condition"]==c]
    acc=sum(1 for r in sub if r["correct"]=="True")/len(sub)
    print(f"Condition {c}: Accuracy={acc:.1%}")