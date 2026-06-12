import csv

def load(filepath):
    with open(filepath, newline='') as f:
        return list(csv.DictReader(f))

results = load("real_experiment_haiku_fixed.csv")

# Build per-question correctness by condition for StrategyQA
sqa = [r for r in results if r["benchmark"] == "strategyqa"]
by_q = {}
for r in sqa:
    by_q.setdefault(r["question"], {})[r["condition"]] = r

# Find cases where A got it right but B got it wrong (the failures we care about)
print("=== A correct, B wrong (scheduled interruption breaks it) ===\n")
count = 0
for q, conds in by_q.items():
    if "A" in conds and "B" in conds:
        a_right = conds["A"]["correct"] == "True"
        b_right = conds["B"]["correct"] == "True"
        if a_right and not b_right:
            count += 1
            print(f"[{count}] Q: {q}")
            print(f"    Correct answer: {conds['A']['correct_answer']}")
            print(f"    --- B's response (got it WRONG) ---")
            print(f"    {conds['B']['response'][:700]}")
            print(f"    {'='*70}\n")
            if count >= 6:
                break