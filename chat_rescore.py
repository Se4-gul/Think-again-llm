import csv
with open("real_experiment_openai_fixed.csv", newline='') as f:
    rows = [r for r in csv.DictReader(f)
            if r["benchmark"]=="arc" and "airplane" in r["question"].lower()]
for r in rows:
    print("COND:", r["condition"], "| CORRECT:", r["correct_answer"], "| EXTRACTED:", r["extracted_answer"], "| CORRECT?:", r["correct"])