import csv
with open("real_experiment_haiku_fixed.csv", newline='') as f:
    rows = [r for r in csv.DictReader(f)
            if r["benchmark"]=="arc" and r["condition"]=="C" and r["correct"]=="False"]
print(f"{len(rows)} incorrect C rows\n")
for r in rows[:6]:
    print("CORRECT:", r["correct_answer"], "| EXTRACTED:", r["extracted_answer"])
    print("END:", r["response"][-250:])
    print("="*70)