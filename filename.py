import csv

with open("test_run.csv") as f:
    reader = csv.DictReader(f)
    results = list(reader)

for benchmark in ["strategyqa", "arc", "gsm8k"]:
    for condition in ["A", "B", "C", "D", "E"]:
        subset = [r for r in results if r["condition"] == condition and r["benchmark"] == benchmark]
        print(f"{benchmark} {condition}: {len(subset)}")