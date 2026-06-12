import csv
for f in ["real_experiment_haiku_fixed.csv", "real_experiment_haiku_fixed_v2.csv"]:
    rows = [r for r in csv.DictReader(open(f)) if r["benchmark"]=="strategyqa" and r["condition"]=="C"]
    acc = sum(1 for r in rows if r["correct"]=="True")/len(rows)
    print(f"{f}: C StrategyQA = {acc:.1%}")