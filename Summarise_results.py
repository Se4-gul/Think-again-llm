import csv

def summarise(filepath, model_name):
    results = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    print(f"\n=== {model_name} ===")
    for benchmark in ["strategyqa", "arc", "gsm8k"]:
        print(f"\n{benchmark.upper()}:")
        for condition in ["A", "B", "C", "D", "E"]:
            subset = [r for r in results if r["condition"] == condition and r["benchmark"] == benchmark]
            if not subset:
                continue
            accuracy = sum(1 for r in subset if r["correct"] == "True") / len(subset)
            avg_score = sum(int(r["score"]) for r in subset) / len(subset)
            print(f"  Condition {condition}: Accuracy={accuracy:.1%}, Avg Score={avg_score:.2f}")

summarise("real_experiment_haiku_fixed.csv", "Claude Haiku")
summarise("real_experiment_openai_fixed.csv", "GPT-4o-mini")