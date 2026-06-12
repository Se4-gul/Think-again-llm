import csv
from itertools import combinations

def load(filepath):
    with open(filepath, newline='') as f:
        return list(csv.DictReader(f))

def mcnemar(results, benchmark, cond1, cond2):
    # build per-question correctness for each condition
    def correctness_map(cond):
        return {r["question"]: (r["correct"] == "True")
                for r in results if r["benchmark"] == benchmark and r["condition"] == cond}
    m1, m2 = correctness_map(cond1), correctness_map(cond2)
    questions = set(m1) & set(m2)
    # discordant pairs
    b = sum(1 for q in questions if m1[q] and not m2[q])  # cond1 right, cond2 wrong
    c = sum(1 for q in questions if not m1[q] and m2[q])  # cond1 wrong, cond2 right
    n = b + c
    if n == 0:
        return None
    # exact binomial p-value (two-sided), no scipy needed
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k+1)) * (0.5**n) * 2
    p = min(p, 1.0)
    return b, c, p

for model_file, name in [("real_experiment_haiku_fixed.csv","HAIKU"),
                         ("real_experiment_openai_fixed.csv","GPT-4o-mini")]:
    results = load(model_file)
    for benchmark in ["strategyqa", "arc", "gsm8k"]:
        print(f"\n{'='*55}\n{name} — McNemar vs baseline A ({benchmark})\n{'='*55}")
        for cond in ["B","C","D","E"]:
            out = mcnemar(results, benchmark, "A", cond)
            if out:
                b, c, p = out
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"A vs {cond}: A-only-right={b}, {cond}-only-right={c}, p={p:.3f} {sig}")
            else:
                print(f"A vs {cond}: no discordant pairs")