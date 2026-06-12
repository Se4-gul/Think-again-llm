import csv

def load(fp):
    with open(fp, newline='') as f:
        return list(csv.DictReader(f))

for fname, name in [("real_experiment_haiku_fixed.csv","HAIKU"),
                    ("real_experiment_openai_fixed.csv","GPT-4o-mini")]:
    r = load(fname)
    print(f"\n{name}")
    print(f"{'Cond':<6}{'StrategyQA':<14}{'ARC':<10}{'GSM8K':<10}")
    for c in ["A","B","C","D","E"]:
        row = f"{c:<6}"
        for b in ["strategyqa","arc","gsm8k"]:
            sub=[x for x in r if x["benchmark"]==b and x["condition"]==c]
            acc=sum(1 for x in sub if x["correct"]=="True")/len(sub)
            row += f"{acc:<14.1%}" if b=="strategyqa" else f"{acc:<10.1%}"
        print(row)
        