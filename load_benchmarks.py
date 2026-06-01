from datasets import load_dataset

dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

for i in range(3):
    print(dataset[i])
    print("---")