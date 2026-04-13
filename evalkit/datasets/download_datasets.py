"""One-time script to download all eval datasets to local disk."""

from datasets import load_dataset
import json, os, random

BASE = os.path.expanduser("~/.evalkit/datasets")
os.makedirs(BASE, exist_ok=True)

# 1. TruthfulQA
print("Downloading TruthfulQA...")
ds = load_dataset("truthful_qa", "generation", split="validation")
rows = [{"prompt": r["question"], "expected": r["best_answer"],
         "category": r["category"], "source": "truthfulqa"} for r in ds]
with open(f"{BASE}/truthfulqa.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"  ✓ {len(rows)} samples")

# 2. MMLU (abstract_algebra subset — small, representative)
print("Downloading MMLU...")
ds = load_dataset("cais/mmlu", "all", split="test")
choices_map = ["A", "B", "C", "D"]
rows = [{"prompt": r["question"],
         "expected": choices_map[r["answer"]],
         "choices": r["choices"],
         "category": r["subject"],
         "source": "mmlu"} for r in ds]
with open(f"{BASE}/mmlu.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"  ✓ {len(rows)} samples")

# 3. AdvBench (harmful prompts for red-teaming)
print("Downloading AdvBench...")
ds = load_dataset("walledai/AdvBench", split="train")
rows = [{"prompt": r["prompt"], "target": r["target"],
         "source": "advbench", "attack_type": "harmful_instruction"} for r in ds]
with open(f"{BASE}/advbench.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"  ✓ {len(rows)} samples")

print("\nAll datasets saved to ~/.evalkit/datasets/")
print("Total files:", os.listdir(BASE))