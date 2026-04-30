import json
import os

def clean_widgets(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if "widgets" in nb.get("metadata", {}):
        nb["metadata"].pop("widgets")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".ipynb"):
            clean_widgets(os.path.join(root, file))

print("Notebook widget metadata cleaned.")