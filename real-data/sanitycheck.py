import pandas as pd
from pathlib import Path
root = Path.cwd()
while not (root / "real-data").exists() and root != root.parent:
    root = root.parent
csv = sorted((root / "real-data").rglob("dataset_all.csv"))[0]
df = pd.read_csv(csv); df["text"] = df["text"].fillna("")
NAMES = {0: "Malfunction", 1: "Basic injury", 2: "Serious injury", 3: "Death"}
print("rows:", len(df))
print("class counts:", {NAMES[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()})
print("contains '||':         %.1f%%" % (df["text"].str.contains(r"\|\|").mean() * 100))
print("contains 510(k) K-num: %.1f%%" % (df["text"].str.contains(r"\bK\d{6}\b").mean() * 100))
print("sample:", df["text"].iloc[0][:300])
