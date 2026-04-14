from pathlib import Path
from datasets import load_from_disk

p = Path("data/hatexplain")
print("Exists:", p.exists())
if p.exists():
    ds = load_from_disk(str(p))
    print("Splits:", list(ds.keys()))
    for s in ds:
        print(s, "rows =", len(ds[s]))
        print("Example:", ds[s][0])
        break
else:
    print("Dataset folder not found. Need to download.")
