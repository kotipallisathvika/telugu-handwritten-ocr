import argparse
import csv
from pathlib import Path


ap = argparse.ArgumentParser(description="Convert Telugu OCR labels to a project manifest")
ap.add_argument("--root", required=True, help="Dataset training directory")
ap.add_argument("--output", required=True, help="Output CSV manifest path")
args = ap.parse_args()
root = Path(args.root)
label_file = root / "train_gt.txt"
rows = []
with label_file.open("r", encoding="utf-8-sig") as f:
    for line_number, line in enumerate(f, 1):
        line = line.rstrip("\n\r")
        if not line.strip():
            continue
        try:
            image, text = line.split("\t", 1)
        except ValueError:
            raise ValueError(f"Invalid label at line {line_number}")
        image_path = root / image
        if image_path.exists():
            rows.append((str(image_path).replace("\\", "/"), text.strip()))
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "text"])
    writer.writerows(rows)
print(f"Wrote {len(rows):,} labeled images to {out}")
