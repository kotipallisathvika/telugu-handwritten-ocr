import argparse
import csv
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/iiit_hw_telugu_train.csv")
ap.add_argument("--limit", type=int, default=5)
ap.add_argument("--output", default="outputs/bodhan_benchmark.json")
args = ap.parse_args()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
with open(args.manifest, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))[:args.limit]
repo = snapshot_download("bodhan-ai/indic-ocr")
sys.path.insert(0, repo)
from indic_ocr import IndicOCR
ocr = IndicOCR.from_pretrained(repo, device="cpu", table_format="markdown", max_new_tokens=128)
results = []
total_chars = total_char_errors = total_words = total_word_errors = 0
for index, row in enumerate(rows, 1):
    predicted = ocr.parse(row["image"]).get("markdown", "").replace("▁", " ").strip()
    reference = row["text"].strip()
    char_errors = edit_distance(reference, predicted)
    ref_words, pred_words = reference.split(), predicted.split()
    word_errors = edit_distance(ref_words, pred_words)
    total_chars += len(reference); total_char_errors += char_errors
    total_words += len(ref_words); total_word_errors += word_errors
    results.append({"image": row["image"], "reference": reference, "prediction": predicted, "char_errors": char_errors})
    print(f"{index}/{len(rows)}: {reference} -> {predicted}")
summary = {"samples": len(rows), "cer": total_char_errors / max(1, total_chars), "wer": total_word_errors / max(1, total_words), "results": results}
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"CER={summary['cer']:.4f} WER={summary['wer']:.4f}")
print(f"Saved: {out}")
