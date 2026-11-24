import argparse
import json
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--input-dir", required=True)
ap.add_argument("--output", default="outputs/batch_ocr.json")
args = ap.parse_args()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
repo = snapshot_download("bodhan-ai/indic-ocr")
sys.path.insert(0, repo)
from indic_ocr import IndicOCR
parser = IndicOCR.from_pretrained(repo, device="cpu", table_format="markdown", max_new_tokens=512)
files = sorted(p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
results = []
for i, path in enumerate(files, 1):
    print(f"Processing {i}/{len(files)}: {path.name}")
    result = parser.parse(str(path))
    results.append({"image": str(path), "text": result.get("markdown", ""), "blocks": result.get("blocks", [])})
out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved: {out}")
