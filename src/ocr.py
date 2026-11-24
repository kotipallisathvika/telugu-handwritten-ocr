import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def load_parser():
    repo = snapshot_download("bodhan-ai/indic-ocr")
    sys.path.insert(0, repo)
    from indic_ocr import IndicOCR
    # Force CPU for machines where the installed PyTorch has no CUDA support.
    # Change to device="cuda" only after installing a CUDA-enabled PyTorch build.
    return IndicOCR.from_pretrained(repo, device="cpu", table_format="markdown")


def main():
    # Windows PowerShell may default to cp1252; OCR output is Telugu Unicode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run Bodhan Indic OCR locally")
    parser.add_argument("--image", required=True, help="Path to PNG/JPG/PDF page image")
    parser.add_argument("--json", dest="json_path", help="Optional JSON output path")
    parser.add_argument("--table-format", choices=["html", "markdown"], default="markdown")
    args = parser.parse_args()
    image = Path(args.image)
    if not image.exists():
        raise FileNotFoundError(f"Input not found: {image}")
    ocr = load_parser()
    result = ocr.parse(str(image))
    result["markdown"] = result.get("markdown", "").replace("▁", " ")
    for block in result.get("blocks", []):
        if isinstance(block.get("text"), str):
            block["text"] = block["text"].replace("▁", " ")
    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.get("markdown", ""))
    if args.json_path:
        print(f"\nSaved JSON: {args.json_path}")


if __name__ == "__main__":
    main()
