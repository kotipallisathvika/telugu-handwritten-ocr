import argparse
import json
import sys
from pathlib import Path

import cv2
from huggingface_hub import snapshot_download


def find_lines(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    # Ink mask; horizontal dilation connects characters within each handwritten line.
    mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    joined = cv2.dilate(mask, kernel, iterations=1)
    projection = (joined > 0).sum(axis=1)
    ranges, start = [], None
    for y, value in enumerate(projection):
        if value > 20 and start is None:
            start = y
        elif value <= 20 and start is not None:
            if y - start >= 5:
                ranges.append((max(0, start - 12), min(image.shape[0], y + 12)))
            start = None
    if start is not None:
        ranges.append((max(0, start - 12), image.shape[0]))
    # Merge overlaps and discard tiny noise bands.
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1] + 5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return image, [(a, b) for a, b in merged if b - a >= 20]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--output", default="outputs/line_ocr.json")
    args = ap.parse_args()
    image, lines = find_lines(args.image)
    repo = snapshot_download("bodhan-ai/indic-ocr")
    sys.path.insert(0, repo)
    from indic_ocr import IndicOCR
    parser = IndicOCR.from_pretrained(repo, device="cpu", table_format="markdown")
    out_dir = Path("outputs/lines")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, (top, bottom) in enumerate(lines, 1):
        crop_path = out_dir / f"line_{index:02d}.png"
        cv2.imwrite(str(crop_path), image[top:bottom, :])
        parsed = parser.parse(str(crop_path))
        results.append({"line": index, "image": str(crop_path), "text": parsed.get("markdown", "").strip()})
        print(f"Line {index}/{len(lines)}: {results[-1]['text']}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"image": args.image, "lines": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
