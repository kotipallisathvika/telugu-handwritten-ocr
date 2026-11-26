"""Evaluate local Telugu OCR against a JSON ground-truth file."""
# The downloaded OCR package has no type stubs; these are type-checker-only
# warnings at the dynamic model boundary, not runtime errors.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

from ocr import load_parser


def normalize(text: str) -> str:
    return " ".join(text.replace("\u2581", " ").split())


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: str, prediction: str, unit: str) -> float:
    reference_units = list(reference) if unit == "char" else reference.split()
    prediction_units = list(prediction) if unit == "char" else prediction.split()
    if not reference_units:
        return 0.0 if not prediction_units else 1.0
    return edit_distance(reference_units, prediction_units) / len(reference_units)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Telugu OCR CER, WER and latency")
    parser.add_argument("--images", required=True, help="Folder containing test images")
    parser.add_argument("--truth", required=True, help="JSON mapping image filename to correct text")
    parser.add_argument("--output", default="metrics.json", help="Report path")
    args = parser.parse_args()

    image_dir = Path(args.images)
    truth: dict[str, str] = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    ocr = cast(Any, load_parser())
    rows: list[dict[str, Any]] = []

    for filename, expected in truth.items():
        image_path = image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in truth file not found: {image_path}")
        start = time.perf_counter()
        result = ocr.parse(str(image_path))
        elapsed = time.perf_counter() - start
        predicted = normalize(str(result.get("markdown", "")))
        expected = normalize(expected)
        rows.append({
            "image": filename,
            "cer": error_rate(expected, predicted, "char"),
            "wer": error_rate(expected, predicted, "word"),
            "exact_match": predicted == expected,
            "seconds": round(elapsed, 3),
        })

    report: dict[str, Any] = {
        "samples": len(rows),
        "average_cer": sum(row["cer"] for row in rows) / len(rows) if rows else 0.0,
        "average_wer": sum(row["wer"] for row in rows) / len(rows) if rows else 0.0,
        "exact_match_accuracy": sum(row["exact_match"] for row in rows) / len(rows) if rows else 0.0,
        "average_seconds": sum(row["seconds"] for row in rows) / len(rows) if rows else 0.0,
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
