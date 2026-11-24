import argparse
from pathlib import Path
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", default="outputs/manual_lines_v2/line_08_tight.png")
args = ap.parse_args()
image = cv2.imread(args.input)
if image is None:
    raise FileNotFoundError(args.input)
h = image.shape[0]
# Tight band around row 8 in the supplied 707x473 sample, scaled by height.
top, bottom = int(h * 0.740), int(h * 0.930)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), image[top:bottom, :])
print(f"Saved: {out} ({top}:{bottom})")
