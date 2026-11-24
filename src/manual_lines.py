import argparse
from pathlib import Path
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", default="outputs/manual_lines")
args = ap.parse_args()
image = cv2.imread(args.input)
if image is None:
    raise FileNotFoundError(args.input)
height = image.shape[0]
# Nine writing rows visible in the supplied 707x473 sample; scaled by image height.
bands = [(0.00, .105), (.105, .215), (.215, .325), (.325, .435),
         (.435, .545), (.545, .655), (.655, .765), (.765, .875), (.875, 1.00)]
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
for i, (a, b) in enumerate(bands, 1):
    top, bottom = max(0, int(a * height)), min(height, int(b * height))
    cv2.imwrite(str(out / f"line_{i:02d}.png"), image[top:bottom, :])
print(f"Created {len(bands)} line crops in {out}")
