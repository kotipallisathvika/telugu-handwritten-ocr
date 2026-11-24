import argparse
from pathlib import Path
import cv2


def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points = cv2.findNonZero(255 - gray)
    if points is None:
        return image
    angle = cv2.minAreaRect(points)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    image = cv2.imread(args.input)
    if image is None:
        raise FileNotFoundError(args.input)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 7, 7, 21)
    clean = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    clean = deskew(clean)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 11)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), binary)
    print(f"Saved preprocessed image: {output}")


if __name__ == "__main__":
    main()
