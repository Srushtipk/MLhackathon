import os
from pathlib import Path

import cv2
import pytesseract

# Automatically configure Tesseract path for Windows
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

for tesseract_path in TESSERACT_PATHS:
    if Path(tesseract_path).exists():
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        break


def preprocess_image(image_path: str) -> any:
    """Preprocess an image for OCR using Otsu’s thresholding and 1.5x rescaling."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    resized = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(resized, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_text(image_path: str, config: str = "--oem 3 --psm 3") -> str:
    """Extract text from a handwritten image using Tesseract OCR."""
    processed = preprocess_image(image_path)
    text = pytesseract.image_to_string(processed, config=config)
    return text.strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Tesseract OCR on a handwriting image.")
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    extracted = extract_text(args.image_path)
    print("OCR Text:")
    print(extracted)
