import argparse
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_NAMES = [
    "IndieFlower",
    "GloriaHallelujah",
    "PatrickHand",
    "JustAnotherHand",
    "Caveat",
    "CoveredByYourGrace",
    "ShadowsIntoLight",
    "Handlee",
    "ReenieBeanie",
    "HomemadeApple",
]

SAMPLE_TEXTS = [
    "The mitochondria is the powerhouse of the cell.",
    "Water boils at 100 degrees Celsius.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Gravity pulls objects toward the Earth.",
    "A triangle has three sides and three angles.",
    "The capital of France is Paris.",
    "Plants need water, light, and carbon dioxide.",
    "The human heart pumps blood through the body.",
    "Sound travels faster in solids than in gases.",
    "The Earth orbits the sun once every year.",
]


def find_font_file(font_name: str) -> str | None:
    """Find a local font file by name, if installed on the machine."""
    windows_font_dir = Path("C:/Windows/Fonts")
    if not windows_font_dir.exists():
        return None

    target = font_name.replace(" ", "").lower()
    for file in windows_font_dir.rglob("*.ttf"):
        if target in file.stem.replace(" ", "").lower():
            return str(file)
    return None


def elastic_distortion(image: np.ndarray, alpha: float = 200, sigma: float = 20) -> np.ndarray:
    """Apply elastic distortion to an image.

    This uses a displacement field (dx, dy) smoothed by Gaussian blur.
    alpha controls intensity and sigma controls smoothing.
    """
    shape = image.shape[:2]
    dx = cv2.GaussianBlur(
        (np.random.rand(*shape).astype(np.float32) * 2 - 1),
        (0, 0),
        sigma,
    )
    dy = cv2.GaussianBlur(
        (np.random.rand(*shape).astype(np.float32) * 2 - 1),
        (0, 0),
        sigma,
    )
    dx *= alpha
    dy *= alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return distorted


def render_text_image(text: str, font_path: str | None, image_size=(280, 80)) -> np.ndarray:
    """Create a structured handwriting-style text image with PIL."""
    img = Image.new("L", image_size, color=255)
    draw = ImageDraw.Draw(img)

    if font_path and Path(font_path).exists():
        try:
            font = ImageFont.truetype(font_path, size=24)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(text, font=font)

    x = max(10, (image_size[0] - text_width) // 2)
    y = max(10, (image_size[1] - text_height) // 2)
    draw.text((x, y), text, fill=0, font=font)

    return np.array(img)


def random_transform(image: np.ndarray) -> np.ndarray:
    """Apply a sequence of image transforms to simulate messy handwriting."""
    angle = random.uniform(-5, 5)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=255)

    blurred = cv2.GaussianBlur(rotated, (3, 3), 0)
    distorted = elastic_distortion(blurred, alpha=200, sigma=20)
    return distorted


def generate_synthetic_dataset(output_dir: str, count: int = 1000) -> None:
    """Generate synthetic handwriting images into output_dir."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    font_files = [find_font_file(name) for name in FONT_NAMES]
    font_files = [f for f in font_files if f is not None]
    if not font_files:
        print("Warning: No handwriting fonts found locally. Falling back to default font.")

    for i in range(1, count + 1):
        text = random.choice(SAMPLE_TEXTS)
        font_path = random.choice(font_files) if font_files else None
        image = render_text_image(text, font_path)
        image = random_transform(image)

        file_name = output_path / f"synthetic_{i:04d}.png"
        cv2.imwrite(str(file_name), image)

        if i % 100 == 0:
            print(f"Generated {i}/{count} images")

    print(f"Dataset generation complete: {count} images in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic handwriting images.")
    parser.add_argument("--output", default="../data_synthetic", help="Output directory for generated images")
    parser.add_argument("--count", type=int, default=1000, help="Number of images to generate")
    args = parser.parse_args()
    generate_synthetic_dataset(args.output, args.count)
