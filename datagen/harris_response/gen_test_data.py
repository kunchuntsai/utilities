#!/usr/bin/env python
"""
Generate test data for Harris corner response.

Usage:
    python gen_test_data.py -i input.jpg -o output_dir/
    python gen_test_data.py --synthetic -o output_dir/ --pattern checkerboard
"""

import argparse
import os
import json
import numpy as np
import cv2


def create_synthetic_image(width=640, height=480, pattern='checkerboard'):
    """
    Create a synthetic test image with corners.

    Args:
        width: Image width
        height: Image height
        pattern: 'checkerboard', 'squares', 'corners'

    Returns:
        Grayscale image with corner features
    """
    if pattern == 'checkerboard':
        block_size = 40
        image = np.zeros((height, width), dtype=np.uint8)
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    image[i:i+block_size, j:j+block_size] = 255

    elif pattern == 'squares':
        image = np.ones((height, width), dtype=np.uint8) * 128
        # Add random squares
        for _ in range(30):
            x = np.random.randint(50, width - 100)
            y = np.random.randint(50, height - 100)
            size = np.random.randint(30, 80)
            color = np.random.choice([0, 255])
            cv2.rectangle(image, (x, y), (x + size, y + size), int(color), -1)

    elif pattern == 'corners':
        image = np.ones((height, width), dtype=np.uint8) * 128
        # Add L-shapes and T-shapes for strong corners
        for _ in range(20):
            x = np.random.randint(50, width - 100)
            y = np.random.randint(50, height - 100)
            color = np.random.choice([0, 255])
            # L-shape
            cv2.rectangle(image, (x, y), (x + 40, y + 10), int(color), -1)
            cv2.rectangle(image, (x, y), (x + 10, y + 40), int(color), -1)

    else:
        image = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    return image


def get_dtype_size(dtype):
    """Get the byte size of a numpy dtype."""
    return np.dtype(dtype).itemsize


def save_outputs(image, output_dir):
    """Save test data outputs."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Save binary frame
    input_path = os.path.join(output_dir, 'input.bin')
    gray.tofile(input_path)

    # Get frame info
    h, w = gray.shape
    dtype_size = get_dtype_size(gray.dtype)

    # Save parameters as JSON
    params = {
        "input": {
            "file": "input.bin",
            "src_width": w,
            "src_height": h,
            "src_channels": 1,
            "src_stride": "{0} * {1}".format(w, dtype_size)
        }
    }

    params_path = os.path.join(output_dir, 'parameters.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

    print("Generated test data:")
    print("  - input.bin ({0} bytes)".format(os.path.getsize(input_path)))
    print("  - parameters.json")
    print("  Image size: {0}x{1}".format(w, h))


def main():
    parser = argparse.ArgumentParser(description='Generate test data for Harris corner response')
    parser.add_argument('-i', '--input', help='Input image path')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic image')
    parser.add_argument('--pattern', default='checkerboard',
                        choices=['checkerboard', 'squares', 'corners'],
                        help='Synthetic pattern type')
    parser.add_argument('--width', type=int, default=640, help='Synthetic image width')
    parser.add_argument('--height', type=int, default=480, help='Synthetic image height')

    args = parser.parse_args()

    if args.synthetic:
        print("Creating synthetic image: {0}x{1}, pattern={2}".format(
            args.width, args.height, args.pattern))
        image = create_synthetic_image(args.width, args.height, args.pattern)
    elif args.input:
        print("Loading image: {0}".format(args.input))
        image = cv2.imread(args.input)
        if image is None:
            print("Error: Could not load image: {0}".format(args.input))
            return 1
    else:
        print("Error: Either --input or --synthetic must be specified")
        return 1

    save_outputs(image, args.output)
    return 0


if __name__ == '__main__':
    exit(main())
