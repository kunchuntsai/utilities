#!/usr/bin/env python
"""
Harris Corner Response - Consume inputs and produce outputs.

Usage:
    python harris.py -i input_dir/ -o output_dir/
    python harris.py -i input_dir/ -o output_dir/ --block-size 2 --ksize 3 --k 0.04
"""

import argparse
import os
import json
import numpy as np
import cv2


def compute_harris_response(image, block_size=2, ksize=3, k=0.04):
    """
    Compute Harris corner response.

    Args:
        image: Grayscale input image
        block_size: Neighborhood size for corner detection
        ksize: Aperture parameter for Sobel operator
        k: Harris detector free parameter

    Returns:
        response: Harris response map (float32)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = np.float32(gray)

    # Compute Harris response
    response = cv2.cornerHarris(gray, block_size, ksize, k)

    return response


def load_frame_from_bin(filepath, width, height, channels, dtype=np.uint8):
    """Load frame from binary file."""
    data = np.fromfile(filepath, dtype=dtype)
    if channels == 1:
        return data.reshape((height, width))
    else:
        return data.reshape((height, width, channels))


def load_inputs(input_dir):
    """Load inputs from directory."""
    params_path = os.path.join(input_dir, 'parameters.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Load input image
    input_info = params['input']
    input_path = os.path.join(input_dir, input_info['file'])
    image = load_frame_from_bin(
        input_path,
        input_info['src_width'],
        input_info['src_height'],
        input_info['src_channels']
    )

    return image, params


def save_outputs(image, response, output_dir, harris_params):
    """Save outputs to directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save binary files
    input_path = os.path.join(output_dir, 'input.bin')
    response_path = os.path.join(output_dir, 'response.bin')

    image.tofile(input_path)
    response.tofile(response_path)

    # Get dimensions
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    dtype_size = np.dtype(image.dtype).itemsize
    response_dtype_size = 4  # float32

    # Save parameters as JSON
    params = {
        "input": {
            "file": "input.bin",
            "src_width": w,
            "src_height": h,
            "src_channels": channels,
            "src_stride": "{0} * {1}".format(w * channels, dtype_size)
        },
        "response": {
            "file": "response.bin",
            "dst_width": w,
            "dst_height": h,
            "dst_channels": 1,
            "dst_stride": "{0} * {1}".format(w, response_dtype_size)
        },
        "harris_params": harris_params
    }

    params_path = os.path.join(output_dir, 'parameters.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

    print("Saved outputs:")
    print("  - input.bin ({0} bytes)".format(os.path.getsize(input_path)))
    print("  - response.bin ({0} bytes)".format(os.path.getsize(response_path)))
    print("  - parameters.json")


def main():
    parser = argparse.ArgumentParser(description='Harris Corner Response')
    parser.add_argument('-i', '--input', required=True, help='Input directory with input.bin, parameters.json')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    # Harris parameters
    parser.add_argument('--block-size', type=int, default=2, help='Neighborhood size (default: 2)')
    parser.add_argument('--ksize', type=int, default=3, help='Aperture parameter for Sobel (default: 3)')
    parser.add_argument('--k', type=float, default=0.04, help='Harris detector free parameter (default: 0.04)')

    args = parser.parse_args()

    # Harris parameters dict
    harris_params = {
        "block_size": args.block_size,
        "ksize": args.ksize,
        "k": args.k
    }

    # Load inputs
    print("Loading inputs from: {0}".format(args.input))
    image, input_params = load_inputs(args.input)
    print("  image: {0}".format(image.shape))

    # Compute Harris response
    print("\nComputing Harris response...")
    print("  block_size={0}, ksize={1}, k={2}".format(
        args.block_size, args.ksize, args.k))

    response = compute_harris_response(
        image,
        block_size=args.block_size,
        ksize=args.ksize,
        k=args.k
    )

    # Statistics
    print("\nResponse statistics:")
    print("  min={0:.6f}, max={1:.6f}".format(response.min(), response.max()))
    print("  mean={0:.6f}, std={1:.6f}".format(response.mean(), response.std()))

    # Count corners (response > threshold)
    threshold = 0.01 * response.max()
    corners = np.sum(response > threshold)
    print("  Corners (response > {0:.6f}): {1}".format(threshold, corners))

    # Save outputs
    print("\nSaving outputs to: {0}".format(args.output))
    save_outputs(image, response, args.output, harris_params)

    return 0


if __name__ == '__main__':
    exit(main())
