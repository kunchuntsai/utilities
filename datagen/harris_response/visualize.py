#!/usr/bin/env python
"""
Visualize Harris corner response inputs and outputs.

Usage:
    python visualize.py -i output_dir/
    python visualize.py -i output_dir/ --save result.png
"""

import argparse
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_frame_from_bin(filepath, width, height, channels, dtype=np.uint8):
    """Load frame from binary file."""
    data = np.fromfile(filepath, dtype=dtype)
    if channels == 1:
        return data.reshape((height, width))
    else:
        return data.reshape((height, width, channels))


def load_data(input_dir):
    """Load all data from directory."""
    params_path = os.path.join(input_dir, 'parameters.json')
    with open(params_path, 'r') as f:
        params = json.load(f)

    data = {'params': params}

    # Load input image
    if 'input' in params:
        info = params['input']
        path = os.path.join(input_dir, info['file'])
        if os.path.exists(path):
            data['input'] = load_frame_from_bin(
                path, info['src_width'], info['src_height'], info['src_channels']
            )

    # Load response
    if 'response' in params:
        info = params['response']
        path = os.path.join(input_dir, info['file'])
        if os.path.exists(path):
            data['response'] = load_frame_from_bin(
                path, info['dst_width'], info['dst_height'], 1, dtype=np.float32
            )

    return data


def visualize_input_response(data, save_path=None):
    """Visualize input image and Harris response side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if 'input' in data:
        img = data['input']
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

    if 'response' in data:
        response = data['response']
        im = axes[1].imshow(response, cmap='hot')
        axes[1].set_title('Harris Response')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], label='Response value')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_response.png'), dpi=150, bbox_inches='tight')

    return fig


def visualize_corners(data, threshold_ratio=0.01, save_path=None):
    """Visualize detected corners overlaid on image."""
    if 'input' not in data or 'response' not in data:
        print("Missing data for corner visualization")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Show image
    img = data['input']
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax.imshow(img_rgb)

    # Find corners above threshold
    response = data['response']
    threshold = threshold_ratio * response.max()
    corners_y, corners_x = np.where(response > threshold)

    # Plot corners
    ax.scatter(corners_x, corners_y, c='lime', s=10, marker='o', alpha=0.7)

    ax.set_title('Detected Corners (threshold={0:.2%} of max)'.format(threshold_ratio))
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_corners.png'), dpi=150, bbox_inches='tight')

    return fig


def visualize_response_histogram(data, save_path=None):
    """Visualize histogram of Harris response values."""
    if 'response' not in data:
        print("No response data")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    response = data['response'].flatten()
    # Remove very small values for better visualization
    response_nonzero = response[np.abs(response) > 1e-10]

    ax.hist(response_nonzero, bins=100, color='steelblue', alpha=0.7)
    ax.set_xlabel('Response Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Harris Response Histogram')
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_histogram.png'), dpi=150, bbox_inches='tight')

    return fig


def print_statistics(data):
    """Print response statistics."""
    print("\nStatistics:")

    if 'response' in data:
        response = data['response']
        print("  Response: min={0:.6f}, max={1:.6f}".format(response.min(), response.max()))
        print("            mean={0:.6f}, std={1:.6f}".format(response.mean(), response.std()))

        # Count corners at different thresholds
        for ratio in [0.01, 0.05, 0.1]:
            threshold = ratio * response.max()
            count = np.sum(response > threshold)
            print("  Corners (>{0:.0%} of max): {1}".format(ratio, count))

    if 'harris_params' in data.get('params', {}):
        params = data['params']['harris_params']
        print("\n  Harris Parameters:")
        print("    block_size: {0}".format(params.get('block_size')))
        print("    ksize: {0}".format(params.get('ksize')))
        print("    k: {0}".format(params.get('k')))


def main():
    parser = argparse.ArgumentParser(description='Visualize Harris corner response')
    parser.add_argument('-i', '--input', required=True, help='Input directory with binary files and parameters.json')
    parser.add_argument('--save', type=str, help='Save visualizations to file (e.g., result.png)')
    parser.add_argument('--threshold', type=float, default=0.01, help='Corner threshold ratio (default: 0.01)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')

    args = parser.parse_args()

    # Load data
    print("Loading data from: {0}".format(args.input))
    data = load_data(args.input)

    # Print what was loaded
    print("Loaded:")
    for key in ['input', 'response']:
        if key in data:
            print("  - {0}: {1}".format(key, data[key].shape))

    # Print statistics
    print_statistics(data)

    # Create visualizations
    visualize_input_response(data, args.save)
    visualize_corners(data, args.threshold, args.save)
    visualize_response_histogram(data, args.save)

    if not args.no_show:
        plt.show()

    return 0


if __name__ == '__main__':
    exit(main())
