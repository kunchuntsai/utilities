#!/usr/bin/env python
"""
Visualize Lucas-Kanade optical flow inputs and outputs.

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

    # Load prev_frame
    if 'prev_frame' in params:
        info = params['prev_frame']
        path = os.path.join(input_dir, info['file'])
        if os.path.exists(path):
            data['prev_frame'] = load_frame_from_bin(
                path, info['src_width'], info['src_height'], info['src_channels']
            )

    # Load curr_frame
    if 'curr_frame' in params:
        info = params['curr_frame']
        path = os.path.join(input_dir, info['file'])
        if os.path.exists(path):
            data['curr_frame'] = load_frame_from_bin(
                path, info['src_width'], info['src_height'], info['src_channels']
            )

    # Load flow_x
    if 'flow_x' in params:
        info = params['flow_x']
        path = os.path.join(input_dir, info['file'])
        if os.path.exists(path):
            data['flow_x'] = load_frame_from_bin(
                path, info['dst_width'], info['dst_height'], 1, dtype=np.float32
            )

    # Load flow_y
    if 'flow_y' in params:
        info = params['flow_y']
        path = os.path.join(input_dir, info['file'])
        if os.path.exists(path):
            data['flow_y'] = load_frame_from_bin(
                path, info['dst_width'], info['dst_height'], 1, dtype=np.float32
            )

    return data


def visualize_frames(data, save_path=None):
    """Visualize prev_frame and curr_frame side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if 'prev_frame' in data:
        img = data['prev_frame']
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[0].set_title('Previous Frame')
        axes[0].axis('off')

    if 'curr_frame' in data:
        img = data['curr_frame']
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        axes[1].set_title('Current Frame')
        axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_frames.png'), dpi=150, bbox_inches='tight')

    return fig


def visualize_flow(data, save_path=None):
    """Visualize flow_x and flow_y."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if 'flow_x' in data:
        flow_x = data['flow_x']
        # Only show non-zero values for better visualization
        vmax = max(abs(flow_x.min()), abs(flow_x.max()))
        if vmax == 0:
            vmax = 1
        im1 = axes[0].imshow(flow_x, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0].set_title('Flow X (Horizontal)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], label='Displacement (pixels)')

    if 'flow_y' in data:
        flow_y = data['flow_y']
        vmax = max(abs(flow_y.min()), abs(flow_y.max()))
        if vmax == 0:
            vmax = 1
        im2 = axes[1].imshow(flow_y, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1].set_title('Flow Y (Vertical)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], label='Displacement (pixels)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_flow.png'), dpi=150, bbox_inches='tight')

    return fig


def visualize_flow_overlay(data, save_path=None, step=20):
    """Visualize flow as arrows overlaid on the image."""
    if 'prev_frame' not in data or 'flow_x' not in data or 'flow_y' not in data:
        print("Missing data for flow overlay visualization")
        return None

    fig, ax = plt.subplots(figsize=(12, 10))

    # Show image
    img = data['prev_frame']
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

    # Draw flow arrows
    flow_x = data['flow_x']
    flow_y = data['flow_y']
    h, w = flow_x.shape

    # Create grid for quiver plot
    y, x = np.mgrid[0:h:step, 0:w:step]
    fx = flow_x[::step, ::step]
    fy = flow_y[::step, ::step]

    # Only show arrows where there's flow
    mask = (fx != 0) | (fy != 0)
    if np.any(mask):
        ax.quiver(x[mask], y[mask], fx[mask], fy[mask],
                  color='lime', angles='xy', scale_units='xy', scale=0.5,
                  width=0.003, headwidth=4)

    ax.set_title('Optical Flow Overlay')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path.replace('.png', '_overlay.png'), dpi=150, bbox_inches='tight')

    return fig


def print_statistics(data):
    """Print flow statistics."""
    print("\nStatistics:")

    if 'flow_x' in data:
        flow_x = data['flow_x']
        nonzero = flow_x[flow_x != 0]
        if len(nonzero) > 0:
            print("  Flow X: min={0:.2f}, max={1:.2f}, mean={2:.2f}, std={3:.2f}".format(
                nonzero.min(), nonzero.max(), nonzero.mean(), nonzero.std()))
            print("          Non-zero points: {0}".format(len(nonzero)))

    if 'flow_y' in data:
        flow_y = data['flow_y']
        nonzero = flow_y[flow_y != 0]
        if len(nonzero) > 0:
            print("  Flow Y: min={0:.2f}, max={1:.2f}, mean={2:.2f}, std={3:.2f}".format(
                nonzero.min(), nonzero.max(), nonzero.mean(), nonzero.std()))


def main():
    parser = argparse.ArgumentParser(description='Visualize Lucas-Kanade optical flow')
    parser.add_argument('-i', '--input', required=True, help='Input directory with binary files and parameters.json')
    parser.add_argument('--save', type=str, help='Save visualizations to file (e.g., result.png)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')

    args = parser.parse_args()

    # Load data
    print("Loading data from: {0}".format(args.input))
    data = load_data(args.input)

    # Print what was loaded
    print("Loaded:")
    for key in ['prev_frame', 'curr_frame', 'flow_x', 'flow_y']:
        if key in data:
            print("  - {0}: {1}".format(key, data[key].shape))

    # Print statistics
    print_statistics(data)

    # Create visualizations
    visualize_frames(data, args.save)
    visualize_flow(data, args.save)
    visualize_flow_overlay(data, args.save)

    if not args.no_show:
        plt.show()

    return 0


if __name__ == '__main__':
    exit(main())
