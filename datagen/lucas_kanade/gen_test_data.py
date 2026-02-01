#!/usr/bin/env python
"""
Generate test data for Lucas-Kanade optical flow.

Usage:
    python gen_test_data.py -i input.jpg -o output_dir/
    python gen_test_data.py -i input.jpg -o output_dir/ --motion rotation --tx 10 --ty 5
"""

import argparse
import os
import json
import numpy as np
import cv2


def translate_image(image, tx, ty):
    """Apply translation to an image."""
    h, w = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (w, h))


def rotate_image(image, angle, center=None, scale=1.0):
    """Apply rotation to an image."""
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, M, (w, h))


def scale_image(image, scale_x, scale_y=None):
    """Apply scaling to an image (zoom in/out from center)."""
    if scale_y is None:
        scale_y = scale_x
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = np.float32([
        [scale_x, 0, center[0] * (1 - scale_x)],
        [0, scale_y, center[1] * (1 - scale_y)]
    ])
    return cv2.warpAffine(image, M, (w, h))


def apply_affine_transform(image, tx=0, ty=0, angle=0, scale=1.0):
    """Apply combined affine transformation."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    cos_a = np.cos(np.radians(angle)) * scale
    sin_a = np.sin(np.radians(angle)) * scale
    M = np.float32([
        [cos_a, -sin_a, center[0] * (1 - cos_a) + center[1] * sin_a + tx],
        [sin_a, cos_a, center[1] * (1 - cos_a) - center[0] * sin_a + ty]
    ])
    return cv2.warpAffine(image, M, (w, h))


def generate_test_frame(prev_frame, motion_type='translation', **kwargs):
    """
    Generate a test current frame from a previous frame with known motion.

    Args:
        prev_frame: The reference/previous frame
        motion_type: 'translation', 'rotation', 'scale', 'affine', 'random'

    Returns:
        curr_frame, motion_params
    """
    tx = kwargs.get('tx', 8)
    ty = kwargs.get('ty', 4)
    angle = kwargs.get('angle', 2)
    scale = kwargs.get('scale', 1.02)

    if motion_type == 'translation':
        curr_frame = translate_image(prev_frame, tx, ty)
        motion_params = {'type': 'translation', 'tx': tx, 'ty': ty}
    elif motion_type == 'rotation':
        curr_frame = rotate_image(prev_frame, angle)
        motion_params = {'type': 'rotation', 'angle': angle}
    elif motion_type == 'scale':
        curr_frame = scale_image(prev_frame, scale)
        motion_params = {'type': 'scale', 'scale': scale}
    elif motion_type == 'affine':
        curr_frame = apply_affine_transform(prev_frame, tx, ty, angle, scale)
        motion_params = {'type': 'affine', 'tx': tx, 'ty': ty, 'angle': angle, 'scale': scale}
    elif motion_type == 'random':
        tx = float(np.random.uniform(-10, 10))
        ty = float(np.random.uniform(-10, 10))
        angle = float(np.random.uniform(-5, 5))
        scale = float(np.random.uniform(0.98, 1.02))
        curr_frame = apply_affine_transform(prev_frame, tx, ty, angle, scale)
        motion_params = {'type': 'random', 'tx': tx, 'ty': ty, 'angle': angle, 'scale': scale}
    else:
        curr_frame = prev_frame.copy()
        motion_params = {'type': 'none'}

    return curr_frame, motion_params


def get_dtype_size(dtype):
    """Get the byte size of a numpy dtype."""
    return np.dtype(dtype).itemsize


def save_outputs(prev_frame, curr_frame, motion_params, output_dir):
    """Save test data outputs."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save binary frames
    prev_path = os.path.join(output_dir, 'prev_frame.bin')
    curr_path = os.path.join(output_dir, 'curr_frame.bin')

    prev_frame.tofile(prev_path)
    curr_frame.tofile(curr_path)

    # Get frame info
    h, w = prev_frame.shape[:2]
    channels = prev_frame.shape[2] if len(prev_frame.shape) == 3 else 1
    dtype_size = get_dtype_size(prev_frame.dtype)

    # Save parameters as JSON
    params = {
        "prev_frame": {
            "file": "prev_frame.bin",
            "src_width": w,
            "src_height": h,
            "src_channels": channels,
            "src_stride": "{0} * {1}".format(w * channels, dtype_size)
        },
        "curr_frame": {
            "file": "curr_frame.bin",
            "src_width": w,
            "src_height": h,
            "src_channels": channels,
            "src_stride": "{0} * {1}".format(w * channels, dtype_size)
        },
        "motion": motion_params
    }

    params_path = os.path.join(output_dir, 'parameters.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

    print("Generated test data:")
    print("  - prev_frame.bin ({0} bytes)".format(os.path.getsize(prev_path)))
    print("  - curr_frame.bin ({0} bytes)".format(os.path.getsize(curr_path)))
    print("  - parameters.json")
    print("  Motion: {0}".format(motion_params))


def main():
    parser = argparse.ArgumentParser(description='Generate test data for Lucas-Kanade optical flow')
    parser.add_argument('-i', '--input', required=True, help='Input image path')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('-m', '--motion', default='translation',
                        choices=['translation', 'rotation', 'scale', 'affine', 'random'],
                        help='Motion type to apply')
    parser.add_argument('--tx', type=float, default=8, help='Translation X')
    parser.add_argument('--ty', type=float, default=4, help='Translation Y')
    parser.add_argument('--angle', type=float, default=2, help='Rotation angle (degrees)')
    parser.add_argument('--scale', type=float, default=1.02, help='Scale factor')

    args = parser.parse_args()

    # Load input image
    prev_frame = cv2.imread(args.input)
    if prev_frame is None:
        print("Error: Could not load image: {0}".format(args.input))
        return 1

    print("Input: {0} ({1})".format(args.input, prev_frame.shape))

    # Generate test frame
    curr_frame, motion_params = generate_test_frame(
        prev_frame,
        motion_type=args.motion,
        tx=args.tx,
        ty=args.ty,
        angle=args.angle,
        scale=args.scale
    )

    # Save outputs
    save_outputs(prev_frame, curr_frame, motion_params, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
