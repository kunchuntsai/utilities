#!/usr/bin/env python
"""
Lucas-Kanade Optical Flow - Consume inputs and produce outputs.

Usage:
    python lk_flow.py -i input_dir/ -o output_dir/
    python lk_flow.py -i input_dir/ -o output_dir/ --max-corners 200
"""

import argparse
import os
import json
import numpy as np
import cv2


def detect_keypoints(image, max_corners=100, quality_level=0.3, min_distance=7, block_size=7):
    """Detect keypoints using Good Features to Track (Shi-Tomasi)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    keypoints = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size
    )
    return keypoints


def compute_optical_flow(prev_frame, curr_frame, keypoints, win_size=(21, 21), max_level=3):
    """Compute Lucas-Kanade optical flow between two frames."""
    # Convert to grayscale if needed
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame

    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame

    if keypoints is None or len(keypoints) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    keypoints = np.asarray(keypoints, dtype=np.float32)
    if len(keypoints.shape) == 2:
        keypoints = keypoints.reshape(-1, 1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    new_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, keypoints, None,
        winSize=win_size, maxLevel=max_level, criteria=criteria
    )

    status = status.flatten()
    old_pts = keypoints.reshape(-1, 2)
    new_pts = new_points.reshape(-1, 2)

    flow_x = new_pts[:, 0] - old_pts[:, 0]
    flow_y = new_pts[:, 1] - old_pts[:, 1]

    return flow_x, flow_y, status, new_points


def create_dense_flow(flow_x_sparse, flow_y_sparse, keypoints, status, image_shape):
    """Create dense flow arrays from sparse keypoint flow."""
    h, w = image_shape[:2]
    flow_x_dense = np.zeros((h, w), dtype=np.float32)
    flow_y_dense = np.zeros((h, w), dtype=np.float32)

    if keypoints is None or len(keypoints) == 0:
        return flow_x_dense, flow_y_dense

    pts = keypoints.reshape(-1, 2)

    for pt, fx, fy, st in zip(pts, flow_x_sparse, flow_y_sparse, status):
        if st == 1:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                flow_x_dense[y, x] = fx
                flow_y_dense[y, x] = fy

    return flow_x_dense, flow_y_dense


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

    # Load prev_frame
    prev_info = params['prev_frame']
    prev_path = os.path.join(input_dir, prev_info['file'])
    prev_frame = load_frame_from_bin(
        prev_path,
        prev_info['src_width'],
        prev_info['src_height'],
        prev_info['src_channels']
    )

    # Load curr_frame
    curr_info = params['curr_frame']
    curr_path = os.path.join(input_dir, curr_info['file'])
    curr_frame = load_frame_from_bin(
        curr_path,
        curr_info['src_width'],
        curr_info['src_height'],
        curr_info['src_channels']
    )

    return prev_frame, curr_frame, params


def save_outputs(prev_frame, curr_frame, flow_x_dense, flow_y_dense, keypoints, status, output_dir, lk_params):
    """Save outputs to directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save binary frames (copy from input)
    prev_path = os.path.join(output_dir, 'prev_frame.bin')
    curr_path = os.path.join(output_dir, 'curr_frame.bin')
    flow_x_path = os.path.join(output_dir, 'flow_x.bin')
    flow_y_path = os.path.join(output_dir, 'flow_y.bin')

    prev_frame.tofile(prev_path)
    curr_frame.tofile(curr_path)
    flow_x_dense.tofile(flow_x_path)
    flow_y_dense.tofile(flow_y_path)

    # Get dimensions
    h, w = prev_frame.shape[:2]
    channels = prev_frame.shape[2] if len(prev_frame.shape) == 3 else 1
    dtype_size = np.dtype(prev_frame.dtype).itemsize
    flow_dtype_size = 4  # float32

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
        "flow_x": {
            "file": "flow_x.bin",
            "dst_width": w,
            "dst_height": h,
            "dst_channels": 1,
            "dst_stride": "{0} * {1}".format(w, flow_dtype_size)
        },
        "flow_y": {
            "file": "flow_y.bin",
            "dst_width": w,
            "dst_height": h,
            "dst_channels": 1,
            "dst_stride": "{0} * {1}".format(w, flow_dtype_size)
        },
        "lk_params": lk_params
    }

    params_path = os.path.join(output_dir, 'parameters.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

    print("Saved outputs:")
    print("  - prev_frame.bin ({0} bytes)".format(os.path.getsize(prev_path)))
    print("  - curr_frame.bin ({0} bytes)".format(os.path.getsize(curr_path)))
    print("  - flow_x.bin ({0} bytes)".format(os.path.getsize(flow_x_path)))
    print("  - flow_y.bin ({0} bytes)".format(os.path.getsize(flow_y_path)))
    print("  - parameters.json")


def main():
    parser = argparse.ArgumentParser(description='Lucas-Kanade Optical Flow')
    parser.add_argument('-i', '--input', required=True, help='Input directory with prev_frame.bin, curr_frame.bin, parameters.json')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    # Feature detection parameters
    parser.add_argument('--max-corners', type=int, default=200, help='Max keypoints to detect')
    parser.add_argument('--quality', type=float, default=0.3, help='Corner quality level (0-1)')
    parser.add_argument('--min-distance', type=float, default=7, help='Min distance between corners')
    # Lucas-Kanade parameters
    parser.add_argument('--win-size', type=int, default=21, help='Search window size')
    parser.add_argument('--max-level', type=int, default=3, help='Max pyramid level')

    args = parser.parse_args()

    # LK parameters dict
    lk_params = {
        "max_corners": args.max_corners,
        "quality_level": args.quality,
        "min_distance": args.min_distance,
        "win_size": args.win_size,
        "max_level": args.max_level
    }

    # Load inputs
    print("Loading inputs from: {0}".format(args.input))
    prev_frame, curr_frame, input_params = load_inputs(args.input)
    print("  prev_frame: {0}".format(prev_frame.shape))
    print("  curr_frame: {0}".format(curr_frame.shape))

    # Detect keypoints
    print("\nDetecting keypoints...")
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame

    keypoints = detect_keypoints(
        prev_gray,
        max_corners=args.max_corners,
        quality_level=args.quality,
        min_distance=args.min_distance
    )
    print("  Detected: {0} keypoints".format(len(keypoints)))

    # Compute optical flow
    print("\nComputing optical flow...")
    win_size = (args.win_size, args.win_size)
    flow_x, flow_y, status, new_points = compute_optical_flow(
        prev_frame, curr_frame, keypoints,
        win_size=win_size,
        max_level=args.max_level
    )

    valid_count = int(np.sum(status == 1))
    print("  Tracked: {0}/{1} points".format(valid_count, len(keypoints)))

    if valid_count > 0:
        valid_mask = status == 1
        print("  Mean flow: x={0:.2f}, y={1:.2f}".format(
            np.mean(flow_x[valid_mask]), np.mean(flow_y[valid_mask])))

    # Create dense flow maps
    flow_x_dense, flow_y_dense = create_dense_flow(
        flow_x, flow_y, keypoints, status, prev_frame.shape
    )

    # Save outputs
    print("\nSaving outputs to: {0}".format(args.output))
    save_outputs(prev_frame, curr_frame, flow_x_dense, flow_y_dense, keypoints, status, args.output, lk_params)

    return 0


if __name__ == '__main__':
    exit(main())
