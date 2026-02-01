# Lucas-Kanade Optical Flow

Compute optical flow using Lucas-Kanade method with Good Features to Track (Shi-Tomasi) keypoint detection.

## Project Structure

```
algorithms/
├── lucas_kanade/
│   ├── gen_test_data.py    # Generate test data
│   ├── lk_flow.py          # Compute optical flow
│   ├── visualize.py        # Visualize inputs/outputs
│   ├── requirements.txt
│   └── README.md
└── tests/                  # Test data directory
```

## Installation

```bash
cd lucas_kanade

# First time (creates virtual environment for this project)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Every subsequent time
source venv/bin/activate
```

## Usage

### 1. Generate Test Data

```bash
python gen_test_data.py -i input.jpg -o ../tests/test1/

# With motion parameters
python gen_test_data.py -i input.jpg -o ../tests/test1/ --motion translation --tx 10 --ty 5
python gen_test_data.py -i input.jpg -o ../tests/test1/ --motion rotation --angle 3
```

**Output:**
- `prev_frame.bin` - Original frame
- `curr_frame.bin` - Transformed frame
- `parameters.json` - Metadata

### 2. Compute Optical Flow

```bash
python lk_flow.py -i ../tests/test1/ -o ../tests/test1_output/

# With LK parameters
python lk_flow.py -i ../tests/test1/ -o ../tests/test1_output/ \
    --max-corners 200 \
    --win-size 21 \
    --max-level 3
```

**LK Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-corners` | 200 | Max keypoints to detect |
| `--quality` | 0.3 | Corner quality level (0-1) |
| `--min-distance` | 7 | Min distance between corners |
| `--win-size` | 21 | Search window size |
| `--max-level` | 3 | Max pyramid level |

**Output:**
- `prev_frame.bin`, `curr_frame.bin` - Input frames
- `flow_x.bin`, `flow_y.bin` - Dense flow maps (float32)
- `parameters.json` - Metadata including LK parameters

### 3. Visualize Results

```bash
python visualize.py -i ../tests/test1_output/

# Save visualizations
python visualize.py -i ../tests/test1_output/ --save result.png
```

## Output Format

### parameters.json

```json
{
    "prev_frame": {
        "file": "prev_frame.bin",
        "src_width": 1920,
        "src_height": 1080,
        "src_channels": 3,
        "src_stride": "5760 * 1"
    },
    "curr_frame": {
        "file": "curr_frame.bin",
        "src_width": 1920,
        "src_height": 1080,
        "src_channels": 3,
        "src_stride": "5760 * 1"
    },
    "flow_x": {
        "file": "flow_x.bin",
        "dst_width": 1920,
        "dst_height": 1080,
        "dst_channels": 1,
        "dst_stride": "1920 * 4"
    },
    "flow_y": {
        "file": "flow_y.bin",
        "dst_width": 1920,
        "dst_height": 1080,
        "dst_channels": 1,
        "dst_stride": "1920 * 4"
    },
    "lk_params": {
        "max_corners": 200,
        "quality_level": 0.3,
        "min_distance": 7,
        "win_size": 21,
        "max_level": 3
    }
}
```

## Example Workflow

```bash
# Generate test data from an image
python gen_test_data.py -i image.jpg -o ../tests/mytest/ --motion translation --tx 8 --ty 4

# Compute optical flow
python lk_flow.py -i ../tests/mytest/ -o ../tests/mytest/

# Visualize results
python visualize.py -i ../tests/mytest/
```

## Requirements

- Python 3.6+
- numpy
- opencv-python
- matplotlib
