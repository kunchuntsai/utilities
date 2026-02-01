# Harris Corner Response

Compute Harris corner response using OpenCV.

## Project Structure

```
datagen/
├── harris_response/
│   ├── gen_test_data.py    # Generate test data
│   ├── harris.py           # Compute Harris response
│   ├── visualize.py        # Visualize inputs/outputs
│   ├── requirements.txt
│   └── README.md
└── tests/                  # Test data directory
```

## Installation

```bash
cd harris_response

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
# From an image
python gen_test_data.py -i input.jpg -o ../tests/harris_test1/

# Synthetic image
python gen_test_data.py --synthetic -o ../tests/harris_test1/ --pattern checkerboard
python gen_test_data.py --synthetic -o ../tests/harris_test1/ --pattern squares
python gen_test_data.py --synthetic -o ../tests/harris_test1/ --pattern corners
```

**Output:**
- `input.bin` - Grayscale input image
- `parameters.json` - Metadata

### 2. Compute Harris Response

```bash
python harris.py -i ../tests/harris_test1/ -o ../tests/harris_test1/

# With parameters
python harris.py -i ../tests/harris_test1/ -o ../tests/harris_test1/ \
    --block-size 2 \
    --ksize 3 \
    --k 0.04
```

**Harris Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--block-size` | 2 | Neighborhood size for corner detection |
| `--ksize` | 3 | Aperture parameter for Sobel operator |
| `--k` | 0.04 | Harris detector free parameter (0.04-0.06) |

**Output:**
- `input.bin` - Input image
- `response.bin` - Harris response map (float32)
- `parameters.json` - Metadata including Harris parameters

### 3. Visualize Results

```bash
python visualize.py -i ../tests/harris_test1/

# Save visualizations
python visualize.py -i ../tests/harris_test1/ --save result.png

# Adjust corner threshold
python visualize.py -i ../tests/harris_test1/ --threshold 0.05
```

## Output Format

### parameters.json

```json
{
    "input": {
        "file": "input.bin",
        "src_width": 640,
        "src_height": 480,
        "src_channels": 1,
        "src_stride": "640 * 1"
    },
    "response": {
        "file": "response.bin",
        "dst_width": 640,
        "dst_height": 480,
        "dst_channels": 1,
        "dst_stride": "640 * 4"
    },
    "harris_params": {
        "block_size": 2,
        "ksize": 3,
        "k": 0.04
    }
}
```

## Example Workflow

```bash
# Generate synthetic test data
python gen_test_data.py --synthetic -o ../tests/harris_test/ --pattern checkerboard

# Compute Harris response
python harris.py -i ../tests/harris_test/ -o ../tests/harris_test/

# Visualize results
python visualize.py -i ../tests/harris_test/
```

## Harris Corner Detection

The Harris corner detector computes a response R for each pixel:

```
R = det(M) - k * trace(M)^2
```

Where M is the structure tensor (sum of squared gradients in a neighborhood).

- **Corners**: R > threshold (high response)
- **Edges**: R < 0
- **Flat regions**: |R| small

## Requirements

- Python 3.6+
- numpy
- opencv-python
- matplotlib
