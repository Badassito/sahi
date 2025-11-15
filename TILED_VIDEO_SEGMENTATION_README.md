# SAHI Tiled Video Segmentation

A high-performance video segmentation system using SAHI (Slicing Aided Hyper Inference) with overlapping tiles and batched inference.

## Features

✅ **Overlapping Tiling**: 1024×1024 tiles with configurable overlap (default 33%)
✅ **Batched Inference**: Process multiple tiles simultaneously for better GPU utilization
✅ **Video Support**: Process video files frame-by-frame
✅ **Dual Output Modes**:
  - Binary masks (white segmentation on black background)
  - Colored overlays (segmentation masks on original images)
✅ **CLI Interface**: Easy-to-use command-line interface
✅ **Frame Control**: Process all frames or skip frames for faster processing
✅ **Flexible Configuration**: Customizable tile sizes, overlap, and batch sizes

## Installation

### Prerequisites

```bash
# Install SAHI
pip install sahi

# Install Ultralytics for YOLO segmentation models
pip install ultralytics

# Install OpenCV
pip install opencv-python

# Optional: Install other model frameworks
pip install mmdet  # For MMDetection models
pip install detectron2  # For Detectron2 models
```

### Download Segmentation Models

```bash
# YOLOv11 segmentation model (recommended)
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n-seg.pt

# OR YOLOv8 segmentation model
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt
```

## Quick Start

### CLI Usage

```bash
# Basic usage - process a video
python sahi_tiled_video_segmentation.py \
    --source video.mp4 \
    --model-path yolo11n-seg.pt

# Process a single image
python sahi_tiled_video_segmentation.py \
    --source image.jpg \
    --model-path yolo11n-seg.pt
```

### Programmatic Usage

```python
from pathlib import Path
from sahi.auto_model import AutoDetectionModel
from sahi_tiled_video_segmentation import BatchedTiledSegmentation

# Initialize model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo11n-seg.pt",
    device="cuda:0",
    confidence_threshold=0.25,
)

# Initialize processor
processor = BatchedTiledSegmentation(
    detection_model=detection_model,
    slice_height=1024,
    slice_width=1024,
    overlap_ratio=0.33,
    batch_size=4,
)

# Process video
processor.process_video(
    video_path="input.mp4",
    output_dir=Path("output"),
    frame_skip_interval=0,
    verbose=True,
)
```

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--source` | Path to input video or image file | `video.mp4` |
| `--model-path` | Path to segmentation model | `yolo11n-seg.pt` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-type` | `ultralytics` | Model type (ultralytics, mmdet, detectron2, torchvision) |
| `--output-dir` | `output` | Output directory for results |
| `--slice-height` | `1024` | Height of each tile in pixels |
| `--slice-width` | `1024` | Width of each tile in pixels |
| `--overlap-ratio` | `0.33` | Overlap ratio for tiles (0.33 = 33%) |
| `--batch-size` | `4` | Number of tiles to process simultaneously |
| `--confidence-threshold` | `0.25` | Confidence threshold for detections |
| `--device` | `cuda:0` | Device for inference (cuda:0, cpu) |
| `--frame-skip` | `0` | Process every Nth frame (0 = all frames) |
| `--quiet` | `False` | Suppress progress output |

## Usage Examples

### 1. Basic Video Processing

Process a video with default settings (1024×1024 tiles, 33% overlap):

```bash
python sahi_tiled_video_segmentation.py \
    --source video.mp4 \
    --model-path yolo11n-seg.pt
```

**Output:**
```
output/
├── binary_masks/
│   ├── frame_000000_binary.png
│   ├── frame_000001_binary.png
│   └── ...
└── colored_overlays/
    ├── frame_000000_colored.jpg
    ├── frame_000001_colored.jpg
    └── ...
```

### 2. Custom Tile Size and Overlap

Use smaller tiles with higher overlap for small objects:

```bash
python sahi_tiled_video_segmentation.py \
    --source video.mp4 \
    --model-path yolo11n-seg.pt \
    --slice-height 640 \
    --slice-width 640 \
    --overlap-ratio 0.5
```

### 3. Large Batch Size for Speed

Increase batch size for faster processing (requires more GPU memory):

```bash
python sahi_tiled_video_segmentation.py \
    --source video.mp4 \
    --model-path yolo11n-seg.pt \
    --batch-size 8 \
    --device cuda:0
```

### 4. Frame Skipping for Fast Preview

Process every 5th frame for quick preview:

```bash
python sahi_tiled_video_segmentation.py \
    --source video.mp4 \
    --model-path yolo11n-seg.pt \
    --frame-skip 4
```

### 5. CPU Inference

Run on CPU (slower but no GPU required):

```bash
python sahi_tiled_video_segmentation.py \
    --source video.mp4 \
    --model-path yolo11n-seg.pt \
    --device cpu
```

### 6. Custom Output Directory

Specify custom output directory:

```bash
python sahi_tiled_video_segmentation.py \
    --source video.mp4 \
    --model-path yolo11n-seg.pt \
    --output-dir results/my_experiment
```

## Output Description

### Binary Masks

- **Format**: PNG (lossless)
- **Content**: White segmentation masks on black background
- **Channel**: 3-channel BGR (for consistency with colored output)
- **Usage**: Instance segmentation, mask analysis, further processing
- **Filename**: `frame_{index:06d}_binary.png`

Example: `frame_000042_binary.png`

### Colored Overlays

- **Format**: JPG (compressed)
- **Content**: Segmentation masks overlaid on original image
- **Colors**: Random colors per object category
- **Alpha**: 0.6 (60% mask opacity)
- **Usage**: Visualization, quality assessment
- **Filename**: `frame_{index:06d}_colored.jpg`

Example: `frame_000042_colored.jpg`

### Key Features

- **Always Generated**: Both outputs are created for every frame, even if no objects are detected
- **Full Frame**: Outputs match the original video frame dimensions
- **Consistent Naming**: Sequential frame numbering with zero-padding

## Performance Considerations

### Tile Size

- **Small tiles (640×640)**: Better for small objects, more tiles to process
- **Medium tiles (1024×1024)**: **Recommended** - good balance
- **Large tiles (1280×1280)**: Faster processing, may miss small objects

### Overlap Ratio

- **Low overlap (0.2-0.25)**: Faster, may miss objects on tile edges
- **Medium overlap (0.33)**: **Recommended** - good balance
- **High overlap (0.5)**: Best accuracy, slower processing

### Batch Size

- **Small batch (2-4)**: Lower GPU memory usage
- **Medium batch (4-8)**: **Recommended** - good GPU utilization
- **Large batch (8-16)**: Fastest, requires high-end GPU

### Recommended Configurations

#### High Accuracy (Small Objects)
```bash
--slice-height 640 \
--slice-width 640 \
--overlap-ratio 0.5 \
--batch-size 4
```

#### Balanced (General Use)
```bash
--slice-height 1024 \
--slice-width 1024 \
--overlap-ratio 0.33 \
--batch-size 4
```

#### High Speed (Large Objects)
```bash
--slice-height 1280 \
--slice-width 1280 \
--overlap-ratio 0.25 \
--batch-size 8
```

## Programmatic API

### BatchedTiledSegmentation Class

```python
from sahi_tiled_video_segmentation import BatchedTiledSegmentation

processor = BatchedTiledSegmentation(
    detection_model,      # SAHI DetectionModel with segmentation support
    slice_height=1024,    # Tile height
    slice_width=1024,     # Tile width
    overlap_ratio=0.33,   # Overlap ratio
    batch_size=4,         # Batch size
)
```

### Process Video

```python
stats = processor.process_video(
    video_path="video.mp4",           # Input video path
    output_dir="output",              # Output directory
    frame_skip_interval=0,            # Frame skip (0 = all frames)
    verbose=True,                     # Show progress
)

# Returns dict with statistics
{
    'video_path': 'video.mp4',
    'total_frames': 1000,
    'processed_frames': 1000,
    'processing_time': 123.45,
    'average_fps': 8.1,
    'output_dir': 'output'
}
```

### Process Image

```python
stats = processor.process_image(
    image_path="image.jpg",           # Input image path
    output_dir="output",              # Output directory
    verbose=True,                     # Show processing info
)

# Returns dict with statistics
{
    'image_path': 'image.jpg',
    'processing_time': 1.23,
    'output_dir': 'output',
    'binary_path': 'output/image_binary.png',
    'colored_path': 'output/image_colored.jpg'
}
```

## Supported Models

### Ultralytics YOLO (Recommended)

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo11n-seg.pt",  # or yolov8{n/s/m/l/x}-seg.pt
    device="cuda:0",
)
```

**Available models:**
- YOLOv11: `yolo11{n/s/m/l/x}-seg.pt`
- YOLOv8: `yolov8{n/s/m/l/x}-seg.pt`

### MMDetection

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="mmdet",
    model_path="mask_rcnn_r50_fpn.py",
    config_path="mask_rcnn_r50_fpn.py",
    device="cuda:0",
)
```

### Detectron2

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="detectron2",
    model_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    device="cuda:0",
)
```

## Troubleshooting

### CUDA Out of Memory

**Solution**: Reduce batch size or tile size

```bash
--batch-size 2 --slice-height 640 --slice-width 640
```

### Processing Too Slow

**Solution**: Increase batch size or use frame skipping

```bash
--batch-size 8 --frame-skip 2
```

### Missing Small Objects

**Solution**: Use smaller tiles with higher overlap

```bash
--slice-height 640 --slice-width 640 --overlap-ratio 0.5
```

### Model Not Found

**Solution**: Download the segmentation model first

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n-seg.pt
```

### No Segmentation Output

**Issue**: Using detection-only model
**Solution**: Use a segmentation model (model name must contain "seg")

```bash
# Wrong: yolo11n.pt (detection only)
# Correct: yolo11n-seg.pt (segmentation)
```

## Technical Details

### Tiling Algorithm

1. **Divide**: Split frame into overlapping tiles using `get_slice_bboxes()`
2. **Process**: Run inference on each tile in batches
3. **Shift**: Map predictions back to full frame coordinates
4. **Merge**: Accumulate all masks into full-frame outputs

### Batching Strategy

- Tiles are processed sequentially in batches
- Each batch contains up to `batch_size` tiles
- Within a batch, tiles are processed one-by-one (model limitation)
- Batching primarily improves memory efficiency

### Mask Merging

- **Binary masks**: Accumulated using `np.maximum()` (union of all masks)
- **Colored overlays**: Alpha blending with random colors per category
- **No NMS**: All detected objects are kept (unlike bounding box detection)

### Output Format

- **Binary masks**: PNG format for lossless quality
- **Colored overlays**: JPG format for smaller file size
- **Frame numbering**: Zero-padded 6-digit indices (000000-999999)

## Examples

See `examples/tiled_segmentation_example.py` for comprehensive usage examples:

```bash
cd examples
python tiled_segmentation_example.py
```

Examples included:
1. Basic video processing
2. Single image processing
3. High overlap for small objects
4. Fast processing with frame skipping
5. Batch process multiple images

## License

This script is part of the SAHI project. See the main SAHI repository for license information.

## Citation

If you use this code, please cite the SAHI paper:

```bibtex
@article{akyon2022sahi,
  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={arXiv preprint arXiv:2202.06934},
  year={2022}
}
```

## Support

For issues and questions:
- SAHI Issues: https://github.com/obss/sahi/issues
- Ultralytics Issues: https://github.com/ultralytics/ultralytics/issues

## Changelog

### Version 1.0.0 (2025-11-15)
- Initial release
- 1024×1024 tiles with 33% overlap
- Batched inference support
- Binary mask and colored overlay outputs
- CLI and programmatic interfaces
- Video and image support
