"""
Example usage of SAHI Tiled Video Segmentation.

This example demonstrates how to use the tiled video segmentation system
both programmatically and via CLI.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sahi.auto_model import AutoDetectionModel
from sahi_tiled_video_segmentation import BatchedTiledSegmentation


def example_video_processing():
    """Example: Process a video with tiled segmentation."""

    print("=" * 80)
    print("Example 1: Video Processing with Default Settings")
    print("=" * 80)

    # Initialize SAHI model with segmentation support
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11n-seg.pt",  # or "yolov8n-seg.pt"
        device="cuda:0",  # or "cpu" for CPU inference
        confidence_threshold=0.25,
    )

    # Initialize the tiled segmentation processor
    processor = BatchedTiledSegmentation(
        detection_model=detection_model,
        slice_height=1024,      # Tile height
        slice_width=1024,       # Tile width
        overlap_ratio=0.33,     # 33% overlap
        batch_size=4,           # Process 4 tiles simultaneously
    )

    # Process video
    stats = processor.process_video(
        video_path="input_video.mp4",
        output_dir=Path("output/video_results"),
        frame_skip_interval=0,  # Process all frames (0 = no skip)
        verbose=True,
    )

    print(f"\nProcessing Statistics:")
    print(f"  Total frames: {stats['total_frames']}")
    print(f"  Processed frames: {stats['processed_frames']}")
    print(f"  Processing time: {stats['processing_time']:.2f}s")
    print(f"  Average FPS: {stats['average_fps']:.2f}")


def example_image_processing():
    """Example: Process a single image with tiled segmentation."""

    print("\n" + "=" * 80)
    print("Example 2: Single Image Processing")
    print("=" * 80)

    # Initialize SAHI model
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

    # Process image
    stats = processor.process_image(
        image_path="input_image.jpg",
        output_dir=Path("output/image_results"),
        verbose=True,
    )

    print(f"\nOutput files:")
    print(f"  Binary mask: {stats['binary_path']}")
    print(f"  Colored overlay: {stats['colored_path']}")


def example_high_overlap():
    """Example: High overlap for small objects."""

    print("\n" + "=" * 80)
    print("Example 3: High Overlap for Small Objects")
    print("=" * 80)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11n-seg.pt",
        device="cuda:0",
        confidence_threshold=0.25,
    )

    # Use higher overlap for better small object detection
    processor = BatchedTiledSegmentation(
        detection_model=detection_model,
        slice_height=640,       # Smaller tiles for small objects
        slice_width=640,
        overlap_ratio=0.5,      # 50% overlap for better coverage
        batch_size=8,           # More tiles, larger batch
    )

    processor.process_video(
        video_path="small_objects_video.mp4",
        output_dir=Path("output/small_objects"),
        frame_skip_interval=0,
        verbose=True,
    )


def example_fast_processing():
    """Example: Fast processing with frame skipping."""

    print("\n" + "=" * 80)
    print("Example 4: Fast Processing with Frame Skipping")
    print("=" * 80)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11n-seg.pt",
        device="cuda:0",
        confidence_threshold=0.3,
    )

    processor = BatchedTiledSegmentation(
        detection_model=detection_model,
        slice_height=1024,
        slice_width=1024,
        overlap_ratio=0.33,
        batch_size=8,           # Larger batch for speed
    )

    # Process every 5th frame for faster processing
    processor.process_video(
        video_path="input_video.mp4",
        output_dir=Path("output/fast_processing"),
        frame_skip_interval=4,  # Process every 5th frame
        verbose=True,
    )


def example_batch_images():
    """Example: Batch process multiple images."""

    print("\n" + "=" * 80)
    print("Example 5: Batch Process Multiple Images")
    print("=" * 80)

    # Initialize model once for efficiency
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11n-seg.pt",
        device="cuda:0",
        confidence_threshold=0.25,
    )

    processor = BatchedTiledSegmentation(
        detection_model=detection_model,
        slice_height=1024,
        slice_width=1024,
        overlap_ratio=0.33,
        batch_size=4,
    )

    # Process all images in a directory
    image_dir = Path("input_images")
    output_base = Path("output/batch_processing")

    if image_dir.exists():
        for image_path in sorted(image_dir.glob("*.jpg")):
            print(f"\nProcessing: {image_path.name}")

            processor.process_image(
                image_path=str(image_path),
                output_dir=output_base / image_path.stem,
                verbose=True,
            )
    else:
        print(f"Directory not found: {image_dir}")
        print("Create the directory and add images to process them in batch.")


def show_cli_examples():
    """Show CLI usage examples."""

    print("\n" + "=" * 80)
    print("CLI Usage Examples")
    print("=" * 80)

    examples = """
# Basic video processing
python sahi_tiled_video_segmentation.py \\
    --source video.mp4 \\
    --model-path yolo11n-seg.pt

# Custom tile size and overlap
python sahi_tiled_video_segmentation.py \\
    --source video.mp4 \\
    --model-path yolo11n-seg.pt \\
    --slice-height 640 \\
    --slice-width 640 \\
    --overlap-ratio 0.25

# Large batch size for faster processing
python sahi_tiled_video_segmentation.py \\
    --source video.mp4 \\
    --model-path yolo11n-seg.pt \\
    --batch-size 8 \\
    --device cuda:0

# Process single image
python sahi_tiled_video_segmentation.py \\
    --source image.jpg \\
    --model-path yolo11n-seg.pt \\
    --output-dir output/my_results

# Process every 5th frame (faster processing)
python sahi_tiled_video_segmentation.py \\
    --source video.mp4 \\
    --model-path yolo11n-seg.pt \\
    --frame-skip 4

# CPU inference
python sahi_tiled_video_segmentation.py \\
    --source video.mp4 \\
    --model-path yolo11n-seg.pt \\
    --device cpu

# Quiet mode (no progress output)
python sahi_tiled_video_segmentation.py \\
    --source video.mp4 \\
    --model-path yolo11n-seg.pt \\
    --quiet
    """

    print(examples)


if __name__ == "__main__":
    print("\nSAHI Tiled Video Segmentation Examples")
    print("========================================\n")

    # Show CLI examples first
    show_cli_examples()

    print("\n" + "=" * 80)
    print("Programmatic Examples")
    print("=" * 80)
    print("\nUncomment the examples below to run them:\n")

    # Example 1: Basic video processing
    # example_video_processing()

    # Example 2: Single image processing
    # example_image_processing()

    # Example 3: High overlap for small objects
    # example_high_overlap()

    # Example 4: Fast processing with frame skipping
    # example_fast_processing()

    # Example 5: Batch process multiple images
    # example_batch_images()

    print("\nTo run an example, uncomment it in the script and run again.")
