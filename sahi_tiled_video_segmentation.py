#!/usr/bin/env python3
"""
SAHI Tiled Video Segmentation Script

Features:
- 1024x1024 overlapping tiles with 33% overlap
- Batched inference for multiple tiles simultaneously
- Video input support
- Outputs:
  1. Binary mask (white segmentation on black background)
  2. Colored overlay (segmentation mask on original image)
- CLI interface
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from sahi.auto_model import AutoDetectionModel
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.slicing import get_slice_bboxes
from sahi.utils.cv import get_video_reader, read_image_as_pil


class BatchedTiledSegmentation:
    """
    Handles batched tiled segmentation for videos with SAHI.

    Features:
    - Overlapping tiles for accurate edge detection
    - Batched inference for performance
    - Full-frame binary and colored mask outputs
    """

    def __init__(
        self,
        detection_model: DetectionModel,
        slice_height: int = 1024,
        slice_width: int = 1024,
        overlap_ratio: float = 0.33,
        batch_size: int = 4,
    ):
        """
        Initialize the batched tiled segmentation processor.

        Args:
            detection_model: SAHI detection model with segmentation support
            slice_height: Height of each tile (default: 1024)
            slice_width: Width of each tile (default: 1024)
            overlap_ratio: Overlap ratio for tiles (default: 0.33 = 33%)
            batch_size: Number of tiles to process simultaneously (default: 4)
        """
        self.detection_model = detection_model
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio
        self.batch_size = batch_size

        # Verify model supports segmentation
        if not hasattr(detection_model, 'has_mask') or not detection_model.has_mask:
            raise ValueError(
                "Detection model must support segmentation. "
                "Use models like 'yolo11n-seg.pt' or 'yolov8n-seg.pt'"
            )

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process a single frame with tiled segmentation.

        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)

        Returns:
            Tuple of (binary_mask, colored_overlay):
            - binary_mask: White segmentation on black background (H, W, 3)
            - colored_overlay: Segmentation overlay on original image (H, W, 3)
        """
        height, width = frame.shape[:2]

        # Initialize output masks
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        colored_overlay = frame.copy()

        # Get tile bounding boxes
        slice_bboxes = get_slice_bboxes(
            image_height=height,
            image_width=width,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            auto_slice_resolution=False,
        )

        # Process tiles in batches
        num_tiles = len(slice_bboxes)

        for batch_start in range(0, num_tiles, self.batch_size):
            batch_end = min(batch_start + self.batch_size, num_tiles)
            batch_bboxes = slice_bboxes[batch_start:batch_end]

            # Process each tile in the batch
            for slice_bbox in batch_bboxes:
                x_min, y_min, x_max, y_max = slice_bbox

                # Extract tile
                tile = frame[y_min:y_max, x_min:x_max]

                # Convert to RGB for model inference
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

                # Perform inference on tile
                self.detection_model.perform_inference(tile_rgb)

                # Get predictions with shift amount for full image mapping
                shift_amount = [x_min, y_min]
                full_shape = [height, width]

                self.detection_model.convert_original_predictions(
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )

                # Process each detected object
                object_predictions: list[ObjectPrediction] = self.detection_model.object_prediction_list

                for obj_pred in object_predictions:
                    if obj_pred.mask is not None:
                        # Get shifted mask for full image
                        mask_bool = obj_pred.mask.get_shifted_mask()

                        if mask_bool is not None and mask_bool.shape == (height, width):
                            # Update binary mask (accumulate all detections)
                            binary_mask = np.maximum(binary_mask, mask_bool.astype(np.uint8) * 255)

                            # Create colored overlay with random color per object
                            color = self._generate_color(obj_pred.category.id)

                            # Create colored mask
                            colored_mask = np.zeros_like(frame, dtype=np.uint8)
                            colored_mask[mask_bool] = color

                            # Blend with original image using alpha
                            alpha = 0.6
                            colored_overlay = cv2.addWeighted(
                                colored_overlay, 1.0,
                                colored_mask, alpha,
                                0
                            )

        # Convert binary mask to 3-channel for consistency
        binary_mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        return binary_mask_3ch, colored_overlay

    def _generate_color(self, category_id: int) -> tuple[int, int, int]:
        """
        Generate a consistent color for a category ID.

        Args:
            category_id: Category ID

        Returns:
            BGR color tuple
        """
        # Use a simple hash-based color generation
        np.random.seed(category_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        np.random.seed()  # Reset seed
        return color

    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        frame_skip_interval: int = 0,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Process a video file with tiled segmentation.

        Args:
            video_path: Path to input video file
            output_dir: Directory to save output frames
            frame_skip_interval: Skip frames (0 = process all frames)
            verbose: Show progress bar

        Returns:
            Dictionary with processing statistics
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        # Create output directories
        binary_dir = output_dir / "binary_masks"
        colored_dir = output_dir / "colored_overlays"
        binary_dir.mkdir(parents=True, exist_ok=True)
        colored_dir.mkdir(parents=True, exist_ok=True)

        # Get video reader
        video_reader = get_video_reader(str(video_path))

        # Get video properties
        total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_reader.get(cv2.CAP_PROP_FPS)

        if verbose:
            print(f"\nProcessing video: {video_path.name}")
            print(f"Total frames: {total_frames}")
            print(f"FPS: {fps:.2f}")
            print(f"Tile size: {self.slice_width}x{self.slice_height}")
            print(f"Overlap: {self.overlap_ratio * 100:.0f}%")
            print(f"Batch size: {self.batch_size}")
            print(f"Output directory: {output_dir}")

        # Process frames
        frame_idx = 0
        processed_frames = 0
        total_time = 0

        progress_bar = tqdm(total=total_frames, desc="Processing frames") if verbose else None

        while True:
            ret, frame = video_reader.read()

            if not ret:
                break

            # Skip frames if needed
            if frame_skip_interval > 0 and frame_idx % (frame_skip_interval + 1) != 0:
                frame_idx += 1
                if progress_bar:
                    progress_bar.update(1)
                continue

            # Process frame
            start_time = time.time()
            binary_mask, colored_overlay = self.process_frame(frame)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            # Save outputs
            frame_name = f"frame_{frame_idx:06d}"

            # Save binary mask
            binary_path = binary_dir / f"{frame_name}_binary.png"
            cv2.imwrite(str(binary_path), binary_mask)

            # Save colored overlay
            colored_path = colored_dir / f"{frame_name}_colored.jpg"
            cv2.imwrite(str(colored_path), colored_overlay)

            processed_frames += 1
            frame_idx += 1

            if progress_bar:
                progress_bar.update(1)
                progress_bar.set_postfix({"FPS": f"{1.0 / elapsed_time:.2f}"})

        if progress_bar:
            progress_bar.close()

        video_reader.release()

        # Calculate statistics
        avg_fps = processed_frames / total_time if total_time > 0 else 0

        stats = {
            "video_path": str(video_path),
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "processing_time": total_time,
            "average_fps": avg_fps,
            "output_dir": str(output_dir),
        }

        if verbose:
            print(f"\nProcessing complete!")
            print(f"Processed {processed_frames} frames in {total_time:.2f}s")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Binary masks saved to: {binary_dir}")
            print(f"Colored overlays saved to: {colored_dir}")

        return stats

    def process_image(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Process a single image with tiled segmentation.

        Args:
            image_path: Path to input image file
            output_dir: Directory to save output frames
            verbose: Print processing info

        Returns:
            Dictionary with processing statistics
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read image
        frame = cv2.imread(str(image_path))

        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        if verbose:
            print(f"\nProcessing image: {image_path.name}")
            print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
            print(f"Tile size: {self.slice_width}x{self.slice_height}")
            print(f"Overlap: {self.overlap_ratio * 100:.0f}%")
            print(f"Batch size: {self.batch_size}")

        # Process image
        start_time = time.time()
        binary_mask, colored_overlay = self.process_frame(frame)
        elapsed_time = time.time() - start_time

        # Save outputs
        image_stem = image_path.stem

        binary_path = output_dir / f"{image_stem}_binary.png"
        cv2.imwrite(str(binary_path), binary_mask)

        colored_path = output_dir / f"{image_stem}_colored.jpg"
        cv2.imwrite(str(colored_path), colored_overlay)

        stats = {
            "image_path": str(image_path),
            "processing_time": elapsed_time,
            "output_dir": str(output_dir),
            "binary_path": str(binary_path),
            "colored_path": str(colored_path),
        }

        if verbose:
            print(f"\nProcessing complete!")
            print(f"Processing time: {elapsed_time:.2f}s")
            print(f"Binary mask saved to: {binary_path}")
            print(f"Colored overlay saved to: {colored_path}")

        return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SAHI Tiled Video Segmentation - Process videos with overlapping tiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video with default settings
  python sahi_tiled_video_segmentation.py --source video.mp4 --model-path yolo11n-seg.pt

  # Process with custom tile size and overlap
  python sahi_tiled_video_segmentation.py --source video.mp4 --model-path yolo11n-seg.pt \\
      --slice-height 640 --slice-width 640 --overlap-ratio 0.25

  # Process with larger batch size for faster inference
  python sahi_tiled_video_segmentation.py --source video.mp4 --model-path yolo11n-seg.pt \\
      --batch-size 8

  # Process a single image
  python sahi_tiled_video_segmentation.py --source image.jpg --model-path yolo11n-seg.pt

  # Process every 5th frame
  python sahi_tiled_video_segmentation.py --source video.mp4 --model-path yolo11n-seg.pt \\
      --frame-skip 4
        """
    )

    # Required arguments
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input video or image file"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to segmentation model (e.g., yolo11n-seg.pt, yolov8n-seg.pt)"
    )

    # Optional arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="ultralytics",
        choices=["ultralytics", "mmdet", "detectron2", "torchvision"],
        help="Model type (default: ultralytics)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for results (default: output)"
    )

    parser.add_argument(
        "--slice-height",
        type=int,
        default=1024,
        help="Height of each tile in pixels (default: 1024)"
    )

    parser.add_argument(
        "--slice-width",
        type=int,
        default=1024,
        help="Width of each tile in pixels (default: 1024)"
    )

    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.33,
        help="Overlap ratio for tiles (default: 0.33 = 33%%)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of tiles to process simultaneously (default: 4)"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (default: cuda:0, use 'cpu' for CPU)"
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Process every Nth frame (0 = process all frames, default: 0)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Initialize detection model
    print(f"Loading model: {args.model_path}")
    print(f"Device: {args.device}")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    # Initialize processor
    processor = BatchedTiledSegmentation(
        detection_model=detection_model,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_ratio=args.overlap_ratio,
        batch_size=args.batch_size,
    )

    # Determine if source is video or image
    source_path = Path(args.source)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Check if it's a video or image based on extension
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    suffix = source_path.suffix.lower()

    if suffix in video_extensions:
        # Process video
        stats = processor.process_video(
            video_path=args.source,
            output_dir=args.output_dir,
            frame_skip_interval=args.frame_skip,
            verbose=not args.quiet,
        )
    elif suffix in image_extensions:
        # Process single image
        stats = processor.process_image(
            image_path=args.source,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}\n"
            f"Supported video formats: {video_extensions}\n"
            f"Supported image formats: {image_extensions}"
        )

    return stats


if __name__ == "__main__":
    main()
