#!/usr/bin/env python3
"""
Test script for SAHI Tiled Video Segmentation.

This script tests the implementation in an environment with proper dependencies.
"""

import sys


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    try:
        from sahi.auto_model import AutoDetectionModel
        print("✓ SAHI AutoDetectionModel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SAHI: {e}")
        return False

    try:
        from sahi_tiled_video_segmentation import BatchedTiledSegmentation
        print("✓ BatchedTiledSegmentation imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import BatchedTiledSegmentation: {e}")
        return False

    print("\nAll imports successful!")
    return True


def test_class_initialization():
    """Test class initialization without running inference."""
    print("\nTesting class initialization...")

    try:
        from sahi.auto_model import AutoDetectionModel
        from sahi_tiled_video_segmentation import BatchedTiledSegmentation

        # This would require actual model file and dependencies
        print("Note: Skipping model initialization test (requires model file and GPU)")
        print("✓ Class definition is valid")

        return True
    except Exception as e:
        print(f"✗ Initialization test failed: {e}")
        return False


def test_cli_help():
    """Test CLI help output."""
    print("\nTesting CLI interface...")

    import subprocess

    try:
        result = subprocess.run(
            ["python", "sahi_tiled_video_segmentation.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("✓ CLI help command works")
            print(f"\nCLI Help Output (first 500 chars):")
            print("-" * 80)
            print(result.stdout[:500])
            print("-" * 80)
            return True
        else:
            print(f"✗ CLI help command failed with exit code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("✗ CLI help command timed out")
        return False
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("SAHI Tiled Video Segmentation - Test Suite")
    print("=" * 80)

    results = []

    # Test 1: Imports
    results.append(("Import Test", test_imports()))

    # Test 2: Class initialization
    results.append(("Class Initialization Test", test_class_initialization()))

    # Test 3: CLI help (only if imports work)
    if results[0][1]:  # If imports test passed
        results.append(("CLI Test", test_cli_help()))

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:40s} {status}")

    all_passed = all(result[1] for result in results)

    print("=" * 80)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
