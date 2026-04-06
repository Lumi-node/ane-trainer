"""Platform detection utilities for ANE availability."""

import platform


def is_apple_silicon() -> bool:
    """
    Detect if running on Apple Silicon (ARM64 Darwin).

    Returns:
        bool: True if platform is Darwin AND processor is ARM64/ARM, else False.

    Implementation:
        - Check platform.system() == 'Darwin'
        - Check 'arm64' or 'arm' in platform.processor().lower()
        - Return conjunction (both must be true)

    Error handling:
        - No exceptions raised; returns False on any detection failure
        - Handles missing or unusual platform.processor() output gracefully
    """
    try:
        # Check if system is Darwin
        if platform.system() != "Darwin":
            return False

        # Check if processor contains 'arm64' or 'arm'
        processor = platform.processor().lower()
        return "arm64" in processor or "arm" in processor
    except Exception:
        # If any error occurs during platform detection, return False
        return False
