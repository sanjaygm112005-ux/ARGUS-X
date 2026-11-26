"""ARGUS-X – TRACK THE UNTRACKABLE.

Minimal CLI entrypoint for environment verification and milestone execution.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import LOGS_DIR, AppConfig, get_config


def setup_logging(level: str, log_dir: Path) -> logging.Logger:
    """Configure console and file logging without failing if log_dir is missing."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "argus.log"

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger("argus_x")


def verify_environment(config: AppConfig, logger: logging.Logger) -> int:
    """Report active compute device (MPS vs CPU fallback) and core package status."""
    logger.info("ARGUS-X initialized | compute device=%s", config.device)
    logger.info("Milestone 1 target: YOLO11n (%s)", config.detection.model_name)

    packages = ("torch", "torchvision", "ultralytics", "cv2", "scipy", "faiss", "fastapi")
    missing: list[str] = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.warning(
            "Missing packages (%s). Install with: pip install -r requirements.txt",
            ", ".join(missing),
        )
        return 1

    logger.info("All core dependencies are available.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="argus-x",
        description="ARGUS-X: Long-Horizon Multi-Object Identity Recovery",
    )
    parser.add_argument(
        "--verify-env",
        action="store_true",
        help="Verify Apple Silicon MPS/CPU device resolution and installed packages.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional video/camera source for upcoming Milestone 1 detection.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run ARGUS-X entrypoint."""
    args = parse_args(argv)
    config = get_config()
    logger = setup_logging(config.log_level, LOGS_DIR)

    if args.verify_env or args.source is None:
        return verify_environment(config, logger)

    logger.info(
        "Source '%s' received. Implement Milestone 1 (detection/) to run YOLO11n inference.",
        args.source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
