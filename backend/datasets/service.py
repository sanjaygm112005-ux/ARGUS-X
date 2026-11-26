"""MOT17 dataset discovery, seqinfo.ini parsing, and ground-truth validation."""

from __future__ import annotations

import configparser
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from backend.config import PROJECT_ROOT
from backend.models import Dataset


@dataclass(frozen=True)
class MOT17GroundTruthSummary:
    """Summary statistics for a MOT17 ground-truth annotation file."""

    gt_path: str
    exists: bool
    total_rows: int
    pedestrian_rows: int
    unique_gt_ids: int
    annotated_frames: int


@dataclass(frozen=True)
class MOT17SequenceInfo:
    """Validated metadata for a single MOT17 sequence."""

    name: str
    split: str
    detector_type: str
    sequence_dir: str
    im_dir: str
    frame_rate: int
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str
    actual_frame_count: int
    has_ground_truth: bool
    gt_summary: Optional[MOT17GroundTruthSummary] = None
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_seqinfo_ini(seq_dir: Path) -> dict[str, str]:
    """Parse MOT17 seqinfo.ini into a dictionary."""
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"Missing seqinfo.ini in {seq_dir}")
    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    if "Sequence" not in parser:
        raise ValueError(f"Invalid seqinfo.ini (missing [Sequence]) in {ini_path}")
    return dict(parser["Sequence"])


def load_mot17_ground_truth(gt_path: Path, pedestrian_only: bool = True) -> dict[int, list[dict[str, float]]]:
    """Load MOT17 gt.txt grouped by 1-based frame index without modifying the file.

    MOT17 gt.txt format:
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    """
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    frames: dict[int, list[dict[str, float]]] = {}
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split(",")
            if len(parts) < 6:
                continue
            frame_idx = int(float(parts[0]))
            obj_id = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls_id = int(float(parts[7])) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0

            if pedestrian_only:
                if conf <= 0 or cls_id != 1:
                    continue

            frames.setdefault(frame_idx, []).append(
                {
                    "id": obj_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": conf,
                    "class": cls_id,
                    "visibility": vis,
                }
            )
    return frames


def summarize_mot17_gt(gt_path: Path) -> MOT17GroundTruthSummary:
    """Compute lightweight summary counts from gt/gt.txt."""
    if not gt_path.is_file():
        return MOT17GroundTruthSummary(
            gt_path=str(gt_path),
            exists=False,
            total_rows=0,
            pedestrian_rows=0,
            unique_gt_ids=0,
            annotated_frames=0,
        )

    total_rows = 0
    ped_rows = 0
    ids: set[int] = set()
    frames: set[int] = set()
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            total_rows += 1
            parts = raw.split(",")
            if len(parts) < 6:
                continue
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls_id = int(float(parts[7])) if len(parts) > 7 else 1
            if conf > 0 and cls_id == 1:
                ped_rows += 1
                ids.add(int(float(parts[1])))
                frames.add(int(float(parts[0])))

    return MOT17GroundTruthSummary(
        gt_path=str(gt_path),
        exists=True,
        total_rows=total_rows,
        pedestrian_rows=ped_rows,
        unique_gt_ids=len(ids),
        annotated_frames=len(frames),
    )


def discover_mot17_sequences(
    mot17_root: Optional[Path] = None,
    detector_filter: Optional[str] = "FRCNN",
    split_filter: Optional[str] = "train",
) -> dict[str, object]:
    """Discover and validate MOT17 sequences under datasets/MOT17/."""
    root = mot17_root or (PROJECT_ROOT / "datasets" / "MOT17")
    if not root.exists() or not root.is_dir():
        return {
            "dataset": "MOT17",
            "status": "MISSING",
            "root_path": str(root),
            "setup_message": (
                "MOT17 dataset not found at datasets/MOT17/. "
                "Download MOT17 from motchallenge.net and extract train/ and test/ folders into datasets/MOT17/."
            ),
            "sequences": [],
            "sequence_count": 0,
        }

    sequences: list[MOT17SequenceInfo] = []
    splits = [split_filter] if split_filter else ["train", "test"]
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir() or seq_dir.name.startswith("."):
                continue
            # Extract detector type (e.g., MOT17-02-FRCNN -> FRCNN)
            parts = seq_dir.name.split("-")
            det_type = parts[-1].upper() if len(parts) >= 3 else "UNKNOWN"
            if detector_filter and det_type != detector_filter.upper():
                continue

            errors: list[str] = []
            try:
                ini = parse_seqinfo_ini(seq_dir)
                name = ini.get("name", seq_dir.name)
                im_dir_name = ini.get("imdir", "img1")
                frame_rate = int(ini.get("framerate", "30"))
                seq_length = int(ini.get("seqlength", "0"))
                im_width = int(ini.get("imwidth", "0"))
                im_height = int(ini.get("imheight", "0"))
                im_ext = ini.get("imext", ".jpg")
            except Exception as exc:
                errors.append(str(exc))
                sequences.append(
                    MOT17SequenceInfo(
                        name=seq_dir.name,
                        split=split,
                        detector_type=det_type,
                        sequence_dir=str(seq_dir),
                        im_dir="img1",
                        frame_rate=0,
                        seq_length=0,
                        im_width=0,
                        im_height=0,
                        im_ext=".jpg",
                        actual_frame_count=0,
                        has_ground_truth=False,
                        is_valid=False,
                        validation_errors=errors,
                    )
                )
                continue

            img_folder = seq_dir / im_dir_name
            actual_frames = 0
            if not img_folder.is_dir():
                errors.append(f"Missing image directory: {im_dir_name}")
            else:
                actual_frames = len(list(img_folder.glob(f"*{im_ext}")))
                if actual_frames == 0:
                    errors.append(f"No {im_ext} frames found in {im_dir_name}/")
                elif seq_length > 0 and actual_frames != seq_length:
                    errors.append(
                        f"Frame count mismatch: seqinfo={seq_length}, found={actual_frames}"
                    )

            gt_path = seq_dir / "gt" / "gt.txt"
            gt_summary = summarize_mot17_gt(gt_path)
            if split == "train" and not gt_summary.exists:
                errors.append("Missing ground-truth file: gt/gt.txt")

            sequences.append(
                MOT17SequenceInfo(
                    name=name,
                    split=split,
                    detector_type=det_type,
                    sequence_dir=str(seq_dir),
                    im_dir=im_dir_name,
                    frame_rate=frame_rate,
                    seq_length=seq_length,
                    im_width=im_width,
                    im_height=im_height,
                    im_ext=im_ext,
                    actual_frame_count=actual_frames,
                    has_ground_truth=gt_summary.exists,
                    gt_summary=gt_summary,
                    is_valid=len(errors) == 0,
                    validation_errors=errors,
                )
            )

    status = "AVAILABLE" if sequences else "MISSING"
    setup_msg = None
    if not sequences:
        setup_msg = (
            "No matching MOT17 sequences found under datasets/MOT17/. "
            "Ensure train/MOT17-*-FRCNN directories with seqinfo.ini, img1/, and gt/gt.txt are present."
        )

    return {
        "dataset": "MOT17",
        "status": status,
        "root_path": str(root),
        "setup_message": setup_msg,
        "sequences": [s.to_dict() for s in sequences],
        "sequence_count": len(sequences),
    }


def sync_mot17_dataset_record(db: Session, mot17_root: Optional[Path] = None) -> dict[str, object]:
    """Discover MOT17 and persist its summary record into Neon datasets table."""
    info = discover_mot17_sequences(mot17_root=mot17_root, detector_filter="FRCNN", split_filter="train")
    row = db.query(Dataset).filter(Dataset.name == "MOT17").first()
    meta_str = json.dumps(
        {
            "setup_message": info["setup_message"],
            "sequence_names": [s["name"] for s in info["sequences"]],
        }
    )
    now = datetime.now(timezone.utc)
    if row is None:
        row = Dataset(
            name="MOT17",
            dataset_type="MOT17",
            root_path=str(info["root_path"]),
            status=str(info["status"]),
            sequence_count=int(info["sequence_count"]),
            metadata_json=meta_str,
            last_scanned_at=now,
        )
        db.add(row)
    else:
        row.root_path = str(info["root_path"])
        row.status = str(info["status"])
        row.sequence_count = int(info["sequence_count"])
        row.metadata_json = meta_str
        row.last_scanned_at = now
    db.commit()
    db.refresh(row)
    info["dataset_id"] = row.id
    return info
