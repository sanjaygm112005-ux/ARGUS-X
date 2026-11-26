"""Dataset management API routes (/api/datasets)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.auth.security import require_roles
from backend.database.session import get_db
from backend.models import User
from backend.datasets.service import (
    discover_mot17_sequences,
    sync_mot17_dataset_record,
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("")
def list_datasets(
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """List available datasets registered in ARGUS-X (scans MOT17 automatically)."""
    mot17 = sync_mot17_dataset_record(db)
    return {
        "datasets": [
            {
                "id": mot17.get("dataset_id"),
                "name": "MOT17",
                "dataset_type": "MOT17",
                "status": mot17["status"],
                "root_path": mot17["root_path"],
                "sequence_count": mot17["sequence_count"],
                "setup_message": mot17["setup_message"],
            }
        ],
        "count": 1,
    }


@router.get("/mot17")
def get_mot17_dataset(
    detector: Optional[str] = Query(default="FRCNN"),
    split: Optional[str] = Query(default="train"),
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Return MOT17 sequence metadata, validation status, and GT summaries."""
    det = None if detector in (None, "", "ALL") else detector
    sp = None if split in (None, "", "ALL") else split
    info = discover_mot17_sequences(detector_filter=det, split_filter=sp)
    if det == "FRCNN" and sp == "train":
        sync_mot17_dataset_record(db)
    return info


@router.get("/mot17/{sequence_name}")
def get_mot17_sequence_detail(
    sequence_name: str,
    _: User = Depends(require_roles("ADMIN", "OPERATOR")),
) -> dict[str, object]:
    """Return detailed validation and annotation summary for a single MOT17 sequence."""
    info = discover_mot17_sequences(detector_filter=None, split_filter=None)
    for seq in info["sequences"]:
        if seq["name"] == sequence_name:
            return {"sequence": seq}
    raise HTTPException(status_code=404, detail=f"MOT17 sequence '{sequence_name}' not found.")
