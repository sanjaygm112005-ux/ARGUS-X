"""Shared AI Engine Bridge wrapping ArgusServiceState without duplicating models."""

from __future__ import annotations

import base64
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sqlalchemy.orm import Session

from api.state import ArgusServiceState
from backend.config import get_backend_settings
from backend.models import Camera, Identity, RecoveryEvent, TrackingEvent
from identity_memory import TrackletManager
from reidentification.feature_extractor import PersonFeatureExtractor
from tracking.tracker import ObjectTracker

logger = logging.getLogger(__name__)


class ArgusAIEngineBridge:
    """Coordinates the single ArgusServiceState instance with the application DB."""

    def __init__(self, service_state: Optional[ArgusServiceState] = None) -> None:
        settings = get_backend_settings()
        self.state = service_state or ArgusServiceState(
            db_path=settings.ai_memory_db_path,
            index_path=settings.ai_memory_index_path,
        )
        self._worker_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._workers: dict[str, threading.Event] = {}
        self._shared_extractor: Optional[PersonFeatureExtractor] = None

        # Synchronize FAISS vector index with SQLite observation storage
        mem = self.state.identity_manager.memory
        if mem.storage.count_observations() > 0 and mem._index.ntotal == 0:
            if settings.ai_memory_index_path.exists():
                try:
                    mem.load_index(settings.ai_memory_index_path)
                except Exception as exc:
                    logger.warning("Could not load FAISS index from %s: %s", settings.ai_memory_index_path, exc)
            if mem._index.ntotal == 0 and mem.storage.count_observations() > 0:
                logger.warning("Clearing orphaned SQLite observations without FAISS vectors.")
                try:
                    with mem.storage._conn:
                        mem.storage._conn.execute("DELETE FROM observations;")
                        mem.storage._conn.execute("DELETE FROM unresolved_observations;")
                    mem._vector_to_identity.clear()
                except Exception as exc:
                    logger.warning("Failed to clear orphaned observations: %s", exc)

    def _get_extractor(self) -> PersonFeatureExtractor:
        if self._shared_extractor is None:
            self._shared_extractor = PersonFeatureExtractor(device="cpu")
        return self._shared_extractor

    def start_video_worker(self, camera_code: str, source_path: str) -> None:
        """Process a local video stream in a background thread using shared AI models."""
        with self._worker_lock:
            if camera_code in self._workers:
                self._workers[camera_code].set()
            stop_event = threading.Event()
            self._workers[camera_code] = stop_event

        # Ensure CameraManager recognizes camera as running
        try:
            cam_state = self.state.camera_manager.get_camera(camera_code)
            cam_state.is_running = True
        except KeyError:
            pass

        def _run() -> None:
            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                logger.warning("Could not open video source for %s: %s", camera_code, source_path)
                with self._worker_lock:
                    self._workers.pop(camera_code, None)
                return
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            frame_delay = min(0.08, 1.0 / max(1.0, fps))
            tracker = ObjectTracker(device="cpu")
            extractor = self._get_extractor()
            tm = TrackletManager(
                camera_id=camera_code,
                feature_extractor=None,
                inactive_timeout_sec=0.0,
                extraction_interval_sec=0.2,
            )
            frame_idx = 0
            settings = get_backend_settings()
            try:
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        # Loop video cleanly for continuous monitoring demo
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    ts = time.time()
                    with self._inference_lock:
                        tracked = tracker.track_frame(frame)
                        person_objs = [o for o in tracked if o.class_id == 0]
                        precomputed: dict[int, np.ndarray] = {}
                        if person_objs:
                            crops, tids = [], []
                            for o in person_objs:
                                try:
                                    crops.append(extractor.crop_person(frame, o.bbox))
                                    tids.append(o.track_id)
                                except ValueError:
                                    continue
                            if crops:
                                feats = extractor.extract_batch(crops)
                                for feat_row, tid in zip(feats, tids):
                                    precomputed[tid] = feat_row

                    upd = tm.update(tracked, timestamp=ts, precomputed_embeddings=precomputed)
                    with self.state._lock:
                        outs = self.state.identity_manager.process_tracklets(
                            upd.active_tracklets, current_timestamp=ts
                        )
                        self.state.ingest_frame_outputs(camera_code, frame, outs, ts)
                        if frame_idx % 120 == 0 and settings.ai_memory_index_path:
                            try:
                                self.state.identity_manager.save_index(settings.ai_memory_index_path)
                            except Exception:
                                pass
                    frame_idx += 1
                    time.sleep(frame_delay)
            except Exception as exc:
                logger.exception("Video worker error for %s: %s", camera_code, exc)
            finally:
                cap.release()
                with self._worker_lock:
                    self._workers.pop(camera_code, None)
                try:
                    self.state.camera_manager.get_camera(camera_code).is_running = False
                except KeyError:
                    pass

        thread = threading.Thread(target=_run, name=f"argus-cam-{camera_code}", daemon=True)
        thread.start()

    def stop_video_worker(self, camera_code: str) -> None:
        with self._worker_lock:
            ev = self._workers.pop(camera_code, None)
            if ev is not None:
                ev.set()
        try:
            self.state.camera_manager.get_camera(camera_code).is_running = False
        except KeyError:
            pass

    def register_or_sync_camera(
        self,
        cam: Camera,
        auto_start: bool = False,
    ) -> dict[str, object]:
        """Register a DB camera in CameraManager if not already registered."""
        try:
            self.state.camera_manager.get_camera(cam.camera_code)
            if auto_start:
                if cam.source and Path(cam.source).is_file():
                    self.start_video_worker(cam.camera_code, cam.source)
                else:
                    self.state.camera_manager.start_camera(cam.camera_code)
            return self.state.get_camera_stream_payload(cam.camera_code)
        except KeyError:
            res = self.state.register_camera(
                camera_id=cam.camera_code,
                source=cam.source,
                name=cam.name,
                fps=cam.fps,
                enabled=cam.enabled,
                auto_start=False,
            )
            if auto_start:
                if cam.source and Path(cam.source).is_file():
                    self.start_video_worker(cam.camera_code, cam.source)
                else:
                    self.state.camera_manager.start_camera(cam.camera_code)
            return res

    def start_camera(self, camera_code: str) -> None:
        self.state.camera_manager.start_camera(camera_code)

    def stop_camera(self, camera_code: str) -> dict[str, object]:
        self.stop_video_worker(camera_code)
        return self.state.disconnect_camera(camera_code)

    def remove_camera(self, camera_code: str) -> None:
        self.stop_video_worker(camera_code)
        try:
            self.state.camera_manager.stop_camera(camera_code)
        except KeyError:
            pass

    def get_monitoring_snapshot(self, db: Session) -> dict[str, object]:
        """Return safe live status, cameras, identities, and alerts (no embeddings)."""
        ai_stats = self.state.get_statistics()
        live_cams = {c["camera_id"]: c for c in self.state.list_cameras()}
        db_cams = db.query(Camera).order_by(Camera.id.asc()).all()

        cameras_payload = []
        for cam in db_cams:
            live = live_cams.get(cam.camera_code)
            is_worker_running = cam.camera_code in self._workers
            is_running = bool(live["is_running"]) if live else is_worker_running
            active_tracks = len(live.get("active_detections", [])) if live else 0

            jpg = self.state.latest_frames_jpeg.get(cam.camera_code)
            frame_b64 = base64.b64encode(jpg).decode("ascii") if jpg else None
            last_ts = self.state.latest_timestamps.get(cam.camera_code)
            if last_ts is None and live:
                last_ts = live.get("last_timestamp")

            is_online = is_running or is_worker_running or (jpg is not None)
            status_str = "ONLINE" if is_online else (cam.status or "OFFLINE")
            cameras_payload.append(
                {
                    "id": cam.id,
                    "camera_code": cam.camera_code,
                    "name": cam.name,
                    "location": cam.location,
                    "source": cam.source,
                    "enabled": cam.enabled,
                    "status": status_str,
                    "is_running": is_online,
                    "active_tracks": active_tracks,
                    "last_frame_timestamp": last_ts,
                    "frame_b64": frame_b64,
                    "last_error": None,
                }
            )

        ai_identities = self.list_safe_identities(db)
        uncertain = self.list_uncertain_items(db)
        return {
            "status": "connected",
            "statistics": {
                "total_cameras": max(len(db_cams), int(ai_stats["total_cameras"])),
                "online_cameras": sum(1 for c in cameras_payload if c["is_running"]),
                "total_identities": len(ai_identities),
                "active_identities": sum(
                    1 for i in ai_identities if i["lifecycle_state"] in ("ACTIVE", "RECOVERED")
                ),
                "recovered_identities": sum(
                    1 for i in ai_identities if i["lifecycle_state"] == "RECOVERED"
                ),
                "uncertain_count": len(uncertain),
            },
            "cameras": cameras_payload,
            "identities": ai_identities,
            "uncertain_alerts": uncertain,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def list_safe_identities(self, db: Session) -> list[dict[str, object]]:
        """Combine live AI identity manager records with DB identity records (no embeddings)."""
        by_gid: dict[str, dict[str, object]] = {}

        try:
            obs_counts = self.state.identity_manager.memory.storage.get_observation_counts()
        except Exception:
            obs_counts = {}

        with self.state._lock:
            for rec in self.state.identity_manager.list_identities():
                cnt = obs_counts.get(rec.global_id, 0)
                by_gid[rec.global_id] = {
                    "global_id": rec.global_id,
                    "lifecycle_state": rec.state.value,
                    "current_camera_code": rec.current_camera_id,
                    "camera_sequence": list(rec.camera_sequence),
                    "observation_count": cnt,
                    "has_conflict": rec.state.value == "UNCERTAIN",
                    "first_seen_at": rec.created_at,
                    "last_seen_at": rec.last_seen_at,
                }

        for row in db.query(Identity).all():
            if row.global_id not in by_gid:
                by_gid[row.global_id] = {
                    "global_id": row.global_id,
                    "lifecycle_state": row.lifecycle_state,
                    "current_camera_code": row.current_camera_code,
                    "camera_sequence": [row.current_camera_code] if row.current_camera_code else [],
                    "observation_count": len(row.observations) if row.observations else 0,
                    "has_conflict": row.has_conflict,
                    "first_seen_at": row.first_seen_at.isoformat() if row.first_seen_at else "",
                    "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else "",
                }

        return list(by_gid.values())

    def list_uncertain_items(self, db: Session) -> list[dict[str, object]]:
        """List pending uncertain observations from AI manager and DB."""
        items: list[dict[str, object]] = []
        with self.state._lock:
            unresolved = self.state.identity_manager.list_unresolved_observations(status="PENDING")
        for u in unresolved:
            items.append(
                {
                    "observation_id": u.tracklet_uid,
                    "source": "ai_engine",
                    "camera_code": u.camera_id,
                    "local_track_id": u.local_track_id,
                    "candidate_global_id": u.candidate_ids[0] if u.candidate_ids else None,
                    "candidate_ids": list(u.candidate_ids),
                    "confidence": round(float(u.confidence), 4),
                    "reason": u.reason,
                    "status": u.status,
                    "timestamp": u.last_seen_at,
                }
            )

        db_pending = (
            db.query(RecoveryEvent)
            .filter(RecoveryEvent.review_status == "PENDING")
            .order_by(RecoveryEvent.recovered_at.desc())
            .all()
        )
        for ev in db_pending:
            items.append(
                {
                    "observation_id": f"db-{ev.id}",
                    "source": "database",
                    "camera_code": ev.to_camera_code,
                    "local_track_id": ev.local_track_id,
                    "candidate_global_id": ev.global_id,
                    "confidence": round(float(ev.confidence), 4),
                    "reason": ev.reason,
                    "status": ev.review_status,
                    "timestamp": ev.recovered_at.isoformat(),
                }
            )
        return items

    def get_recent_tracking_events(self, db: Session, limit: int = 50) -> list[dict[str, object]]:
        rows = (
            db.query(TrackingEvent)
            .order_by(TrackingEvent.observed_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "camera_id": r.camera_id,
                "local_track_id": r.local_track_id,
                "global_id": r.global_id,
                "bbox": [r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2],
                "confidence": r.confidence,
                "decision": r.decision,
                "observed_at": r.observed_at.isoformat(),
            }
            for r in rows
        ]

    def close(self) -> None:
        self.state.close()
