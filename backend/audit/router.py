"""Audit log and system settings routes (/api/audit-logs, /api/settings)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from backend.auth.security import require_roles, verify_csrf
from backend.database.session import get_db
from backend.models import AuditLog, SystemSetting, User
from backend.schemas import SystemSettingUpdateRequest
from backend.services.audit_service import record_audit_event

router = APIRouter(tags=["audit-and-settings"])


@router.get("/api/audit-logs")
def get_audit_logs(
    action: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    _: User = Depends(require_roles("ADMIN")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    query = db.query(AuditLog)
    if action:
        query = query.filter(AuditLog.action == action.strip())
    rows = query.order_by(AuditLog.created_at.desc()).limit(limit).all()
    return {
        "audit_logs": [
            {
                "id": r.id,
                "actor_username": r.actor_username,
                "action": r.action,
                "resource_type": r.resource_type,
                "resource_id": r.resource_id,
                "details": r.details,
                "ip_address": r.ip_address,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.get("/api/settings")
def list_settings(
    _: User = Depends(require_roles("ADMIN")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    rows = db.query(SystemSetting).order_by(SystemSetting.key.asc()).all()
    return {
        "settings": [
            {
                "key": s.key,
                "value": s.value,
                "description": s.description,
                "updated_at": s.updated_at.isoformat(),
            }
            for s in rows
        ]
    }


@router.put("/api/settings/{key}")
def update_setting(
    key: str,
    payload: SystemSettingUpdateRequest,
    request: Request,
    admin: User = Depends(require_roles("ADMIN")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if setting is None:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found.")

    if key == "auto_merge_identities" and payload.value.strip().lower() in ("1", "true", "yes"):
        raise HTTPException(
            status_code=400,
            detail="Automatic identity merging on appearance similarity alone is disabled by safety policy.",
        )

    setting.value = payload.value.strip()
    db.commit()
    db.refresh(setting)
    record_audit_event(
        db,
        action="SETTING_UPDATED",
        resource_type="system_setting",
        resource_id=key,
        details=f"value={setting.value}",
        actor=admin,
        ip_address=request.client.host if request.client else None,
    )
    return {
        "setting": {
            "key": setting.key,
            "value": setting.value,
            "description": setting.description,
            "updated_at": setting.updated_at.isoformat(),
        }
    }
