"""Application-level audit logging helper for administrative and security events."""

from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from backend.models import AuditLog, User


def record_audit_event(
    db: Session,
    action: str,
    resource_type: str,
    resource_id: str = "",
    details: str = "",
    actor: Optional[User] = None,
    ip_address: Optional[str] = None,
) -> AuditLog:
    """Persist an administrative or security audit event in the application DB."""
    entry = AuditLog(
        actor_user_id=actor.id if actor else None,
        actor_username=actor.username if actor else "system",
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
