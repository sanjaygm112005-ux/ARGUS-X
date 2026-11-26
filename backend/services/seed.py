"""Database seeding utilities for roles, default settings, and initial admin."""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.auth.security import hash_password
from backend.config import get_backend_settings
from backend.models import Role, SystemSetting, User

DEFAULT_ROLES = {
    "ADMIN": "Full administrative access to users, cameras, identities, audit logs, and settings.",
    "OPERATOR": "Monitoring access to live camera feeds, tracking, identity recovery, and review queue.",
}

DEFAULT_SETTINGS = {
    "allow_public_registration": (
        "true",
        "Controls whether new users can self-register via /api/auth/register.",
    ),
    "default_registration_role": (
        "OPERATOR",
        "Default role assigned to self-registered users.",
    ),
    "auto_merge_identities": (
        "false",
        "Safety guard: never automatically merge identities on appearance similarity alone.",
    ),
    "dashboard_url": (
        "http://localhost:5173/dashboard",
        "Configured ARGUS-X monitoring dashboard URL.",
    ),
}


def seed_database(db: Session, create_bootstrap_admin: bool = True) -> None:
    """Idempotently seed roles, system settings, and optional bootstrap admin."""
    role_map: dict[str, Role] = {}
    for role_name, desc in DEFAULT_ROLES.items():
        role = db.query(Role).filter(Role.name == role_name).first()
        if role is None:
            role = Role(name=role_name, description=desc)
            db.add(role)
            db.flush()
        role_map[role_name] = role

    for key, (value, desc) in DEFAULT_SETTINGS.items():
        setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
        if setting is None:
            db.add(SystemSetting(key=key, value=value, description=desc))

    if create_bootstrap_admin:
        settings = get_backend_settings()
        existing_admin = (
            db.query(User)
            .join(Role)
            .filter(Role.name == "ADMIN")
            .first()
        )
        if existing_admin is None:
            admin_user = User(
                email=settings.bootstrap_admin_email.lower(),
                username=settings.bootstrap_admin_username,
                full_name="System Administrator",
                password_hash=hash_password(settings.bootstrap_admin_password),
                role_id=role_map["ADMIN"].id,
                is_active=True,
            )
            db.add(admin_user)

    db.commit()
