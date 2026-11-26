"""Admin-only user management routes (/api/users)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.auth.security import hash_password, require_roles, verify_csrf
from backend.database.session import get_db
from backend.models import Role, User
from backend.schemas import UserCreateRequest, UserOut, UserUpdateRequest
from backend.services.audit_service import record_audit_event

router = APIRouter(prefix="/api/users", tags=["users"])


def _to_out(u: User) -> dict[str, object]:
    return UserOut(
        id=u.id,
        email=u.email,
        username=u.username,
        full_name=u.full_name,
        role=u.role.name,
        is_active=u.is_active,
        last_login_at=u.last_login_at,
        created_at=u.created_at,
    ).model_dump()


@router.get("")
def list_users(
    _: User = Depends(require_roles("ADMIN")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    users = db.query(User).order_by(User.id.asc()).all()
    return {"users": [_to_out(u) for u in users], "count": len(users)}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreateRequest,
    request: Request,
    admin: User = Depends(require_roles("ADMIN")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    email_norm = payload.email.strip().lower()
    username_norm = payload.username.strip()
    if "@" not in email_norm:
        raise HTTPException(status_code=422, detail="Valid email address is required.")

    existing = (
        db.query(User)
        .filter(or_(User.email == email_norm, User.username == username_norm))
        .first()
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="Email or username already exists.")

    role = db.query(Role).filter(Role.name == payload.role).first()
    if role is None:
        raise HTTPException(status_code=400, detail=f"Unknown role '{payload.role}'.")

    user = User(
        email=email_norm,
        username=username_norm,
        full_name=payload.full_name.strip(),
        password_hash=hash_password(payload.password),
        role_id=role.id,
        is_active=payload.is_active,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    record_audit_event(
        db,
        action="USER_CREATED",
        resource_type="user",
        resource_id=str(user.id),
        details=f"username={user.username}, role={role.name}",
        actor=admin,
        ip_address=request.client.host if request.client else None,
    )
    return {"user": _to_out(user)}


@router.get("/{user_id}")
def get_user(
    user_id: int,
    _: User = Depends(require_roles("ADMIN")),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"user": _to_out(user)}


@router.put("/{user_id}")
def update_user(
    user_id: int,
    payload: UserUpdateRequest,
    request: Request,
    admin: User = Depends(require_roles("ADMIN")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if payload.full_name is not None:
        user.full_name = payload.full_name.strip()
    if payload.is_active is not None:
        if user.id == admin.id and not payload.is_active:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own admin account.")
        user.is_active = payload.is_active
    if payload.role is not None:
        role = db.query(Role).filter(Role.name == payload.role).first()
        if role is None:
            raise HTTPException(status_code=400, detail=f"Unknown role '{payload.role}'.")
        user.role_id = role.id
    if payload.password is not None:
        user.password_hash = hash_password(payload.password)

    db.commit()
    db.refresh(user)
    record_audit_event(
        db,
        action="USER_UPDATED",
        resource_type="user",
        resource_id=str(user.id),
        details=f"role={user.role.name}, active={user.is_active}",
        actor=admin,
        ip_address=request.client.host if request.client else None,
    )
    return {"user": _to_out(user)}


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    request: Request,
    admin: User = Depends(require_roles("ADMIN")),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account.")

    username = user.username
    db.delete(user)
    db.commit()
    record_audit_event(
        db,
        action="USER_DELETED",
        resource_type="user",
        resource_id=str(user_id),
        details=f"username={username}",
        actor=admin,
        ip_address=request.client.host if request.client else None,
    )
    return {"message": "User deleted."}
