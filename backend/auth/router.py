"""Authentication, registration, password reset, profile, and session routes."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.auth.security import (
    ACCESS_COOKIE_NAME,
    CSRF_COOKIE_NAME,
    create_access_token,
    generate_csrf_token,
    get_current_user,
    hash_password,
    hash_token,
    login_rate_limiter,
    verify_csrf,
    verify_password,
)
from backend.config import get_backend_settings
from backend.database.session import get_db
from backend.models import Role, SystemSetting, User, UserSession
from backend.schemas import (
    ForgotPasswordRequest,
    LoginRequest,
    ProfileUpdateRequest,
    RegisterRequest,
    ResetPasswordRequest,
    UserOut,
)
from backend.services.audit_service import record_audit_event

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _serialize_user(user: User) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        role=user.role.name,
        is_active=user.is_active,
        last_login_at=user.last_login_at,
        created_at=user.created_at,
    )


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _set_auth_cookies(
    response: Response,
    token: str,
    csrf_token: str,
    max_age_seconds: int,
) -> None:
    settings = get_backend_settings()
    response.set_cookie(
        key=ACCESS_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=max_age_seconds,
        path="/",
    )
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        httponly=False,  # Readable by frontend JS to echo in X-CSRF-Token header
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=max_age_seconds,
        path="/",
    )


@router.post("/login")
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Authenticate user with email or username and set HttpOnly JWT + CSRF cookies."""
    ident = payload.identifier.strip()
    rate_key = f"{_client_ip(request)}:{ident.lower()}"
    login_rate_limiter.check_and_record(rate_key)

    user = (
        db.query(User)
        .filter(or_(User.email.ilike(ident), User.username.ilike(ident)))
        .first()
    )
    if user is None or not verify_password(payload.password, user.password_hash):
        record_audit_event(
            db,
            action="LOGIN_FAILED",
            resource_type="auth",
            resource_id=ident,
            details="Invalid credentials",
            ip_address=_client_ip(request),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled.",
        )

    login_rate_limiter.reset(rate_key)
    settings = get_backend_settings()
    ttl_minutes = (
        settings.access_token_expire_minutes * 24 * 7
        if payload.remember_me
        else settings.access_token_expire_minutes
    )
    session_jti = secrets.token_urlsafe(32)
    token, expires_at = create_access_token(
        user,
        session_jti=session_jti,
        expires_delta=timedelta(minutes=ttl_minutes),
    )
    csrf_token = generate_csrf_token()

    db.add(
        UserSession(
            user_id=user.id,
            session_jti=session_jti,
            ip_address=_client_ip(request),
            user_agent=(request.headers.get("user-agent") or "")[:255],
            remember_me=payload.remember_me,
            expires_at=expires_at,
        )
    )
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)

    _set_auth_cookies(response, token, csrf_token, max_age_seconds=ttl_minutes * 60)
    record_audit_event(
        db,
        action="LOGIN_SUCCESS",
        resource_type="user",
        resource_id=str(user.id),
        details=f"role={user.role.name}",
        actor=user,
        ip_address=_client_ip(request),
    )
    return {
        "user": _serialize_user(user).model_dump(),
        "csrf_token": csrf_token,
    }


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Register a new account when enabled by administrator system_settings."""
    setting = (
        db.query(SystemSetting)
        .filter(SystemSetting.key == "allow_public_registration")
        .first()
    )
    is_allowed = setting is None or setting.value.strip().lower() in ("1", "true", "yes")
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public registration is currently disabled by administrator settings.",
        )

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
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email or username is already registered.",
        )

    role_setting = (
        db.query(SystemSetting)
        .filter(SystemSetting.key == "default_registration_role")
        .first()
    )
    role_name = role_setting.value.strip().upper() if role_setting else "OPERATOR"
    if role_name not in ("OPERATOR", "ADMIN"):
        role_name = "OPERATOR"

    role = db.query(Role).filter(Role.name == role_name).first()
    if role is None:
        raise HTTPException(status_code=500, detail="Default role is not configured.")

    user = User(
        email=email_norm,
        username=username_norm,
        full_name=payload.full_name.strip(),
        password_hash=hash_password(payload.password),
        role_id=role.id,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    record_audit_event(
        db,
        action="USER_REGISTERED",
        resource_type="user",
        resource_id=str(user.id),
        details=f"username={user.username}, role={role.name}",
        actor=user,
        ip_address=_client_ip(request),
    )
    return {"user": _serialize_user(user).model_dump()}


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Revoke the active session and clear authentication cookies."""
    jti: Optional[str] = getattr(request.state, "session_jti", None)
    if jti:
        session_row = (
            db.query(UserSession)
            .filter(UserSession.session_jti == jti, UserSession.user_id == current_user.id)
            .first()
        )
        if session_row is not None:
            session_row.is_revoked = True
            db.commit()

    response.delete_cookie(ACCESS_COOKIE_NAME, path="/")
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")
    record_audit_event(
        db,
        action="LOGOUT",
        resource_type="user",
        resource_id=str(current_user.id),
        actor=current_user,
        ip_address=_client_ip(request),
    )
    return {"message": "Logged out successfully."}


@router.get("/me")
def get_me(
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
) -> dict[str, object]:
    """Return the authenticated user's profile and refresh CSRF cookie if absent."""
    csrf = request.cookies.get(CSRF_COOKIE_NAME) or generate_csrf_token()
    if CSRF_COOKIE_NAME not in request.cookies:
        settings = get_backend_settings()
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=csrf,
            httponly=False,
            secure=settings.cookie_secure,
            samesite=settings.cookie_samesite,
            path="/",
        )
    return {"user": _serialize_user(current_user).model_dump(), "csrf_token": csrf}


@router.put("/profile")
def update_profile(
    payload: ProfileUpdateRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Update the authenticated user's full name or password."""
    if payload.full_name is not None:
        current_user.full_name = payload.full_name.strip()

    if payload.new_password is not None:
        if not payload.current_password or not verify_password(
            payload.current_password, current_user.password_hash
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect.",
            )
        current_user.password_hash = hash_password(payload.new_password)

    db.commit()
    db.refresh(current_user)
    record_audit_event(
        db,
        action="PROFILE_UPDATED",
        resource_type="user",
        resource_id=str(current_user.id),
        actor=current_user,
        ip_address=_client_ip(request),
    )
    return {"user": _serialize_user(current_user).model_dump()}


@router.get("/sessions")
def list_my_sessions(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """List active and recent sessions for the current user."""
    current_jti = getattr(request.state, "session_jti", None)
    rows = (
        db.query(UserSession)
        .filter(UserSession.user_id == current_user.id)
        .order_by(UserSession.created_at.desc())
        .limit(25)
        .all()
    )
    return {
        "sessions": [
            {
                "id": s.id,
                "ip_address": s.ip_address,
                "user_agent": s.user_agent,
                "remember_me": s.remember_me,
                "is_revoked": s.is_revoked,
                "is_current": s.session_jti == current_jti,
                "expires_at": s.expires_at,
                "created_at": s.created_at,
            }
            for s in rows
        ]
    }


@router.delete("/sessions/{session_id}")
def revoke_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    _: None = Depends(verify_csrf),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Revoke one of the current user's sessions."""
    session_row = (
        db.query(UserSession)
        .filter(UserSession.id == session_id, UserSession.user_id == current_user.id)
        .first()
    )
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    session_row.is_revoked = True
    db.commit()
    return {"message": "Session revoked."}


@router.post("/forgot-password")
def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """Generate a time-limited password reset token (never leaks user existence)."""
    user = db.query(User).filter(User.email == payload.email.strip().lower()).first()
    reset_token: Optional[str] = None
    if user is not None and user.is_active:
        reset_token = secrets.token_urlsafe(32)
        user.reset_token_hash = hash_token(reset_token)
        user.reset_token_expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
        db.commit()
        record_audit_event(
            db,
            action="PASSWORD_RESET_REQUESTED",
            resource_type="user",
            resource_id=str(user.id),
            actor=user,
            ip_address=_client_ip(request),
        )

    # In local/hackathon mode without SMTP configured, return a generic message
    # (and include dev_reset_token only if non-production cookie_secure is False).
    settings = get_backend_settings()
    resp: dict[str, object] = {
        "message": "If an account with that email exists, a password reset token has been generated.",
    }
    if not settings.cookie_secure and reset_token is not None:
        resp["dev_reset_token"] = reset_token
    return resp


@router.post("/reset-password")
def reset_password(
    payload: ResetPasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Reset password using a valid, unexpired reset token and revoke old sessions."""
    token_digest = hash_token(payload.token.strip())
    user = db.query(User).filter(User.reset_token_hash == token_digest).first()
    if user is None or user.reset_token_expires_at is None:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")

    exp = user.reset_token_expires_at
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    if exp < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Reset token has expired.")

    user.password_hash = hash_password(payload.new_password)
    user.reset_token_hash = None
    user.reset_token_expires_at = None
    db.query(UserSession).filter(UserSession.user_id == user.id).update({"is_revoked": True})
    db.commit()

    record_audit_event(
        db,
        action="PASSWORD_RESET_COMPLETED",
        resource_type="user",
        resource_id=str(user.id),
        actor=user,
        ip_address=_client_ip(request),
    )
    return {"message": "Password reset successfully. Please sign in with your new password."}
