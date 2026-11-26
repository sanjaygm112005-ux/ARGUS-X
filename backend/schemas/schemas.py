"""Pydantic v2 request and response schemas for ARGUS-X full-stack API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    identifier: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=256)
    remember_me: bool = False


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    username: str = Field(..., min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    full_name: str = Field(default="", max_length=128)
    password: str = Field(..., min_length=8, max_length=256)


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)


class ResetPasswordRequest(BaseModel):
    token: str = Field(..., min_length=10, max_length=256)
    new_password: str = Field(..., min_length=8, max_length=256)


class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=128)
    current_password: Optional[str] = Field(default=None, max_length=256)
    new_password: Optional[str] = Field(default=None, min_length=8, max_length=256)


class UserCreateRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    username: str = Field(..., min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    full_name: str = Field(default="", max_length=128)
    password: str = Field(..., min_length=8, max_length=256)
    role: Literal["ADMIN", "OPERATOR"] = "OPERATOR"
    is_active: bool = True


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=128)
    role: Optional[Literal["ADMIN", "OPERATOR"]] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=8, max_length=256)


class UserOut(BaseModel):
    id: int
    email: str
    username: str
    full_name: str
    role: str
    is_active: bool
    last_login_at: Optional[datetime] = None
    created_at: datetime


class CameraCreateRequest(BaseModel):
    camera_code: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    name: str = Field(..., min_length=1, max_length=128)
    source: Optional[Union[str, int]] = None
    location: str = Field(default="", max_length=128)
    fps: Optional[float] = Field(default=None, gt=0.0)
    enabled: bool = True
    auto_start: bool = False


class CameraUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    source: Optional[Union[str, int]] = None
    location: Optional[str] = Field(default=None, max_length=128)
    fps: Optional[float] = Field(default=None, gt=0.0)
    enabled: Optional[bool] = None
    action: Optional[Literal["start", "stop"]] = None


class IdentityReviewPayload(BaseModel):
    decision: Literal["ACCEPT", "REJECT"]
    target_global_id: Optional[str] = Field(default=None, max_length=64)
    reason: str = Field(..., min_length=3, max_length=256)
    operator_confirmed: bool = False


class SystemSettingUpdateRequest(BaseModel):
    value: str = Field(..., min_length=1, max_length=512)
