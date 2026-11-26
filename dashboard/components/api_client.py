"""REST client wrapper communicating with the ARGUS-X FastAPI backend."""

from __future__ import annotations

from typing import Any, Optional

import requests


class ArgusApiClient:
    """Thin HTTP client keeping the Streamlit frontend decoupled from backend internals."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}{path}", json=payload or {}, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def statistics(self) -> dict[str, Any]:
        return self._get("/statistics")

    def list_cameras(self) -> list[dict[str, Any]]:
        return self._get("/cameras").get("cameras", [])

    def register_camera(
        self,
        camera_id: str,
        source: Optional[str] = None,
        name: Optional[str] = None,
        auto_start: bool = False,
    ) -> dict[str, Any]:
        return self._post(
            "/cameras/register",
            {
                "camera_id": camera_id,
                "source": source or None,
                "name": name or camera_id,
                "auto_start": auto_start,
            },
        )

    def disconnect_camera(self, camera_id: str) -> dict[str, Any]:
        return self._post(f"/cameras/{camera_id}/disconnect")

    def get_camera_frame_bytes(self, camera_id: str) -> Optional[bytes]:
        resp = requests.get(
            f"{self.base_url}/cameras/{camera_id}/frame", timeout=self.timeout
        )
        if resp.status_code == 200:
            return resp.content
        return None

    def list_identities(
        self, state: Optional[str] = None, query: Optional[str] = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if state and state != "ALL":
            params["state"] = state
        if query:
            params["query"] = query
        return self._get("/identities", params=params).get("identities", [])

    def get_identity_details(self, global_id: str) -> dict[str, Any]:
        return self._get(f"/identities/{global_id}")

    def list_unresolved(self, status: str = "PENDING") -> list[dict[str, Any]]:
        return self._get("/unresolved", params={"status": status}).get("unresolved", [])

    def review_unresolved(
        self,
        observation_id: str,
        decision: str,
        target_global_id: Optional[str],
        reason: str,
        operator_confirmed: bool,
    ) -> dict[str, Any]:
        return self._post(
            f"/unresolved/{observation_id}/review",
            {
                "decision": decision,
                "target_global_id": target_global_id,
                "reason": reason,
                "operator_confirmed": operator_confirmed,
            },
        )

    def merge_identities(
        self,
        source_global_id: str,
        target_global_id: str,
        reason: str,
        operator_approved: bool,
    ) -> dict[str, Any]:
        return self._post(
            "/identities/merge",
            {
                "source_global_id": source_global_id,
                "target_global_id": target_global_id,
                "reason": reason,
                "operator_approved": operator_approved,
            },
        )

    def get_audit_logs(
        self,
        global_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        decision: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if global_id:
            params["global_id"] = global_id
        if camera_id:
            params["camera_id"] = camera_id
        if decision and decision != "ALL":
            params["decision"] = decision
        return self._get("/audit", params=params).get("audit_logs", [])
