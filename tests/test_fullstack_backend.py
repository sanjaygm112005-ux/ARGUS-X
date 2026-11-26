"""Integration tests for ARGUS-X full-stack auth, RBAC, DB, and AI bridge."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.state import ArgusServiceState
from backend.auth.security import CSRF_COOKIE_NAME, CSRF_HEADER_NAME, login_rate_limiter
from backend.database.base import Base
from backend.database.session import build_engine, get_db
from backend.main import create_fullstack_app
from backend.models import Identity, RecoveryEvent, SystemSetting, User
from backend.services.seed import seed_database


class TestFullStackBackend(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        db_url = f"sqlite:///{tmp_path / 'test_app.db'}"
        self.engine = build_engine(db_url)
        Base.metadata.create_all(bind=self.engine)
        self.TestSession = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False
        )

        db = self.TestSession()
        seed_database(db, create_bootstrap_admin=True)
        db.close()

        self.ai_state = ArgusServiceState(db_path=":memory:", index_path=None)
        self.app = create_fullstack_app(service_state=self.ai_state, init_db=False)

        def _override_get_db():
            s = self.TestSession()
            try:
                yield s
            finally:
                s.close()

        self.app.dependency_overrides[get_db] = _override_get_db
        self.client = TestClient(self.app)
        login_rate_limiter._attempts.clear()

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        self.ai_state.close()
        self.engine.dispose()
        self._tmp.cleanup()

    def _csrf_headers(self) -> dict[str, str]:
        token = self.client.cookies.get(CSRF_COOKIE_NAME, "")
        return {CSRF_HEADER_NAME: token}

    def _login(self, identifier: str, password: str) -> dict[str, object]:
        resp = self.client.post(
            "/api/auth/login",
            json={"identifier": identifier, "password": password, "remember_me": False},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def test_unauthenticated_visitor_blocked_from_protected_routes(self) -> None:
        for path in [
            "/api/auth/me",
            "/api/monitoring/status",
            "/api/cameras",
            "/api/users",
            "/api/identities",
            "/api/audit-logs",
        ]:
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 401, f"Expected 401 for {path}")

    def test_admin_login_csrf_logout_and_password_hashing(self) -> None:
        # Verify plaintext password is never stored
        db = self.TestSession()
        admin = db.query(User).filter(User.username == "admin").first()
        self.assertIsNotNone(admin)
        self.assertNotEqual(admin.password_hash, "ChangeMeNow!123")
        self.assertTrue(admin.password_hash.startswith("$argon2"))
        db.close()

        data = self._login("admin", "ChangeMeNow!123")
        self.assertEqual(data["user"]["role"], "ADMIN")

        # Missing CSRF header should be rejected on POST
        no_csrf = self.client.post(
            "/api/cameras",
            json={"camera_code": "cam_1", "name": "Lobby Cam"},
        )
        self.assertEqual(no_csrf.status_code, 403)

        # Logout revokes session
        out = self.client.post("/api/auth/logout", headers=self._csrf_headers())
        self.assertEqual(out.status_code, 200)
        me_after = self.client.get("/api/auth/me")
        self.assertEqual(me_after.status_code, 401)

    def test_registration_setting_and_operator_rbac(self) -> None:
        # Register an operator
        reg = self.client.post(
            "/api/auth/register",
            json={
                "email": "op1@argusx.local",
                "username": "operator1",
                "full_name": "Operator One",
                "password": "StrongPassword!123",
            },
        )
        self.assertEqual(reg.status_code, 201, reg.text)
        self.assertEqual(reg.json()["user"]["role"], "OPERATOR")

        # Login as operator
        self._login("operator1", "StrongPassword!123")
        self.assertEqual(self.client.get("/api/monitoring/status").status_code, 200)
        self.assertEqual(self.client.get("/api/cameras").status_code, 200)
        self.assertEqual(self.client.get("/api/identities").status_code, 200)

        # Operator forbidden from admin routes
        self.assertEqual(self.client.get("/api/users").status_code, 403)
        self.assertEqual(self.client.get("/api/audit-logs").status_code, 403)
        self.assertEqual(self.client.get("/api/settings").status_code, 403)
        self.assertEqual(
            self.client.post(
                "/api/cameras",
                json={"camera_code": "cam_x", "name": "Cam X"},
                headers=self._csrf_headers(),
            ).status_code,
            403,
        )

        # Admin disables public registration
        self._login("admin", "ChangeMeNow!123")
        upd = self.client.put(
            "/api/settings/allow_public_registration",
            json={"value": "false"},
            headers=self._csrf_headers(),
        )
        self.assertEqual(upd.status_code, 200)

        blocked_reg = self.client.post(
            "/api/auth/register",
            json={
                "email": "op2@argusx.local",
                "username": "operator2",
                "password": "StrongPassword!123",
            },
        )
        self.assertEqual(blocked_reg.status_code, 403)

    def test_login_rate_limiting_and_password_reset(self) -> None:
        for _ in range(5):
            r = self.client.post(
                "/api/auth/login",
                json={"identifier": "admin", "password": "WrongPassword!"},
            )
            self.assertEqual(r.status_code, 401)
        blocked = self.client.post(
            "/api/auth/login",
            json={"identifier": "admin", "password": "WrongPassword!"},
        )
        self.assertEqual(blocked.status_code, 429)

        # Forgot & reset password flow
        forgot = self.client.post(
            "/api/auth/forgot-password",
            json={"email": "admin@argusx.local"},
        )
        self.assertEqual(forgot.status_code, 200)
        token = forgot.json().get("dev_reset_token")
        self.assertTrue(token)

        reset = self.client.post(
            "/api/auth/reset-password",
            json={"token": token, "new_password": "NewAdminPassword!999"},
        )
        self.assertEqual(reset.status_code, 200)

        login_rate_limiter._attempts.clear()
        self._login("admin", "NewAdminPassword!999")

    def test_camera_ai_bridge_and_uncertain_review_no_embeddings(self) -> None:
        self.assertIs(self.app.state.ai_bridge.state, self.app.state.argus)
        self._login("admin", "ChangeMeNow!123")

        cam_resp = self.client.post(
            "/api/cameras",
            json={
                "camera_code": "cam_lobby",
                "name": "Main Lobby",
                "location": "Floor 1",
                "auto_start": False,
            },
            headers=self._csrf_headers(),
        )
        self.assertEqual(cam_resp.status_code, 201, cam_resp.text)
        # Registered in shared CameraManager
        cam_state = self.ai_state.camera_manager.get_camera("cam_lobby")
        self.assertEqual(cam_state.config.name, "Main Lobby")

        # Seed a DB identity + pending recovery event and verify no embeddings leak
        db = self.TestSession()
        ident = Identity(global_id="GID-001", lifecycle_state="RECOVERED", current_camera_code="cam_lobby")
        db.add(ident)
        db.flush()
        ev = RecoveryEvent(
            identity_id=ident.id,
            global_id="GID-001",
            from_camera_code="cam_gate",
            to_camera_code="cam_lobby",
            local_track_id=12,
            decision="UNCERTAIN",
            confidence=0.61,
            review_status="PENDING",
            reason="Low margin across cameras",
        )
        db.add(ev)
        db.commit()
        ev_id = ev.id
        db.close()

        ids_resp = self.client.get("/api/identities")
        self.assertEqual(ids_resp.status_code, 200)
        payload_str = str(ids_resp.json())
        self.assertNotIn("embedding", payload_str.lower())

        review_resp = self.client.post(
            f"/api/identities/db-{ev_id}/review",
            json={
                "decision": "ACCEPT",
                "target_global_id": "GID-001",
                "reason": "Verified by operator",
                "operator_confirmed": True,
            },
            headers=self._csrf_headers(),
        )
        self.assertEqual(review_resp.status_code, 200)
        self.assertEqual(review_resp.json()["review_status"], "ACCEPTED")


if __name__ == "__main__":
    unittest.main()
