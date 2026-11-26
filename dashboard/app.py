"""ARGUS-X Live Multi-Camera Monitoring Dashboard (Streamlit)."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.components.api_client import ArgusApiClient
from dashboard.components.theme import inject_dark_theme, render_kpi_card

st.set_page_config(
    page_title="ARGUS-X | Track the Untrackable",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_dark_theme()


def _fmt_ts(ts: object) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(ts)


with st.sidebar:
    st.markdown("### ARGUS-X Control")
    default_api = os.getenv("ARGUS_API_URL", "http://127.0.0.1:8000")
    api_url = st.text_input("FastAPI Backend URL", value=default_api)
    client = ArgusApiClient(base_url=api_url)

    if st.button("Refresh Telemetry", use_container_width=True):
        st.rerun()

    st.divider()
    st.markdown("#### Register Camera Stream")
    new_cam_id = st.text_input("Camera ID", placeholder="cam_lobby")
    new_cam_source = st.text_input("Source (path / index)", placeholder="0 or video.mp4")
    new_cam_start = st.checkbox("Auto-start stream", value=False)
    if st.button("Register Camera", use_container_width=True) and new_cam_id.strip():
        try:
            client.register_camera(
                camera_id=new_cam_id.strip(),
                source=new_cam_source.strip() or None,
                auto_start=new_cam_start,
            )
            st.success(f"Registered {new_cam_id.strip()}")
        except Exception as exc:
            st.error(f"Registration failed: {exc}")

st.markdown(
    """
    <div class="argus-header">
        <div class="argus-title">ARGUS-X — TRACK THE UNTRACKABLE</div>
        <div class="argus-subtitle">
            Multi-Camera Long-Horizon Identity Recovery &amp; Confidence-Aware Governance
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    health = client.health()
    stats = client.statistics()
except Exception as exc:
    st.error(
        f"Unable to reach ARGUS-X FastAPI backend at `{api_url}`: {exc}. "
        "Start it with `uvicorn api.main:app --host 127.0.0.1 --port 8000`."
    )
    st.stop()

tabs = st.tabs(
    [
        "1. Overview",
        "2. Multi-Camera View",
        "3. Global Identity Management",
        "4. Identity Recovery",
        "5. Manual Review",
        "6. Audit Logs",
    ]
)

# ---------------------------------------------------------------------------
# 1. OVERVIEW
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_kpi_card("Total Cameras", stats["total_cameras"])
    with c2:
        render_kpi_card("Online Cameras", stats["online_cameras"])
    with c3:
        render_kpi_card("Active Identities", stats["active_identities"])
    with c4:
        render_kpi_card("Temporarily Lost", stats["temporarily_lost_identities"])
    with c5:
        render_kpi_card("Recovered Identities", stats["recovered_identities"])
    with c6:
        render_kpi_card("Uncertain Obs.", stats["uncertain_observations"])

    st.markdown("")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(
            f"Backend Status: **{health['status'].upper()}** | "
            f"FAISS Vectors: **{stats['total_observations_in_memory']}** | "
            f"Conflicts Blocked: **{stats['total_conflicts']}**"
        )
    with col_b:
        st.info(
            f"Total Global Identities: **{stats['total_identities']}** | "
            f"Expired: **{stats['expired_identities']}** | "
            f"Audit Entries: **{stats['total_audit_logs']}**"
        )

# ---------------------------------------------------------------------------
# 2. MULTI-CAMERA VIEW
# ---------------------------------------------------------------------------
with tabs[1]:
    cameras = client.list_cameras()
    if not cameras:
        st.warning("No cameras registered yet. Use the sidebar to register a camera stream.")
    else:
        cols = st.columns(min(3, len(cameras)))
        for idx, cam in enumerate(cameras):
            with cols[idx % len(cols)]:
                badge = (
                    '<span class="badge-online">ONLINE</span>'
                    if cam["is_running"]
                    else '<span class="badge-offline">OFFLINE</span>'
                )
                st.markdown(
                    f"**{cam['name']}** (`{cam['camera_id']}`) &nbsp; {badge}",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Last timestamp: {_fmt_ts(cam['last_timestamp'])} | "
                    f"Frames: {cam['frame_count']}"
                )
                frame_bytes = client.get_camera_frame_bytes(cam["camera_id"])
                if frame_bytes:
                    st.image(frame_bytes, use_container_width=True)
                else:
                    st.markdown(
                        "<div class='kpi-card'>No live frame buffered yet.</div>",
                        unsafe_allow_html=True,
                    )

                dets = cam.get("active_detections", [])
                if dets:
                    det_rows = [
                        {
                            "Local ID": d["local_track_id"],
                            "Global ID": d["global_id"] or "UNCERTAIN",
                            "State": d["identity_state"],
                            "Conf": round(float(d["confidence"]), 3),
                            "BBox": [round(float(x), 1) for x in d["bounding_box"]],
                        }
                        for d in dets
                    ]
                    st.dataframe(pd.DataFrame(det_rows), use_container_width=True, hide_index=True)

                if cam["is_running"]:
                    if st.button(f"Disconnect {cam['camera_id']}", key=f"disc_{cam['camera_id']}"):
                        client.disconnect_camera(cam["camera_id"])
                        st.rerun()

# ---------------------------------------------------------------------------
# 3. GLOBAL IDENTITY MANAGEMENT
# ---------------------------------------------------------------------------
with tabs[2]:
    f1, f2 = st.columns([2, 1])
    with f1:
        search_gid = st.text_input("Search Global Identity ID", placeholder="gid_...")
    with f2:
        state_opt = st.selectbox(
            "Lifecycle Filter",
            ["ALL", "ACTIVE", "TEMPORARILY_LOST", "RECOVERED", "EXPIRED", "UNKNOWN"],
        )

    identities = client.list_identities(state=state_opt, query=search_gid.strip() or None)
    if identities:
        st.dataframe(pd.DataFrame(identities), use_container_width=True, hide_index=True)
        selected_gid = st.selectbox(
            "Inspect Global Identity Details",
            [i["global_id"] for i in identities],
        )
        if selected_gid:
            details = client.get_identity_details(selected_gid)
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("#### Observation History")
                obs_df = pd.DataFrame(details["observations"])
                st.dataframe(obs_df, use_container_width=True, hide_index=True)
            with d2:
                st.markdown("#### Camera Transition & Confidence Audit")
                st.write("Camera Sequence:", details["identity"]["camera_sequence"])
                audit_df = pd.DataFrame(details["audit_logs"])
                st.dataframe(audit_df, use_container_width=True, hide_index=True)
    else:
        st.info("No matching global identities found.")

# ---------------------------------------------------------------------------
# 4. IDENTITY RECOVERY
# ---------------------------------------------------------------------------
with tabs[3]:
    all_logs = client.get_audit_logs(limit=200)
    r_col1, r_col2, r_col3 = st.columns(3)
    recovered_logs = [e for e in all_logs if e["decision"] == "RECOVERED_ACCEPT"]
    uncertain_logs = [e for e in all_logs if e["decision"] == "UNCERTAIN"]
    rejected_logs = [
        e
        for e in all_logs
        if e["decision"] in ("REJECT_NEW_IDENTITY", "CONFLICT_BLOCKED", "MANUAL_REVIEW_REJECT")
    ]

    with r_col1:
        st.markdown("#### Recovered Identities")
        st.dataframe(pd.DataFrame(recovered_logs), use_container_width=True, hide_index=True)
    with r_col2:
        st.markdown("#### Uncertain Matches")
        st.dataframe(pd.DataFrame(uncertain_logs), use_container_width=True, hide_index=True)
    with r_col3:
        st.markdown("#### Rejected / Conflict-Blocked")
        st.dataframe(pd.DataFrame(rejected_logs), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 5. MANUAL REVIEW
# ---------------------------------------------------------------------------
with tabs[4]:
    unresolved = client.list_unresolved(status="PENDING")
    if not unresolved:
        st.success("No pending unresolved observations.")
    else:
        st.dataframe(pd.DataFrame(unresolved), use_container_width=True, hide_index=True)
        obs_choice = st.selectbox(
            "Select Unresolved Observation",
            [u["observation_id"] for u in unresolved],
        )
        selected_u = next(u for u in unresolved if u["observation_id"] == obs_choice)
        all_gids = [i["global_id"] for i in client.list_identities()]

        rev_decision = st.radio("Operator Decision", ["ACCEPT", "REJECT"], horizontal=True)
        target_gid = None
        if rev_decision == "ACCEPT":
            default_candidates = selected_u["candidate_ids"] or all_gids
            target_gid = st.selectbox("Assign to Global Identity", default_candidates or [""])
        rev_reason = st.text_input("Audit Reason", value="operator_visual_verification")
        rev_confirm = st.checkbox(
            "I confirm this manual review decision should be recorded in the audit log."
        )
        if st.button("Submit Review Decision", type="primary"):
            try:
                res = client.review_unresolved(
                    observation_id=obs_choice,
                    decision=rev_decision,
                    target_global_id=target_gid,
                    reason=rev_reason,
                    operator_confirmed=rev_confirm,
                )
                st.success(f"Review recorded: {res}")
                st.rerun()
            except Exception as exc:
                st.error(f"Review failed: {exc}")

    st.divider()
    st.markdown("#### Controlled Identity Merge (Requires Operator Confirmation)")
    all_gids = [i["global_id"] for i in client.list_identities()]
    if len(all_gids) >= 2:
        m1, m2 = st.columns(2)
        with m1:
            src_gid = st.selectbox("Source Global ID", all_gids, key="merge_src")
        with m2:
            dst_gid = st.selectbox("Target Global ID", all_gids, index=1, key="merge_dst")
        merge_reason = st.text_input("Merge Justification", value="operator_confirmed_same_person")
        merge_confirm = st.checkbox("Confirm controlled merge (logged permanently)")
        if st.button("Execute Controlled Merge"):
            try:
                res = client.merge_identities(src_gid, dst_gid, merge_reason, merge_confirm)
                st.success(f"Controlled merge logged: {res}")
            except Exception as exc:
                st.error(f"Merge blocked: {exc}")

# ---------------------------------------------------------------------------
# 6. AUDIT LOGS
# ---------------------------------------------------------------------------
with tabs[5]:
    a1, a2, a3 = st.columns(3)
    with a1:
        a_gid = st.text_input("Filter by Global ID", key="audit_gid")
    with a2:
        a_cam = st.text_input("Filter by Camera ID", key="audit_cam")
    with a3:
        a_dec = st.selectbox(
            "Decision Filter",
            [
                "ALL",
                "NEW_IDENTITY",
                "RECOVERED_ACCEPT",
                "UNCERTAIN",
                "REJECT_NEW_IDENTITY",
                "CONFLICT_BLOCKED",
                "CAMERA_DISCONNECTED",
                "EXPIRED",
                "MANUAL_REVIEW_ACCEPT",
                "MANUAL_REVIEW_REJECT",
                "MANUAL_MERGE",
            ],
        )
    audit_rows = client.get_audit_logs(
        global_id=a_gid.strip() or None,
        camera_id=a_cam.strip() or None,
        decision=a_dec,
        limit=300,
    )
    if audit_rows:
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No audit log entries match the current filters.")
