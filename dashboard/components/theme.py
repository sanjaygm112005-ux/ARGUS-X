"""Dark surveillance-analytics theme and UI helpers for Streamlit."""

from __future__ import annotations

import streamlit as st

DARK_THEME_CSS = """
<style>
    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    .argus-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 22px;
        margin-bottom: 18px;
    }
    .argus-title {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #38bdf8;
        margin: 0;
    }
    .argus-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .kpi-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: left;
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    .badge-online {
        background-color: #064e3b;
        color: #34d399;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-offline {
        background-color: #450a0a;
        color: #f87171;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
"""


def inject_dark_theme() -> None:
    """Inject custom dark surveillance-analytics CSS."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def render_kpi_card(label: str, value: object) -> None:
    """Render a compact dark-mode KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
