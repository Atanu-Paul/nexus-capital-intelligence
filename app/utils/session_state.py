"""Manages chat/query session state."""

import streamlit as st

DEFAULTS = {
    "stage": "idle",
    "query": "",
    "sector": None,
    "engine": "auto",
    "current_record": None,
    "last_report": None,
    "error_message": None,
    "error_detail": None,
    "status_lines": {},
}


def init_session_state() -> None:
    """Initialize Streamlit session state keys."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_stage(stage: str) -> None:
    st.session_state["stage"] = stage


def update_status_line(stage: str, message: str) -> None:
    st.session_state["status_lines"][stage] = message
