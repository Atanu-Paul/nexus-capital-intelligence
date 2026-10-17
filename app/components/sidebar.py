"""Sidebar UI component."""

import os

import streamlit as st

from rag.build_index import ensure_index


def render_sidebar() -> tuple[str | None, str]:
    """Render sidebar controls and return (sector, engine)."""
    st.sidebar.header("Controls")

    sector_map = {
        "All": None,
        "Technology": "technology",
        "Banking": "banking",
        "Pharma": "pharma",
    }
    sector_label = st.sidebar.selectbox(
        "Sector", list(sector_map.keys()), index=0, key="sidebar_sector"
    )
    sector = sector_map[sector_label]

    engine = st.sidebar.selectbox(
        "Engine", ["auto", "crewai", "standalone"], index=0, key="sidebar_engine"
    )
    os.environ["ENGINE"] = engine

    st.sidebar.markdown("**Example queries**")
    examples = [
        "Compare technology sector companies Q3 performance",
        "Which banking firms show the strongest growth?",
        "Summarize pharma sector risks and opportunities",
    ]
    for index, example in enumerate(examples):
        if st.sidebar.button(
            example,
            use_container_width=True,
            key=f"sidebar_example_{index}",
        ):
            st.session_state["query"] = example

    try:
        chunk_count = ensure_index()
        st.sidebar.success(f"Knowledge base: {chunk_count} chunks indexed")
    except Exception as exc:
        st.sidebar.error(f"Knowledge base error: {exc}")

    from deployment.s3_upload import S3Storage

    storage = S3Storage()
    if storage.available():
        st.sidebar.success("Storage: S3 connected")
    else:
        st.sidebar.warning("Storage: local-only (S3 unavailable)")

    if st.sidebar.button("Rebuild index", use_container_width=True, key="sidebar_rebuild_index"):
        with st.spinner("Rebuilding index..."):
            count = ensure_index(force=True)
        st.sidebar.success(f"Rebuilt index: {count} chunks")

    return sector, engine
