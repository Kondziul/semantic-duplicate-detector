"""
app.py — Streamlit dashboard for the Semantic Duplicate Detector.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from data.sample_docs import SAMPLE_FRAGMENTS
from src.embedder import FragmentEmbedder
from src.detector import detect_duplicates

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Semantic Duplicate Detector",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

threshold = st.sidebar.slider(
    "Similarity threshold",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.01,
    help="Pairs with cosine similarity above this value are flagged as duplicates.",
)

model_name = st.sidebar.selectbox(
    "Embedding model",
    options=[
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
    ],
    index=0,
    help="Larger models are slower but may capture subtler paraphrases.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About:** Uses [sentence-transformers](https://sbert.net) to encode "
    "fragments into dense embeddings, then computes pairwise cosine similarity "
    "to surface near-duplicate content."
)

# ---------------------------------------------------------------------------
# Main — fragment input
# ---------------------------------------------------------------------------
st.title("🔍 Semantic Duplicate Detector")
st.caption("Identify near-duplicate content in technical documentation using sentence embeddings.")

tab_sample, tab_custom = st.tabs(["📄 Sample data", "✏️ Custom input"])

with tab_sample:
    st.info(
        f"Using **{len(SAMPLE_FRAGMENTS)} built-in fragments** that simulate "
        "Dolby-style audio/video documentation. Edit the `data/sample_docs.py` "
        "file to replace them with real content."
    )
    with st.expander("Show sample fragments"):
        for i, frag in enumerate(SAMPLE_FRAGMENTS):
            st.markdown(f"**[{i}]** {frag}")
    fragments = SAMPLE_FRAGMENTS

with tab_custom:
    raw = st.text_area(
        "Paste fragments here — one per line:",
        height=260,
        placeholder="The encoder compresses audio using perceptual coding.\n"
                    "Perceptual audio encoding reduces bitrate by exploiting hearing limits.\n"
                    "HDR video preserves highlight detail beyond SDR capabilities.",
    )
    custom_fragments = [l.strip() for l in raw.splitlines() if l.strip()]
    if custom_fragments:
        fragments = custom_fragments
        st.success(f"{len(fragments)} fragments ready.")

# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------
run = st.button("🚀 Detect duplicates", type="primary", use_container_width=True)

if run:
    with st.spinner("Encoding fragments …"):
        # Cache the embedder so Streamlit doesn't reload the model on every click
        @st.cache_resource
        def get_embedder(model: str) -> FragmentEmbedder:
            return FragmentEmbedder(model_name=model)

        embedder = get_embedder(model_name)
        embeddings = embedder.encode(fragments)

    with st.spinner("Computing similarities …"):
        result = detect_duplicates(fragments, embeddings, threshold=threshold)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Fragments analysed", result.n_fragments)
    col2.metric("Threshold", f"{threshold:.2f}")
    col3.metric("Duplicate candidates", result.n_pairs)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    st.subheader("Similarity heatmap")
    sim_df = pd.DataFrame(
        result.similarity_matrix,
        index=[f"[{i}]" for i in range(result.n_fragments)],
        columns=[f"[{i}]" for i in range(result.n_fragments)],
    )
    st.dataframe(
        sim_df.style.background_gradient(cmap="YlOrRd", vmin=0.0, vmax=1.0).format("{:.2f}"),
        use_container_width=True,
    )

    # ------------------------------------------------------------------
    # Duplicate pairs table
    # ------------------------------------------------------------------
    st.subheader(f"Candidate duplicate pairs (≥ {threshold:.2f})")

    if result.n_pairs == 0:
        st.warning("No pairs found above the current threshold. Try lowering it.")
    else:
        rows = [
            {
                "Pair": f"{p.idx_a} ↔ {p.idx_b}",
                "Similarity": round(p.similarity, 4),
                "Fragment A": p.text_a,
                "Fragment B": p.text_b,
            }
            for p in result.pairs
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # CSV download
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name="duplicates.csv",
            mime="text/csv",
        )
