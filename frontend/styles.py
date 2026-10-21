"""Load the interface stylesheet independently of workflow code."""

from pathlib import Path

import streamlit as st


def apply_card_styles():
    css = Path(__file__).with_name("styles.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
