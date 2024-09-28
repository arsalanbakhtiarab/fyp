import time
import numpy as np
import streamlit as st

# Streamed response emulator
def response_generator():
    response = np.random.choice(
        [
            "Hello there! How can I assist you today? 🌟💼",
            "Hi, human! Is there anything I can help you with? 🤖👋",
            "Do you need help? 🤔❓"
            "Hi there! How's the market treating you? 📈💼",
            "Hey! How's your trading day going? 💹💰",
            "Hello! How's the stock world treating you today? 🌍📊",
            "Hey! How's your portfolio holding up? 💼💸"
         ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.09)



def display_image_with_value(uptrend, downtrend, uptrend_value, downtrend_value):
    col1, col2 = st.columns(2)
    width = 100
    with col1:
        st.image(uptrend, width=width, caption='Uptrend', use_column_width=True)
        st.markdown("<style>div.row-widget.stHorizontal {padding-top: 30px; margin-bottom: -20px;}</style>", unsafe_allow_html=True)
        st.markdown(f"<div style='position:relative; text-align:center; top:-25px; color:white;'>{uptrend_value}</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.image(downtrend, width=width, caption='Downtrend', use_column_width=True)
        st.markdown("<style>div.row-widget.stHorizontal { padding-top: 30px; margin-bottom: -20px;}</style>", unsafe_allow_html=True)
        st.markdown(f"<div style='position:relative; text-align:center; top:-25px; color:white;'>{downtrend_value}</div>",
            unsafe_allow_html=True
        )
