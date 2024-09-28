from PIL import Image
import streamlit as st
from my_module import response_generator
from my_module import display_image_with_value
from main import get_up_down_trend_values

st.title('Trade :blue[Genie] :chart_with_upwards_trend:')
# MarketBuddy

prompt = st.chat_input("How i assist you.. ")

if not prompt:  # Check if prompt is empty
    st.chat_message("assistant").write(response_generator())
    uptrend_value = 0
    downtrend_value = 0
    uptrend = Image.open(r"D:\Streamlit\icons\bull.png")
    downtrend = Image.open(r"D:\Streamlit\icons\bear.png")
    display_image_with_value(uptrend, downtrend, uptrend_value, downtrend_value)
    
else:
    pr = f'Accoding To The Input "{prompt}" The Stocks Will BE '
    st.chat_message("assistant").write(pr)
    
    downtrend_value ,uptrend_value = get_up_down_trend_values(prompt)

    if uptrend_value > downtrend_value:
        uptrend = Image.open(r"D:\Streamlit\icons\bull_green.png")
        downtrend = Image.open(r"D:\Streamlit\icons\bear.png")
        display_image_with_value(uptrend, downtrend, uptrend_value, downtrend_value)

    elif uptrend_value < downtrend_value:
        uptrend = Image.open(r"D:\Streamlit\icons\bull.png")
        downtrend = Image.open(r"D:\Streamlit\icons\bear_red.png")
        display_image_with_value(uptrend, downtrend, uptrend_value, downtrend_value)

    else:
        uptrend = Image.open(r"D:\Streamlit\icons\bull.png")
        downtrend = Image.open(r"D:\Streamlit\icons\bear.png")
        display_image_with_value(uptrend, downtrend, uptrend_value, downtrend_value)
