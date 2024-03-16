import pandas as pd
import streamlit as st

data = pd.read_csv('./notebook/data/stockx/stockx_v2.csv')

dfcolor = data.groupby('Color2').size().reset_index(name='counts')['Color2']

def proper_case(s):
    return s.title()

dfcolor = dfcolor.apply(proper_case)

def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
        <h1 style="color:white;text-align:center;">Sneaker Resale Value Predictor</h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    st.markdown('<p style="margin-top: 10px">Hi, This is a Resale Value Calculator for Newly Released Sneakers ! Please give us a minute to input some parameters, then we will give you a value of the sneaker you would like to buy.</p>', unsafe_allow_html=True)

    s1 = st.selectbox('Brand', ['Nike', 'Adidas', 'Jordan'])

    s2 = st.selectbox('Color', dfcolor)

    s3 = st.text_input('Model', "")

    s4 = st.selectbox('Gender', ['Man', 'Woman'])

    if s4 == "Man":
        s4 = 1
    else:
        s4 = 2

    s3 = st.number_input('US Size', 1)

    if st.button('Predict'):
        st.balloons()
        st.success('Predicted Value: $1000')

if __name__ == '__main__':
    main()