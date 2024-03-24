import pandas as pd
import streamlit as st

from prediction import predict

# import joblib

data = pd.read_csv('./data/df_clean.csv')

dfcolor = data.groupby('ori_color1').size().reset_index(name='counts')['ori_color1']

dfbrand = data.groupby('ori_brand').size().reset_index(name='counts')['ori_brand']

dfgender = data.groupby('ori_gender').size().reset_index(name='counts')['ori_gender']

def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
        <h1 style="color:white;text-align:center;">Sneaker Resale Value Predictor</h1>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    st.markdown('<p style="margin-top: 10px">Hi, This is a Resale Value Calculator for Newly Released Sneakers ! Please give us a minute to input some parameters, then we will give you a value of the sneaker you would like to buy.</p>', unsafe_allow_html=True)

    txbrand = st.selectbox('Brand', dfbrand)

    txcolor = st.selectbox('Color', ['Black', 'White', 'Red', 'Blue', 'Yellow'])

    txmodel = st.text_input('Model', "")

    txgender = st.selectbox('Gender', dfgender)

    txsize = st.number_input('US Size', 4)

    if st.button('Predict'):
        price = predict(data, txmodel, txsize, txbrand, txgender, txcolor)
        st.success('Predicted Value: $' + str(price))
        st.balloons()


if __name__ == '__main__':
    main()

#To Do:
#1. Target prediction ? average annual or last sale?
#2. Codespace
#3. Hyperparameter
#4. To many color, shoes type 
#5. case sensitive
#6. Validasi input form
#7. Show picture
#8. Show graphic price
#9. Show recommend buy or no
#10. resize RF model