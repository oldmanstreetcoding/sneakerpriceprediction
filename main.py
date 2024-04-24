import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nicegui import ui

import pandas as pd
import joblib

ui.add_head_html('''
    <style>
        desktop {
            width: 50%;
        }
    </style>
''')

rfmodel = 'rf_model.joblib'

data = pd.read_csv("data/input_df.csv")

global_model = None
global_color = None
global_imgurl = None
global_price = None

async def predict_price(input_features_dict):
    # Load the trained model and the imputer
    model = joblib.load(rfmodel)

    # Convert the input features dictionary to a DataFrame
    input_df = pd.DataFrame(input_features_dict, index=[0])

    # Ensure that the order of the features matches the training data
    top_features = ['averagePrice_Annual_Statistics', 'size', 'days_since_release', 'salesCount_Annual', 'retailPrice', 'brand_adidas', 'brand_Nike', 'gender_men', 'brand_ASICS', 'gender_women', 'brand_Jordan', 'brand_New Balance']
    
    input_df = input_df[top_features]

    # Make prediction
    prediction = model.predict(input_df)

    global global_price

    global_price = prediction[0]

    show_spinner.refresh(False)


def fetch_features(brand, model, gender, color, size):
    # Compute the average 'retailPrice' for the specified brand
    brand_average_price = data[(data['brand'] == brand) & (data['primaryTitle'] == model)]['retailPrice'].mean()

    # Filter the DataFrame based on the user input
    filtered_df = data[(data['brand'] == brand) &
                                 (data['primaryTitle'] == model) &
                                 (data['gender'] == gender) &
                                 (data['color1'] == color) &
                                 (data['size'] == size)]

    # Check if the filtered DataFrame is empty
    if filtered_df.empty:
        # Return the average values for the specified brand
        return {
            'averagePrice_Annual_Statistics': brand_average_price,
            'size': size,  # size is provided as a parameter
            'days_since_release': data[(data['brand'] == brand) & (data['primaryTitle'] == model)]['days_since_release'].mean(),
            'salesCount_Annual': data[(data['brand'] == brand) & (data['primaryTitle'] == model)]['salesCount_Annual'].mean(),
            'retailPrice': brand_average_price,
            'brand_adidas': 1 if brand == 'adidas' else 0,
            'brand_Nike': 1 if brand == 'Nike' else 0,
            'gender_men': 1 if gender == 'men' else 0,
            'gender_women': 1 if gender == 'women' else 0,
            'brand_ASICS': 1 if brand == 'ASICS' else 0,
            'brand_Jordan': 1 if brand == 'Jordan' else 0,
            'brand_New Balance': 1 if brand == 'New Balance' else 0
        }

    # If the DataFrame is not empty, return the average value, except retailPrice using the latest release price
    most_recent_release = filtered_df.sort_values('days_since_release', ascending=True).iloc[0]

    return {
        'averagePrice_Annual_Statistics': filtered_df['averagePrice_Annual_Statistics'].mean(),
        'size': size,
        'days_since_release': most_recent_release['days_since_release'],
        'salesCount_Annual': filtered_df['salesCount_Annual'].mean(),
        'retailPrice': most_recent_release['retailPrice'],
        'brand_adidas': 1 if brand == 'adidas' else 0,
        'brand_Nike': 1 if brand == 'Nike' else 0,
        'gender_men': 1 if gender == 'men' else 0,
        'gender_women': 1 if gender == 'women' else 0,
        'brand_ASICS': 1 if brand == 'ASICS' else 0,
        'brand_Jordan': 1 if brand == 'Jordan' else 0,
        'brand_New Balance': 1 if brand == 'New Balance' else 0
    }

async def getPrediction() -> None:

    show_spinner.refresh(True)

    global global_model
    global global_color
    global global_imgurl
    global global_price

    brand = txbrand.value
    gender = txgender.value
    size = txsize.value

    model = global_model
    color = global_color
    tximageURL = global_imgurl

    if brand == None or model == None or color == None or size == 0 or gender == None:
        ui.notify('Please fill in all the fields')
        show_spinner.refresh(False)
    else:

        dfPriceChart = data[(data['brand'] == brand) & (data['primaryTitle'] == model) & (data['gender'] == gender.lower()) & (data['color1'] == color)]
        
        dfPriceChart = dfPriceChart[['releaseDate', 'lastSale']].sort_values(by='releaseDate')
        # print(dfPriceChart)

        # Fetch other features
        input_features_dict = fetch_features(brand, model, gender, color, size)

        print(input_features_dict)

        # Predict price
        await predict_price(input_features_dict)

        fprice = "{:,.1f}".format(global_price)

        with ui.dialog() as dialog, ui.card():
            with ui.row().classes('w-full justify-center').style('padding:5px;border:2px solid tomato;'):
                ui.html('Prediction Result').style('font-size:30px;font-weight:bold;color:tomato;')
                ui.html('Here is the result of prediction for 'f"{brand}"'-'f"{model}").style('margin-top:-15px;')
                ui.image(f"{tximageURL}").classes('w-64').style('margin-top:-20px')
                with ui.row().classes('w-full justify-center'):
                    ui.html(f"$ {fprice}").style('font-size:50px;margin-top:-30px;color:tomato;font-weight:bolder').style('text-decoration: underline')
                ui.html('Price History').style('font-size:20px;font-weight:bold;color:black;margin-bottom:-20px;margin-top:-10px;')
                with ui.pyplot(figsize=(5, 2)):
                    plt.plot(dfPriceChart['releaseDate'], dfPriceChart['lastSale'], '-')
                ui.html('Does this help? if you would like to continue, please push the button below').style('margin-top:-10px;')
                ui.button('Back', on_click=dialog.close).style('margin-top:-10px;')

        dialog.open()

def update_color(brand, model, color):
    global global_color
    global global_imgurl

    global_color = color
    global_imgurl = data[(data['brand'] == brand) & (data['primaryTitle'] == model) & (data['color1'] == color)]['thumbUrl'].unique()[0]

@ui.refreshable
def show_color(brand = '', model = '') -> None:
    global global_model
    if brand == '' or model == '':
        txcolor = ui.select([], label='Color').style('width: 100%;').disable()
    else:
        dfcolor = data[(data['brand'] == brand) & (data['primaryTitle'] == model)]['color1'].unique().tolist()
        txcolor = ui.select(dfcolor, label='Color', on_change=lambda e: {update_color(brand, model, e.value)}).style('width: 100%;')
        global_model = model

@ui.refreshable
def show_model(brand = '') -> None:
    if brand == '':
        txmodel = ui.select([], label='Model').style('width: 100%;').disable()
    else:
        dfmodel = data[data['brand'] == brand]['primaryTitle'].unique().tolist()
        txmodel = ui.select(dfmodel, label='Model', on_change=lambda e: show_color.refresh(brand, e.value)).style('width: 100%;')

@ui.refreshable
def show_spinner(status=False) -> None:
    ui.spinner(size='sm').style('margin-top: 5px;').set_visibility(status)


with ui.card().classes('desktop').style('margin:auto;margin-top:10px;'):
    with ui.row().classes('justify-center w-full').style('font-weight: bold;font-size:30px;background-color:tomato;color:white;padding:10px;'):
        ui.html('Sneaker Resale Value Prediction')
    with ui.row().classes('w-full').style('border: 2px solid red; padding: 15px;'):
                
        ui.html('<p>Hi, This is a Resale Value Calculator for Newly Released Sneakers ! Please give us a minute to input some parameters, then we will give you a value of the sneaker you would like to buy.<p>')
        
        txbrand = ui.select(['Nike', 'New Balance', 'adidas', 'Jordan', 'ASICS'], label='Brand', on_change=lambda e: show_model.refresh(e.value)).style('width: 100%;')
        
        txmodel = show_model()
        txcolor = show_color()

        txgender = ui.select(['Men', 'Women'], label='Gender').style('width: 100%;')
        txsize = ui.number(label='US Size', 
                           value=0, 
                           format='%.1f').style('width: 100%;')

        with ui.row().classes('w-full'):
            ui.button('Predict', on_click=lambda e: getPrediction())
            show_spinner()

ui.run()
