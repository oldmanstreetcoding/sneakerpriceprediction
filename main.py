# Title: Sneaker Resale Value Predictor
# Author: Ahmad Furqan (afurqan@andrew.cmu.edu)
# Date: 04-26-2024
# Description: This is a Python script to create a web-based application using NiceGUI library to predict the resale value of newly released sneakers. The application will take the brand, model, color, gender type and size from user input and predict the resale value of the sneaker based on random forest model. The application will also display the price history of the selected sneaker from stocx dataset.

# Import python libraries
import pandas as pd
from matplotlib import pyplot as plt
from nicegui import ui
import joblib

# Define global variables
rfmodel = 'rf_model.joblib'
global_model = None
global_color = None
global_imgurl = None
global_price = None

# Load the data
data = pd.read_csv("input_data.csv")

# Define the function to predict the price
def predict_price(input_features_dict):
    
    # Load the trained model
    model = joblib.load(rfmodel)

    # Convert the input features dictionary to a DataFrame
    input_df = pd.DataFrame(input_features_dict, index=[0])

    # Ensure that the order of the features matches the training data
    top_features = ['averagePrice_Annual_Statistics', 'size', 'days_since_release', 'salesCount_Annual', 'retailPrice', 'brand_adidas', 'brand_Nike', 'gender_men', 'brand_ASICS', 'gender_women', 'brand_Jordan', 'brand_New Balance']
    
    # Filter the DataFrame based on the top features
    input_df = input_df[top_features]

    # Make prediction
    prediction = model.predict(input_df)

    # Update the global variable to be used in the UI
    global global_price
    global_price = prediction[0]

    # Hide the loading spinner
    show_spinner.refresh(False)


# Define the function to fetch the features
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


# Define the function to get the prediction
def getPrediction() -> None:

    # Show the loading spinner
    show_spinner.refresh(True)

    # localizing the global variables
    global global_model
    global global_color
    global global_imgurl
    global global_price

    # Get the input values
    brand = txbrand.value
    gender = txgender.value
    size = txsize.value
    model = global_model
    color = global_color
    tximageURL = global_imgurl

    # Check if any of the fields are empty
    if brand == None or model == None or color == None or size == 0 or gender == None:
        # Show an error message in toast message
        ui.notify('Please fill in all the fields')
        show_spinner.refresh(False)

    else:

        # create dataframe for price chart based on the input
        dfPriceChart = data[(data['brand'] == brand) & (data['primaryTitle'] == model) & (data['gender'] == gender.lower()) & (data['color1'] == color)]
        dfPriceChart = dfPriceChart[['releaseDate', 'lastSale']].sort_values(by='releaseDate')
        
        # Print the price history for debugging
        print("")
        print("====================================")
        print("Selected Sneaker Details :"+brand+"-"+model)
        print("")
        print("Price History: ")
        print(dfPriceChart)

        # Fetch other features
        input_features_dict = fetch_features(brand, model, gender, color, size)

        print("")
        print("Selected Features :")
        print(input_features_dict)

        # Wait for the prediction
        predict_price(input_features_dict)

        # get the predicted price
        fprice = "{:,.1f}".format(global_price)
        print("")
        print("Predicted Price : $", fprice)
        print("====================================")

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


# Define the function to update the selected color
def update_color(brand, model, color):
    # localizing the global variables
    global global_color
    global global_imgurl

    global_color = color
    global_imgurl = data[(data['brand'] == brand) & (data['primaryTitle'] == model) & (data['color1'] == color)]['thumbUrl'].unique()[0]


# Define the function to update the selected model
@ui.refreshable
def show_color(brand = '', model = '') -> None:
    global global_model

    # Check if the brand and model are empty
    if brand == '' or model == '':
        txcolor = ui.select([], label='Color').style('width: 100%;').disable()
    else:
        # Get the unique colors for the selected brand and model
        dfcolor = data[(data['brand'] == brand) & (data['primaryTitle'] == model)]['color1'].unique().tolist()
        txcolor = ui.select(dfcolor, label='Color', on_change=lambda e: {update_color(brand, model, e.value)}).style('width: 100%;').props('use-chips')
        global_model = model


# Initially, input model is in disabled mode, it will active after brand is selected
@ui.refreshable
def show_model(brand = '') -> None:
    # If brand is empty, disable the model input
    if brand == '':
        txmodel = ui.select([], label='Model').style('width: 100%;').disable()
    else:
        # Get the unique models for the selected brand
        dfmodel = data[data['brand'] == brand]['primaryTitle'].unique().tolist()
        txmodel = ui.select(dfmodel, label='Model', on_change=lambda e: show_color.refresh(brand, e.value)).style('width: 100%;').props('use-chips')


# Define the function to show the loading spinner
@ui.refreshable
def show_spinner(status=False) -> None:
    ui.spinner(size='sm').style('margin-top: 5px;').set_visibility(status)

# The main UI layout
with ui.card().classes('desktop').style('margin:auto;margin-top:10px;'):
    with ui.row().classes('justify-center w-full').style('font-weight: bold;font-size:30px;background-color:tomato;color:white;padding:10px;'):
        ui.html('Sneaker Resale Value Predictor')
    with ui.row().classes('w-full justify-center'):
        with ui.card().style('width:20%;float:left; margin-top: 0px;'):
            ui.image("heroapp.jpg").style('margin:auto;')
            ui.image("heroapp.jpg").style('margin:auto;')
            ui.html("<small><i>Image Generated by: Meta Llama 3 Model<i></small>")
        with ui.row().style('width:78%;float:left'):
            with ui.card().classes('w-full justify-center'):
                ui.html('<p>Hi, This is a Resale Value Calculator for Newly Released Sneakers ! Please give us a minute to input some parameters, then we will give you a value of the sneaker you would like to buy.<p>').style("font-size: 20px;font-weight: bold;margin-top:")
            with ui.card().classes('w-full justify-center'):
                with ui.row().classes('w-full').style('border: 2px solid red; border-radius: 5px 5px; padding: 15px;background-color: #F0F0F0;'):
                    txbrand = ui.select(['Nike', 'New Balance', 'adidas', 'Jordan', 'ASICS'], label='Brand', on_change=lambda e: show_model.refresh(e.value)).style('width: 100%;margin-top: -10px').props('use-chips')
                    
                    txmodel = show_model()
                    txcolor = show_color()

                    txgender = ui.select(['Men', 'Women'], label='Gender').style('width: 100%;').props('use-chips')
                    txsize = ui.number(label='US Size', 
                                    value=0, 
                                    format='%.1f').style('width: 100%;')

                    with ui.row().classes('w-full'):
                        ui.button('Predict', on_click=lambda e: getPrediction())
                        show_spinner()
ui.run()
