import pandas as pd
from nicegui import ui

import pandas as pd
import joblib

def getFeatures(model, size, brand, gender, color):

    data = pd.read_csv('./data/df_clean.csv')

    txbrand = data[data['ori_brand'] == brand]['brand'].unique()

    txcolor = data[data['ori_color1']== color]['color1'].unique()[0]

    txmodel = data[data['ori_primaryTitle'].str.contains(model)]['primaryTitle'].unique()[0]

    txgender = data[data['ori_gender']== gender]['gender'].unique()[0]

    txretailPrice = data[(data['ori_primaryTitle'].str.contains(model)) & (data['size'] == size) & (data['ori_brand'] == brand) & (data['ori_gender'] == gender) & (data['ori_color1'] == color)]['retailPrice'].mean()

    txdaysPostRelease = data[(data['ori_primaryTitle'].str.contains(model)) & (data['size'] == size) & (data['ori_brand'] == brand) & (data['ori_gender'] == gender) & (data['ori_color1'] == color)]['retailPrice'].mean()

    txaverageAnnualPrice = data[(data['ori_primaryTitle'].str.contains(model)) & (data['size'] == size) & (data['ori_brand'] == brand) & (data['ori_gender'] == gender) & (data['ori_color1'] == color)]['retailPrice'].mean()

    df = pd.DataFrame({
        'brand': txbrand,
        'gender': txgender,
        'retailPrice': txretailPrice, 
        'days_postRelease': txdaysPostRelease, 
        'averagePrice_Annual_Statistics': txaverageAnnualPrice,
        'primaryTitle': txmodel,
        'color1': txcolor,
        'size': size
    }, index=[0])

    return df

def predict(data, txmodel, txsize, txbrand, txgender, txcolor):
    clf = joblib.load('./model/xgb_model.joblib')
    return clf.predict(getFeatures(data, txmodel, txsize, txbrand, txgender, txcolor))[0]

ui.add_head_html('''
    <style>
        desktop {
            width: 50%;
        }
    </style>
''')

data = pd.read_csv('./data/df_clean.csv')

dfcolor = data.groupby('ori_color1').size().reset_index(name='counts')['ori_color1'].unique().tolist()

dfgender = data.groupby('ori_gender').size().reset_index(name='counts')['ori_gender'].unique().tolist()

dfbrand = data.groupby('ori_brand').size().reset_index(name='counts')['ori_brand'].unique().tolist()

def getPrediction() -> None:
    
    data = pd.read_csv('./data/df_clean.csv')
    brand = txbrand.value
    color = txcolor.value
    size = txsize.value
    model = txmodel.value
    gender = txgender.value

    txbrandf = data[data['ori_brand'] == brand]['brand'].unique()

    txcolorf = data[data['ori_color1']== color]['color1'].unique()[0]

    txmodelf = data[data['ori_primaryTitle'].str.contains(model)]['primaryTitle'].unique()[0]

    txgenderf = data[data['ori_gender']== gender]['gender'].unique()[0]

    tximageURL = data[(data['ori_primaryTitle'].str.contains(model)) & (data['size'] == size) & (data['ori_brand'] == brand) & (data['ori_gender'] == gender) & (data['ori_color1'] == color)]['ori_thumbUrl'].unique()[0]

    txretailPricef = data[(data['ori_primaryTitle'].str.contains(model)) & (data['size'] == size) & (data['ori_brand'] == brand) & (data['ori_gender'] == gender) & (data['ori_color1'] == color)]['retailPrice'].mean()

    txdaysPostReleasef = data[(data['ori_primaryTitle'].str.contains(model)) & (data['size'] == size) & (data['ori_brand'] == brand) & (data['ori_gender'] == gender) & (data['ori_color1'] == color)]['retailPrice'].mean()

    txaverageAnnualPricef = data[(data['ori_primaryTitle'].str.contains(model)) & (data['size'] == size) & (data['ori_brand'] == brand) & (data['ori_gender'] == gender) & (data['ori_color1'] == color)]['retailPrice'].mean()

    df = pd.DataFrame({
        'brand': txbrandf,
        'gender': txgenderf,
        'retailPrice': txretailPricef, 
        'days_postRelease': txdaysPostReleasef, 
        'averagePrice_Annual_Statistics': txaverageAnnualPricef,
        'primaryTitle': txmodelf,
        'color1': txcolorf,
        'size': size
    }, index=[0])

    clf = joblib.load('./model/xgb_model.joblib')
    price = round(clf.predict(df)[0])

    fprice = "{:,.1f}".format(price)

    with ui.dialog() as dialog, ui.card():
        with ui.row().classes('w-full').style('padding:5px;'):
            ui.html('Prediction Result').style('font-size:30px')
            ui.html('Here is the result of prediction based on the data you provided')
            ui.image(f"{tximageURL}")
            with ui.row().classes('w-full justify-center'):
                ui.html(f"$ {fprice}").style('font-size:50px').style('text-decoration: underline')
            ui.html('Does this help? if you would like to continue, please push the button below')
            ui.button('Back', on_click=dialog.close)

    dialog.open()

with ui.card().classes('desktop').style('margin:auto;margin-top:10px;'):
    with ui.row().classes('justify-center w-full').style('font-weight: bold;font-size:30px;background-color:tomato;color:white;padding:10px;'):
        ui.html('Resale Value Prediction')
    with ui.row().classes('w-full').style('border: 2px solid red; padding: 15px;'):
                
        ui.html('<p>Hi, This is a Resale Value Calculator for Newly Released Sneakers ! Please give us a minute to input some parameters, then we will give you a value of the sneaker you would like to buy.<p>')
        
        txbrand = ui.select(dfbrand, label='Brand').style('width: 100%;')
        txcolor = ui.select(['Black', 'White', 'Red', 'Blue', 'Yellow'], label='Color').style('width: 100%;')
        txmodel = ui.input(label='Model', 
                placeholder='Type your Model...', 
                validation={'Input too long': lambda value: len(value) < 100}).style('width: 100%;').props('clearable')
        txgender = ui.select(dfgender, label='Gender').style('width: 100%;')
        txsize = ui.number(label='US Size', 
                           value=5, 
                           format='%.1f').style('width: 100%;')

        with ui.row().classes('w-full'):
            ui.button('Predict', on_click=lambda e: getPrediction())


ui.run()

#!/usr/bin/env python3
# from nicegui import ui

# with ui.header().classes(replace='row items-center') as header:
#     ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
#     with ui.tabs() as tabs:
#         ui.tab('A')
#         ui.tab('B')
#         ui.tab('C')

# with ui.footer(value=False) as footer:
#     ui.label('Footer')

# with ui.left_drawer().classes('bg-blue-100') as left_drawer:
#     ui.label('Side menu')

# with ui.page_sticky(position='bottom-right', x_offset=20, y_offset=20):
#     ui.button(on_click=footer.toggle, icon='contact_support').props('fab')

# with ui.tab_panels(tabs, value='A').classes('w-full'):
#     with ui.tab_panel('A'):
#         ui.label('Content of A')
#     with ui.tab_panel('B'):
#         ui.label('Content of B')
#     with ui.tab_panel('C'):
#         ui.label('Content of C')

# ui.run()
