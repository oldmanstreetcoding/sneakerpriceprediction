import pandas as pd
from nicegui import ui

from prediction import predict

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

with ui.card().classes('desktop').style('margin:auto;margin-top:10px;'):
    with ui.row().classes('justify-center w-full').style('font-weight: bold;font-size:30px;background-color:tomato;color:white;padding:10px;'):
        ui.html('Resale Value Prediction')
    with ui.row().classes('w-full').style('border: 2px solid red; padding: 15px;'):
                
        ui.html('<p>Hi, This is a Resale Value Calculator for Newly Released Sneakers ! Please give us a minute to input some parameters, then we will give you a value of the sneaker you would like to buy.<p>')
        
        txbrand = ui.select(dfbrand, label='Brand').style('width: 100%;')
        txcolor = ui.select(['Black', 'White', 'Red', 'Blue', 'Yellow'], label='Color').style('width: 100%;')
        txmodel = ui.input(label='Model', 
                placeholder='Type your Model...', 
                validation={'Input too long': lambda value: len(value) < 20}).style('width: 100%;')
        txgender = ui.select(dfgender, label='Gender').style('width: 100%;')
        txsize = ui.number(label='US Size', 
                           value=5, 
                           format='%.1f').style('width: 100%;')

        with ui.dialog() as dialog, ui.card():
            with ui.row().classes('w-full').style('padding:5px;'):
                ui.html('Prediction Result').style('font-size:30px')
                ui.html('Here is the result of prediction based on the data you provided')
                ui.html('$ XXX').style('font-size:50px').style('text-decoration: underline')
                ui.html('Does this help? if you would like to continue, please push the button below')
                ui.button('Back', on_click=dialog.close)
        
        with ui.row().classes('w-full'):
            # ui.button('Predict', on_click=lambda: ui.notify('You clicked me!')).classes('justify-right')
            ui.button('Predict', on_click=dialog.open)


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
