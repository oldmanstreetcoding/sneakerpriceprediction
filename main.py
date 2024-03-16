<<<<<<< HEAD
from nicegui import ui

class Demo:
    def __init__(self):
        self.number = 1

demo = Demo()
v = ui.checkbox('visible', value=True)
with ui.column().bind_visibility_from(v, 'value'):
    ui.slider(min=1, max=3).bind_value(demo, 'number')
    ui.toggle({1: 'A', 2: 'B', 3: 'C'}).bind_value(demo, 'number')
    ui.number().bind_value(demo, 'number')

ui.run()
=======
import pandas as pd
from nicegui import ui

ui.add_head_html('''
    <style>
        desktop {
            width: 50%;
        }
    </style>
''')

data = pd.read_csv('./data/stockx_v2.csv')
dfcolor = data.groupby('Color2').size().reset_index(name='counts')['Color2']
dfcolor
# np = dfcolor.values
# print(np)
# def proper_case(s):
#     return s.title()
# dfcolor = dfcolor.apply(proper_case)

with ui.card().classes('desktop').style('margin:auto;margin-top:10px;'):
    with ui.row().classes('justify-center w-full').style('font-weight: bold;font-size:30px;background-color:tomato;color:white;padding:10px;'):
        ui.html('Resale Value Prediction')
    with ui.row().classes('w-full').style('border: 2px solid red; padding: 15px;'):

        # with ui.tabs().classes('w-full') as tabs:
        #     one = ui.tab('Form')
        #     # two = ui.tab('Result')
        # with ui.tab_panels(tabs, value=one).classes('w-full'):
        #     with ui.tab_panel(one):
        #         ui.html('<p>Hi, This is a Resale Value Calculator for Newly Released Sneakers ! Please give us a minute to input some parameters, then we will give you a value of the sneaker you would like to buy.<p>')
        
        #         txbrand = ui.select(['Nike', 'Adidas', 'Jordan'], label='Brand').style('width: 100%;')
        #         txcolor = ui.select(['Black'], label='Color').style('width: 100%;')
        #         txmodel = ui.input(label='Model', 
        #                 placeholder='Type your Model...', 
        #                 validation={'Input too long': lambda value: len(value) < 20}).style('width: 100%;')
        #         txgender = ui.select(['Man', 'Woman'], label='Gender').style('width: 100%;')
        #         txsize = ui.number(label='US Size', 
        #                         value=5, 
        #                         format='%.2f').style('width: 100%;')

        #         with ui.dialog() as dialog, ui.card():
        #             with ui.row().classes('w-full').style('padding:5px;'):
        #                 ui.html('Prediction Result').style('font-size:30px')
        #                 ui.html('Here is the result of prediction based on the data you provided')
        #                 ui.html('$ XXX').style('font-size:50px').style('text-decoration: underline')
        #                 ui.html('Does this help? if you would like to continue, please push the button below')
        #                 ui.button('Back', on_click=dialog.close)
                
        #         with ui.row().classes('w-full'):
        #             ui.button('Predict', on_click=dialog.open)
            # with ui.tab_panel(two):
            #     ui.label('Second tab')
                
        ui.html('<p>Hi, This is a Resale Value Calculator for Newly Released Sneakers ! Please give us a minute to input some parameters, then we will give you a value of the sneaker you would like to buy.<p>')
        
        txbrand = ui.select(['Nike', 'Adidas', 'Jordan'], label='Brand').style('width: 100%;')
        txcolor = ui.select(['Black'], label='Color').style('width: 100%;')
        txmodel = ui.input(label='Model', 
                placeholder='Type your Model...', 
                validation={'Input too long': lambda value: len(value) < 20}).style('width: 100%;')
        txgender = ui.select(['Man', 'Woman'], label='Gender').style('width: 100%;')
        txsize = ui.number(label='US Size', 
                           value=5, 
                           format='%.2f').style('width: 100%;')

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
>>>>>>> 9a8ddb1 (form v1)
