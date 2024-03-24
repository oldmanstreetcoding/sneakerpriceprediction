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