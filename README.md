# Title: 
Sneaker Resale Value Predictor

# Author: 
Ahmad Furqan (afurqan@andrew.cmu.edu)

# Date: 
04-26-2024

# Description: 
This Python script creates a web-based application using the NiceGUI library to predict the resale value of newly released sneakers. The application takes input from the user, including brand, model, color, gender type, and size, and predicts the resale value of the sneaker based on a random forest model. Additionally, the application displays the price history of the selected sneaker from the stocx dataset.

# Installation on Windows Machine
To run this application, you should have the internet connection and following instructions below:

1. Download and install Python from https://www.python.org/downloads/windows/.
2. Click on RunApp.bat (Windows Machine Only).
3. The Windows command prompt will open automatically.
4. The command prompt will extract rf_model.zip. If not, you need to manually extract this model using zip/rar.
5. The command prompt will download Python packages: pandas, numpy, scikit-learn, seaborn, matplotlib, nicegui, and joblib. If not, you need to download each package beforehand.
6. It will run main.py and the Python server.
7. The browser will open and direct to the local sneaker web application: http://127.0.0.1:8080/
8. Input all the fields: brand, model, color, gender, and size.
9. Click the prediction button and you can see the prediciton result on windows command prompt
10. It will also open the prediction result on the browser. If not, you need to reclick the prediction button
11. niceGUI is a simple Python library for handling Python frontend and Python webserver. However, sometimes it takes time to process large input model.
12. Therefore, you may need to wait and reclick the prediction button if the popup prediction result does not appear.
13. The command prompt will also display program debugging information, including input features, price history, and price prediction.
14. You can also run manually this program by running: python main.py on the windows command prompt
15. To stop the programm, CTRL+C in windows command
16. If you encounter any problems running this app, please contact me via the email provided above.