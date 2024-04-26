@echo off
Echo "Extracting the model..."
tar -xf rf_model.zip

Echo "Installing the required packages..."
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install seaborn
pip install nicegui
pip install joblib

Echo "Running the app and Opening the Browser on http://127.0.0.1:8080/"
python main.py
pause