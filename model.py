import pandas as pd
import numpy as np

df = pd.read_csv('./data/stockx_vyuki.csv')

# List of 'size' and 'lastSale' columns
size_columns = ['size{}'.format(i) for i in range(1, 31)]
lastSale_columns = ['lastSale{}'.format(i) for i in range(1, 31)]

# Clean 'size' columns by removing 'W's and converting to numeric
for column in size_columns:
    df[column] = df[column].astype(str).str.replace('W', '', regex=True)
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Convert 'lastSale' columns to numeric
for column in lastSale_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

price_columns = [
    'averagePrice_Annual_Statistics',
    'averagePrice_Dead_Stock',
    'retailPrice'
]
for column in price_columns:
  df[column] = pd.to_numeric(df[column], errors='coerce')

df['releaseDate'] = pd.to_datetime(df['releaseDate'])
df['days_since_release'] = (pd.Timestamp.now() - df['releaseDate']).dt.days

#Create dummy variables for the gender & brand
df_with_dummies = pd.get_dummies(df, columns=['gender','brand'], drop_first=True)

# Create dummy variables without dropping the original columns
dummies = pd.get_dummies(df[['gender', 'brand']], drop_first=False)

# Concatenate the dummy variables with the original DataFrame
df_with_dummies = pd.concat([df, dummies], axis=1)

# Define a list of size columns and lastSale columns
size_columns = [col for col in df_with_dummies.columns if 'size' in col]
lastSale_columns = [col for col in df_with_dummies.columns if 'lastSale' in col]

# Pair the size columns and lastSale columns correctly
pairs = list(zip(size_columns, lastSale_columns))

additional_columns = ['urlKey','condition','gender','brand','retailPrice','days_since_release', 'gender_men','gender_women','gender_infant','gender_preschool','gender_toddler','gender_unisex','brand_Nike','averagePrice_Dead_Stock','averagePrice_Annual_Statistics','salesCount_Annual','color1']

# Use pd.melt to transform each pair and concatenate them into a new dataframe
melted_dfs = []
for size_col, sale_col in pairs:
    # Melt the DataFrame with the specific size and lastSale columns
    melted_df = df_with_dummies.melt(id_vars=additional_columns, value_vars=[size_col, sale_col], var_name='variable', value_name='value')

    # Determine if the row is from size or lastSale based on 'variable' and then assign to 'size' or 'lastSale'
    melted_df['size'] = melted_df.apply(lambda row: row['value'] if 'size' in row['variable'] else np.nan, axis=1)
    melted_df['lastSale'] = melted_df.apply(lambda row: row['value'] if 'lastSale' in row['variable'] else np.nan, axis=1)

    # Drop the intermediate columns
    melted_df.drop(['variable', 'value'], axis=1, inplace=True)
    melted_dfs.append(melted_df)

# Concatenate all the small dataframes into one
long_format_df = pd.concat(melted_dfs, ignore_index=True)

# Forward fill the NaNs for the 'size' column as each 'lastSale' entry should correspond to its above 'size'
long_format_df['size'] = long_format_df['size'].ffill()

# Drop the rows where 'lastSale' is NaN because we only want complete size-lastSale pairs
long_format_df.dropna(subset=['size', 'lastSale'], inplace=True)

# Remove rows where 'size' or 'lastSale' is 0.0
long_format_df = long_format_df[(long_format_df['size'] != 0.0) & (long_format_df['lastSale'] != 0.0)]

# Reset the index of the final DataFrame
long_format_df.reset_index(drop=True, inplace=True)

# Check for null values in the DataFrame
null_values = long_format_df.isnull().sum()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Select features and target
X = long_format_df.drop('lastSale', axis=1)
y = long_format_df['lastSale']

# Encode categorical variables
categorical_features = ['color1','condition','urlKey','gender','brand']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing for numerical data: fill in any missing values with the median
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data: fill in missing values with the most frequent value and apply one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = LinearRegression()

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = mse ** 0.5

print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)

features = ['retailPrice', 'days_since_release', 'gender_men', 'brand_Nike','averagePrice_Annual_Statistics']
target = 'lastSale'

X = long_format_df[features]
y = long_format_df[target]

imputer = SimpleImputer(strategy='median')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute the missing values in the feature set
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create and fit the linear model on the imputed data
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Predict on the test data
y_pred = model.predict(X_test_imputed)

# Calculate the performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Print out the metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Print the coefficients of the model
coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print(coefficients)

from sklearn.linear_model import LassoCV, RidgeCV

# Lasso regression with cross-validation to find the best alpha
lasso = LassoCV(cv=5).fit(X_train_imputed, y_train)
lasso_pred = lasso.predict(X_test_imputed)

# Ridge regression with cross-validation to find the best alpha
ridge = RidgeCV(cv=5).fit(X_train_imputed, y_train)
ridge_pred = ridge.predict(X_test_imputed)

# Evaluate Lasso performance
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = lasso_mse ** 0.5
lasso_r2 = r2_score(y_test, lasso_pred)

# Evaluate Ridge performance
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = ridge_mse ** 0.5
ridge_r2 = r2_score(y_test, ridge_pred)

# Print out the metrics for Lasso
print(f"Lasso - Mean Squared Error (MSE): {lasso_mse}")
print(f"Lasso - Root Mean Squared Error (RMSE): {lasso_rmse}")
print(f"Lasso - R-squared (R2): {lasso_r2}")

# Print out the metrics for Ridge
print(f"Ridge - Mean Squared Error (MSE): {ridge_mse}")
print(f"Ridge - Root Mean Squared Error (RMSE): {ridge_rmse}")
print(f"Ridge - R-squared (R2): {ridge_r2}")

# Coefficients
lasso_coefficients = pd.DataFrame(lasso.coef_, features, columns=['Lasso Coefficient'])
ridge_coefficients = pd.DataFrame(ridge.coef_, features, columns=['Ridge Coefficient'])

print(lasso_coefficients)
print(ridge_coefficients)

from sklearn.ensemble import RandomForestRegressor

# Fit Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_imputed, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test_imputed)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - Mean Squared Error (MSE): {mse_rf}")
print(f"Random Forest - Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"Random Forest - R-squared (R2): {r2_rf}")

import joblib

# Save the trained model and the imputer for later use
joblib.dump(rf_model, 'rf_model.joblib')
joblib.dump(imputer, 'imputer.joblib')

# def predict_price(input_features_dict):
#     # Load the trained model and the imputer
#     model = joblib.load('rf_model.joblib')
#     imputer = joblib.load('imputer.joblib')

#     # Convert the input features dictionary to a DataFrame
#     input_df = pd.DataFrame(input_features_dict, index=[0])

#     # Ensure that the order of the features matches the training data
#     input_df = input_df[features]

#     # Impute any missing values
#     input_imputed = imputer.transform(input_df)

#     # Make prediction
#     prediction = model.predict(input_imputed)

#     return prediction[0]

# def fetch_features(brand, gender, color, size):
#     # Filter the DataFrame based on the user input
#     filtered_df = long_format_df[(long_format_df['brand'] == brand) &
#                                  (long_format_df['gender'] == gender) &
#                                  (long_format_df['color1'] == color) &
#                                  (long_format_df['size'] == size)]

#     # Assuming we take the latest or an average of the relevant features for simplicity
#     retailPrice = filtered_df['retailPrice'].median()
#     days_since_release = filtered_df['days_since_release'].median()
#     averagePrice_Annual_Statistics = filtered_df['averagePrice_Annual_Statistics'].median()

#     return {
#         'retailPrice': retailPrice,
#         'days_since_release': days_since_release,
#         'averagePrice_Annual_Statistics': averagePrice_Annual_Statistics,
#         'gender_men': 1 if gender == 'men' else 0,
#         'brand_Nike': 1 if brand == 'Nike' else 0

#     }

# def main():
#     # User input
#     brand = input("Brand: ")
#     gender = input("Gender: ")
#     color = input("Color: ")
#     size = input("Size: ")

#     # Fetch other features
#     input_features_dict = fetch_features(brand, gender, color, size)

#     # Predict price
#     predicted_price = predict_price(input_features_dict)

#     print(f"The predicted price is: {predicted_price}")

# if __name__ == "__main__":
#     main()