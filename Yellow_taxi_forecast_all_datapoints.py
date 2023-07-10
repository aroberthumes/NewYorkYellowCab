import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import holidays

# Function to create time series features
def create_features(df, label=None):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    df['is_holiday'] = df.index.isin(us_holidays).astype(int)
    X = df[['hour', 'dayofweek', 'month', 'is_weekend', 'is_holiday']]
    if label:
        y = df[label]
        return X, y
    return X

# Load the data
taxi_zone_df = pd.read_csv('C:/Users/arobe/Downloads/taxi+_zone_lookup.csv')
yellow_taxi_df = pd.read_csv('C:/Users/arobe/OneDrive/Documents/yellow_taxi.csv')

# Define holiday list
us_holidays = holidays.US()

# Merge and preprocess the data
df = pd.merge(yellow_taxi_df, taxi_zone_df, left_on='PULocationID', right_on='LocationID', how='left')
df = df.drop('LocationID', axis=1)
df = df.rename(columns={'Borough': 'PUBorough', 'Zone': 'PUZone', 'service_zone': 'PUservice_zone'})
df = pd.merge(df, taxi_zone_df, left_on='DOLocationID', right_on='LocationID', how='left')
df = df.drop('LocationID', axis=1)
df = df.rename(columns={'Borough': 'DOBorough', 'Zone': 'DOZone', 'service_zone': 'DOservice_zone'})
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df.set_index('tpep_pickup_datetime', inplace=True)

# Filter the data for only Manhattan borough and January 2022
df = df[(df['PUBorough'] == 'Manhattan') & (df.index.year == 2022) & (df.index.month == 1)]

# Add a rides_count column to the dataframe
df['rides_count'] = 1

# Group the data by PULocationID and hour, and calculate the total fare amount, total tip amount, total trip_distance, total amount, and the number of rides
df_hourly = df.groupby([pd.Grouper(freq='H'), 'PULocationID']).agg({'trip_distance': 'sum', 'fare_amount': 'sum', 'tip_amount': 'sum', 'total_amount': 'sum', 'rides_count': 'sum'}).reset_index().rename(columns={'trip_distance': 'total_trip_distance', 'fare_amount': 'total_fare_amount', 'tip_amount': 'total_tip_amount', 'total_amount': 'total_amount', 'rides_count': 'num_rides'})

# Create an XGBoost regressor object for num_rides
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')

# Create an XGBoost regressor object for total_trip_distance
xg_reg_distance = xgb.XGBRegressor(objective ='reg:squarederror')

# Create an XGBoost regressor object for total_fare_amount
xg_reg_fare = xgb.XGBRegressor(objective ='reg:squarederror')

# Create an XGBoost regressor object for total_tip_amount
xg_reg_tip = xgb.XGBRegressor(objective ='reg:squarederror')

# Create an XGBoost regressor object for total_amount
xg_reg_total = xgb.XGBRegressor(objective ='reg:squarederror')

# Define the parameter grid for GridSearchCV
param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 10],
    'alpha': [1, 10],
    'n_estimators': [10, 50, 100]
}

# Create a GridSearchCV object for num_rides
grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Create a GridSearchCV object for total_trip_distance
grid_search_distance = GridSearchCV(estimator=xg_reg_distance, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Create a GridSearchCV object for total_fare_amount
grid_search_fare = GridSearchCV(estimator=xg_reg_fare, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Create a GridSearchCV object for total_tip_amount
grid_search_tip = GridSearchCV(estimator=xg_reg_tip, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Create a GridSearchCV object for total_amount
grid_search_total = GridSearchCV(estimator=xg_reg_total, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Get unique PULocationIDs
unique_ids = df_hourly['PULocationID'].unique()

# Create a DataFrame to store all predictions
all_predictions = []

for location_id in unique_ids:
    # Subset the training data to the current PULocationID
    subset_train = df_hourly[df_hourly['PULocationID'] == location_id]

    # If there's not enough data for this PULocationID, skip it
    if len(subset_train) < 10:
        continue

    subset_train.set_index('tpep_pickup_datetime', inplace=True)

    X_train, y_train = create_features(subset_train, label='num_rides')
    X_train_distance, y_train_distance = create_features(subset_train, label='total_trip_distance')
    X_train_fare, y_train_fare = create_features(subset_train, label='total_fare_amount')
    X_train_tip, y_train_tip = create_features(subset_train, label='total_tip_amount')
    X_train_total, y_train_total = create_features(subset_train, label='total_amount')

    # Fit GridSearchCV for num_rides
    grid_search.fit(X_train, y_train)

    # Fit GridSearchCV for total_trip_distance
    grid_search_distance.fit(X_train_distance, y_train_distance)

    # Fit GridSearchCV for total_fare_amount
    grid_search_fare.fit(X_train_fare, y_train_fare)

    # Fit GridSearchCV for total_tip_amount
    grid_search_tip.fit(X_train_tip, y_train_tip)

    # Fit GridSearchCV for total_amount
    grid_search_total.fit(X_train_total, y_train_total)

    # Print the best parameters for num_rides
    print('Best parameters found for num_rides for PULocationID {}: '.format(location_id), grid_search.best_params_)

    # Print the best parameters for total_trip_distance
    print('Best parameters found for total_trip_distance for PULocationID {}: '.format(location_id), grid_search_distance.best_params_)

    # Print the best parameters for total_fare_amount
    print('Best parameters found for total_fare_amount for PULocationID {}: '.format(location_id), grid_search_fare.best_params_)

    # Print the best parameters for total_tip_amount
    print('Best parameters found for total_tip_amount for PULocationID {}: '.format(location_id), grid_search_tip.best_params_)

    # Print the best parameters for total_amount
    print('Best parameters found for total_amount for PULocationID {}: '.format(location_id), grid_search_total.best_params_)

    # Create a dataframe for February 1st, 2022
    feb_1 = pd.date_range('2022-02-01', '2022-02-02', freq='H')
    feb_1_df = pd.DataFrame({'tpep_pickup_datetime': feb_1})

    # Add PULocationID to the dataframe
    forecast_df = feb_1_df.copy()
    forecast_df['PULocationID'] = location_id

    # Set tpep_pickup_datetime as the index and create features
    forecast_df.set_index('tpep_pickup_datetime', inplace=True)
    X_forecast = create_features(forecast_df)

    # Predict the number of rides, total distance, total fare amount, total tip amount, and total amount for February 1st using the best models
    y_pred = grid_search.best_estimator_.predict(X_forecast)
    y_pred_distance = grid_search_distance.best_estimator_.predict(X_forecast)
    y_pred_fare = grid_search_fare.best_estimator_.predict(X_forecast)
    y_pred_tip = grid_search_tip.best_estimator_.predict(X_forecast)
    y_pred_total = grid_search_total.best_estimator_.predict(X_forecast)

    # Create a DataFrame to hold the predictions
    predictions_df = X_forecast.copy()
    predictions_df['predicted_rides'] = y_pred
    predictions_df['predicted_total_distance'] = y_pred_distance
    predictions_df['predicted_total_fare'] = y_pred_fare
    predictions_df['predicted_total_tip'] = y_pred_tip
    predictions_df['predicted_total_amount'] = y_pred_total
    predictions_df['PULocationID'] = location_id

    # Append the predictions to the overall predictions DataFrame
    all_predictions.append(predictions_df)

all_predictions = pd.concat(all_predictions)
all_predictions['predicted_rides'] = all_predictions['predicted_rides'].round()
all_predictions['predicted_total_distance'] = all_predictions['predicted_total_distance'].round(2)
all_predictions['predicted_total_fare'] = all_predictions['predicted_total_fare'].round(2)
all_predictions['predicted_total_tip'] = all_predictions['predicted_total_tip'].round(2)
all_predictions['predicted_total_amount'] = all_predictions['predicted_total_amount'].round(2)
print('Predictions for February 1st, 2022:')
print(all_predictions)

all_predictions.to_csv('C:/Users/arobe/OneDrive/Documents/Feb1_taxi_predictions.csv')

