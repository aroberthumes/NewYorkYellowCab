# NewYorkYellowCab
**NYC Yellow Taxi Dashboard and Forecasting Project**

This project creates a dashboard showing the performance of yellow taxi cabs in Manhattan, New York City for January 2022, and provides a predictive dashboard for February 1st, 2022. The dashboards are made using Tableau, and the forecasting is performed using XGBoost in Python.

**Getting Started**

To get started with this project, you need to clone the repository to your local machine.

**Prerequisites**

Tableau Desktop or Tableau Public for viewing the dashboards
Python (with the libraries pandas, numpy, XGBoost, holidays, sklearn, requests and os )
A code editor such as VSCode

**Installation**

Clone the repo: git clone https://github.com/aroberthumes/NewYorkYellowCab.git

Install Python dependencies: pip install pandas, numpy, xgboost, holidays, sklearn, requests and os

Open the .py file in your preferred code editor.

Usage

Open these links to view the dashboards:
https://public.tableau.com/app/profile/anthony.humes/viz/NewYorkCityYellowTaxiRides/Dashboard1
https://public.tableau.com/app/profile/anthony.humes/viz/ForecastedNewYorkCityYellowTaxiRides/Dashboard1Prediction

To run the predictive model, run the Python script Yellow_taxi_forecast_all_datapoints.py

**Data**

The data for this project was retrieved from the NYC Taxi and Limousine Commission.

**Methodology**

The data was scraped from the NYC Taxi and Limousine Commission website and preprocessed to clean it up for modeling. Features used in the model include the features in the CSV file as well as hour, weekday, whether the day is a holiday, and the month. A grid search was performed to find the best parameters for the XGBoost model.

To validate the model, a baseline test was run to compare the XGBoost model with a baseline average model. The model was trained on data up to January 24th, and tested on data from January 25th through 31st.

To improve forecasting for February 1st 2022 you can scrape or download more data from the NYC Taxi and Limousine Commission website so XGBoost has more data to model on. 

**Deployment**

The dashboards can be viewed in Tableau, and the forecast can be run from the command line using Python.

https://public.tableau.com/app/profile/anthony.humes/viz/NewYorkCityYellowTaxiRides/Dashboard1
https://public.tableau.com/app/profile/anthony.humes/viz/ForecastedNewYorkCityYellowTaxiRides/Dashboard1Prediction

**Contributing**

If you would like to contribute to this project, please fork this repository, make your changes, and open a pull request to propose your changes.

**Acknowledgments**

Thanks to the NYC Taxi and Limousine Commission for providing the data.
