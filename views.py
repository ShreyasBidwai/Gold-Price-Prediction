from django.shortcuts import render
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import plotly.graph_objs as go

def forecast_view(request):
    
    # Load and preprocess data
    data = pd.read_csv("C:/Users/hp/Desktop/DSBDA_mini/project/ml_integration/gold_monthly_csv.csv")  # Replace with the correct file path
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m')  # Ensure date format matches
    data = data.set_index(['Date'])
    
    # Seasonal decomposition
    decompose_data = seasonal_decompose(data['Price'], model="additive")
    
    # Augmented Dickey–Fuller test
    dftest = adfuller(data['Price'], autolag='AIC')
    
    # Rolling mean and differencing
    rolling_mean = data['Price'].rolling(window=12).mean()
    data['rolling_mean_diff'] = rolling_mean - rolling_mean.shift()
    
    # SARIMA model fitting
    model = sm.tsa.statespace.SARIMAX(data['Price'], order=(1, 1, 1), seasonal_order=(1,1,1,12))
    results = model.fit()
    
    # Forecasting
    # Calculate the forecast end date dynamically based on the last date in the dataset
    forecast_end_date = data.index[-1] + pd.DateOffset(months=48)  # Extend forecast period by 12 months
    forecast_dates = pd.date_range(start=data.index[-1], end=forecast_end_date, freq='M')
    forecast = results.predict(start=len(data), end=len(data) + len(forecast_dates) - 1, dynamic=True)


    
    # Plotting using Plotly
    original_trace = go.Scatter(x=data.index, y=data['Price'], mode='lines', name='Original')
    forecast_trace = go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecast')
    layout = go.Layout(title='Gold Price Forecast till 2024', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    fig = go.Figure(data=[original_trace, forecast_trace], layout=layout)
    fig.update_layout(template="plotly_dark") #Setting the template to plotly_dark
    graph = fig.to_html(full_html=False, default_height=500, default_width=1200)

    
    # Calculate average of past 3 and 6 months' data
    past_3_months_average = round(data['Price'].iloc[-3:].mean(), 2)
    past_6_months_average = round(data['Price'].iloc[-6:].mean(), 2)
    
    # Calculate average of forecasted 3 and 6 months' data
    forecasted_3_months_average = round(forecast.iloc[:3].mean(), 2)
    forecasted_6_months_average = round(forecast.iloc[:6].mean(), 2)
    
    context = {
        'graph': graph,
        'past_3_months_average': past_3_months_average,
        'past_6_months_average': past_6_months_average,
        'forecasted_3_months_average': forecasted_3_months_average,
        'forecasted_6_months_average': forecasted_6_months_average,
    }
    
    return render(request, 'forecast.html', context)
