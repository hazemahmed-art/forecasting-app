# price_forecast.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io

def forecast_price(df):
    """
    Performs linear regression forecasting for the next year's average price.
    Assumes df has 'date' (datetime) and 'price' columns.
    Aggregates data annually by mean price.
    """
    # Ensure 'date' is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year
    df['year'] = df['date'].dt.year
    
    # Aggregate by year: mean price
    annual_df = df.groupby('year')['price'].mean().reset_index()
    annual_df.columns = ['Year', 'Actual Price']
    
    # Prepare data for linear regression
    X = annual_df['Year'].values.reshape(-1, 1)
    y = annual_df['Actual Price'].values
    
    # Fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get the next year (assuming current year is the max in data + 1)
    current_year = annual_df['Year'].max()
    next_year = current_year + 1
    
    # Forecast for next year
    forecasted_price = model.predict(np.array([[next_year]]))[0]
    
    # Create forecasted row
    forecasted_row = pd.DataFrame({'Year': [next_year], 'Forecasted Price': [forecasted_price]})
    
    # Combine actual and forecasted
    result_df = pd.concat([annual_df, forecasted_row], ignore_index=True)
    result_df['Actual Price'] = result_df['Actual Price'].where(result_df['Year'] != next_year, np.nan)
    result_df['Forecasted Price'] = result_df['Forecasted Price'].where(result_df['Year'] == next_year, np.nan)
    
    # Generate time series plot
    plt.figure(figsize=(10, 6))
    plt.plot(result_df['Year'], result_df['Actual Price'], marker='o', label='Actual Price', color='blue')
    plt.plot(result_df['Year'], result_df['Forecasted Price'], marker='o', linestyle='--', label='Forecasted Price', color='red')
    plt.title('Annual Price: Actual vs Forecasted')
    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.legend()
    plt.grid(True)
    
    # Save plot to bytes
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()
    
    return forecasted_price, result_df, img_buf
