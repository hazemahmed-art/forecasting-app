# price_forecast.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io


def forecast_price(df):
    # Ensure date conversion
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # 1. Calculate Weighted Average Components
    # We use the cleaned column names here
    df['total_value'] = df['price received'] * df['quantity received']
    
    # 2. Aggregate by year
    annual_agg = df.groupby('year').agg({
        'total_value': 'sum',
        'quantity received': 'sum'
    }).reset_index()
    
    # 3. Calculate the Weighted Price
    # This creates the 'Actual Price' column used in the regression
    annual_agg['Actual Price'] = annual_agg['total_value'] / annual_agg['quantity received']
    
    # Prepare data for Linear Regression
    X = annual_agg['year'].values.reshape(-1, 1)
    y = annual_agg['Actual Price'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    current_year = annual_agg['year'].max()
    next_year = current_year + 1
    forecasted_price = model.predict(np.array([[next_year]]))[0]
    
    # Prepare result dataframe for the UI
    result_df = annual_agg[['year', 'Actual Price']].copy()
    result_df.columns = ['Year', 'Actual Price']
    
    forecasted_row = pd.DataFrame({'Year': [next_year], 'Forecasted Price': [forecasted_price]})
    result_df = pd.concat([result_df, forecasted_row], ignore_index=True)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(result_df['Year'], result_df['Actual Price'], marker='o', label='Actual (Weighted)')
    plt.plot(result_df['Year'], result_df['Forecasted Price'], marker='o', linestyle='--', color='red', label='Forecast')
    plt.legend()
    plt.grid(True)
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    plt.close()
    
    return forecasted_price, result_df, img_buf
