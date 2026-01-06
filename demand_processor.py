import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import streamlit as st
import os
from statsmodels.tsa.stattools import adfuller, kpss
from pandas.plotting import autocorrelation_plot
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import levene

#========================================BEFORE DATA PREPARATION PLOTS============================================================================================================================
def plot_before_preparation(df: pd.DataFrame):
    """
    يرسم الـ 4 رسوم البيانية لحالة البيانات قبل التنظيف
    ويعيد الـ figure عشان يتعرض في Streamlit
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('BEFORE DATA PREPARATION', fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: Raw demand data
    axes[0, 0].plot(range(len(df)), df['demand'], linewidth=1, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Raw Demand Data (Before Cleaning)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Record Index')
    axes[0, 0].set_ylabel('Demand')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Distribution of demand
    axes[0, 1].hist(df['demand'].dropna(), bins=30, edgecolor='black', color='coral', alpha=0.7)
    axes[0, 1].set_title('Demand Distribution (Before Cleaning)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Demand')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Box plot for outlier detection
    axes[1, 0].boxplot(df['demand'].dropna(), vert=True, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 0].set_title('Box Plot - Outlier Detection (Before)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Demand')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Missing values info
    columns_to_check = ['date', 'demand'] if 'date' in df.columns else ['demand']
    missing_info = df[columns_to_check].isnull().sum()
    
    colors = ['skyblue' if col == 'date' else 'salmon' for col in missing_info.index]
    axes[1, 1].bar(missing_info.index, missing_info.values, 
                   color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Missing Values Count (Before)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # إضافة الأرقام فوق الأعمدة
    for i, v in enumerate(missing_info.values):
        axes[1, 1].text(i, v + max(missing_info.values)*0.01, str(v), 
                        ha='center', fontweight='bold')

    plt.tight_layout()
    
    return fig


# ===================== DATA PREPARATION ==========================================================================================
def prepare_demand_data(df: pd.DataFrame):

    prep_results = {}

    # ------------------- STEP 1: DATE -------------------
    if 'date' not in df.columns:
        raise ValueError("Column 'date' not found in dataframe")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    # ------------------- STEP 2: MISSING VALUES -------------------
    prep_results['missing_values_before'] = df['demand'].isnull().sum()
    # Forward fill small gaps
    df['demand'] = df['demand'].fillna(method='ffill', limit=3)
    # Interpolation
    df['demand'] = df['demand'].interpolate(method='time', limit=7)
    # Backward fill any remaining
    df['demand'] = df['demand'].fillna(method='bfill')
    prep_results['missing_values_after'] = df['demand'].isnull().sum()

    # ------------------- STEP 3: OUTLIERS -------------------
    Q1 = df['demand'].quantile(0.25)
    Q3 = df['demand'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 2*IQR
    upper = Q3 + 2*IQR
    outliers = df[(df['demand'] < lower) | (df['demand'] > upper)]
    prep_results['outliers_count'] = len(outliers)
    prep_results['outliers_percent'] = len(outliers)/len(df)*100
    prep_results['outliers_bounds'] = [round(lower,2), round(upper,2)]
    df['demand_original'] = df['demand'].copy()
    df['demand'] = df['demand'].clip(lower=lower, upper=upper)

    # ------------------- STEP 4: REGULAR TIME FREQUENCY -------------------
    date_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
    prep_results['expected_records'] = len(date_range)
    prep_results['actual_records'] = len(df)
    prep_results['missing_days_filled'] = len(date_range) - len(df)
    df = df.reindex(date_range)
    df['demand'] = df['demand'].interpolate(method='time', limit=7)
    df['demand'] = df['demand'].fillna(method='ffill').fillna(method='bfill')

    # ------------------- STEP 5: FEATURE ENGINEERING -------------------
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_week'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['week_of_year'] = df.index.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    # Lag features
    for lag in [1,7,14,30]:
        df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
    # Rolling stats
    for window in [7,14,30]:
        df[f'demand_ma_{window}'] = df['demand'].rolling(window=window).mean()
        df[f'demand_std_{window}'] = df['demand'].rolling(window=window).std()
    # Expanding mean
    df['demand_expanding_mean'] = df['demand'].expanding().mean()
    prep_results['features_created'] = len(df.columns)

    # ------------------- STEP 6: HANDLE NaN FROM LAGS -------------------
    df_clean = df.dropna(subset=['demand'])

    # ------------------- STEP 7: STATIONARITY -------------------
    def check_stationarity(series):
        result = adfuller(series.dropna())
        return result[1] < 0.05, result[1]

    stationary, pvalue = check_stationarity(df_clean['demand'])
    prep_results['is_stationary'] = stationary
    if not stationary:
        df_clean['demand_diff'] = df_clean['demand'].diff()
        stationary_diff, pvalue_diff = check_stationarity(df_clean['demand_diff'].dropna())
        prep_results['first_diff_pvalue'] = round(pvalue_diff,4)

    return df_clean, prep_results


def plot_after_preparation(df_clean):

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('AFTER DATA PREPARATION', fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Cleaned demand time series
    axes[0, 0].plot(df_clean.index, df_clean['demand'], linewidth=1.5, alpha=0.8)
    axes[0, 0].set_title('Cleaned Demand Time Series', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Demand')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Distribution after cleaning
    axes[0, 1].hist(df_clean['demand'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Demand Distribution (After Cleaning)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Demand')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Moving averages
    axes[1, 0].plot(df_clean.index, df_clean['demand'], alpha=0.4, label='Demand', linewidth=1)
    axes[1, 0].plot(df_clean.index, df_clean['demand_ma_7'], linewidth=2, label='7-Day MA')
    axes[1, 0].plot(df_clean.index, df_clean['demand_ma_30'], linewidth=2, label='30-Day MA')
    axes[1, 0].set_title('Demand with Moving Averages', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Demand')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Box plot after cleaning
    axes[1, 1].boxplot(df_clean['demand'], vert=True, patch_artist=True,
                       boxprops=dict(alpha=0.7))
    axes[1, 1].set_title('Box Plot (After Cleaning)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Demand')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Plot 5: Autocorrelation
    autocorrelation_plot(df_clean['demand'], ax=axes[2, 0])
    axes[2, 0].set_title('Autocorrelation Plot', fontsize=12, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Weekly pattern
    weekly_avg = df_clean.groupby('day_of_week')['demand'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[2, 1].bar(range(7), weekly_avg.values, alpha=0.7)
    axes[2, 1].set_xticks(range(7))
    axes[2, 1].set_xticklabels(day_names)
    axes[2, 1].set_title('Average Demand by Day of Week', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Day of Week')
    axes[2, 1].set_ylabel('Average Demand')
    axes[2, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig

#============================================================================

def aggregate_demand(df, period_selected):
    """
    df: prepared demand DataFrame (after cleaning)
    period_selected: user's selection ['Daily', 'Weekly', ...]
    """
    # Map user choice to pandas resample frequency
    freq_map = {
        'Daily': 'D',
        'Weekly': 'W',  
        'Monthly': 'M',
        'Quarterly': 'Q',
        'Semi-Annual': '6M',
        'Annual': 'Y'
    }

    if period_selected not in freq_map:
        st.error(f"Unknown period selected: {period_selected}")
        return None

    freq = freq_map[period_selected]

    # Daily is just the cleaned data itself
    if period_selected == 'Daily':
        agg_df = df[['demand']].copy()
        agg_df.columns = ['Total']
    else:
        agg_df = df['demand'].resample(freq).agg(['sum', 'mean', 'std', 'min', 'max'])
        agg_df.columns = ['Total', 'Average', 'Std Dev', 'Min', 'Max']

    return agg_df

# ==================== Streamlit UI ====================
def show_aggregated_table(df_prepared):

    period = st.selectbox(
        "Select Period",
        ["Daily","Weekly", "Monthly", "Quarterly", "Semi-Annual", "Annual"],
        index=2,  # Default Monthly
        key="aggregated_period_select"  # مفتاح فريد عشان الاستقرار
    )

    agg_df = aggregate_demand(df_prepared, period)

    if agg_df is not None:
        st.dataframe(agg_df)

        # Export to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            agg_df.to_excel(writer, sheet_name='Aggregated Data')
        processed_data = output.getvalue()

        st.download_button(
            label="Download Excel",
            data=processed_data,
            file_name=f"aggregated_demand_{period}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    return period, agg_df

# =====================COMPARATIVE VISUALIZATION ==================================================================================================================================

def plot_comparative_demand(df_prepared, selected_period):
    """
    Plots comparative demand across periods starting from selected_period.
    Only shows the selected period and larger aggregations (not smaller ones).
    """
    # Map periods to pandas resample frequencies
    period_map = {
        'Daily': 'D',
        'Weekly': 'W',
        'Monthly': 'M',
        'Quarterly': 'Q',
        'Semi-Annual': '6M',
        'Annual': 'Y'
    }

    # Define the order of periods (smallest to largest)
    period_order = ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annual', 'Annual']
    colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'orange', 'crimson']

    # Find the starting index and get periods from selected onward
    start_idx = period_order.index(selected_period)
    show_periods = period_order[start_idx:]  # This will exclude smaller periods

    # Create aggregations only for periods we want to show
    aggregations = {}
    for p in show_periods:
        if p == 'Daily':
            agg = df_prepared[['demand']].copy()
            agg.columns = ['Total']
        else:
            agg = df_prepared['demand'].resample(period_map[p]).agg(['sum','mean','min','max'])
            agg.columns = ['Total','Average','Min','Max']
        aggregations[p] = agg

    # Calculate grid layout
    n_plots = len(show_periods)
    n_rows = (n_plots + 1) // 2
    n_cols = 2

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    
    # Handle different axes configurations
    if n_rows == 1 and n_cols == 2:
        axes = axes.reshape(1, -1)
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    fig.suptitle('TOTAL DEMAND ACROSS TIME PERIODS', fontsize=16, fontweight='bold', y=0.995)

    # Plot each period
    for idx, period in enumerate(show_periods):
        row = idx // 2
        col = idx % 2
        ax = axes[row][col] if n_rows > 1 else axes[col]

        # Use the color corresponding to the period's position in the original order
        color_idx = period_order.index(period)
        
        ax.plot(aggregations[period].index, aggregations[period]['Total'],
                linewidth=2, color=colors[color_idx], marker='o', markersize=6)
        ax.set_ylabel('Total Demand')
        ax.set_title(period, fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

    # Hide empty subplot if odd number of plots
    if n_plots % 2 != 0:
        empty_ax = axes[-1][-1] if n_rows > 1 else axes[-1]
        empty_ax.axis('off')

    plt.tight_layout()
    return fig


# =====================tests ==================================================================================================================================

# =========================================
# =====================noise test=============
# =========================================

def test_noise_level(demand_series):#=========================================================================================================================
    """
    Perform Noise Test on demand series.
    
    Args:
        demand_series (pd.Series or np.array): The demand values (cleaned)
    
    Returns:
        dict: Classification and metrics
    """
    values = demand_series.values if hasattr(demand_series, 'values') else np.array(demand_series)
    
    # إزالة القيم الصفرية لو كانت موجودة (عشان ما تؤثرش على المتوسط)
    non_zero = values[values > 0]
    if len(non_zero) == 0:
        return {
            'Classification': 'Undefined (No positive demand)',
            'SNR (dB)': None,
            'Coefficient of Variation (%)': None
        }
    
    # Signal power (mean of squared values)
    signal_power = np.mean(non_zero ** 2)
    
    # Noise estimation using first differences
    noise = np.diff(non_zero)
    noise_power = np.mean(noise ** 2) if len(noise) > 0 else 1e-10
    
    # Signal-to-Noise Ratio in dB
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # Coefficient of Variation (CV %)
    mean_demand = np.mean(non_zero)
    std_demand = np.std(non_zero)
    cv = (std_demand / mean_demand) * 100 if mean_demand > 0 else 0
    
    # Classification logic
    if snr > 20 or cv < 20:
        classification = "Low Noise"
    elif snr > 10 or cv < 40:
        classification = "Moderate Noise"
    else:
        classification = "High Noise"
    
    return {
        'Classification': classification,
        'SNR (dB)': round(snr, 2) if np.isfinite(snr) else "Very High",
        'Coefficient of Variation (%)': round(cv, 2)
    }

def show_noise_test_results(demand_series):
    """
    Display the noise test results in Streamlit with nice formatting
    """
    st.subheader("📊 N.oise Level Test Results")
    
    result = test_noise_level(demand_series)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Signal-to-Noise Ratio (SNR)",
            value=f"{result['SNR (dB)']} dB" if result['SNR (dB)'] is not None else "N/A"
        )
    
    with col2:
        st.metric(
            label="Coefficient of Variation (CV)",
            value=f"{result['Coefficient of Variation (%)']}%"
        )
    
    with col3:
        # لون التصنيف حسب النتيجة
        color = "green" if "Low" in result['Classification'] else \
                "orange" if "Moderate" in result['Classification'] else "red"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: {color}; color: white; border-radius: 10px; font-weight: bold; font-size: 1.1rem;">
            {result['Classification']}
        </div>
        """, unsafe_allow_html=True)

    return result

# =========================================
# =====================stationarity test===
# =========================================

import streamlit as st
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

warnings.filterwarnings("ignore")  # لتجنب تحذيرات KPSS

def test_stationarity(demand_series: pd.Series):
    """
    Perform comprehensive stationarity test using ADF and KPSS tests.
    
    Args:
        demand_series (pd.Series): سلسلة الطلب الزمنية (Demand time series)
    
    Returns:
        dict: يحتوي على التصنيف و q-values
    """
    # إزالة القيم المفقودة
    values = demand_series.dropna()
    
    # حالة البيانات غير كافية
    if len(values) < 2:
        return {
            'Classification': 'Insufficient Data',
            'ADF p-value': None,
            'KPSS p-value': None
        }
    
    # اختبار ADF (الفرضية الصفرية: غير ثابتة)
    adf_result = adfuller(values)
    adf_pvalue = adf_result[1]
    
    # اختبار KPSS (الفرضية الصفرية: ثابتة)
    kpss_result = kpss(values, regression='c', nlags="auto")
    kpss_pvalue = kpss_result[1]
    
    # منطق التصنيف
    if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
        classification = "Stationary"
    elif adf_pvalue >= 0.05 and kpss_pvalue <= 0.05:
        classification = "Non-Stationary"
    else:
        # حالات التعارض أو الاتجاه فقط (trend-stationary)
        classification = "Trend-Stationary"
    
    return {
        'Classification': classification,
        'ADF p-value': round(adf_pvalue, 4),
        'KPSS p-value': round(kpss_pvalue, 4)
    }

def show_stationarity_test_results(demand_series):
    """
    Display stationarity test results in Streamlit with consistent card style
    """
    st.subheader("Stationarity Test Results")
    
    result = test_stationarity(demand_series)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        adf_val = result['ADF p-value'] if result['ADF p-value'] is not None else "N/A"
        st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
            <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">ADF Test p-value</span><br>
            <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                {adf_val}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        kpss_val = result['KPSS p-value'] if result['KPSS p-value'] is not None else "N/A"
        st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: #f0f8ff; height: 100%;">
            <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">KPSS Test p-value</span><br>
            <span style="font-size: 1.8rem; font-weight: bold; color: #333;">
                {kpss_val}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Color logic
        if "Stationary" in result['Classification']:
            bg_color = "#d4edda"
            text_color = "#155724"
        elif "Non-Stationary" in result['Classification']:
            bg_color = "#f8d7da"
            text_color = "#721c24"
        else:
            bg_color = "#fff3cd"
            text_color = "#856404"
            
        st.markdown(f"""
        <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 8px; background-color: {bg_color}; height: 100%; text-align: center;">
            <span style="font-weight:bold; color:#1565C0; font-size:1.1rem;">Stationarity Classification</span><br>
            <span style="font-size: 1.6rem; font-weight: bold; color: {text_color};">
                {result['Classification']}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    return result

# =========================================
# =====================linearity test===
# =========================================

def test_linearity(demand_series):#=========================================================================================================================
    """
    Test if the demand series follows a linear or non-linear trend.
    
    Args:
        demand_series (pd.Series): Cleaned demand time series
    
    Returns:
        dict: Classification and R² scores
    """
    values = demand_series.dropna().values
    
    if len(values) < 4:
        return {
            'Classification': 'Insufficient Data',
            'Linear R²': None,
            'Polynomial R²': None
        }
    
    X = np.arange(len(values)).reshape(-1, 1)
    y = values
    
    # Linear fit
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_linear = lr.predict(X)
    r2_linear = r2_score(y, y_pred_linear)
    
    # Polynomial fit (degree 3)
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly, y)
    y_pred_poly = lr_poly.predict(X_poly)
    r2_poly = r2_score(y, y_pred_poly)
    
    # Improvement ratio
    improvement = (r2_poly - r2_linear) / (r2_linear + 0.001)
    
    if improvement > 0.1 or r2_linear < 0.3:
        classification = "Non-Linear"
    else:
        classification = "Linear"
    
    return {
        'Classification': classification,
        'Linear R²': round(r2_linear, 4),
        'Polynomial R²': round(r2_poly, 4)
    }

# =========================================
# =====================Demand Type test===
# =========================================

def classify_demand_type(demand_series):#=========================================================================================================================
    """
    Classify demand type using ADI and CV² (Syntetos classification)
    """
    series = demand_series.dropna()
    
    if len(series) == 0:
        return {
            'Classification': 'Insufficient Data',
            'ADI': None,
            'CV²': None,
            'Zero-Demand Periods': 0
        }
    
    zero_demands = (series == 0).sum()
    non_zero_demands = (series > 0).sum()
    
    adi = len(series) / non_zero_demands if non_zero_demands > 0 else len(series)
    
    non_zero_values = series[series > 0]
    if len(non_zero_values) > 0 and non_zero_values.mean() > 0:
        cv_squared = (non_zero_values.std() / non_zero_values.mean()) ** 2
    else:
        cv_squared = 0
    
    if adi < 1.32 and cv_squared < 0.49:
        demand_type = "Smooth"
    elif adi < 1.32 and cv_squared >= 0.49:
        demand_type = "Erratic"
    elif adi >= 1.32 and cv_squared >= 0.49:
        demand_type = "Lumpy"
    else:
        demand_type = "Intermittent"
    
    return {
        'Classification': demand_type,
        'ADI': round(adi, 4),
        'CV²': round(cv_squared, 4),
        'Zero-Demand Periods': int(zero_demands)
    }

# =========================================
# =====================trend test===
# =========================================

def analyze_trend(demand_series):#=========================================================================================================================

    """
    Analyze trend using Mann-Kendall and linear slope
    """
    series = demand_series.dropna().values
    
    if len(series) < 2:
        return {
            'Classification': 'Insufficient Data',
            'Slope': None,
            'Trend Strength (%)': None,
            'p-value': None
        }
    
    x = np.arange(len(series))
    tau, p_value = kendalltau(x, series)
    
    X = x.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, series)
    slope = lr.coef_[0]
    
    if p_value < 0.05:
        if slope > 0:
            trend = "Positive Trend"
        else:
            trend = "Negative Trend"
    else:
        trend = "No Significant Trend"
    
    trend_strength = (slope * len(series)) / series.mean() * 100 if series.mean() != 0 else 0
    
    return {
        'Classification': trend,
        'Slope': round(slope, 4),
        'Trend Strength (%)': round(trend_strength, 2),
        'p-value': round(p_value, 4)
    }

# =========================================
# =====================seasonality test===
# =========================================

def analyze_seasonality(demand_series, max_period=365): #==================================================================================
    series = demand_series.dropna()
    
    if len(series) < 10:
        return {
            'Classification': 'Insufficient Data',
            'Seasonal Period': 'None',
            'Period Length (days)': 0,
            'Seasonal Strength': 0
        }
    
    # Autocorrelation for peak detection
    acf_values = acf(series, nlags=min(len(series)//2, max_period), fft=True)
    peaks, _ = find_peaks(acf_values[1:], height=0.1, distance=5)
    
    best_period = None
    best_strength = 0
    
    # Try common periods
    common_periods = [7, 30, 90, 365]
    for period in common_periods:
        if len(series) >= 2 * period:
            try:
                decomposition = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
                seasonal_var = np.var(decomposition.seasonal)
                residual_var = np.var(decomposition.resid.dropna())
                strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0
                
                if strength > best_strength:
                    best_strength = strength
                    best_period = period
            except:
                continue
    
    if best_strength > 0.6:
        classification = "Strong Seasonality"
    elif best_strength > 0.3:
        classification = "Moderate Seasonality"
    elif best_strength > 0.1:
        classification = "Weak Seasonality"
    else:
        classification = "No Seasonality"
    
    period_name = "None"
    if best_period:
        if best_period == 7:
            period_name = "Weekly"
        elif 28 <= best_period <= 31:
            period_name = "Monthly"
        elif 88 <= best_period <= 92:
            period_name = "Quarterly"
        elif 360 <= best_period <= 370:
            period_name = "Yearly"
        else:
            period_name = f"{best_period} days"
    
    return {
        'Classification': classification,
        'Seasonal Period': period_name,
        'Period Length (days)': best_period if best_period else 0,
        'Seasonal Strength': round(best_strength, 4)
    }

# ================================================
# =====================Temporal Dependency test===
# ================================================

def analyze_temporal_dependency(demand_series): #==================================================================================
    series = demand_series.dropna()
    
    if len(series) < 10:
        return {
            'Classification': 'Insufficient Data',
            'Ljung-Box p-value': None,
            'Avg ACF (1-10 lags)': None,
            'Significant Lags': 0
        }
    
    lb_test = acorr_ljungbox(series, lags=[10], return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].values[0]
    
    acf_values = acf(series, nlags=min(30, len(series)//4), fft=True)
    significant_lags = np.sum(np.abs(acf_values[1:]) > 1.96 / np.sqrt(len(series)))
    
    avg_acf = np.mean(np.abs(acf_values[1:11]))
    
    if lb_pvalue < 0.05 and avg_acf > 0.3:
        dependency = "Strong Dependency"
    elif lb_pvalue < 0.05 and avg_acf > 0.1:
        dependency = "Moderate Dependency"
    elif lb_pvalue < 0.05:
        dependency = "Weak Dependency"
    else:
        dependency = "No Dependency (Random)"
    
    return {
        'Classification': dependency,
        'Ljung-Box p-value': round(lb_pvalue, 4),
        'Avg ACF (1-10 lags)': round(avg_acf, 4),
        'Significant Lags': int(significant_lags)
    }

# TEST 8: Change Point Detection (simplified fallback - no ruptures) #==================================================================================
def detect_change_points(demand_series):
    series = demand_series.dropna()
    
    if len(series) < 20:
        return {
            'Classification': 'Insufficient Data',
            'Number of Change Points': 0,
            'Max Change (%)': 0,
            'CV of Segments (%)': 0,
            'Change Point Dates': []
        }
    
    # Simple CUSUM-based fallback
    mean_val = series.mean()
    cusum = np.cumsum(series - mean_val)
    cusum_normalized = cusum / series.std() if series.std() > 0 else cusum
    
    threshold = 3.0
    diffs = np.abs(np.diff(cusum_normalized))
    potential_changes = np.where(diffs > threshold)[0]
    
    # Filter distant changes
    change_points = []
    min_distance = len(series) // 10
    for cp in potential_changes:
        if not change_points or cp - change_points[-1] > min_distance:
            change_points.append(cp + 1)  # +1 because diff
    
    n_changes = len(change_points)
    
    if n_changes == 0:
        classification = "Stable Level (No Change Points)"
    elif n_changes == 1:
        classification = "Single Change Detected"
    elif n_changes <= 3:
        classification = "Few Changes Detected"
    else:
        classification = "Multiple Changes (High Instability)"
    
    change_dates = [str(series.index[cp].date()) for cp in change_points[:5]]
    
    return {
        'Classification': classification,
        'Number of Change Points': n_changes,
        'Max Change (%)': 0,  # placeholder
        'CV of Segments (%)': 0,  # placeholder
        'Change Point Dates': change_dates
    }

# TEST 9: Variance Structure #==================================================================================
def analyze_variance_structure(demand_series): 
    series = demand_series.dropna()
    
    if len(series) < 20:
        return {
            'Classification': 'Insufficient Data',
            'Levene p-value': None,
            'CV of Variance (%)': 0,
            'Mean Variance': 0
        }
    
    n_segments = 5
    segment_size = len(series) // n_segments
    segments = [series.iloc[i*segment_size:(i+1)*segment_size] for i in range(n_segments)]
    
    variances = [seg.var() for seg in segments if len(seg) > 1]
    
    if len(segments) >= 2:
        valid_segments = [seg.values for seg in segments if len(seg) > 1]
        if len(valid_segments) >= 2:
            levene_stat, levene_pvalue = levene(*valid_segments)
        else:
            levene_pvalue = 1.0
    else:
        levene_pvalue = 1.0
    
    cv_variance = (np.std(variances) / np.mean(variances)) * 100 if np.mean(variances) > 0 else 0
    
    if levene_pvalue < 0.05 or cv_variance > 50:
        classification = "Heteroscedastic (Non-constant)"
    else:
        classification = "Homoscedastic (Constant)"
    
    return {
        'Classification': classification,
        'Levene p-value': round(levene_pvalue, 4),
        'CV of Variance (%)': round(cv_variance, 2),
        'Mean Variance': round(np.mean(variances), 2) if variances else 0
    }

