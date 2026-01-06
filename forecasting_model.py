# forecasting_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib
try:
    matplotlib.use('Agg') 
except:
    pass

# CatBoost - اختياري
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None

# Deep Learning (GRU) - اختياري
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("TensorFlow غير متوفر. نموذج GRU مش هيشتغل.")


# ========================================
# بتحول التاريخ لمجموعة متغيرات 
# ========================================
def create_features(df, target_col='value', date_col='date'):
    """
    إنشاء ميزات إضافية من التاريخ لتحسين أداء النماذج
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['quarter'] = df[date_col].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

#بتحضر البيانات خصيصاً لنموذج الـ GRU
def prepare_data_for_gru(series, look_back=60):
    """
    تحضير البيانات لنموذج GRU
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# ========================================
# تدريب النماذج وتقييمها
# ========================================

def train_and_evaluate_models(df, target_col='value', date_col='date', forecast_steps=30):
    """
    تدريب عدة نماذج وإرجاع نتائج التقييم والتنبؤات
    """
    df = create_features(df, target_col, date_col)
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # تقسيم البيانات
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    X_train = train_df.drop(columns=[target_col, date_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col, date_col])
    y_test = test_df[target_col]
    
    models = {}
    predictions = {}
    metrics = {}
    
    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    models['Random Forest'] = rf
    predictions['Random Forest'] = pred_rf
    metrics['Random Forest'] = {
        'MAE': mean_absolute_error(y_test, pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred_rf)),
        'R2': r2_score(y_test, pred_rf)
    }
    
    # 2. XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    pred_xgb = xgb_model.predict(X_test)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = pred_xgb
    metrics['XGBoost'] = {
        'MAE': mean_absolute_error(y_test, pred_xgb),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred_xgb)),
        'R2': r2_score(y_test, pred_xgb)
    }
    
    # 3. CatBoost (لو متوفر)
    if CATBOOST_AVAILABLE:
        cat = CatBoostRegressor(iterations=100, verbose=0, random_state=42)
        cat.fit(X_train, y_train)
        pred_cat = cat.predict(X_test)
        models['CatBoost'] = cat
        predictions['CatBoost'] = pred_cat
        metrics['CatBoost'] = {
            'MAE': mean_absolute_error(y_test, pred_cat),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred_cat)),
            'R2': r2_score(y_test, pred_cat)
        }
    
    # 4. ARIMA (على القيم الأصلية فقط)
    try:
        arima_model = ARIMA(train_df[target_col], order=(5,1,0))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=len(test_df))
        metrics['ARIMA'] = {
            'MAE': mean_absolute_error(y_test, arima_forecast),
            'RMSE': np.sqrt(mean_squared_error(y_test, arima_forecast)),
            'R2': r2_score(y_test, arima_forecast)
        }
        predictions['ARIMA'] = arima_forecast.values
    except Exception as e:
        metrics['ARIMA'] = {'Error': str(e)}
        predictions['ARIMA'] = None
    
    # 5. GRU (لو TensorFlow متوفر)
    if DEEP_LEARNING_AVAILABLE and len(df) > 100:
        try:
            series = df[target_col]
            X_gru, y_gru, scaler = prepare_data_for_gru(series)
            
            model_gru = Sequential([
                GRU(50, return_sequences=True, input_shape=(X_gru.shape[1], 1)),
                Dropout(0.2),
                GRU(50),
                Dropout(0.2),
                Dense(1)
            ])
            model_gru.compile(optimizer='adam', loss='mse')
            model_gru.fit(X_gru, y_gru, epochs=20, batch_size=32, verbose=0)
            
            # تنبؤ على جزء الاختبار
            last_sequence = X_gru[-len(y_test):]
            gru_pred_scaled = model_gru.predict(last_sequence)
            gru_pred = scaler.inverse_transform(gru_pred_scaled)
            
            metrics['GRU'] = {
                'MAE': mean_absolute_error(y_test[-len(gru_pred):], gru_pred.flatten()),
                'RMSE': np.sqrt(mean_squared_error(y_test[-len(gru_pred):], gru_pred.flatten())),
                'R2': r2_score(y_test[-len(gru_pred):], gru_pred.flatten())
            }
            predictions['GRU'] = gru_pred.flatten()
            models['GRU'] = (model_gru, scaler)
        except Exception as e:
            metrics['GRU'] = {'Error': str(e)}
            predictions['GRU'] = None
    
    return {
        'models': models,
        'predictions': predictions,
        'metrics': metrics,
        'test_dates': test_df[date_col],
        'actual': y_test.values,
        'train_size': train_size
    }

# ========================================
# رسم النتائج (للاستخدام في Streamlit)
# ========================================

def plot_forecasts(results, title="مقارنة النماذج في التنبؤ"):

    """
    رسم التنبؤات مقابل القيم الحقيقية
    """
    plt.figure(figsize=(14, 8))
    test_dates = results['test_dates']
    actual = results['actual']
    
    plt.plot(test_dates, actual, label='القيم الحقيقية', color='black', linewidth=2)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, (name, pred) in enumerate(results['predictions'].items()):
        if pred is not None and len(pred) == len(test_dates):
            plt.plot(test_dates, pred, label=name, color=colors[i % len(colors)], alpha=0.8)
    
    plt.title(title, fontsize=16)
    plt.xlabel('التاريخ')
    plt.ylabel('القيمة')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

# ==============================
# Base Forecasting Model Class
# ==============================

class ForecastingModel:
    """
    Base class for all forecasting models.
    This class handles feature engineering and data splitting.
    """

    def __init__(self, name):
        self.name = name
        self.model = None

    def prepare_features(self, data, lookback=12):
        """
        Create lagged features for ML models
        data: DataFrame with column 'demand'
        """
        df = data.copy()

        for i in range(1, lookback + 1):
            df[f'lag_{i}'] = df['demand'].shift(i)

        df = df.dropna().reset_index(drop=True)
        return df

    def split_data(self, df, test_size=0.2):
        """
        Split data into train and test sets
        """
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test

# ==============================
# ARIMAModel
# ==============================
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMAModel(ForecastingModel):
    """
    ARIMA Time Series Forecasting Model
    """

    def __init__(self, order=(1, 1, 1)):
        super().__init__("ARIMA")
        self.order = order
        self.fitted_model = None

    def fit(self, train_data):
        """
        Train the ARIMA model
        train_data: DataFrame with column 'demand'
        """
        series = train_data['demand'].astype(float).values
        self.model = ARIMA(series, order=self.order)
        self.fitted_model = self.model.fit()
        return self

    def predict(self, steps):
        """
        Forecast future demand
        """
        if self.fitted_model is None:
            raise Exception("Model must be fitted before calling predict().")

        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)

# ========================================
# SARIMA Model
# ========================================

class SARIMAModel(ForecastingModel):
    """SARIMA Time Series Model with Seasonality"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        super().__init__("SARIMA")
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, train_data):
        """Fit SARIMA model"""
        self.model = SARIMAX(
            train_data['demand'].values,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = self.model.fit(disp=False)
        return self

    def predict(self, steps):
        """Forecast future values"""
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

# ========================================
# SARIMAX Model
# ========================================
class SARIMAXModel(ForecastingModel):
    """SARIMAX Time Series Model with Exogenous Variables"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        super().__init__("SARIMAX")
        self.order = order
        self.seasonal_order = seasonal_order
        self.train_length = None  # لحفظ طول بيانات التدريب

    def fit(self, train_data):
        """Fit SARIMAX model"""
        exog_train = self._create_exog_vars(train_data)
        
        self.model = SARIMAX(
            train_data['demand'].values, 
            exog=exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = self.model.fit(disp=False)
        self.train_length = len(train_data)
        return self

    def _create_exog_vars(self, data):
        """Create exogenous variables (trend + month)"""
        exog = pd.DataFrame()
        exog['trend'] = np.arange(len(data))
        exog['month'] = pd.to_datetime(data['date']).dt.month
        return exog

    def predict(self, steps):
        """Forecast future values"""
        exog_forecast = pd.DataFrame()
        exog_forecast['trend'] = np.arange(self.train_length, self.train_length + steps)
        exog_forecast['month'] = [(self.train_length + i) % 12 + 1 for i in range(steps)]
        
        forecast = self.fitted_model.forecast(steps=steps, exog=exog_forecast)
        return forecast

# ========================================
# Decision Tree Model
# ========================================
from sklearn.tree import DecisionTreeRegressor

class DecisionTreeModel(ForecastingModel):
    """Decision Tree Regressor using lag features of 'demand'"""
    
    def __init__(self, lookback=12, max_depth=10):
        super().__init__("Decision Tree")
        self.lookback = lookback
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        self.lag_columns = []
        self.last_values = None

    def prepare_features(self, data):
        """
        Create lag features based on 'demand' column only.
        Ignores any extra columns like 'price'.
        """
        df = data.copy()
        for i in range(1, self.lookback + 1):
            col_name = f'lag_{i}'
            df[col_name] = df['demand'].shift(i)
            self.lag_columns.append(col_name)
        df = df.dropna().reset_index(drop=True)
        return df

    def fit(self, train_data):
        """Fit Decision Tree model"""
        df = self.prepare_features(train_data)
        X = df[self.lag_columns].values
        y = df['demand'].values
        self.model.fit(X, y)
        # Save last 'lookback' values for iterative prediction
        self.last_values = train_data['demand'].tail(self.lookback).values.tolist()
        return self

    def predict(self, steps):
        """Forecast future values iteratively"""
        predictions = []
        current_values = self.last_values.copy()
        
        for _ in range(steps):
            X_pred = np.array(current_values[-self.lookback:]).reshape(1, -1)
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            current_values.append(pred)
        
        return np.array(predictions)
    
# ========================================
# XGBoost Model
# ========================================

class XGBoostModel(ForecastingModel):
    """XGBoost Regressor for Time Series Forecasting - Fixed Version"""

    def __init__(self, lookback=12, n_estimators=200):
        super().__init__("XGBoost")
        self.lookback = lookback
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42
        )
        self.feature_names = None  # مهم جدًا: نحفظ أسماء الأعمدة

    def fit(self, train_data):
        df = self.prepare_features(train_data, self.lookback)  # lags فقط

        # إضافة features إضافية (اختياري لتحسين الأداء)
        df['month'] = pd.to_datetime(train_data['date']).dt.month
        df['trend'] = np.arange(len(df))

        df = df.dropna().reset_index(drop=True)

        X = df.drop(['date', 'demand'], axis=1)
        y = df['demand']

        # حفظ أسماء وترتيب الأعمدة بالظبط
        self.feature_names = X.columns.tolist()

        self.model.fit(X, y, verbose=False)

        # حفظ آخر lookback قيم demand للتنبؤ المتكرر
        self.last_values = train_data['demand'].tail(self.lookback).values.tolist()
        self.last_date = train_data['date'].iloc[-1]
        self.last_trend = len(train_data) - 1

        return self

    def predict(self, steps):
        if self.feature_names is None:
            raise Exception("Model must be fitted first!")

        predictions = []
        current_lags = self.last_values.copy()
        current_trend = self.last_trend

        for i in range(steps):
            # بناء row جديد
            row = {}
            for j in range(1, self.lookback + 1):
                row[f'lag_{j}'] = current_lags[-j] if len(current_lags) >= j else 0

            # إضافة نفس الـ extra features
            next_date = pd.to_datetime(self.last_date) + pd.DateOffset(months=i+1)
            row['month'] = next_date.month
            row['trend'] = current_trend + i + 1

            # تحويل لـ DataFrame وترتيب الأعمدة زي التدريب + تعبئة الناقص
            X_pred = pd.DataFrame([row])
            X_pred = X_pred.reindex(columns=self.feature_names, fill_value=0)

            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            current_lags.append(pred)

        return np.array(predictions)  
# ========================================
# Cat Boost Model 
# ========================================
class CatBoostModel(ForecastingModel):
    """CatBoost Time Series Regressor - Fixed & Safe Version"""

    def __init__(self, lookback=12, iterations=200):
        super().__init__("CatBoost")
        self.lookback = lookback

        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is required. Install with: pip install catboost")

        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            verbose=False
        )
        self.feature_names = None  # سيتم تعيينه في كل fit

    def _create_features(self, data):
        df = data.copy().reset_index(drop=True)

        for i in range(1, self.lookback + 1):
            df[f"lag_{i}"] = df['demand'].shift(i)

        df['month'] = pd.to_datetime(df['date']).dt.month
        df['trend'] = np.arange(len(df))

        df.dropna(inplace=True)
        return df

    def fit(self, train_data):
        df = self._create_features(train_data)

        X = df.drop(['date', 'demand'], axis=1)
        y = df['demand']

        # إعادة تعيين feature_names في كل مرة fit
        self.feature_names = X.columns.tolist()

        self.model.fit(X, y)

        # حفظ البيانات اللازمة للتنبؤ
        self.last_demand_values = train_data['demand'].tail(self.lookback).values.tolist()
        self.last_date = train_data['date'].iloc[-1]
        self.last_trend = len(train_data) - 1

        return self

    def predict(self, steps):
        if self.feature_names is None:
            raise Exception("Model must be fitted first!")

        preds = []
        current_lags = self.last_demand_values.copy()
        current_date = self.last_date

        for i in range(steps):
            row = {}
            # Lags (نفس الترتيب)
            for j in range(1, self.lookback + 1):
                row[f"lag_{j}"] = current_lags[-j]

            # Time features
            next_date = pd.to_datetime(current_date) + pd.DateOffset(months=1)
            row['month'] = next_date.month
            row['trend'] = self.last_trend + i + 1

            # تحويل إلى DataFrame مع ضمان الترتيب الصحيح
            X_pred = pd.DataFrame([row])
            X_pred = X_pred.reindex(columns=self.feature_names, fill_value=0)  # هنا الحل الآمن!

            pred = self.model.predict(X_pred)[0]
            preds.append(pred)

            current_lags.append(pred)
            current_date = next_date

        return np.array(preds)
# ========================================
# GRUModel Model
# ========================================

# تحقق من وجود TensorFlow
DEEP_LEARNING_AVAILABLE = True
try:
    import tensorflow as tf
except ImportError:
    DEEP_LEARNING_AVAILABLE = False


class GRUModel(ForecastingModel):
    """GRU Neural Network for Time Series Forecasting"""
    
    def __init__(self, lookback=12, units=50, epochs=50):
        super().__init__("GRU")
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.model = None
    
    def fit(self, train_data):
        """Fit GRU model on training data"""
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("TensorFlow is required for GRU model. Install with: pip install tensorflow")
        
        # Prepare sequences
        data = train_data['demand'].values
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build GRU model
        self.model = Sequential([
            GRU(self.units, activation='relu', input_shape=(self.lookback, 1)),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train
        self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
        self.last_values = train_data['demand'].tail(self.lookback).values
        return self
    
    def predict(self, steps):
        """Forecast future values"""
        predictions = []
        current_values = list(self.last_values)
        
        for _ in range(steps):
            X_pred = np.array(current_values[-self.lookback:]).reshape(1, self.lookback, 1)
            pred = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred)
            current_values.append(pred)
        
        return np.array(predictions)

# ======================================================================
# ============================== TCN MODEL ==============================
# ======================================================================

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D
    DEEP_LEARNING_AVAILABLE = True
except:
    DEEP_LEARNING_AVAILABLE = False

class TCNModel(ForecastingModel):
    """Temporal Convolutional Network"""

    def __init__(self, lookback=12, filters=64, kernel_size=3, epochs=50):
        super().__init__("TCN")
        self.lookback = lookback
        self.filters = filters
        self.kernel_size = kernel_size
        self.epochs = epochs

    def fit(self, train_data):
        if not DEEP_LEARNING_AVAILABLE:
            raise ImportError("TensorFlow is required for TCN model")

        data = train_data['demand'].values

        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])

        X = np.array(X).reshape(-1, self.lookback, 1)
        y = np.array(y)

        self.model = Sequential([
            Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                   activation='relu', padding='causal',
                   input_shape=(self.lookback, 1)),
            Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                   activation='relu', padding='causal'),
            GlobalAveragePooling1D(),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse')

        self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)

        self.last_values = data[-self.lookback:]
        return self

    def predict(self, steps):
        predictions = []
        current_values = list(self.last_values)

        for _ in range(steps):
            X_pred = np.array(current_values[-self.lookback:]).reshape(1, self.lookback, 1)
            pred = self.model.predict(X_pred, verbose=0)[0, 0]
            predictions.append(pred)
            current_values.append(pred)

        return np.array(predictions)

# ======================================================================
# =========================== RANDOM FOREST =============================
# ======================================================================

class RandomForestModel(ForecastingModel):
    """Random Forest Regressor"""

    def __init__(self, lookback=12, n_estimators=100):
        super().__init__("Random Forest")
        self.lookback = lookback
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    def fit(self, train_data):
        df = self.prepare_features(train_data, self.lookback)

        X = df.drop(['date', 'demand'], axis=1)
        y = df['demand']

        self.model.fit(X, y)
        self.last_values = train_data['demand'].tail(self.lookback).values
        return self

    def predict(self, steps):
        predictions = []
        current_values = list(self.last_values)

        for _ in range(steps):
            X_pred = np.array(current_values[-self.lookback:]).reshape(1, -1)
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            current_values.append(pred)

        return np.array(predictions)
    

# ======================================================================
# =========================== RANDOM FOREST ================================================================================================
# ======================================================================

def aggregate_demand(df):
    """
    Prepare aggregated demand data for multiple time domains.
    Output format is compatible with run_forecast().
    """

    data = df.copy()

    # Ensure datetime
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')

    # ------------------ DAILY ------------------
    daily = data[['date', 'demand']].copy()

    # ------------------ WEEKLY ------------------
    weekly = data.resample('W', on='date')['demand'].sum().reset_index()

    # ------------------ MONTHLY ------------------
    monthly = data.resample('MS', on='date')['demand'].sum().reset_index()

    # ------------------ QUARTERLY ------------------
    quarterly = data.resample('QS', on='date')['demand'].sum().reset_index()

    # ------------------ HALF YEARLY ------------------
    half_yearly = data.resample('6MS', on='date')['demand'].sum().reset_index()

    # ------------------ YEARLY ------------------
    yearly = data.resample('YS', on='date')['demand'].sum().reset_index()

    return {
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "quarterly": quarterly,
        "half_yearly": half_yearly,
        "yearly": yearly
    }

   
def run_forecast(aggregated_data, time_domain, model_name, forecast_periods, lookback=12, seasonal_period=12):

    data = aggregated_data[time_domain].copy()
    data = data.sort_values('date').reset_index(drop=True)

    # ---------------- Split data ----------------
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]

    # ---------------- Initialize model ----------------
    if model_name == "ARIMA":
        model = ARIMAModel(order=(1,1,1))

    elif model_name == "SARIMA":
        model = SARIMAModel(order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))

    elif model_name == "SARIMAX":
        model = SARIMAXModel(order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))

    elif model_name == "Decision Tree":
        model = DecisionTreeModel(lookback=lookback)

    elif model_name == "Random Forest":
        model = RandomForestModel(lookback=lookback)

    elif model_name == "XGBoost":
        model = XGBoostModel(lookback=lookback)

    elif model_name == "CatBoost":
        model = CatBoostModel(lookback=lookback)

    elif model_name == "GRU":
        model = GRUModel(lookback=lookback)

    elif model_name == "TCN":
        model = TCNModel(lookback=lookback)

    else:
        raise ValueError("Unknown model")

    # ---------------- Fit & Predict ----------------
    
    model.fit(train_data)

    test_forecast = model.predict(len(test_data))
    future_forecast = model.predict(forecast_periods)

    # ---------------- Metrics ----------------
    metrics = {
        "MAE": mean_absolute_error(test_data["demand"], test_forecast),
        "RMSE": np.sqrt(mean_squared_error(test_data["demand"], test_forecast)),
        "R2": r2_score(test_data["demand"], test_forecast)
    }

    # ---------------- Create Future Dates ----------------
    last_date = data["date"].iloc[-1]

    freq_map = {
        "daily": "D",
        "weekly": "W",
        "monthly": "MS",
        "quarterly": "QS",
        "half_yearly": "6MS",
        "yearly": "YS"
    }

    future_dates = pd.date_range(start=last_date, periods=forecast_periods+1, freq=freq_map[time_domain])[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted_Demand": future_forecast
    })

    return test_data, test_forecast, forecast_df, metrics


