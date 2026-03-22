#!/usr/bin/env python3
"""
Time Series Forecasting Script
Runs multiple forecasting models on monthly cost data with expanding window backtesting.
"""

import argparse
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Optional imports with graceful handling
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except (ImportError, ValueError, Exception):
    PROPHET_AVAILABLE = False

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except (ImportError, ValueError, Exception):
    PMDARIMA_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, ValueError, Exception):
    XGBOOST_AVAILABLE = False

# TensorFlow removed due to compatibility issues
TENSORFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')


def clean_currency(value):
    """Clean currency strings to numeric values."""
    if pd.isna(value) or value == '':
        return 0.0
    
    # Convert to string and strip whitespace
    value_str = str(value).strip()
    
    # Handle special cases
    if value_str in ['-', '$-', '$', '']:
        return 0.0
    
    # Remove $ and commas
    value_str = value_str.replace('$', '').replace(',', '')
    
    # Handle percentage (convert to decimal)
    if value_str.endswith('%'):
        return float(value_str[:-1]) / 100.0
    
    try:
        return float(value_str)
    except ValueError:
        return 0.0


def clean_data(df, date_col, value_col, count_col):
    """Clean and preprocess the data."""
    # Strip header whitespace
    df.columns = df.columns.str.strip()
    
    # Use first column as date if date_col not found
    if date_col not in df.columns:
        date_col = df.columns[0]
        print(f"Warning: Date column '{date_col}' not found, using first column")
    
    # Parse date column
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Clean value column
    df[value_col] = df[value_col].apply(clean_currency)
    
    # Auto-detect count column if not found
    count_cols_to_try = ["Inspection/Audit/Review Count", "Inspection Count", "Count"]
    if count_col not in df.columns:
        for col in count_cols_to_try:
            if col in df.columns:
                count_col = col
                print(f"Using count column: {col}")
                break
        else:
            print("Warning: No count column found, proceeding without exogenous variables")
            count_col = None
    
    # Clean count column if it exists
    if count_col and count_col in df.columns:
        df[count_col] = df[count_col].apply(clean_currency)
    
    # Clean additional cost columns for fiscal year cumulative means
    cost_columns = {
        'Federal Labor Costs': 'Federal Labor Costs',
        'Contract Labor Costs': 'Contract Labor Costs', 
        'Fed Travel': 'Fed Travel',
        'Contractor Travel': 'Contractor Travel'
    }
    
    # Clean the cost columns
    for col_name, col_key in cost_columns.items():
        if col_key in df.columns:
            df[col_key] = df[col_key].apply(clean_currency)
        else:
            print(f"Warning: {col_name} column not found")
    
    # Aggregate to monthly (sum costs and counts)
    agg_dict = {value_col: 'sum'}
    if count_col and count_col in df.columns:
        agg_dict[count_col] = 'sum'
    
    # Add cost columns to aggregation
    for col_key in cost_columns.values():
        if col_key in df.columns:
            agg_dict[col_key] = 'sum'
    
    df_monthly = df.groupby(df[date_col].dt.to_period('M')).agg(agg_dict).reset_index()
    
    # Convert period index to datetime
    df_monthly[date_col] = df_monthly[date_col].dt.to_timestamp()
    
    # Set date as index
    df_monthly = df_monthly.set_index(date_col)
    
    # Reindex to continuous monthly timeline and interpolate internal gaps
    start_date = df_monthly.index.min()
    end_date = df_monthly.index.max()
    full_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    df_monthly = df_monthly.reindex(full_range)
    
    # Interpolate only internal gaps (not leading/trailing NaNs)
    df_monthly[value_col] = df_monthly[value_col].interpolate(method='linear')
    if count_col and count_col in df_monthly.columns:
        df_monthly[count_col] = df_monthly[count_col].interpolate(method='linear')
    
    # Add fiscal year cumulative means
    df_monthly = add_fiscal_year_rolling_averages(df_monthly, cost_columns)
    
    return df_monthly, value_col, count_col


def add_fiscal_year_rolling_averages(df, cost_columns):
    """Add fiscal year-to-date cumulative means that reset at the start of each fiscal year (October)."""
    # Create fiscal year groups (October to September)
    df['fiscal_year'] = df.index.to_series().apply(
        lambda x: x.year if x.month >= 10 else x.year - 1
    )
    
    # Add cumulative means for each cost category
    for col_name, col_key in cost_columns.items():
        if col_key in df.columns:
            # Calculate the fiscal year-to-date mean within each fiscal year
            df[f'{col_name}_FY_RollingAvg'] = (
                df.groupby('fiscal_year')[col_key]
                .expanding()
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Fill any remaining NaNs with the original values
            df[f'{col_name}_FY_RollingAvg'] = df[f'{col_name}_FY_RollingAvg'].fillna(df[col_key])
        else:
            # Create zero column if original column doesn't exist
            df[f'{col_name}_FY_RollingAvg'] = 0.0
    
    # Remove the temporary fiscal_year column
    df = df.drop('fiscal_year', axis=1)
    
    return df


def create_features(df, value_col, count_col, max_lag=12):
    """Create features for tree-based models."""
    features_df = pd.DataFrame(index=df.index)
    
    # Target variable
    features_df['y'] = df[value_col]
    
    # Lags of y
    for lag in [1, 3, 6, 12]:
        if lag <= max_lag:
            features_df[f'y_lag_{lag}'] = df[value_col].shift(lag)
    
    # Lags of count (if available)
    if count_col and count_col in df.columns:
        for lag in [0, 1, 2]:
            features_df[f'count_lag_{lag}'] = df[count_col].shift(lag)
    
    # Rolling means of y
    for window in [3, 6, 12]:
        if window <= max_lag:
            features_df[f'y_rolling_{window}'] = df[value_col].rolling(window=window).mean()
    
    # Add fiscal year cumulative mean features
    fy_rolling_cols = [col for col in df.columns if col.endswith('_FY_RollingAvg')]
    for col in fy_rolling_cols:
        features_df[col] = df[col]
    
    # Month-of-year one-hot encoding
    features_df['month'] = df.index.month
    month_dummies = pd.get_dummies(features_df['month'], prefix='month')
    features_df = pd.concat([features_df, month_dummies], axis=1)
    features_df = features_df.drop('month', axis=1)
    
    return features_df


def expanding_window_backtest(model_func, df, value_col, count_col, horizon):
    """Perform expanding window backtest."""
    n = len(df)
    start_idx = n - horizon
    
    predictions = []
    actuals = []
    
    for i in range(start_idx, n):
        # Training data: all data up to t-1
        train_data = df.iloc[:i]
        
        # Test point: t
        test_actual = df.iloc[i][value_col]
        
        try:
            # Get prediction for time t
            pred = model_func(train_data, value_col, count_col, i)
            if np.isnan(pred):
                print(f"Model returned NaN at step {i}")
            predictions.append(pred)
            actuals.append(test_actual)
        except Exception as e:
            print(f"Error in backtest at step {i}: {e}")
            predictions.append(np.nan)
            actuals.append(test_actual)
    
    # Calculate MAPE
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Remove NaN predictions
    valid_mask = ~np.isnan(predictions) & ~np.isnan(actuals) & (actuals != 0)
    
    if np.sum(valid_mask) == 0:
        return np.nan
    
    return mean_absolute_percentage_error(actuals[valid_mask], predictions[valid_mask])


def prophet_model(train_data, value_col, count_col, test_idx):
    """Prophet model wrapper."""
    if not PROPHET_AVAILABLE:
        print("Prophet not available - skipping")
        return np.nan
    
    try:
        # Prepare data for Prophet
        df_prophet = train_data.reset_index()
        # Rename columns properly
        df_prophet = df_prophet.rename(columns={df_prophet.columns[0]: 'ds', value_col: 'y'})
        
        # Keep only ds, y, and count columns (exclude fiscal year cumulative means as they are endogenous)
        keep_cols = ['ds', 'y']
        if count_col and count_col in df_prophet.columns:
            keep_cols.append(count_col)
        
        df_prophet = df_prophet[keep_cols]
        
        # Initialize Prophet with multiplicative seasonality
        model = Prophet(seasonality_mode='multiplicative')
        
        # Add count as regressor if available
        if count_col and count_col in df_prophet.columns:
            # Standardize the count variable
            count_values = df_prophet[count_col].values
            if not np.isnan(count_values).all():
                count_mean = np.nanmean(count_values)
                count_std = np.nanstd(count_values)
                if count_std > 0:
                    df_prophet['count_std'] = (count_values - count_mean) / count_std
                    model.add_regressor('count_std')
        
        # Fit model
        model.fit(df_prophet)
        
        # Make prediction for next period
        future = model.make_future_dataframe(periods=1, freq='MS')
        
        # Add count regressor if available
        if count_col and count_col in df_prophet.columns and 'count_std' in df_prophet.columns:
            last_count = df_prophet['count_std'].iloc[-1]
            future['count_std'] = last_count
        
        forecast = model.predict(future)
        return forecast['yhat'].iloc[-1]
    
    except Exception as e:
        print(f"Prophet error: {e}")
        return np.nan


def sarimax_model(train_data, value_col, count_col, test_idx):
    """SARIMAX model wrapper."""
    try:
        # Prepare data
        y = train_data[value_col].dropna()
        
        # Add exogenous variables if count is available (exclude fiscal year cumulative means as they are endogenous)
        exog = None
        if count_col and count_col in train_data.columns:
            count_data = train_data[count_col].dropna()
            if len(count_data) > 0:
                exog = count_data.values.reshape(-1, 1)
        
        # Fit SARIMAX with fixed order
        model = SARIMAX(y, exog=exog, order=(0, 1, 2), seasonal_order=(0, 1, 1, 12))
        fitted_model = model.fit(disp=False)
        
        # Make prediction
        if exog is not None:
            # Use the last count value for prediction
            last_count = exog[-1].reshape(1, -1)
            forecast = fitted_model.forecast(steps=1, exog=last_count)
        else:
            forecast = fitted_model.forecast(steps=1)
        
        return forecast[0]
    
    except Exception as e:
        print(f"SARIMAX error: {e}")
        return np.nan


def ets_model(train_data, value_col, count_col, test_idx):
    """ETS/Holt-Winters model wrapper."""
    try:
        y = train_data[value_col].dropna()
        
        # Try multiplicative first, fallback to additive
        try:
            model = ExponentialSmoothing(y, seasonal='multiplicative', seasonal_periods=12)
            fitted_model = model.fit()
        except:
            model = ExponentialSmoothing(y, seasonal='additive', seasonal_periods=12)
            fitted_model = model.fit()
        
        forecast = fitted_model.forecast(steps=1)
        return forecast[0]
    
    except Exception as e:
        print(f"ETS error: {e}")
        return np.nan


def exponential_smoothing_simple(train_data, value_col, count_col, test_idx):
    """Simple Exponential Smoothing with higher alpha for recent data emphasis."""
    try:
        y = train_data[value_col].dropna()
        
        if len(y) < 3:
            return np.nan
        
        # Use higher alpha (0.3-0.5) to emphasize recent data more
        alpha = 0.4
        model = ExponentialSmoothing(y, trend=None, seasonal=None)
        fitted_model = model.fit(smoothing_level=alpha)
        
        forecast = fitted_model.forecast(steps=1)
        return forecast[0]
    
    except Exception as e:
        print(f"Simple Exponential Smoothing error: {e}")
        return np.nan


def exponential_smoothing_trend(train_data, value_col, count_col, test_idx):
    """Double Exponential Smoothing (Holt's method) with trend."""
    try:
        y = train_data[value_col].dropna()
        
        if len(y) < 4:
            return np.nan
        
        # Use higher smoothing parameters to emphasize recent data
        model = ExponentialSmoothing(y, trend='add', seasonal=None)
        fitted_model = model.fit(smoothing_level=0.3, smoothing_trend=0.2)
        
        forecast = fitted_model.forecast(steps=1)
        return forecast[0]
    
    except Exception as e:
        print(f"Double Exponential Smoothing error: {e}")
        return np.nan


def exponential_smoothing_adaptive(train_data, value_col, count_col, test_idx):
    """Adaptive Exponential Smoothing with dynamic alpha."""
    try:
        y = train_data[value_col].dropna()
        
        if len(y) < 6:
            return np.nan
        
        # Calculate adaptive alpha based on recent volatility
        recent_data = y[-6:]  # Last 6 months
        volatility = recent_data.std() / recent_data.mean()
        
        # Higher volatility = higher alpha (more weight to recent data)
        alpha = min(0.6, max(0.1, 0.2 + volatility * 0.3))
        
        model = ExponentialSmoothing(y, trend='add', seasonal=None)
        fitted_model = model.fit(smoothing_level=alpha, smoothing_trend=0.15)
        
        forecast = fitted_model.forecast(steps=1)
        return forecast[0]
    
    except Exception as e:
        print(f"Adaptive Exponential Smoothing error: {e}")
        return np.nan


def weighted_moving_average(train_data, value_col, count_col, test_idx):
    """Weighted Moving Average with exponential weights for recent data."""
    try:
        y = train_data[value_col].dropna()
        
        if len(y) < 6:
            return np.nan
        
        # Use last 6 months with exponential weights
        window = min(6, len(y))
        recent_data = y[-window:]
        
        # Create exponential weights (more recent = higher weight)
        weights = np.exp(np.linspace(-1, 0, window))
        weights = weights / weights.sum()
        
        # Calculate weighted average
        forecast = np.sum(recent_data * weights)
        return forecast
    
    except Exception as e:
        print(f"Weighted Moving Average error: {e}")
        return np.nan


def autoarima_model(train_data, value_col, count_col, test_idx):
    """AutoARIMA model wrapper."""
    if not PMDARIMA_AVAILABLE:
        print("AutoARIMA not available - skipping")
        return np.nan
    
    try:
        y = train_data[value_col].dropna()
        
        if len(y) < 24:  # Need minimum data for seasonal model
            print("AutoARIMA: Insufficient data for seasonal model")
            return np.nan
        
        # Add exogenous variables if count is available (exclude fiscal year cumulative means as they are endogenous)
        exog = None
        if count_col and count_col in train_data.columns:
            count_data = train_data[count_col].dropna()
            if len(count_data) > 0:
                exog = count_data.values.reshape(-1, 1)
        
        # Fit AutoARIMA
        model = auto_arima(y, exogenous=exog, seasonal=True, m=12, 
                          suppress_warnings=True, error_action='ignore',
                          max_p=3, max_q=3, max_P=2, max_Q=2)
        
        # Make prediction
        if exog is not None:
            last_count = exog[-1].reshape(1, -1)
            forecast = model.predict(n_periods=1, exogenous=last_count)
        else:
            forecast = model.predict(n_periods=1)
        
        return forecast[0]
    
    except Exception as e:
        print(f"AutoARIMA error: {e}")
        return np.nan


def tree_model(train_data, value_col, count_col, test_idx):
    """Tree-based model wrapper."""
    try:
        # Create features
        features_df = create_features(train_data, value_col, count_col)
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        if len(features_df) < 10:  # Need minimum data
            return np.nan
        
        # Prepare training data
        X = features_df.drop('y', axis=1)
        y = features_df['y']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Choose model
        if XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit model
        model.fit(X_scaled, y)
        
        # Create features for prediction (use last available data)
        last_features = X.iloc[-1:].copy()
        
        # Scale the last features
        last_features_scaled = scaler.transform(last_features)
        
        # Make prediction
        pred = model.predict(last_features_scaled)[0]
        return pred
    
    except Exception as e:
        print(f"Tree model error: {e}")
        return np.nan


def neural_network_model(train_data, value_col, count_col, test_idx):
    """Neural network (LSTM) model wrapper with enhanced features."""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available - skipping")
        return np.nan
    
    try:
        # Prepare data
        y = train_data[value_col].dropna().values
        
        if len(y) < 24:  # Need minimum data for LSTM
            print("Neural Network: Insufficient data for LSTM model")
            return np.nan
        
        # Create enhanced features
        features_df = create_features(train_data, value_col, count_col)
        features_df = features_df.dropna()
        
        if len(features_df) < 15:  # Need minimum data
            print("Neural Network: Insufficient data for enhanced model")
            return np.nan
        
        # Prepare features for neural network
        feature_cols = [col for col in features_df.columns if col != 'y']
        X_features = features_df[feature_cols].values
        y_values = features_df['y'].values
        
        # Normalize features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X_features)
        y_scaled = scaler_y.fit_transform(y_values.reshape(-1, 1)).flatten()
        
        # Create sequences for LSTM (use last 6 months of features to predict next month)
        sequence_length = min(6, len(X_scaled) - 1)
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y_scaled[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        if len(X_seq) < 5:  # Need minimum sequences
            print("Neural Network: Insufficient sequences for training")
            return np.nan
        
        # Build enhanced LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(sequence_length, X_scaled.shape[1])),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model
        model.fit(X_seq, y_seq, epochs=80, batch_size=2, verbose=0, validation_split=0.2)
        
        # Make prediction
        last_sequence = X_scaled[-sequence_length:].reshape(1, sequence_length, X_scaled.shape[1])
        pred_scaled = model.predict(last_sequence, verbose=0)[0][0]
        
        # Denormalize prediction
        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        
        return pred
    
    except Exception as e:
        print(f"Neural Network error: {e}")
        return np.nan


def main():
    parser = argparse.ArgumentParser(description='Time Series Forecasting with Multiple Models')
    parser.add_argument('--input', default='pseudo data.csv', help='Path to input CSV file')
    parser.add_argument('--date-col', default='Date', help='Date column name')
    parser.add_argument('--value-col', default='Total Cost', help='Value column name')
    parser.add_argument('--count-col', default='Inspection/Audit/Review Count', help='Count column name')
    
    args = parser.parse_args()
    
    print("Loading and cleaning data...")
    df = pd.read_csv(args.input)
    df_clean, value_col, count_col = clean_data(df, args.date_col, args.value_col, args.count_col)
    
    print(f"Data shape: {df_clean.shape}")
    print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    
    # Calculate backtest horizon
    n = len(df_clean)
    horizon = min(12, max(6, n // 4))
    print(f"Backtest horizon: {horizon} months")
    
    # Define models
    models = [
        ('Prophet', prophet_model),
        ('SARIMAX', sarimax_model),
        ('ETS', ets_model),
        ('SimpleExpSmooth', exponential_smoothing_simple),
        ('DoubleExpSmooth', exponential_smoothing_trend),
        ('AdaptiveExpSmooth', exponential_smoothing_adaptive),
        ('WeightedMA', weighted_moving_average),
        ('AutoARIMA', autoarima_model),
        ('TreeModel', tree_model)
    ]
    
    # Run backtests
    results = []
    print("\nRunning expanding window backtests...")
    
    for model_name, model_func in models:
        print(f"Testing {model_name}...")
        mape = expanding_window_backtest(model_func, df_clean, value_col, count_col, horizon)
        results.append((model_name, mape))
        print(f"{model_name} MAPE: {mape:.4f}" if not np.isnan(mape) else f"{model_name} MAPE: NaN")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results, columns=['model', 'MAPE'])
    
    # Sort by MAPE (NaN values go to the end)
    results_df['MAPE_numeric'] = results_df['MAPE'].replace([np.nan], [np.inf])
    results_df = results_df.sort_values('MAPE_numeric').drop('MAPE_numeric', axis=1)
    
    # Print results
    print("\n" + "="*50)
    print("=== Forecast MAPE Leaderboard (expanding-window 1-step) ===")
    print("="*50)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Save results
    results_df.to_csv('forecast_mape_leaderboard.csv', index=False)
    print(f"\nResults saved to forecast_mape_leaderboard.csv")


if __name__ == "__main__":
    main()
