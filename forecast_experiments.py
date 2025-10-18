#!/usr/bin/env python3
"""
Forecast Experiments Script
Systematically tests multiple feature engineering configs and model variants.
"""

import argparse
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HoltWinters

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
    from tbats import TBATS
    TBATS_AVAILABLE = True
except (ImportError, ValueError, Exception):
    TBATS_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, ValueError, Exception):
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')


def clean_currency(value):
    """Clean currency strings to numeric values."""
    if pd.isna(value) or value == '':
        return 0.0
    
    value_str = str(value).strip()
    if value_str in ['-', '$-', '$', '']:
        return 0.0
    
    value_str = value_str.replace('$', '').replace(',', '')
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
    
    # Aggregate to monthly (sum costs and counts)
    agg_dict = {value_col: 'sum'}
    if count_col and count_col in df.columns:
        agg_dict[count_col] = 'sum'
    
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
    
    return df_monthly, value_col, count_col


def apply_transformation(y, transform_type):
    """Apply transformation to data."""
    if transform_type == 'identity':
        return y, lambda x: x
    elif transform_type == 'log1p':
        return np.log1p(np.maximum(y, 0)), lambda x: np.expm1(np.maximum(x, 0))
    elif transform_type == 'boxcox':
        y_positive = np.maximum(y, 1e-6)  # Ensure positive
        if len(y_positive) > 1:
            try:
                transformed, lambda_val = stats.boxcox(y_positive)
                return transformed, lambda x: np.maximum(stats.boxcox(x, lambda_val), 1e-6)
            except:
                return np.log1p(y_positive), lambda x: np.expm1(np.maximum(x, 0))
        else:
            return np.log1p(y_positive), lambda x: np.expm1(np.maximum(x, 0))
    else:
        return y, lambda x: x


def apply_outlier_handling(y, method):
    """Apply outlier handling to data."""
    if method == 'none':
        return y
    elif method == 'winsorize_1pct':
        return stats.mstats.winsorize(y, limits=[0.01, 0.01])
    elif method == 'zscore_clip_3sd':
        z_scores = np.abs(stats.zscore(y))
        return np.where(z_scores > 3, np.median(y), y)
    else:
        return y


def create_features(df, value_col, count_col, toggles):
    """Create features based on toggles."""
    features_df = pd.DataFrame(index=df.index)
    
    # Target variable
    features_df['y'] = df[value_col]
    
    # Target lags
    for lag in toggles.get('target_lags', [1, 3, 6, 12]):
        features_df[f'y_lag_{lag}'] = df[value_col].shift(lag)
    
    # Count lags (if count exists)
    if count_col and count_col in df.columns:
        for lag in toggles.get('count_lags', [0]):
            features_df[f'count_lag_{lag}'] = df[count_col].shift(lag)
    else:
        # Create zero columns if count doesn't exist
        for lag in toggles.get('count_lags', [0]):
            features_df[f'count_lag_{lag}'] = 0.0
    
    # Rolling means of y
    for window in toggles.get('rolling_means', [3, 6, 12]):
        features_df[f'y_rolling_{window}'] = df[value_col].rolling(window=window).mean()
    
    # Calendar features
    if toggles.get('month_dummies', True):
        features_df['month'] = df.index.month
        month_dummies = pd.get_dummies(features_df['month'], prefix='month')
        features_df = pd.concat([features_df, month_dummies], axis=1)
        features_df = features_df.drop('month', axis=1)
    
    if toggles.get('quarter_dummies', False):
        features_df['quarter'] = df.index.quarter
        quarter_dummies = pd.get_dummies(features_df['quarter'], prefix='quarter')
        features_df = pd.concat([features_df, quarter_dummies], axis=1)
        features_df = features_df.drop('quarter', axis=1)
    
    # Fiscal calendar (shift months by +3)
    if toggles.get('fiscal_calendar', False):
        fiscal_month = ((df.index.month - 1 + 3) % 12) + 1
        features_df['fiscal_month'] = fiscal_month
        fiscal_dummies = pd.get_dummies(features_df['fiscal_month'], prefix='fiscal_month')
        features_df = pd.concat([features_df, fiscal_dummies], axis=1)
        features_df = features_df.drop('fiscal_month', axis=1)
    
    # Holiday/EOQ flags
    if toggles.get('holiday_flags', False):
        features_df['is_mar'] = (df.index.month == 3).astype(int)
        features_df['is_jun'] = (df.index.month == 6).astype(int)
        features_df['is_sep'] = (df.index.month == 9).astype(int)
        features_df['is_dec'] = (df.index.month == 12).astype(int)
    
    return features_df


def prepare_series(df, toggles, value_col, count_col):
    """Prepare series with transformations and features."""
    # Apply outlier handling
    y = apply_outlier_handling(df[value_col].values, toggles.get('outlier_handling', 'none'))
    
    # Apply transformation
    y_transformed, inverse_fn = apply_transformation(y, toggles.get('transformation', 'identity'))
    
    # Create features
    features_df = create_features(df, value_col, count_col, toggles)
    
    # Prepare exogenous variables
    exog_cols = []
    if count_col and count_col in df.columns:
        exog_cols.append(count_col)
        # Add count lags if specified
        for lag in toggles.get('count_lags', [0]):
            if lag > 0:
                exog_cols.append(f'count_lag_{lag}')
    
    X_exog = None
    if exog_cols:
        # Filter to only existing columns
        available_exog_cols = [col for col in exog_cols if col in features_df.columns]
        if available_exog_cols:
            exog_data = features_df[available_exog_cols].dropna()
            if len(exog_data) > 0:
                X_exog = exog_data.values
    
    return y_transformed, X_exog, inverse_fn, features_df


def expanding_window_backtest(model_func, df, value_col, count_col, toggles, horizon, seed=42):
    """Perform expanding window backtest."""
    np.random.seed(seed)
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
            pred = model_func(train_data, value_col, count_col, toggles, i)
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


def prophet_model(train_data, value_col, count_col, toggles, test_idx):
    """Prophet model wrapper."""
    if not PROPHET_AVAILABLE:
        return np.nan
    
    try:
        # Prepare data
        y_transformed, X_exog, inverse_fn, features_df = prepare_series(train_data, toggles, value_col, count_col)
        
        # Create Prophet dataframe
        df_prophet = pd.DataFrame({
            'ds': train_data.index,
            'y': y_transformed
        })
        
        # Initialize Prophet
        seasonality_mode = toggles.get('seasonality_mode', 'multiplicative')
        changepoint_prior_scale = toggles.get('changepoint_prior_scale', 0.1)
        
        model = Prophet(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale
        )
        
        # Add regressors
        if X_exog is not None:
            for i, col in enumerate(['count', 'count_lag1']):
                if i < X_exog.shape[1]:
                    df_prophet[col] = X_exog[:, i]
                    model.add_regressor(col)
        
        # Fit model
        model.fit(df_prophet)
        
        # Make prediction
        future = model.make_future_dataframe(periods=1, freq='MS')
        
        # Add regressors for prediction
        if X_exog is not None:
            for i, col in enumerate(['count', 'count_lag1']):
                if i < X_exog.shape[1]:
                    future[col] = X_exog[-1, i]
        
        forecast = model.predict(future)
        pred_transformed = forecast['yhat'].iloc[-1]
        
        # Apply inverse transformation
        pred = inverse_fn(pred_transformed)
        return max(pred, 1e-6)  # Ensure non-negative
    
    except Exception as e:
        print(f"Prophet error: {e}")
        return np.nan


def sarimax_model(train_data, value_col, count_col, toggles, test_idx):
    """SARIMAX model wrapper."""
    try:
        # Prepare data
        y_transformed, X_exog, inverse_fn, features_df = prepare_series(train_data, toggles, value_col, count_col)
        
        # Get order from toggles
        order = toggles.get('sarimax_order', (0, 1, 2))
        seasonal_order = toggles.get('sarimax_seasonal_order', (0, 1, 1, 12))
        
        # Fit SARIMAX
        model = SARIMAX(y_transformed, exog=X_exog, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        # Make prediction
        if X_exog is not None:
            last_exog = X_exog[-1].reshape(1, -1)
            forecast = fitted_model.forecast(steps=1, exog=last_exog)
        else:
            forecast = fitted_model.forecast(steps=1)
        
        # Apply inverse transformation
        pred = inverse_fn(forecast[0])
        return max(pred, 1e-6)  # Ensure non-negative
    
    except Exception as e:
        print(f"SARIMAX error: {e}")
        return np.nan


def ets_model(train_data, value_col, count_col, toggles, test_idx):
    """ETS/Holt-Winters model wrapper."""
    try:
        # Prepare data
        y_transformed, X_exog, inverse_fn, features_df = prepare_series(train_data, toggles, value_col, count_col)
        
        # Get ETS parameters
        trend = toggles.get('ets_trend', 'add')
        seasonal = toggles.get('ets_seasonal', 'add')
        
        # Fit ETS
        model = ExponentialSmoothing(y_transformed, trend=trend, seasonal=seasonal, seasonal_periods=12)
        fitted_model = model.fit()
        
        # Make prediction
        forecast = fitted_model.forecast(steps=1)
        
        # Apply inverse transformation
        pred = inverse_fn(forecast[0])
        return max(pred, 1e-6)  # Ensure non-negative
    
    except Exception as e:
        print(f"ETS error: {e}")
        return np.nan


def autoarima_model(train_data, value_col, count_col, toggles, test_idx):
    """AutoARIMA model wrapper."""
    if not PMDARIMA_AVAILABLE:
        return np.nan
    
    try:
        # Prepare data
        y_transformed, X_exog, inverse_fn, features_df = prepare_series(train_data, toggles, value_col, count_col)
        
        if len(y_transformed) < 24:
            return np.nan
        
        # Fit AutoARIMA
        model = auto_arima(y_transformed, exogenous=X_exog, seasonal=True, m=12, 
                          suppress_warnings=True, error_action='ignore')
        
        # Make prediction
        if X_exog is not None:
            last_exog = X_exog[-1].reshape(1, -1)
            forecast = model.predict(n_periods=1, exogenous=last_exog)
        else:
            forecast = model.predict(n_periods=1)
        
        # Apply inverse transformation
        pred = inverse_fn(forecast[0])
        return max(pred, 1e-6)  # Ensure non-negative
    
    except Exception as e:
        print(f"AutoARIMA error: {e}")
        return np.nan


def tbats_model(train_data, value_col, count_col, toggles, test_idx):
    """TBATS model wrapper."""
    if not TBATS_AVAILABLE:
        return np.nan
    
    try:
        # Prepare data
        y_transformed, X_exog, inverse_fn, features_df = prepare_series(train_data, toggles, value_col, count_col)
        
        if len(y_transformed) < 24:
            return np.nan
        
        # Fit TBATS
        use_box_cox = toggles.get('tbats_boxcox', False)
        model = TBATS(seasonal_periods=[12], use_box_cox=use_box_cox)
        fitted_model = model.fit(y_transformed)
        
        # Make prediction
        forecast = fitted_model.forecast(steps=1)
        
        # Apply inverse transformation
        pred = inverse_fn(forecast[0])
        return max(pred, 1e-6)  # Ensure non-negative
    
    except Exception as e:
        print(f"TBATS error: {e}")
        return np.nan


def tree_model(train_data, value_col, count_col, toggles, test_idx):
    """Tree-based model wrapper."""
    try:
        # Prepare data
        y_transformed, X_exog, inverse_fn, features_df = prepare_series(train_data, toggles, value_col, count_col)
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        if len(features_df) < 10:
            return np.nan
        
        # Prepare training data
        X = features_df.drop('y', axis=1)
        y = features_df['y']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Choose model
        n_estimators = toggles.get('n_estimators', 400)
        max_depth = toggles.get('max_depth', 6)
        
        if XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
        # Fit model
        model.fit(X_scaled, y)
        
        # Create features for prediction
        last_features = X.iloc[-1:].copy()
        last_features_scaled = scaler.transform(last_features)
        
        # Make prediction
        pred_transformed = model.predict(last_features_scaled)[0]
        
        # Apply inverse transformation
        pred = inverse_fn(pred_transformed)
        return max(pred, 1e-6)  # Ensure non-negative
    
    except Exception as e:
        print(f"Tree model error: {e}")
        return np.nan


def generate_experiment_configs(max_experiments=80):
    """Generate experiment configurations."""
    configs = []
    
    # Base configurations
    transformations = ['identity', 'log1p', 'boxcox']
    outlier_methods = ['none', 'winsorize_1pct', 'zscore_clip_3sd']
    count_lag_sets = [[0], [0, 1], [0, 1, 2]]
    target_lag_sets = [[1, 3, 6, 12]]
    rolling_mean_sets = [[3, 6, 12]]
    
    # Model-specific configs
    prophet_configs = [
        {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05},
        {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.1},
        {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.5},
        {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.1},
    ]
    
    sarimax_configs = [
        {'sarimax_order': (0, 1, 1), 'sarimax_seasonal_order': (0, 1, 1, 12)},
        {'sarimax_order': (0, 1, 2), 'sarimax_seasonal_order': (0, 1, 1, 12)},
        {'sarimax_order': (1, 1, 1), 'sarimax_seasonal_order': (1, 1, 1, 12)},
    ]
    
    ets_configs = [
        {'ets_trend': 'add', 'ets_seasonal': 'add'},
        {'ets_trend': 'mul', 'ets_seasonal': 'mul'},
    ]
    
    tree_configs = [
        {'n_estimators': 400, 'max_depth': 6},
        {'n_estimators': 800, 'max_depth': 8},
    ]
    
    # Generate combinations
    experiment_count = 0
    
    for transform in transformations:
        for outlier in outlier_methods:
            for count_lags in count_lag_sets:
                for target_lags in target_lag_sets:
                    for rolling_means in rolling_mean_sets:
                        base_toggles = {
                            'transformation': transform,
                            'outlier_handling': outlier,
                            'count_lags': count_lags,
                            'target_lags': target_lags,
                            'rolling_means': rolling_means,
                            'month_dummies': True,
                            'quarter_dummies': False,
                            'fiscal_calendar': False,
                            'holiday_flags': False,
                        }
                        
                        # Prophet experiments
                        if PROPHET_AVAILABLE and experiment_count < max_experiments:
                            for prop_config in prophet_configs:
                                config = base_toggles.copy()
                                config.update(prop_config)
                                config['model'] = 'Prophet'
                                configs.append(config)
                                experiment_count += 1
                                if experiment_count >= max_experiments:
                                    break
                        
                        # SARIMAX experiments
                        if experiment_count < max_experiments:
                            for sarimax_config in sarimax_configs:
                                config = base_toggles.copy()
                                config.update(sarimax_config)
                                config['model'] = 'SARIMAX'
                                configs.append(config)
                                experiment_count += 1
                                if experiment_count >= max_experiments:
                                    break
                        
                        # ETS experiments
                        if experiment_count < max_experiments:
                            for ets_config in ets_configs:
                                config = base_toggles.copy()
                                config.update(ets_config)
                                config['model'] = 'ETS'
                                configs.append(config)
                                experiment_count += 1
                                if experiment_count >= max_experiments:
                                    break
                        
                        # AutoARIMA experiments
                        if PMDARIMA_AVAILABLE and experiment_count < max_experiments:
                            config = base_toggles.copy()
                            config['model'] = 'AutoARIMA'
                            configs.append(config)
                            experiment_count += 1
                        
                        # TBATS experiments
                        if TBATS_AVAILABLE and experiment_count < max_experiments:
                            for boxcox in [True, False]:
                                config = base_toggles.copy()
                                config['tbats_boxcox'] = boxcox
                                config['model'] = 'TBATS'
                                configs.append(config)
                                experiment_count += 1
                                if experiment_count >= max_experiments:
                                    break
                        
                        # Tree experiments
                        if experiment_count < max_experiments:
                            for tree_config in tree_configs:
                                config = base_toggles.copy()
                                config.update(tree_config)
                                config['model'] = 'Tree'
                                configs.append(config)
                                experiment_count += 1
                                if experiment_count >= max_experiments:
                                    break
                        
                        if experiment_count >= max_experiments:
                            break
                    if experiment_count >= max_experiments:
                        break
                if experiment_count >= max_experiments:
                    break
            if experiment_count >= max_experiments:
                break
        if experiment_count >= max_experiments:
            break
    
    return configs


def create_model_variant_name(config):
    """Create a readable model variant name."""
    model = config['model']
    
    if model == 'Prophet':
        seasonality = config.get('seasonality_mode', 'mul')
        cps = config.get('changepoint_prior_scale', 0.1)
        transform = config.get('transformation', 'identity')
        return f"Prophet[{seasonality},cps={cps},{transform}]"
    
    elif model == 'SARIMAX':
        order = config.get('sarimax_order', (0, 1, 2))
        seasonal = config.get('sarimax_seasonal_order', (0, 1, 1, 12))
        return f"SARIMAX[{order}x{seasonal}]"
    
    elif model == 'ETS':
        trend = config.get('ets_trend', 'add')
        seasonal = config.get('ets_seasonal', 'add')
        return f"ETS[{trend},{seasonal}]"
    
    elif model == 'AutoARIMA':
        return "AutoARIMA"
    
    elif model == 'TBATS':
        boxcox = config.get('tbats_boxcox', False)
        return f"TBATS[boxcox={boxcox}]"
    
    elif model == 'Tree':
        n_est = config.get('n_estimators', 400)
        max_d = config.get('max_depth', 6)
        return f"Tree[{n_est},{max_d}]"
    
    else:
        return model


def main():
    parser = argparse.ArgumentParser(description='Forecast Experiments')
    parser.add_argument('--input', default='psuedo data.csv', help='Path to input CSV file')
    parser.add_argument('--date-col', default='Date', help='Date column name')
    parser.add_argument('--value-col', default='Total Cost', help='Value column name')
    parser.add_argument('--count-col', default='Inspection/Audit/Review Count', help='Count column name')
    parser.add_argument('--folds', type=int, default=12, help='Number of backtest folds')
    parser.add_argument('--top-k', type=int, default=20, help='Number of top results to show')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-experiments', type=int, default=80, help='Maximum number of experiments')
    
    args = parser.parse_args()
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    print("Loading and cleaning data...")
    df = pd.read_csv(args.input)
    df_clean, value_col, count_col = clean_data(df, args.date_col, args.value_col, args.count_col)
    
    # Save cleaned data
    df_clean.to_csv('outputs/clean_history.csv')
    
    print(f"Data shape: {df_clean.shape}")
    print(f"Date range: {df_clean.index.min()} to {df_clean.index.max()}")
    
    # Calculate backtest horizon
    n = len(df_clean)
    horizon = min(args.folds, max(6, n // 4))
    print(f"Backtest horizon: {horizon} months")
    
    # Generate experiment configurations
    print("Generating experiment configurations...")
    configs = generate_experiment_configs(args.max_experiments)
    print(f"Generated {len(configs)} experiment configurations")
    
    # Model functions
    model_functions = {
        'Prophet': prophet_model,
        'SARIMAX': sarimax_model,
        'ETS': ets_model,
        'AutoARIMA': autoarima_model,
        'TBATS': tbats_model,
        'Tree': tree_model,
    }
    
    # Run experiments
    results = []
    print(f"\nRunning {len(configs)} experiments...")
    
    for i, config in enumerate(configs):
        model_name = config['model']
        model_func = model_functions.get(model_name)
        
        if model_func is None:
            continue
        
        print(f"Experiment {i+1}/{len(configs)}: {create_model_variant_name(config)}")
        
        try:
            mape = expanding_window_backtest(
                model_func, df_clean, value_col, count_col, config, horizon, args.seed
            )
            results.append({
                'model_variant': create_model_variant_name(config),
                'MAPE': mape,
                'config': config
            })
            print(f"  MAPE: {mape:.4f}" if not np.isnan(mape) else "  MAPE: NaN")
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'model_variant': create_model_variant_name(config),
                'MAPE': np.nan,
                'config': config
            })
    
    # Sort results by MAPE
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAPE', na_position='last')
    
    # Print leaderboard
    print("\n" + "="*80)
    print("=== Forecast Experiments — MAPE Leaderboard ===")
    print("="*80)
    print(f"{'rank':<4} {'model_variant':<50} {'MAPE':<8}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(results_df.head(args.top_k).iterrows()):
        rank = i + 1
        model_variant = row['model_variant']
        mape = f"{row['MAPE']:.4f}" if not np.isnan(row['MAPE']) else "NaN"
        print(f"{rank:<4} {model_variant:<50} {mape:<8}")
    
    # Save CSV
    results_df.to_csv('outputs/experiments_leaderboard.csv', index=False)
    print(f"\nResults saved to outputs/experiments_leaderboard.csv")
    
    # Create markdown summary
    with open('outputs/experiments_summary.md', 'w') as f:
        f.write("# Forecast Experiments Summary\n\n")
        f.write(f"**Data Range:** {df_clean.index.min().strftime('%Y-%m-%d')} to {df_clean.index.max().strftime('%Y-%m-%d')}\n")
        f.write(f"**Total Experiments:** {len(configs)}\n")
        f.write(f"**Successful Experiments:** {len(results_df.dropna())}\n")
        f.write(f"**Backtest Folds:** {horizon}\n")
        f.write(f"**Random Seed:** {args.seed}\n\n")
        
        f.write("## Top 10 Results\n\n")
        f.write("| Rank | Model Variant | MAPE |\n")
        f.write("|------|---------------|------|\n")
        
        for i, (_, row) in enumerate(results_df.head(10).iterrows()):
            rank = i + 1
            model_variant = row['model_variant']
            mape = f"{row['MAPE']:.4f}" if not np.isnan(row['MAPE']) else "NaN"
            f.write(f"| {rank} | {model_variant} | {mape} |\n")
        
        f.write("\n## Reproducibility\n\n")
        f.write(f"- **Random Seed:** {args.seed}\n")
        f.write(f"- **Backtest Folds:** {horizon}\n")
        f.write(f"- **Data Range:** {df_clean.index.min().strftime('%Y-%m-%d')} to {df_clean.index.max().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Total Data Points:** {len(df_clean)}\n")
    
    print("Markdown summary saved to outputs/experiments_summary.md")


if __name__ == "__main__":
    main()
