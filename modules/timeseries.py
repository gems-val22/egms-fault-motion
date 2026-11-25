
import pandas as pd
import numpy as np
import datetime

from patsy import dmatrix
from scipy.stats import linregress
from statsmodels.tsa.seasonal import STL


# Define window lengths outside the function to avoid recalculating them each time (for temporal variability metric)
revisit_time = 6 # days

n_obs_2yrs = int(2*365.25/revisit_time) # how many observations in 2 years
n_obs_3yrs = int(3*365.25/revisit_time) # how many observations in 3 years
month_shift = int(30/revisit_time) # how many observations per month


def drop_non_ts_cols(df):

    to_drop = [col for col in df.columns if '20' not in col] # Assuming time series columns contain '20' (for years), this will identify non-time-series columns
    df = df.drop(columns=to_drop) # Drop the identified non-time-series columns

    return df


def differential_ts(ts_left, ts_right, uncertainty_left, uncertainty_right): 

    diff_ts = ts_left - ts_right 
    diff_ts_uncertainty = np.sqrt(uncertainty_left**2 + uncertainty_right**2) 

    differential_df = pd.DataFrame(np.array(diff_ts), index = np.array(diff_ts.index), columns = ['differential_ts']) 
    differential_df['lower'] = diff_ts - diff_ts_uncertainty
    differential_df['upper'] = diff_ts + diff_ts_uncertainty
    differential_df['uncertainty'] = diff_ts_uncertainty 

    return differential_df 


def get_interval_df(mean_ts, uncertainty):

    interval_df = pd.DataFrame(np.array(mean_ts), index = np.array(mean_ts.index), columns = ['mean_ts'])
    interval_df['lower'] = mean_ts - uncertainty
    interval_df['upper'] =  mean_ts + uncertainty
    interval_df['uncertainty'] = uncertainty

    return interval_df


def stl_decomposition(ts_series):
    stl = STL(ts_series, period=60) # period=60 because we have ca. 60 data points per year; STL needs this to find seasonality 
    stl_result = stl.fit() 
 
    trend_component = stl_result.trend
    seasonal_component = stl_result.seasonal
    residual_component = stl_result.resid
    
    return trend_component, seasonal_component, residual_component


def ensure_zero_start(ts, start_idx = 0):
    """Ensure that the time series starts at zero by adjusting all values accordingly."""
    start_value = ts.iloc[start_idx]
    adjusted_ts = ts - start_value
    return adjusted_ts


def precompute_harmonic_seasonality(n, P=60, K=2):

    t = np.arange(n)

    seasonal_terms = np.column_stack([
        func(2 * np.pi * k * t / P)
        for k in range(1, K + 1)
        for func in (np.sin, np.cos)
    ])
    return seasonal_terms


def precompute_spline_bases(n, max_df_trend=6):

    t = np.arange(n)

    # Precompute spline basis for all df_trend values
    spline_bases = {}
    for df in range(2, max_df_trend+1):  # for max. df_trend = 6
        spline_bases[df] = dmatrix(
            f"bs(t, df={df}, degree=1, include_intercept=True)",
            {"t": t},
            return_type="dataframe"
        ).values

    return spline_bases


def decompose_spline_harmonic_fast(ts, spline_bases=None, seasonal_terms=None, P=60, K=2, df_trend=6):
    
    n = len(ts)

    if spline_bases is None:
        spline_bases = precompute_spline_bases(n, df_trend) 
    if seasonal_terms is None:
        seasonal_terms = precompute_harmonic_seasonality(n, P, K)

    best_score = np.inf
    best_trend = None
    best_seasonal = None
    best_resid = None
    best_df_trend = None

    for df in range(df_trend, df_trend + 1): # currently only testing one df_trend value! 
        
        # try:
            spline_basis = spline_bases[df]

            X = np.column_stack((spline_basis, seasonal_terms))

            # Solve least squares instead of statsmodels OLS
            beta, _, _, _ = np.linalg.lstsq(X, ts, rcond=None)

            k = X.shape[1]

            fitted = X @ beta
            resid = ts - fitted
            ssr = resid @ resid

            bic = n * np.log(ssr / n) + k * np.log(n)

            if bic < best_score:

                best_score = bic
                best_df_trend = df

                trend = spline_basis @ beta[:spline_basis.shape[1]]
                seasonal = seasonal_terms @ beta[spline_basis.shape[1]:]

                best_trend = trend
                best_seasonal = seasonal
                best_resid = resid

        # except Exception as e:
        #     print(f"Skipping df_trend={df} due to error: {e}")

    if best_df_trend is None:
        raise ValueError("All df_trend values failed.")

    return best_trend, best_seasonal, best_resid


def temporal_variability_metric(df):

    ts_df = drop_non_ts_cols(df)
    date_objects = [datetime.date.fromisoformat(col) for col in ts_df.columns]
    numeric_dates = [d.toordinal() for d in date_objects]
    
    tvms = []

    # Iterate over rows to calculate the temporal variability metric for each time series
    for i in range(len(ts_df)):
        
        x = ts_df.iloc[i].to_numpy(dtype=float)
        tvm = calculate_tvm(x, numeric_dates)
        tvms.append(tvm)
        
        if i%1000 == 0:
            print(f'{i}/{len(df)}')
            
    return tvms


def calculate_tvm(x, numeric_dates): 
    
    slope_overall = get_slope(numeric_dates, x) 
    window_slopes = moving_window_slopes(x, numeric_dates)

    return np.mean(np.abs(window_slopes - slope_overall)) 


def moving_window_slopes(x, numeric_dates, shift = month_shift):
    
    slopes = []

    for window_length in [n_obs_2yrs, n_obs_3yrs]:
        
        start_idx = 0
        end_idx = window_length

        while end_idx <= len(numeric_dates):

            slope_year = get_slope(time = numeric_dates[start_idx:end_idx], x = x[start_idx:end_idx])
            slopes.append(slope_year)

            start_idx += shift
            end_idx += shift
    
    return slopes


def get_slope(time, x):
    
    slope_day, intercept, r_val, p_val, std_err = linregress(time, x)
    slope_year = slope_day * 365.25
    
    return slope_year