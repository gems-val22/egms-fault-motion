
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from patsy import dmatrix


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