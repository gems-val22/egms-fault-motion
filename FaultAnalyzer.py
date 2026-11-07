'''
Classes for creating polygons around faults, extracting EGMS points inside them, averaging and merging time series. 
Errors on time series are calculated as the bootstrap error on the mean. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import modules.coordinates as crd
import modules.timeseries as ts

from shapely.geometry import LineString, Polygon, Point
import datetime


class FaultPolygons:

    def __init__(self, fault_df, n_segments = 5, offset_distance = 2, line_spacing = 0.25, smoothing_window=1):
        # offset distance and line spacing in km

        self.n_segments = n_segments
        self.offset_distance = crd.km_to_lat(offset_distance)
        self.line_spacing = crd.km_to_lat(line_spacing)
        self.smoothing_window = smoothing_window

        self.spaced_fault = self.space_line(fault_df['longitude'], fault_df['latitude'])
        self.offset_lines = self.calculate_perpendicular_points()

        self.cut_indices = self.get_cut_indices()
        self.left_polygons, self.right_polygons = self.create_polygons()


    def space_line(self, xs, ys):

        assert len(xs) == len(ys), "Longitude and latitude lists must be of the same length"

        spaced_line = [(xs[0], ys[0])]  # Start with the first point

        for i in range(1, len(xs)): # iterate through points along fault line
            if LineString([spaced_line[-1], (xs[i], ys[i])]).length >= self.line_spacing or i == len(xs) - 1:  # if the points are not too close or it's the last point
                spaced_line.append((xs[i], ys[i]))  # add the point to the spaced fault
            else:
                continue

        spaced_line = LineString(spaced_line)
        return spaced_line
       

    def get_cut_indices(self):

        n = len(self.spaced_fault.coords) # number of points along the spaced fault line
        polygon_length = self.spaced_fault.length / self.n_segments 

        current_length = 0
        cut_indices = [0] # Start with the first point as the first cut index

        for i in range(1, n):
            segment_length = LineString([self.spaced_fault.coords[i-1], self.spaced_fault.coords[i]]).length
            current_length += segment_length

            if current_length >= polygon_length:
                cut_indices.append(i)
                current_length = 0

        if cut_indices[-1] != n-1:
            cut_indices.append(n-1)  # Ensure the last point is included

        return cut_indices
    

    def calculate_perpendicular_points(self):
        # code by Claude

        coords = np.array(self.spaced_fault.coords)
        left_points = []
        right_points = []
        
        for i in range(len(coords)):
            # Get perpendicular direction at this point (with smoothing)
            perp_dir = self.get_smoothed_perpendicular_direction(coords, i)
            
            # Calculate offset points
            left_point = coords[i] + perp_dir * self.offset_distance
            right_point = coords[i] - perp_dir * self.offset_distance
            
            left_points.append(tuple(left_point))
            right_points.append(tuple(right_point))
        
        return {
            'left_points': left_points,
            'right_points': right_points
        }


    def get_perpendicular_direction_from_neighbours(self, coords, index):
        # code by Claude
        
        n = len(coords)
        
        # Get neighbouring points
        if index == 0: # First point: use direction to next point
            prev_point = coords[0]
            next_point = coords[1]
        elif index == n - 1: # Last point: use direction from previous point
            prev_point = coords[n - 2]
            next_point = coords[n - 1]
        else: # Middle point: use direction between previous and next
            prev_point = coords[index - 1]
            next_point = coords[index + 1]
        
        # Calculate tangent direction
        tangent = next_point - prev_point
        tangent_norm = np.linalg.norm(tangent)
        
        if tangent_norm < 1e-10: # Handle degenerate case
            return np.array([0.0, 1.0])
        
        tangent = tangent / tangent_norm
        perpendicular = np.array([-tangent[1], tangent[0]]) # Perpendicular direction (rotate 90 degrees counterclockwise)
        
        return perpendicular


    def get_smoothed_perpendicular_direction(self, coords, index):
        # code by Claude
        
        if self.smoothing_window <= 0:
            # No smoothing
            return self.get_perpendicular_direction_from_neighbours(coords, index)
        
        n = len(coords)
        
        # Calculate perpendicular directions for points in the window
        perp_directions = []
        
        for offset in range(-self.smoothing_window, self.smoothing_window + 1):
            i = index + offset
            
            # Clamp to valid range
            if i < 0 or i >= n:
                continue
            
            perp_dir = self.get_perpendicular_direction_from_neighbours(coords, i)
            perp_directions.append(perp_dir)
        
        # Average the perpendicular directions
        avg_perp = np.mean(perp_directions, axis=0)
        
        # Normalize to unit vector
        avg_norm = np.linalg.norm(avg_perp)
        if avg_norm < 1e-10:
            return np.array([0.0, 1.0])
        
        return avg_perp / avg_norm
    

    def create_polygons(self):

        left_polygons = []
        right_polygons = []

        for i in range(len(self.cut_indices)-1):
            start_idx = self.cut_indices[i]
            end_idx = self.cut_indices[i+1] + 1 # Include the end index in the slice

            # slice center and offset lines
            center_coords = np.array(self.spaced_fault.coords[start_idx:end_idx] )
            left_coords = np.array(self.offset_lines['left_points'][start_idx:end_idx])
            right_coords = np.array(self.offset_lines['right_points'][start_idx:end_idx])

            # Respace the offset points for smoother polygons
            left_coords = self.space_line(left_coords[:,0], left_coords[:,1]).coords # Respace the left offset points for smoother polygons
            right_coords = self.space_line(right_coords[:,0], right_coords[:,1]).coords # Respace the right offset points for smoother polygons

            # create polygons and add to list
            left_coords_list = list(center_coords) + list(left_coords)[::-1] # reverse
            left_polygon = self.draw_polygon(left_coords_list)

            right_coords_list = list(center_coords) + list(right_coords)[::-1] # reverse
            right_polygon = self.draw_polygon(right_coords_list)

            left_polygons.append(left_polygon)
            right_polygons.append(right_polygon)

        return left_polygons, right_polygons
    

    def draw_polygon(self, polygon_coords_list):

        polygon = Polygon(polygon_coords_list)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)  # Attempt to fix invalid polygon
        return polygon


class PolygonTS:

    def __init__(self, polygon, egms_df, yearly_ts_error = 1):
        
        self.polygon = polygon
        self.egms_df = egms_df
        self.yearly_ts_error = yearly_ts_error

        # get points within the polygon
        self.polygon_points = self.get_points_within_polygon()

        # get average time series
        self.tsdata = ts.drop_non_ts_cols(self.polygon_points.copy(deep=True)).reset_index(drop=True) # time series data for points within the polygon, with non-time-series columns dropped
        
        self.n_series, self.n_time = self.tsdata.shape
        self.end_ts_error = yearly_ts_error * np.round(self.n_time / 60.875)  # scale error to the length of the time series
                                                                        # 60.875 = avg. measurements in a year (6-day intervals)
        self.polygon_ts = self.get_ts_df()


    def get_points_within_polygon(self):

        points_within_polygon = []

        for i in range(len(self.egms_df)):
            point_geom = Point(self.egms_df.iloc[i].longitude, self.egms_df.iloc[i].latitude)

            if self.polygon.contains(point_geom):
                points_within_polygon.append(i)

        polygon_points_df = self.egms_df.iloc[points_within_polygon]

        return polygon_points_df
    

    def get_ts_df(self):

        mean_ts = self.get_mean_ts()
        correlation_uncertainty = self.bootstrap_with_noise()

        ts_df = pd.DataFrame({
            'mean_ts': mean_ts['mean_ts'],
            'uncertainty': correlation_uncertainty
        }, index=mean_ts.index) 

        return ts_df


    def get_mean_ts(self):

        timeseries = self.tsdata

        mean_values = np.array(timeseries.mean())
        df = pd.DataFrame(mean_values, columns=['mean_ts'])

        dates = [datetime.date.fromisoformat(col) for col in timeseries.columns]
        df['date'] = pd.DatetimeIndex(dates)
        df = df.set_index('date')
        df.index.name = None

        return df



    def bootstrap_with_noise(self, n_boot=500):
            """
            Hybrid bootstrap of time series with measurement noise.
            
            Parameters:
                n_boot : number of bootstrap resamples
    
            Returns:
                correlation_uncertainty : array (n_time,) — uncertainty on the regional mean
                                        accounting for spatial correlation between points.
            """
    
            data = self.tsdata
            sample_means = np.zeros((n_boot, self.n_time))
    
            for b in range(n_boot):
                # 1. Resample the time series with replacement
                idx = np.random.choice(self.n_series, size=self.n_series, replace=True)
                sample = data.iloc[idx]
    
                # 2. Add simulated measurement noise
                sample_noisy = self.random_trend_uncertainty(sample)
    
                # 3. Store the mean of the bootstrapped sample
                sample_mean = np.mean(sample_noisy, axis=0)
                sample_means[b] = sample_mean
    
            # Average across bootstrap iterations
            bootstrap_se = sample_means.std(axis=0, ddof=1)
    
            return bootstrap_se


    def random_trend_uncertainty(self, sample):

        sample = np.atleast_2d(sample)  # ensure 2D, shape (n_series, n_time)

        uncertainty_strength = (np.random.random(size=(self.n_series, 1)) * 2 * self.end_ts_error) - self.end_ts_error
        time_scale = np.arange(self.n_time) / self.n_time  # grows from 0 to ~1
        uncertainty = uncertainty_strength * time_scale  # (n_series, 1) * (1, n_time)

        return sample + uncertainty 


class TSShift():

    def __init__(self, ts1_df, ts2_df, ts3_df, join_index_1 = 30, join_index_2 = 30):

        # dfs have columns 'mean_ts', 'uncertainty', indexed by date
        # join indices refer to the second and third time series

        self.ts1_df = ts1_df
        self.ts2_df = ts2_df
        self.ts3_df = ts3_df

        self.join_index_1 = join_index_1
        self.join_index_2 = join_index_2

        self.ts1, self.ts2, self.ts3 = self.get_shifted_dfs()
        self.merged_ts = self.get_merged_ts()
        

    def get_shifted_dfs(self):

        ts1 = ts.ensure_zero_start(self.ts1_df.mean_ts)
        ts2 = self.shift_ts(timeseries = self.ts2_df.mean_ts, ref_timeseries = ts1, join_index = self.join_index_1)
        ts3 = self.shift_ts(timeseries = self.ts3_df.mean_ts, ref_timeseries = ts2, join_index = self.join_index_2)

        ts1 = ts.get_interval_df(ts1, self.ts1_df.uncertainty)
        ts2 = ts.get_interval_df(ts2, self.ts2_df.uncertainty)
        ts3 = ts.get_interval_df(ts3, self.ts3_df.uncertainty)

        return ts1, ts2, ts3

    
    def get_merged_ts(self):

        merge_date_1 = self.ts2.index[self.join_index_1]
        merge_date_2 = self.ts3.index[self.join_index_2]

        merged_ts = pd.concat([self.ts1.loc[:merge_date_1], self.ts2.loc[merge_date_1:merge_date_2], self.ts3.loc[merge_date_2:]])
        return merged_ts


    def shift_ts(self, timeseries, ref_timeseries, join_index):

        timeseries = ts.ensure_zero_start(timeseries, start_idx=join_index)
        trend, seasonal, resid = ts.stl_decomposition(timeseries)

        shift_date = timeseries.index[join_index]
        shift_value = self.get_trend_value(ref_timeseries, shift_date)
        
        trend_adj = trend + shift_value # shift the trend
        timeseries_adjusted = trend_adj + seasonal + resid # reconstruct the time series with shifted trend

        return timeseries_adjusted 

    
    def get_trend_value(self, timeseries, date):

        trend, seasonal, resid = ts.stl_decomposition(timeseries)
        trend_value = trend.loc[date]

        return trend_value


class TrendPValues():

    def __init__(self, tsfile, verbose=False, num_bootstrap=500, max_df_trend=6):

        # Read data from tsfile
        self.ts1 = tsfile['left_mean_ts'].values[:-2] # remove last 2 values because bootstrap causes dimension issues
        self.ts2 = tsfile['right_mean_ts'].values[:-2] # remove last 2 values because bootstrap causes dimension issues
        self.uncertainty1 = tsfile['left_uncertainty'].values[:-2]
        self.uncertainty2 = tsfile['right_uncertainty'].values[:-2]

        # Decompose time series to extract trends and residuals
        self.trend1, _, self.resid1 = ts.decompose_spline_harmonic_fast(self.ts1)
        self.trend2, _, self.resid2 = ts.decompose_spline_harmonic_fast(self.ts2)

        # Calculate observed trend difference
        trend_diff_obs = self.trend_difference(self.trend1, self.trend2)

        # Calculate bootstrap trend differences assuming a common underlying trend 
        trend_diffs_bootstrap = self.block_bootstrap(num_bootstrap=num_bootstrap, max_df_trend = max_df_trend)

        # Calculate p-value: estimate likelihood of the time series sharing a common trend
        self.p_value = len(trend_diffs_bootstrap[trend_diffs_bootstrap > trend_diff_obs])/len(trend_diffs_bootstrap)

        # Give results if prompted
        if verbose:
            if self.p_value < 0.05:
                print(f"Reject null hypothesis: trends are different (p-value: {self.p_value:.4f})")
            else:
                print(f"Fail to reject null hypothesis: trends are not significantly different (p-value: {self.p_value:.4f})")
            plt.hist(trend_diffs_bootstrap, label='Bootstrap Trend Differences')
            plt.vlines(x=trend_diff_obs, ymin=0, ymax=plt.ylim()[1], color='red', label='Observed Trend Difference')
            plt.legend()


    def trend_difference(self, trend1, trend2):
        return np.sum((trend1 - trend2) ** 2)


    def block_bootstrap(self, num_bootstrap=500, max_df_trend=6):

        common_trend = (self.trend1 + self.trend2) / 2
        n = len(common_trend)
        trend_diffs_bootstrap = []

        resid_blocks1 = self.get_residual_bootstrap_blocks(self.resid1)
        resid_blocks2 = self.get_residual_bootstrap_blocks(self.resid2)
        n_resid_blocks = len(resid_blocks1)

        # Precompute spline bases and seasonality terms 
        spline_bases = ts.precompute_spline_bases(n=n, max_df_trend=max_df_trend)
        seasonality_terms = ts.precompute_harmonic_seasonality(n=n, P=60, K=2)

        for i in range(num_bootstrap):

            # Resample residual blocks
            resid_resample = np.random.choice(np.arange(n_resid_blocks), size=n_resid_blocks, replace=True)
            resid_bootstrap1 = np.concatenate([resid_blocks1[j] for j in resid_resample])
            resid_bootstrap2 = np.concatenate([resid_blocks2[j] for j in resid_resample])

            # Add a random uncertainty to the trends - to reflect measurement uncertainty 
            common_trend_noisy_1 = self.add_uncertainty(common_trend, error=self.uncertainty1)
            common_trend_noisy_2 = self.add_uncertainty(common_trend, error=self.uncertainty2)

            # Add random seasonality to the noisy trends
            common_trend_noisy_1 += self.add_random_seasonality(n=n)
            common_trend_noisy_2 += self.add_random_seasonality(n=n)

            # Construct bootstrap time series
            ts_bootstrap1 = common_trend_noisy_1 + resid_bootstrap1
            ts_bootstrap2 = common_trend_noisy_2 + resid_bootstrap2

            # Decompose bootstrap time series and calculate trend difference
            trend_bootstrap1, _, _ = ts.decompose_spline_harmonic_fast(ts=ts_bootstrap1, spline_bases=spline_bases, \
                                                                        seasonal_terms=seasonality_terms)
            trend_bootstrap2, _, _ = ts.decompose_spline_harmonic_fast(ts=ts_bootstrap2, spline_bases=spline_bases, \
                                                                        seasonal_terms=seasonality_terms)

            trend_diff_bootstrap = self.trend_difference(trend_bootstrap1, trend_bootstrap2)
            trend_diffs_bootstrap.append(trend_diff_bootstrap.item())

        return np.array(trend_diffs_bootstrap)


    def get_residual_bootstrap_blocks(self, residual, sample_length = 15):

        n_samples = len(residual) // sample_length
        blocks = [residual[i*sample_length:(i+1)*sample_length] for i in range(n_samples-1)]
        blocks.append(residual[(n_samples-1)*sample_length:])  # add the last block, which may be shorter or longer

        return blocks


    def add_uncertainty(self, trend, error):

        trend_uncertainty = np.random.randn(1) * 1.96 * error  # provided error scaled by a random normal variable
        return trend + trend_uncertainty


    def add_random_seasonality(self, n, period = 60):

        t = np.arange(n)
        amplitude = np.random.random(size=(1))*2 # random amplitude between 0 and 2
        offset = np.random.random(size=(1))*np.pi # random phase offset between 0 and pi (half a period)
        seasonality = amplitude * np.sin(2 * np.pi * t / period + offset)

        return seasonality

