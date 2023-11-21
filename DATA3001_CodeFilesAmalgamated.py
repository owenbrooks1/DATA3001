#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:56:55 2023
    # Amalgamated files for Group 4, Racing Simulator Project -  "CrticialF1" 
    
    # Note: A lot of the code written over the course of this project 
            has been excluded. This is an abbreviated version, intended to 
            outline the key steps in the project and some of the checks and 
            tests undertaken 
    
    
    
@author: Owenbrooks
"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import packages 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error
from scipy.interpolate import UnivariateSpline

from sklearn.decomposition import PCA

from shapely.geometry import Point, Polygon, LineString, MultiPoint

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

from sklearn.model_selection import GridSearchCV

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Read Files  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

f1sim_left = pd.read_csv('/Users/Owenbrooks/Desktop/DATA3001/Project/Data/Other Data/f1sim-ref-left.csv')
f1sim_right = pd.read_csv('/Users/Owenbrooks/Desktop/DATA3001/Project/Data/Other Data/f1sim-ref-right.csv')

f1sim_line = pd.read_csv('/Users/Owenbrooks/Desktop/DATA3001/Project/Data/Other Data/f1sim-ref-line.csv')
f1sim_turns = pd.read_csv('/Users/Owenbrooks/Desktop/DATA3001/Project/Data/Other Data/f1sim-ref-turns.csv')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Track Plot
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#df['DIST_FROM_START'] = df.groupby(['SESSION_IDENTIFIER', 'LAP_NUM'])['LAP_DISTANCE'].transform(lambda x: x - x.min())
#df['DIST_FROM_START'].describe()

halfway = len(f1sim_turns) // 2

# Plot track w boundaries 
plt.figure(figsize=(16, 14))
plt.scatter(f1sim_left['WORLDPOSX'], f1sim_left['WORLDPOSY'], c='black', label='Left Boundary', s=.05)
plt.scatter(f1sim_right['WORLDPOSX'], f1sim_right['WORLDPOSY'], c='black', label='Right Boundary', s=.05)

# Annotate apex
for i, row in f1sim_turns.iterrows():
    
    offset = (-60, 60) if i < halfway else (60, -50)


    # Annotyate arrows
    plt.annotate(
        f"Turn {int(row['TURN'])} Apex", 
        xy=(row['APEX_X1'], row['APEX_Y1']), 
        xytext=offset, 
        textcoords="offset points", 
        ha='center', 
        fontsize=11, 
        color='red', 
        weight='bold',
        arrowprops=dict(facecolor='red', shrink=0.05)
    )
    plt.annotate(f"Turn {int(row['TURN'])} Corner 1", (row['CORNER_X1'], row['CORNER_Y1']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7, color='#004C99')
    plt.annotate(f"Turn {int(row['TURN'])} Corner 2", (row['CORNER_X2'], row['CORNER_Y2']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7, color='#004C99')


plt.xlabel('WORLDPOSX')
plt.ylabel('WORLDPOSY')
plt.title('Track')
plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# FEATURE ENG: TRACK DISTANCE 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Compute centreline (based on track boundaries)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def compute_centerline(left, right):
    centerline = []
    for lx, ly in zip(left['WORLDPOSX'], left['WORLDPOSY']):
        distances = ((right['WORLDPOSX'] - lx) ** 2 + (right['WORLDPOSY'] - ly) ** 2)
        closest_idx = distances.idxmin()
        center_x = (lx + right['WORLDPOSX'].iloc[closest_idx]) / 2
        center_y = (ly + right['WORLDPOSY'].iloc[closest_idx]) / 2
        centerline.append((center_x, center_y))
    return pd.DataFrame(centerline, columns=['WORLDPOSX', 'WORLDPOSY'])

centerline = compute_centerline(left_boundary, right_boundary)

#------------------------------------------------------------------------------
# 1, SORT CENTRELINE

start_line_data = car_data[car_data['LAP_DISTANCE'] == car_data['LAP_DISTANCE'].min()].iloc[0]
start_line_x = start_line_data['WORLDPOSX']
start_line_y = start_line_data['WORLDPOSY']

# Find  closest point in centerline to the start_line_x\y
distances = np.sqrt((centerline['WORLDPOSX'] - start_line_x)**2 + (centerline['WORLDPOSY'] - start_line_y)**2)
closest_idx = distances.idxmin()

# Get seg of start/fin line 
segment_start = max(0, closest_idx - 5)
segment_end = min(len(centerline) - 1, closest_idx + 5)
segment = centerline.iloc[segment_start:segment_end+1]

# Midpoint of seg
midpoint_x = segment['WORLDPOSX'].mean()
midpoint_y = segment['WORLDPOSY'].mean()


# Using start/finish line for direction, sort 
def sort_centerline_from_start(df, start_point):
    sorted_df = pd.DataFrame(columns=['WORLDPOSX', 'WORLDPOSY'])
    remaining = df.copy()
    current_point = start_point
    
    while not remaining.empty:
        distances = ((remaining['WORLDPOSX'] - current_point[0]) ** 2 + 
                     (remaining['WORLDPOSY'] - current_point[1]) ** 2)
        nearest_idx = distances.idxmin()
        current_point = (remaining.loc[nearest_idx, 'WORLDPOSX'], 
                         remaining.loc[nearest_idx, 'WORLDPOSY'])
        sorted_df = sorted_df.append(remaining.loc[nearest_idx])
        remaining.drop(nearest_idx, inplace=True)
        
    return sorted_df.reset_index(drop=True)


start_point = (midpoint_x, midpoint_y)
sorted_centerline = sort_centerline_from_start(centerline, start_point)

#------------------------------------------------------------------------------
# 2. Compute the scaled cumulative distances

# MEthod1 
#------------------------------------------------------------------------------
# Computing Raw Cumulative Distance
def compute_raw_cumulative_distance(data):
    distances = [0]
    for i in range(1, len(data)):
        dx = data['WORLDPOSX'].iloc[i] - data['WORLDPOSX'].iloc[i-1]
        dy = data['WORLDPOSY'].iloc[i] - data['WORLDPOSY'].iloc[i-1]
        dist = np.sqrt(dx**2 + dy**2)
        distances.append(distances[-1] + dist)
    data['RAW_X_TRACK_DIST'] = distances
    return data

centerline_raw_distance = compute_raw_cumulative_distance(centerline)

# Scaling Factor
distances_to_midpoint = np.sqrt((centerline_raw_distance['WORLDPOSX'] - midpoint_x)**2 + (centerline_raw_distance['WORLDPOSY'] - midpoint_y)**2)
start_finish_idx = distances_to_midpoint.idxmin()

# Calculate the raw cumu dist of 1 lap using this point as the start/fin
raw_distance_of_one_lap = centerline_raw_distance['RAW_X_TRACK_DIST'].iloc[start_finish_idx]

# scaling factor
scaling_factor = known_track_distance / raw_distance_of_one_lap

# Apply the scaling factor
centerline_raw_distance['SCALED_X_TRACK_DIST'] = centerline_raw_distance['RAW_X_TRACK_DIST'] * scaling_factor

#------------------------------------------------------------------------------
# Method2 

# Seq Dist Calc
sorted_centerline['sequential_distance'] = np.sqrt(
    np.diff(sorted_centerline['WORLDPOSX'], prepend=sorted_centerline['WORLDPOSX'].iloc[0])**2 +
    np.diff(sorted_centerline['WORLDPOSY'], prepend=sorted_centerline['WORLDPOSY'].iloc[0])**2
)

# Cum Sum
sorted_centerline['cumulative_distance'] = sorted_centerline['sequential_distance'].cumsum()

# Scale by known track len from LAP_DISTANCE 
known_track_length = car_data['LAP_DISTANCE'].max() - car_data['LAP_DISTANCE'].min()
scaling_factor = known_track_length / sorted_centerline['cumulative_distance'].iloc[-1]
sorted_centerline['scaled_cumulative_distance'] = sorted_centerline['cumulative_distance'] * scaling_factor

sorted_centerline.drop(columns=['sequential_distance', 'cumulative_distance'], inplace=True)


sorted_centerline_for_plotting = sorted_centerline.iloc[::-1].reset_index(drop=True)

# Plotting
left_x, left_y = left_boundary['WORLDPOSX'].values, left_boundary['WORLDPOSY'].values
right_x, right_y = right_boundary['WORLDPOSX'].values, right_boundary['WORLDPOSY'].values
mid_x, mid_y = sorted_centerline_for_plotting['WORLDPOSX'].values, sorted_centerline_for_plotting['WORLDPOSY'].values
track_dist = sorted_centerline_for_plotting['scaled_cumulative_distance'].values

# PLOT 
plt.figure(figsize=(12, 12))
plt.scatter(left_x, left_y, c='blue', label='Left Boundary')
plt.scatter(right_x, right_y, c='red', label='Right Boundary')
plt.scatter(mid_x, mid_y, c=track_dist, cmap='viridis', marker='.', label='Centerline')
plt.colorbar(label='X_TRACK_DIST')
plt.legend()
plt.title('Track with X_TRACK_DIST Mapped on Centerline')
plt.show()


# WORKING 

#------------------------------------------------------------------------------
# 3. Reorder & Checks 
first_half = sorted_centerline.iloc[:closest_idx]
second_half = sorted_centerline.iloc[closest_idx:]

# Reverse the order 
first_half = first_half.iloc[::-1].reset_index(drop=True)
second_half = second_half.iloc[::-1].reset_index(drop=True)

# Concat
reordered_centerline = pd.concat([second_half, first_half]).reset_index(drop=True)


# Recompute the scaled cumulative distances on reordered centerline


#------------------------------------------------------------------------------
#  SAVE
#------------------------------------------------------------------------------

reordered_centerline.columns

output_path = "/Users/Owenbrooks/Desktop/reordered_centerline.csv"
reordered_centerline.to_csv(output_path, index=False)


reordered_centerline.columns.to_list()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#  APPLY SORTED CENTRELINE TO CAR DATSA 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def closest_distance(row, centerline):
    distances = np.sqrt((centerline['WORLDPOSX'] - row['WORLDPOSX'])**2 + 
                        (centerline['WORLDPOSY'] - row['WORLDPOSY'])**2)
    if distances.isnull().any():
        return np.nan
    closest_idx = distances.idxmin()
    if np.isnan(closest_idx):
        return np.nan
    return centerline.iloc[int(closest_idx)]['scaled_cumulative_distance']

car_data['X_TRACK_DIST'] = car_data.apply(lambda row: closest_distance(row, reordered_centerline), axis=1)
car_data['X_TRACK_DIST'].isna().sum()
car_data.shape

#------------------------------------------------------------------------------
#       Checks 
#------------------------------------------------------------------------------

car_data['X_TRACK_DIST'] = car_data['X_TRACK_DIST'].max() - car_data['X_TRACK_DIST']


# PLotting 
left_x, left_y = left_boundary['WORLDPOSX'].values, left_boundary['WORLDPOSY'].values
right_x, right_y = right_boundary['WORLDPOSX'].values, right_boundary['WORLDPOSY'].values

car_x = car_data['WORLDPOSX'].values
car_y = car_data['WORLDPOSY'].values
car_track_dist = car_data['X_TRACK_DIST'].values


plt.figure(figsize=(12, 12))
plt.scatter(left_x, left_y, c='blue', label='Left Boundary')
plt.scatter(right_x, right_y, c='red', label='Right Boundary')
plt.scatter(car_x, car_y, c=car_track_dist, cmap='viridis', marker='.', label='Car Position')
plt.colorbar(label='X_TRACK_DIST')
plt.legend()
plt.title('Track with X_TRACK_DIST Mapped on Car\'s Positions')
plt.show()


# Plot dual hist 
plt.figure(figsize=(12, 6))
plt.hist(car_data['LAP_DISTANCE'], bins=50, alpha=0.5, label='LAP_DISTANCE')
plt.hist(car_data['X_TRACK_DIST'], bins=50, alpha=0.5, label='X_TRACK_DIST')
plt.xlabel('Distance')
plt.ylabel('Number of Data Points')
plt.title('Distribution of LAP_DISTANCE vs X_TRACK_DIST')
plt.legend()
plt.show()

# Both look good, working as intended 

#------------------------------------------------------------------------------
#  SAVE
#------------------------------------------------------------------------------


reordered_centerline.columns

output_path = "/Users/Owenbrooks/Desktop/x_track_dist_solved.csv"
car_data.to_csv(output_path, index=False)


reordered_centerline.columns.to_list()
output_path_pickle = "/Users/Owenbrooks/Desktop/x_track_dist_solved.pkl"
car_data.to_pickle(output_path_pickle)

#------------------------------------------------------------------------------
# Further Checks 

import matplotlib.pyplot as plt

# Extracting X and Y coordinates for plotting
left_x, left_y = left_boundary['WORLDPOSX'].values, left_boundary['WORLDPOSY'].values
right_x, right_y = right_boundary['WORLDPOSX'].values, right_boundary['WORLDPOSY'].values
mid_x, mid_y = centerline_raw_distance['WORLDPOSX'].values, centerline_raw_distance['WORLDPOSY'].values
track_dist_scaled = centerline_raw_distance['SCALED_X_TRACK_DIST'].values

# Create a figure and axis
plt.figure(figsize=(12, 12))

# Plot the left and right boundaries
plt.scatter(left_x, left_y, c='blue', label='Left Boundary')
plt.scatter(right_x, right_y, c='red', label='Right Boundary')

# Plot the centerline points with a colormap based on SCALED_X_TRACK_DIST
plt.scatter(mid_x, mid_y, c=track_dist_scaled, cmap='viridis', marker='.', label='Scaled Centerline Distance')
plt.colorbar(label='SCALED_X_TRACK_DIST')

# Legend, title, and show the plot
plt.legend()
plt.title('Track with Scaled Distance Mapped on Centerline')
plt.show()

#------------------------------------------------------------------------------

#Mapping Each Car's Position to the Reference Path
#car_data = pd.read_csv('first_lap_first_session.csv')

def map_to_reference(data, reference):
    distances = np.sqrt((np.array(data['WORLDPOSX'])[:, None] - np.array(reference['WORLDPOSX']))**2 + 
                    (np.array(data['WORLDPOSY'])[:, None] - np.array(reference['WORLDPOSY']))**2)
    idx = distances.argmin(axis=1)
    data['X_TRACK_DIST'] = reference['X_TRACK_DIST'].iloc[idx].values
    return data

car_data_mapped = map_to_reference(car_data, dense_centerline)

#smoothing
def smooth_data(data, column, window_size=5):
    data[column + '_smoothed'] = data[column].rolling(window=window_size).mean()
    return data

car_data_smoothed = smooth_data(car_data_mapped, 'X_TRACK_DIST')

#handling outliers
def remove_outliers(data, column, threshold=100):
    data = data[data[column].diff().abs() < threshold]
    return data

car_data_cleaned = remove_outliers(car_data_smoothed, 'X_TRACK_DIST_smoothed')

car_data_cleaned.to_csv('first_lap_first_session.csv', index=False)


# Plotting the histogram for X_TRACK_DIST
plt.figure(figsize=(10, 6))
plt.hist(car_data_mapped['X_TRACK_DIST'], bins=50, density=True, alpha=0.75)
plt.title('Distribution of X_TRACK_DIST')
plt.xlabel('Distance from Start Line (X_TRACK_DIST)')
plt.ylabel('Density')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

#------------------------------------------------------------------------------
# Interpolation step (discont.)

#interpolating centreline

def interpolate_path(data):
    t = range(len(data))
    cs_x = CubicSpline(t, data['WORLDPOSX'])
    cs_y = CubicSpline(t, data['WORLDPOSY'])
    t_new = np.linspace(0, len(data) - 1, len(data) * 3) # 10 times denser
    return pd.DataFrame({'WORLDPOSX': cs_x(t_new), 'WORLDPOSY': cs_y(t_new)})

dense_centerline = interpolate_path(centerline)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# FEATURE ENG: LAP VALID/INVALID BINARY 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# X_LAP_INVALID_LEFT

car_df['X_LAP_INVALID_LEFT'] = 0

for index, row in car_df.iterrows():

    distances = np.sqrt((left_boundary['WORLDPOSX'] - row['WORLDPOSX'])**2 + (left_boundary['WORLDPOSY'] - row['WORLDPOSY'])**2)
    nearest_idx = distances.idxmin()
    
    # Check if nearest_idx is not nan
    if not pd.isna(nearest_idx):
        nearest_point = left_boundary.iloc[int(nearest_idx)]

        # conditions
        if (row['WORLDPOSX'] > nearest_point['WORLDPOSX']) and (row['WORLDPOSY'] < nearest_point['WORLDPOSY']):
            if distances.iloc[int(nearest_idx)] > 1:  # If the car is more than 1m outside the boundary
                car_df.at[index, 'X_LAP_INVALID_LEFT'] = 1
        elif (row['WORLDPOSX'] > nearest_point['WORLDPOSX']) and (row['WORLDPOSY'] > nearest_point['WORLDPOSY']):
            if distances.iloc[int(nearest_idx)] > 1:  # If the car is more than 1m outside the boundary
                car_df.at[index, 'X_LAP_INVALID_LEFT'] = 1
        

#------------------------------------------------------------------------------
# X_LAP_INVALID_RIGHT


car_df['X_LAP_INVALID_RIGHT'] = 0

for index, row in car_df.iterrows():

    distances = np.sqrt((right_boundary['WORLDPOSX'] - row['WORLDPOSX'])**2 + (right_boundary['WORLDPOSY'] - row['WORLDPOSY'])**2)
    nearest_idx = distances.idxmin()
    # Check if nearest_idx is not nan
    if not pd.isna(nearest_idx):
        nearest_point = right_boundary.iloc[int(nearest_idx)]

        # Check conditions
        if (row['WORLDPOSX'] < nearest_point['WORLDPOSX']) and (row['WORLDPOSY'] > nearest_point['WORLDPOSY']):
            if distances.iloc[int(nearest_idx)] > 1:  # If the car is more than 1m outside the boundary
                car_df.at[index, 'X_LAP_INVALID_RIGHT'] = 1
        elif (row['WORLDPOSX'] < nearest_point['WORLDPOSX']) and (row['WORLDPOSY'] < nearest_point['WORLDPOSY']):
            if distances.iloc[int(nearest_idx)] > 1:  # If the car is more than 1m outside the boundary
                car_df.at[index, 'X_LAP_INVALID_RIGHT'] = 1
        
        
#------------------------------------------------------------------------------        
# CHECKS FOR X_LAP_INVALID_LEFT/RIGHT        

# Invalid 
invalid_left = car_df[car_df['X_LAP_INVALID_LEFT'] == 1]
invalid_right = car_df[car_df['X_LAP_INVALID_RIGHT'] == 1]

plt.figure(figsize=(10, 10))
plt.scatter(left_x, left_y, c='black', s=.05, label='Left Boundary')
plt.scatter(right_x, right_y, c='black', s=.05, label='Right Boundary')
#plt.scatter(car_x, car_y, c='green', marker='.', label='Turns 1 & 2 df')
plt.scatter(invalid_left['WORLDPOSX'], invalid_left['WORLDPOSY'], c='red', marker='x', s=0.05.label='Invalid Left')
plt.scatter(invalid_right['WORLDPOSX'], invalid_right['WORLDPOSY'], c='blue', marker='x', s=0.05 label='Invalid Right')

plt.xlim(250,600)
plt.ylim(250, -200)
plt.legend()
plt.title('Invalid')
plt.show()

# Valid  Left / Right

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

# Valid Left 
ax[0].scatter(left_x, left_y, c='black', s=2, label='Left Boundary')
ax[0].scatter(right_x, right_y, c='black', s=2, label='Right Boundary')
ax[0].scatter(valid_left['WORLDPOSX'], valid_left['WORLDPOSY'], c='red', marker='x', s=0.05, label='Valid Left')
ax[0].set_xlim(250, 600)
ax[0].set_ylim(250, -200)
ax[0].legend()
ax[0].set_title('Valid Left)')

# Valid Right 
ax[1].scatter(left_x, left_y, c='black', s=2, label='Left Boundary')
ax[1].scatter(right_x, right_y, c='black', s=2, label='Right Boundary')
ax[1].scatter(valid_right['WORLDPOSX'], valid_right['WORLDPOSY'], c='blue', marker='x', s=0.05, label='Valid Right')
ax[1].set_xlim(250, 600)
ax[1].set_ylim(250, -200)
ax[1].legend()
ax[1].set_title('Valid (Right)')

plt.tight_layout()
plt.show()

# Good 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# FEATURE ENG: DISTANCE FROM CENTRE LINE (LATERAL DIST)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------        
# X_DIST_FROM_CENTRE 

# USing same sorted centreline functions from track dist calcs 

#------------------------------------------------------------------------------     
def closest_segment_distance(x, y, line_segments):
    min_distance = float('inf')
    on_left = True  
    final_on_left = True  
    
    segment_found = False # INdicator 

    for (x1, y1, x2, y2) in line_segments:
        A = np.array([x1, y1])
        B = np.array([x2, y2])
        P = np.array([x, y])

        if np.dot(P - A, B - A) < 0 or np.dot(P - B, A - B) < 0:
            continue  # Point's projection doesn't lie on the line segment

        segment_found = True  # else found a segment that can project the point
        
        normal_length = np.linalg.norm(np.cross(B - A, A - P)) / np.linalg.norm(B - A)
        direction = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)

        if direction < 0:
            on_left = True
        else:
            on_left = False

        if normal_length < min_distance:
            min_distance = normal_length
            final_on_left = on_left  

    # If no suitable segment is found,  return NaN 
    if not segment_found:
        return float('nan')

    return min_distance if final_on_left else -min_distance


# Calc the distance, i.e. APPPL TO DF 
car_df['DIST_FROM_CENTERLINE'] = car_df.apply(lambda row: closest_segment_distance(row['WORLDPOSX'], row['WORLDPOSY'], line_segments), axis=1)


#car_df['DIST_FROM_CENTERLINE'].head()

##car_df['DIST_FROM_CENTERLINE'].describe()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# FEATURE ENG: DISTANCE FROM REFERENCE LINE (LATERAL DIST)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def calculate_distance_from_rl(row):
    lx, ly = row['WORLDPOSX'], row['WORLDPOSY']
    
    rl_squared_distance = ((ref_line['WORLDPOSX'] - lx) ** 2 +
                         (ref_line['WORLDPOSY'] - ly) ** 2)

    closest_idx = rl_squared_distance.idxmin()
    if not pd.isna(closest_idx):
    
        rl_squared_distance = ((ref_line['WORLDPOSX'].iloc[int(closest_idx)] - lx) ** 2 +
                         (ref_line['WORLDPOSY'].iloc[int(closest_idx)] - ly) ** 2)
    
        rl_distance = np.sqrt(rl_squared_distance)
    
        #on the reference line
        if (rl_distance < 1):
            return 0

        if ((lx < ref_line['WORLDPOSX'].iloc[closest_idx]) and 
             ((ly > ref_line['WORLDPOSY'].iloc[closest_idx]) or 
             (ly < ref_line['WORLDPOSY'].iloc[closest_idx])) and 
             rl_distance > 1) :            
            return rl_distance

        else:
            return -rl_distance
    
# Apply to actual car df 
car_df['X_DIST_FROM_RL'] = car_df.apply(calculate_distance_from_rl, axis=1) 



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aggregate Features [2] - With added constraint (whole lap aggs)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# make primary key from sessionID & lap num 
car_df['X_LAP_ID'] = car_df['SESSION_IDENTIFIER'].astype(str) + "_" + car_df['LAP_NUM'].astype(str)

#------------------------------------------------------------------------------
# ADDED CONSTRAINT 

#During the computation of TRACK_DISTANCE_DIFF (the difference in X_TRACK_DIST
#between consecutive rows), if we find a value that is negative indicating
#the car moved backward or there was a data discrepancy):

# Add feature: X_CAR_FIRST_REVERSED_AT_DIST:
#If the car never moved backward during the lap, the feature value should be 0.
#If the car moved backward, the feature value should be the X_TRACK_DIST from
# the FIRST occurrence when this happened.


#After identifying such a row, we want to skip (continue) subsequent 
#calculations for this particular row and proceed to the next one, waiting 
#until the car is moving forward again before resuming calcs.
#------------------------------------------------------------------------------
def aggregate_features(group):
    results = {}
    
    group_sorted = group.sort_values(by='CURRENT_LAP_TIME_MS')
    track_distance_diffs = []
    previous_dist = None
    first_reversed_at_dist = 0
    valid_rows = []  # To store valid records 
    
    for index, row in group_sorted.iterrows():
        current_dist = row['X_TRACK_DIST']
        
        # Check for backward movement
        if previous_dist and current_dist < previous_dist:
            if first_reversed_at_dist == 0:
                first_reversed_at_dist = previous_dist
            continue
        
        valid_rows.append(row)
        previous_dist = current_dist

    # DF for better processsiing 
    valid_df = pd.DataFrame(valid_rows)
    
    total_braking = valid_df['BRAKE'].sum()
    avg_speed = valid_df['SPEED_KPH'].mean()
    avg_gear = valid_df['GEAR'].mean()
    gear_changes = (valid_df['GEAR'].diff() != 0).sum()
    avg_throttle = valid_df['THROTTLE'].mean()
    avg_yaw = valid_df['YAW'].mean()
    avg_pitch = valid_df['PITCH'].mean()
    avg_roll = valid_df['ROLL'].mean()
    avg_yaw_speed_ratio = (valid_df['YAW'] / valid_df['SPEED_KPH']).mean()
    first_brake_idx = valid_df[valid_df['BRAKE'] > 0].index.min()
    
    distance_before_brake = valid_df.loc[first_brake_idx, 'X_TRACK_DIST'] if pd.notna(first_brake_idx) else 0
    
    valid_df['TRACK_DISTANCE_DIFF'] = valid_df['X_TRACK_DIST'].diff()
    distance_per_gear = valid_df.groupby('GEAR')['TRACK_DISTANCE_DIFF'].sum()

    for gear, distance in distance_per_gear.items():
        results[f'X_DISTANCE_IN_GEAR_{gear}'] = distance

    results.update({
        'X_TOTAL_BRAKING': total_braking,
        'X_AVG_SPEED': avg_speed,
        'X_AVG_GEAR': avg_gear,
        'X_GEAR_CHANGES': gear_changes,
        'X_AVG_THROTTLE': avg_throttle,
        'X_AVG_YAW': avg_yaw,
        'X_AVG_PITCH': avg_pitch,
        'X_AVG_ROLL': avg_roll,
        'X_AVG_YAW_SPEED_RATIO': avg_yaw_speed_ratio,
        'X_DISTANCE_BEFORE_BRAKE': distance_before_brake,
        'X_CAR_FIRST_REVERSED_AT_DIST': first_reversed_at_dist
    })
    
    mean_dist_from_rl = group['X_DIST_FROM_RL'].mean()
    max_dist_from_rl = group['X_DIST_FROM_RL'].max()
    min_dist_from_rl = group['X_DIST_FROM_RL'].min()
    range_dist_from_rl = max_dist_from_rl - min_dist_from_rl


    # ADD DISTANCE FROM REF LINE AGGS 
    results.update({
        'X_MEAN_DIST_FROM_RL': mean_dist_from_rl,
        'X_MAX_DIST_FROM_RL': max_dist_from_rl,
        'X_MIN_DIST_FROM_RL': min_dist_from_rl,
        'X_RANGE_DIST_FROM_RL': range_dist_from_rl,
    })
    
    return pd.Series(results)



# Group by session and lap, APPLY AGG FUNC
aggregated_df = car_df.groupby(['SESSION_IDENTIFIER', 'LAP_NUM']).apply(aggregate_features)

# Restruct..
aggregated_df = aggregated_df.reset_index()
aggregated_df['X_LAP_ID'] = aggregated_df['SESSION_IDENTIFIER'].astype(str) + "_" + aggregated_df['LAP_NUM'].astype(str)
aggregated_df.rename(columns={"level_2": "FEATURE_NAME", 0: "FEATURE_VALUE"}, inplace=True)

pivoted_df = aggregated_df.pivot(index='X_LAP_ID', columns='FEATURE_NAME', values='FEATURE_VALUE').reset_index()


pivoted_df.columns.to_list()


#------------------------------------------------------------------------------
# Checks on tot lap agg feats 
#------------------------------------------------------------------------------
tot_aggs_desc = tot_lap_df.describe()


nan_counts = tot_lap_df.isnull().sum()
cols_with_nan = nan_counts[nan_counts > 0]

#FEATURE_NAME
#TOT_LAP_AVG_YAW_SPEED_RATIO        3
#TOT_LAP_DISTANCE_IN_GEAR_-1.0    494
#TOT_LAP_DISTANCE_IN_GEAR_0.0     494
#TOT_LAP_DISTANCE_IN_GEAR_1.0     466
#TOT_LAP_DISTANCE_IN_GEAR_2.0     419
##TOT_LAP_DISTANCE_IN_GEAR_3.0     295
#TOT_LAP_DISTANCE_IN_GEAR_4.0      40
#TOT_LAP_DISTANCE_IN_GEAR_5.0      21
#TOT_LAP_DISTANCE_IN_GEAR_6.0      43
#TOT_LAP_DISTANCE_IN_GEAR_7.0      61
#TOT_LAP_DISTANCE_IN_GEAR_8.0      98
#TOT_LAP_AVG_PITCH                 2
#TOT_LAP__AVG_ROLL                  2
#TOT_LAP_AVG_YAW                   2
#dtype: int64

# nans for distance in gear are not an issue  

#---------------------------
   # Check Distance in gear 
distance_gear_cols = [col for col in tot_lap_df.columns if 'DISTANCE_IN_GEAR' in col]
tot_lap_df['TOTAL_DISTANCE_IN_GEAR'] = tot_lap_df[distance_gear_cols].sum(axis=1)

tot_lap_df['TOTAL_DISTANCE_IN_GEAR'].describe()
#count    517.000000
#mean     497.450862
#std        9.305796
#min      304.689919
#25%      497.201662
#50%      498.435340
#75%      498.703391
#max      499.937069
# Looks ok so far 


tot_lap_df['TOT_LAP_DISTANCE_IN_GEAR_1.0'].isna().sum()

car_df['GEAR'].value_counts()

sample_laps = tot_lap_df['LAP_ID'].sample(5)
for lap in sample_laps:
    lap_data = car_df[car_df['X_LAP_ID'] == lap]
    plt.plot(lap_data['CURRENT_LAP_TIME_MS'], lap_data['GEAR'])
    plt.title(f'Gear Changes Over Time for Lap {lap}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Gear')
    plt.show()
    
    
for gear in [-1, 0]:
    col_name = f'TOT_LAP_DISTANCE_IN_GEAR_{gear}'
    print(f"Gear {gear} Statistics:")
    print(f"Average Distance: {tot_lap_df[col_name].mean()}")
    print(f"Minimum Distance: {tot_lap_df[col_name].min()}")
    print(f"Maximum Distance: {tot_lap_df[col_name].max()}")
    print("-"*50)

# These all seem fine 

#---------------------------
# Impute the mean for the missing vals im the pitch yaw roll cols, 
     # since there are only a couple there will be no issue with learning the
     # test data
     
cols_to_impute = [
    'TOT_LAP_AVG_PITCH',
    'TOT_LAP_AVG_ROLL',
    'TOT_LAP_AVG_YAW',
    'TOT_LAP_AVG_YAW_SPEED_RATIO'
]

# avgs 
means = {}
for col in cols_to_impute:
    valid_values = tot_lap_df[col].replace([np.inf, -np.inf], np.nan)  
    means[col] = valid_values.mean()

# Dont do this step before TT split, else introdcuing bias form learning train set in test set.. 
#for col in cols_to_impute:
#    tot_lap_df[col] = tot_lap_df[col].replace([np.nan, np.inf, -np.inf], means[col])     
     

#---------------------------

tot_lap_df['TOT_LAP_TOTAL_BRAKING'].describe()

tot_lap_df['TOT_LAP_GEAR_CHANGES'].value_counts()

cols_to_check = [col for col in tot_lap_df.columns if 'DISTANCE_IN_GEAR' not in col and 'LAP_ID' not in col]


# Check cols with neg vals 
negative_counts = {}
for col in cols_to_check:
    negative_counts[col] = (tot_lap_df[col] < 0).sum()


negative_counts

# Looks good 


# Visualisations
import os 
save_directory = "/Users/Owenbrooks/Desktop/DATA3001/Project/[6] Segmentation:Modelling/tot_lap_agg_plots"
for col in cols_to_check:
    plt.figure(figsize=(10, 5))
    plt.hist(tot_lap_df[col], bins=50, edgecolor='k', alpha=0.7)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    file_path = os.path.join(save_directory, f"{col}.png")
    plt.savefig(file_path)
    plt.close() 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# SEGMENTATION - Method 3 ***USING THIS METHOD *****
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Determine the bin edges using QUANTILES
quantiles = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
bin_edges = car_df['X_TRACK_DIST'].quantile(quantiles).values

# Adjust the first and last bin edges for full coverage
bin_edges[0] = car_df['X_TRACK_DIST'].min()
bin_edges[-1] = car_df['X_TRACK_DIST'].max()

# Rounding bin edges to avoid precision issues
bin_edges = np.round(bin_edges, 2)  

# Remove any duppes
bin_edges = np.unique(bin_edges)

# BIN
car_df['X_TRACK_SEGMENT'] = pd.cut(car_df['X_TRACK_DIST'], bins=bin_edges, labels=range(len(bin_edges)-1), include_lowest=True)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# FLatten the dffor seg spec calcs 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Similar to tot lap aggs, some extra feats included.. 
def segment_aggregate_features(group):
    # Fetch the segment
    segment = group['X_TRACK_SEGMENT'].iloc[0]
    prefix = f"Y_S{segment}"
    
    # Same non-reverse constraint
    results = {}
    valid_rows = []
    previous_dist = None
    
    for index, row in group.sort_values(by='CURRENT_LAP_TIME_MS').iterrows():
        current_dist = row['X_TRACK_DIST']
        if previous_dist and current_dist < previous_dist:
            continue
        valid_rows.append(row)
        previous_dist = current_dist

    valid_df = pd.DataFrame(valid_rows)
    
    total_braking = valid_df['BRAKE'].sum()
    avg_speed = valid_df['SPEED_KPH'].mean()
    avg_gear = valid_df['GEAR'].mean()
    gear_changes = (valid_df['GEAR'].diff() != 0).sum()
    avg_throttle = valid_df['THROTTLE'].mean()
    avg_yaw = valid_df['YAW'].mean()
    avg_pitch = valid_df['PITCH'].mean()
    avg_roll = valid_df['ROLL'].mean()
    avg_yaw_speed_ratio = (valid_df['YAW'] / valid_df['SPEED_KPH']).mean()
    
    # Braking duration
    valid_df['SPEED_DIFF'] = valid_df['SPEED_KPH'].diff()
    valid_df['BRAKING'] = (valid_df['SPEED_DIFF'] < 0).astype(int)
    valid_df['TIME_DIFF_MS'] = valid_df['CURRENT_LAP_TIME_MS'].diff()
    valid_df['BRAKING_DURATION_MS'] = valid_df['BRAKING'] * valid_df['TIME_DIFF_MS']
    braking_duration_ms = valid_df['BRAKING_DURATION_MS'].sum()
    
    first_brake_idx = valid_df[valid_df['BRAKE'] > 0].index.min()
    distance_before_brake = valid_df.loc[first_brake_idx, 'X_TRACK_DIST'] if pd.notna(first_brake_idx) else 0
    
    # Gears
    valid_df['TRACK_DISTANCE_DIFF'] = valid_df['X_TRACK_DIST'].diff()
    distance_per_gear = valid_df.groupby('GEAR')['TRACK_DISTANCE_DIFF'].sum()
    for gear, distance in distance_per_gear.items():
        results[f'{prefix}_DISTANCE_IN_GEAR_{gear}'] = distance

    results.update({
        f'{prefix}_TOTAL_BRAKING': total_braking,
        f'{prefix}_AVG_SPEED': avg_speed,
        f'{prefix}_AVG_GEAR': avg_gear,
        f'{prefix}_GEAR_CHANGES': gear_changes,
        f'{prefix}_AVG_THROTTLE': avg_throttle,
        f'{prefix}_BRAKING_DURATION_MS': braking_duration_ms,
        f'{prefix}_AVG_YAW': avg_yaw,
        f'{prefix}_AVG_PITCH': avg_pitch,
        f'{prefix}_AVG_ROLL': avg_roll,
        f'{prefix}_AVG_YAW_SPEED_RATIO': avg_yaw_speed_ratio,
        f'{prefix}_DISTANCE_BEFORE_BRAKE': distance_before_brake
    })
    
    
    # ADD dist from RL metrics 
    mean_dist_from_rl = group['X_DIST_FROM_RL'].mean()
    max_dist_from_rl = group['X_DIST_FROM_RL'].max()
    min_dist_from_rl = group['X_DIST_FROM_RL'].min()
    range_dist_from_rl = max_dist_from_rl - min_dist_from_rl

    results.update({
        f'Y_S{prefix}_MEAN_DIST_FROM_RL': mean_dist_from_rl,
        f'Y_S{prefix}_MAX_DIST_FROM_RL': max_dist_from_rl,
        f'Y_S{prefix}_MIN_DIST_FROM_RL': min_dist_from_rl,
        f'Y_S{prefix}_RANGE_DIST_FROM_RL': range_dist_from_rl,
    })
    
    return pd.Series(results)


segmented_df = car_df.groupby(['X_LAP_ID', 'X_TRACK_SEGMENT']).apply(segment_aggregate_features).reset_index()

#-------------------
# Pivot 


segmented_pivot_df = segmented_df.pivot_table(index=['X_LAP_ID', 'X_TRACK_SEGMENT'],
                                              columns='level_2', 
                                              values=0).reset_index()

segmented_pivot_df.columns.name = None  
segmented_pivot_df.columns.to_list()
segmented_pivot_df.shape

pivoted_df.columns.to_list()

#------------------------------------------------------------------------------
# Checks 

segmented_df.columns.to_list()


#---------------------
# Suss out inf values 

inf_cols = pivoted_df.select_dtypes(include=[np.number]).columns[pivoted_df.select_dtypes(include=[np.number]).apply(lambda x: np.isinf(x).sum() > 0)]


for col in inf_cols:
    print(f"{col}: {np.isinf(pivoted_df[col]).sum()} infinite values")

#SEG5_AVG_YAW_SPEED_RATIO: 10 infinite values
#SEG6_AVG_YAW_SPEED_RATIO: 8 infinite values
#SEG7_AVG_YAW_SPEED_RATIO: 8 infinite values
#SEG8_AVG_YAW_SPEED_RATIO: 5 infinite values
#EG9_AVG_YAW_SPEED_RATIO: 3 infinite values
#SEG10_AVG_YAW_SPEED_RATIO: 1 infinite values

# Replace with nan 

pivoted_df[inf_cols] = pivoted_df[inf_cols].replace([np.inf, -np.inf], np.nan)

#---------------------
# Save plots to folder for each segment 

base_dir = "/Users/Owenbrooks/Desktop/DATA3001/Project/[6] Segmentation:Modelling/seg_spec_plots"

# Get all segment prefixes from the column names
segments = set(col.split("_")[0] for col in pivoted_df.columns if "SEG" in col)

for segment in segments:
    # Create segment spec dir 
    segment_dir = os.path.join(base_dir, segment)
    os.makedirs(segment_dir, exist_ok=True)
    
    # Columns spec to the seg. .
    segment_cols = [col for col in pivoted_df.columns if segment in col]
    
    for col in segment_cols:
        plt.figure(figsize=(10, 5))
        plt.hist(pivoted_df[col], bins=50, edgecolor='k', alpha=0.7)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        file_path = os.path.join(segment_dir, f"{col}.png")
        plt.savefig(file_path)
        plt.close()


pivoted_df.columns.to_list()



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Join flattened df with whole lap agg df (pivot_df)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

flattened_df_all = pd.merge(pivoted_df, segmented_pivot_df, on="X_LAP_ID", how="inner")


flattened_df_all.columns.to_list()

#------------------------------------------------------------------------------
# Rename cols & reorder [house cleaning ]

def rename_column(col_name):
 
    if col_name.startswith('X_'):
        return 'TOT_LAP_' + col_name[2:]

    elif col_name.startswith('Y_S') and col_name[3].isdigit():

        return 'SEG' + col_name[3] + '_' + col_name[5:]

    else:
        return col_name

new_columns = [rename_column(col) for col in flattened_df_all.columns.to_list()]
flattened_df_all.columns = new_columns

flattened_df_all.columns.to_list()

columns = ['MAX', 'MEAN', 'MIN', 'RANGE']
stages = range(10)  

rename_dict = {
    'Y_SY_S{}_{}_DIST_FROM_RL'.format(s, col): 'SEG{}_{}_DIST_FROM_RL'.format(s, col) 
    for s in stages for col in columns
}


flattened_df_all.rename(columns=rename_dict, inplace=True)
flattened_df_all.rename(columns={'TOT_LAP_LAP_ID': 'LAP_ID'}, inplace=True)

flattened_df_all.columns.to_list()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Add LAP_INVALID_IN_SEGMENT - 0 if valid lap, num seg first invalidated if invalid
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


car_df.rename(columns={'X_LAP_ID': 'LAP_ID'}, inplace=True)


# Convert X_TRACK_SEGMENT to int type and update segment numbers in car_df
car_df['X_TRACK_SEGMENT'] = car_df['X_TRACK_SEGMENT'].astype('int') + 1

# Similarly, if there's an X_TRACK_SEGMENT in flattened_df_all
if 'X_TRACK_SEGMENT' in flattened_df_all.columns:
    flattened_df_all['X_TRACK_SEGMENT'] = flattened_df_all['X_TRACK_SEGMENT'].astype('int') + 1

# Calculate the TOT_LAP_INVALID_IN_SEGMENT col
first_invalid_segment_df = car_df[car_df['X_LAP_INVALID'] == 1].groupby('LAP_ID')['X_TRACK_SEGMENT'].min().reset_index()
first_invalid_segment_df = first_invalid_segment_df.rename(columns={'X_TRACK_SEGMENT': 'TOT_LAP_INVALID_IN_SEGMENT'})

# Merge info
flattened_df_all = flattened_df_all.merge(first_invalid_segment_df, on='LAP_ID', how='left')

# Fill NaN values ( valid laps) with 0
flattened_df_all['TOT_LAP_INVALID_IN_SEGMENT'] = flattened_df_all['TOT_LAP_INVALID_IN_SEGMENT'].fillna(0).astype('int')


flattened_df_all['TOT_LAP_INVALID_IN_SEGMENT'].value_counts()

#0     2486
#3     1720
#4      530
#2      190
#5       70
#8       60
#1       40
#9       40
#7       20
#10      10


del flattened_df_all['TOT_LAP_INVALID_IN_SEGMENT_x']
del flattened_df_all['TOT_LAP_INVALID_IN_SEGMENT_y']


# Fix err seg range 0-9, change 1-10
renamed_columns = {}
for i in range(10):
    for col in flattened_df_all.columns:
        if f'SEG{i}_' in col:
            renamed_columns[col] = col.replace(f'SEG{i}_', f'SEG{i+1}_')

flattened_df_all.rename(columns=renamed_columns, inplace=True)

#------------------------------------------------------------------------------
# Save intermedioate 

output_path = "/Users/Owenbrooks/Desktop/flattened_2.csv"
flattened_df_all.to_csv(output_path, index=False)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Add 'TOT_LAP_CAR_FIRST_REVERSED_IN_SEG' - if 0, then car did not reverse 
    # Note that TOT_LAP_CAR_FIRST_REVERSED_AT_DIST == 0 means no reversal at all 
        # Should probably change this to be n/a if no reverse # TODO 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Initialize the column to indicate no reversal with 0
flattened_df_all['TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'] = 0

# Condition 
for idx, edge in enumerate(bin_edges[:-1]):
    next_edge = bin_edges[idx + 1]
    mask = (flattened_df_all['TOT_LAP_CAR_FIRST_REVERSED_AT_DIST'] > edge) & (flattened_df_all['TOT_LAP_CAR_FIRST_REVERSED_AT_DIST'] <= next_edge)
    
    # Assign segment values starting from 1 to 10
    flattened_df_all.loc[mask, 'TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'] = idx + 1


flattened_df_all['TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'].value_counts()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# CURRENT LAP TIME AT THE START AND END OF EACH SEGMENT 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


# Extracting the CURRENT_LAP_TIME_MS at the minimum X_TRACK_DISTANCE within each interval
min_times = car_df.loc[car_df.groupby(['LAP_ID', 'X_TRACK_SEGMENT'])['X_TRACK_DIST'].idxmin()][['LAP_ID', 'X_TRACK_SEGMENT', 'CURRENT_LAP_TIME_MS']]
min_times = min_times.pivot(index='LAP_ID', columns='X_TRACK_SEGMENT', values='CURRENT_LAP_TIME_MS')
min_times.columns = [f"SEG{col}_MIN_CURR_TIME" for col in min_times.columns]

# Extracting the CURRENT_LAP_TIME_MS at the maximum X_TRACK_DISTANCE within each interval
max_times = car_df.loc[car_df.groupby(['LAP_ID', 'X_TRACK_SEGMENT'])['X_TRACK_DIST'].idxmax()][['LAP_ID', 'X_TRACK_SEGMENT', 'CURRENT_LAP_TIME_MS']]
max_times = max_times.pivot(index='LAP_ID', columns='X_TRACK_SEGMENT', values='CURRENT_LAP_TIME_MS')
max_times.columns = [f"SEG{col}_MAX_CURR_TIME" for col in max_times.columns]

# Merging to flattened_df_all
flattened_df_all = flattened_df_all.merge(min_times, on='LAP_ID', how='left')
flattened_df_all = flattened_df_all.merge(max_times, on='LAP_ID', how='left')

flattened_df_all.columns.to_list()

#------------------------------------------------------------------------------
# Checks -  PROBLEM 
cols_to_check = [
 'SEG1_MIN_CURR_TIME',
 'SEG2_MIN_CURR_TIME',
 'SEG3_MIN_CURR_TIME',
 'SEG4_MIN_CURR_TIME',
 'SEG5_MIN_CURR_TIME',
 'SEG6_MIN_CURR_TIME',
 'SEG7_MIN_CURR_TIME',
 'SEG8_MIN_CURR_TIME',
 'SEG9_MIN_CURR_TIME',
 'SEG10_MIN_CURR_TIME',
 'SEG1_MAX_CURR_TIME',
 'SEG2_MAX_CURR_TIME',
 'SEG3_MAX_CURR_TIME',
 'SEG4_MAX_CURR_TIME',
 'SEG5_MAX_CURR_TIME',
 'SEG6_MAX_CURR_TIME',
 'SEG7_MAX_CURR_TIME',
 'SEG8_MAX_CURR_TIME',
 'SEG9_MAX_CURR_TIME',
 'SEG10_MAX_CURR_TIME'
]

na_counts = flattened_df_all[cols_to_check].isna().sum()
print(na_counts)

#SEG1_MIN_CURR_TIME     16
#SEG2_MIN_CURR_TIME      7

#SEG1_MAX_CURR_TIME     16
#SEG2_MAX_CURR_TIME      7

# Investigate why there are missing values in these segments * TO DO ****
#------------------------------------------------------------------------------
# CHecks - PROBLEM 

#  Consistency of Segment Times
inconsistencies = []

for i in range(1, 11):
    # Check max > min for each segment
    inconsistent_max_min = flattened_df_all[flattened_df_all[f"SEG{i}_MAX_CURR_TIME"] <= flattened_df_all[f"SEG{i}_MIN_CURR_TIME"]]
    if not inconsistent_max_min.empty:
        inconsistencies.append(f"Inconsistent max/min times in SEG{i}")

    # Check min of the following seg > max of prev seg
    if i < 10:
        inconsistent_min_next = flattened_df_all[flattened_df_all[f"SEG{i+1}_MIN_CURR_TIME"] <= flattened_df_all[f"SEG{i}_MAX_CURR_TIME"]]
        if not inconsistent_min_next.empty:
            inconsistencies.append(f"Inconsistent min time in SEG{i+1} compared to max time in SEG{i}")

if inconsistencies:
    for msg in inconsistencies:
        print("\n" + msg)
else:
    print("\nAll segment times are consistent.")

#Inconsistent min time in SEG2 compared to max time in SEG1

#Inconsistent min time in SEG3 compared to max time in SEG2

#Inconsistent min time in SEG4 compared to max time in SEG3

#Inconsistent max/min times in SEG5

#Inconsistent min time in SEG7 compared to max time in SEG6

#Inconsistent max/min times in SEG9

#Inconsistent max/min times in SEG10

# Possibly due to cars reversing? ****** NEEDS FURTHER INVESTIGATION ** TODO 


#  Consistency of Segment Times [2] more detail 
inconsistencies = {}
total_rows_with_inconsistencies = set()

for i in range(1, 11):
    # Check max is greater than min for each segment
    inconsistent_max_min = flattened_df_all[flattened_df_all[f"SEG{i}_MAX_CURR_TIME"] <= flattened_df_all[f"SEG{i}_MIN_CURR_TIME"]]
    if not inconsistent_max_min.empty:
        inconsistencies[f"Inconsistent max/min times in SEG{i}"] = len(inconsistent_max_min)
        total_rows_with_inconsistencies.update(inconsistent_max_min.index.tolist())

    # Check min of the following segment is greater than the max of the previous segment
    if i < 10:
        inconsistent_min_next = flattened_df_all[flattened_df_all[f"SEG{i+1}_MIN_CURR_TIME"] <= flattened_df_all[f"SEG{i}_MAX_CURR_TIME"]]
        if not inconsistent_min_next.empty:
            inconsistencies[f"Inconsistent min time in SEG{i+1} compared to max time in SEG{i}"] = len(inconsistent_min_next)
            total_rows_with_inconsistencies.update(inconsistent_min_next.index.tolist())

print(f"\nTotal rows with inconsistencies: {len(total_rows_with_inconsistencies)}")

# Print inconsistencies count for each type of inconsistency
for msg, count in inconsistencies.items():
    print(f"{msg}: {count}")

# Identify which segments are most commonly involved in inconsistencies
from collections import defaultdict
segment_counts = defaultdict(int)

for key in inconsistencies.keys():
    segments = [int(s[-1]) for s in key.split() if s.startswith("SEG")]
    for s in segments:
        segment_counts[s] += inconsistencies[key]

print("\nInconsistencies by segment:")
for seg, count in segment_counts.items():
    print(f"SEG{seg}: {count}")


#Total rows with inconsistencies: 50
#Inconsistent min time in SEG2 compared to max time in SEG1: 10
#Inconsistent min time in SEG3 compared to max time in SEG2: 20
#Inconsistent min time in SEG4 compared to max time in SEG3: 20
#Inconsistent max/min times in SEG5: 10
#Inconsistent min time in SEG7 compared to max time in SEG6: 10
#Inconsistent max/min times in SEG9: 20
#Inconsistent max/min times in SEG10: 10

#Inconsistencies by segment:
#SEG2: 30
#SEG1: 10
#SEG3: 40
#SEG4: 20
#SEG5: 10
#SEG7: 10
#SEG6: 10
#SEG9: 20
#SEG0: 10

# more in seg3 and 2, likely that reversing is at least part of the problem 

# Check against reversing laps 


# Extract LAP_IDs of rows where drivers reversed
laps_where_car_reversed = flattened_df_all[flattened_df_all['TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'] != 0]['LAP_ID'].unique()

# Compare them against LAP_IDs of rows with inconsistencies
laps_with_inconsistencies = flattened_df_all.iloc[list(total_rows_with_inconsistencies)]['LAP_ID'].unique()

laps_responsible_for_inconsistencies = set(laps_where_car_reversed).intersection(laps_with_inconsistencies)

print(f"Number of laps where drivers reversed that are responsible for inconsistencies: {len(laps_responsible_for_inconsistencies)}")

#Number of laps where drivers reversed that are responsible for inconsistencies: 5


     # That doesn't seem to be the problem ..
     
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# RATE OF CHANGE IN LAP TIME PER SEGMENT 
    # The rate of change is the difference in lap time divided by 
    # the distance of the segment interval
    
    # TIME_DELTA is how long the car took to complete the segment 
    
    # This will need to be re run when MIN/MAX Segment time is fixed # TODO 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


segment_intervals = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]

# Calculate TIME_DELTA and RATE_OF_CHANGE for each segment
for segment in range(1, 11): 
    print(segment)
    min_time_col = f"SEG{segment}_MIN_CURR_TIME"
    max_time_col = f"SEG{segment}_MAX_CURR_TIME"
    delta_time_col = f"SEG{segment}_TIME_DELTA"
    rate_col = f"SEG{segment}_RATE_OF_CHANGE"

    # Calculate diff in CURRENT_LAP_TIME_MS between max and min times
    flattened_df_all[delta_time_col] = flattened_df_all[max_time_col] - flattened_df_all[min_time_col]

    # Dist for the current segment interval
    segment_length = segment_intervals[segment-1][1] - segment_intervals[segment-1][0]  # Adjusted index to start from 0

    # Calc ROC 
    flattened_df_all[rate_col] = flattened_df_all[delta_time_col] / segment_length


flattened_df_all.columns.to_list()

#------------------------------------------------------------------------------
# Add resp suffix to time cols for clarity in modelling [ house cleaning ]


all_columns = flattened_df_all.columns

corrected_columns = [col.replace(" (RESP)_(RESP)", "_(RESP)") for col in all_columns]

# Update 
flattened_df_all.columns = corrected_columns

flattened_df_all.columns.to_list()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Add Max & Min Speed in each segment 
    # NOTE * THIS DIDNT WORK - COLS DELETED AND FIXED 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
car_df.columns.to_list()


grouped = car_df.groupby('X_TRACK_SEGMENT')['SPEED_KPH'].agg([('MAX_SPEED', 'max'), ('MIN_SPEED', 'min')]).reset_index()

for i, row in grouped.iterrows():
    segment = row['X_TRACK_SEGMENT']
    flattened_df_all[f'SEG{segment}_MAX_SPEED'] = row['MAX_SPEED']
    flattened_df_all[f'SEG{segment}_MIN_SPEED'] = row['MIN_SPEED']


def find_max_speed_dist(group):
    max_idx = group['SPEED_KPH'].idxmax()
    return group.loc[max_idx, 'X_TRACK_DIST']

def find_min_speed_dist(group):
    min_idx = group['SPEED_KPH'].idxmin()
    return group.loc[min_idx, 'X_TRACK_DIST']

grouped = car_df.groupby('X_TRACK_SEGMENT').apply(lambda group: pd.Series({
    'MAX_SPEED': group['SPEED_KPH'].max(),
    'MIN_SPEED': group['SPEED_KPH'].min(),
    'MAX_SPEED_CURR_DIST': find_max_speed_dist(group),
    'MIN_SPEED_CURR_DIST': find_min_speed_dist(group)
})).reset_index()

# 2. Join computed X_TRACK_DIST values to flattened_df_all

        # TODO THERE MAY BE A PROBLEM WITH OVERWRITING VALUES IN THIS CODE SNIPPET
        #   CHECK THIS 
for i, row in grouped.iterrows():
    segment = row['X_TRACK_SEGMENT']
    flattened_df_all[f'SEG{segment}_MAX_SPEED_CURR_DIST'] = row['MAX_SPEED_CURR_DIST']
    flattened_df_all[f'SEG{segment}_MIN_SPEED_CURR_DIST'] = row['MIN_SPEED_CURR_DIST']



cols_to_rename = {}

for i in range(1, 11):
    cols_to_rename[f'SEG{i}.0_MAX_SPEED'] = f'SEG{i}_MAX_SPEED'
    cols_to_rename[f'SEG{i}.0_MIN_SPEED'] = f'SEG{i}_MIN_SPEED'
    cols_to_rename[f'SEG{i}.0_MAX_SPEED_CURR_DIST'] = f'SEG{i}_MAX_SPEED_CURR_DIST'
    cols_to_rename[f'SEG{i}.0_MIN_SPEED_CURR_DIST'] = f'SEG{i}_MIN_SPEED_CURR_DIST'

flattened_df_all = flattened_df_all.rename(columns=cols_to_rename)


flattened_df_all.columns.to_list()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Check point 2 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#output_path = "/Users/Owenbrooks/Desktop/flattened_3.csv"
#flattened_df_all.to_csv(output_path, index=False)

input_path = "/Users/Owenbrooks/Desktop/flattened_3.csv"
flattened_df_all = pd.read_csv(input_path)

flattened_df_all['LAP_ID'].nunique()

flattened_df_all.shape

flattened_df_all.columns.to_list()



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Adding more Agg Features (for each seg)
    # For ENGINE_RPM & STEERING
        # TO DO ********
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

car_df.columns.to_list()


car_df['ENGINE_RPM'].corr(car_df['SPEED_KPH'])

#0.7620757594348201 so correlated but still distinct. probably worth adding 


missing_data = car_df.groupby('X_TRACK_SEGMENT').apply(lambda g: 'ENGINE_RPM' not in g.columns)
print(missing_data)
nan_data = car_df.groupby('X_TRACK_SEGMENT')['ENGINE_RPM'].apply(lambda g: g.isna().all())
print(nan_data)



car_df['ENGINE_RPM'].isna().sum()


def get_distance_for_max_rpm(series):
    idx = series.idxmax()
    return car_df.loc[idx, 'X_TRACK_DIST']

def get_distance_for_min_rpm(series):
    idx = series.idxmin()
    return car_df.loc[idx, 'X_TRACK_DIST']

def get_distance_for_max_steering(series):
    idx = series.idxmax()
    return car_df.loc[idx, 'X_TRACK_DIST']

def get_distance_for_min_steering(series):
    idx = series.idxmin()
    return car_df.loc[idx, 'X_TRACK_DIST']



def compute_metrics(group):
    return pd.Series({
        'ENGINE_RPM_MEAN': group['ENGINE_RPM'].mean(),
        'ENGINE_RPM_STD': group['ENGINE_RPM'].std(),
        'ENGINE_RPM_MAX': group['ENGINE_RPM'].max(),
        'ENGINE_RPM_MIN': group['ENGINE_RPM'].min(),
        'MAX_ENGINE_RPM_CURR_DISTANCE': get_distance_for_max_rpm(group['ENGINE_RPM']),
        'MIN_ENGINE_RPM_CURR_DISTANCE': get_distance_for_min_rpm(group['ENGINE_RPM']),
        'STEERING_MEAN': group['STEERING'].mean(),
        'STEERING_STD': group['STEERING'].std(),
        'STEERING_MAX': group['STEERING'].max(),
        'STEERING_MIN': group['STEERING'].min(),
        'MAX_STEERING_CURR_DISTANCE': get_distance_for_max_steering(group['STEERING']),
        'MIN_STEERING_CURR_DISTANCE': get_distance_for_min_steering(group['STEERING'])
    })

# Apply func 
grouped = car_df.groupby(['LAP_ID', 'X_TRACK_SEGMENT']).apply(compute_metrics).reset_index()

for _, row in grouped.iterrows():
    segment = row['X_TRACK_SEGMENT']
    lap_id = row['LAP_ID']
    for col in grouped.columns.difference(['LAP_ID', 'X_TRACK_SEGMENT']):
        flattened_df_all.loc[flattened_df_all['LAP_ID'] == lap_id, f'SEG{segment}_{col}'] = row[col]



#------------------------------------------------------------------------------
# Checks 

flattened_df_all.columns.to_list()
flattened_df_all.shape

new_columns = [f'SEG{segment}_{col}' for segment in grouped['X_TRACK_SEGMENT'].unique() 
               for col in grouped.columns.difference(['LAP_ID', 'X_TRACK_SEGMENT'])]
missing_values = flattened_df_all[new_columns].isnull().sum()
print("Missing values for each new metric:")
print(missing_values[missing_values > 0])



# SAME 16 and 7 issues here TODO 


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Check point 3
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#output_path = "/Users/Owenbrooks/Desktop/flattened_4_checksreq.csv"
#flattened_df_all.to_csv(output_path, index=False)

input_path = "/Users/Owenbrooks/Desktop/flattened_4_checksreq.csv"
flattened_df_all = pd.read_csv(input_path)

flattened_df_all.shape

flattened_df_all['LAP_ID'].nunique()

     
#------------------------------------------------------------------------------
# PROBLEM WITH NUM ROWS - SHOULD BE 517 

# Diagnose: 
    
duplicated_lap_ids = flattened_df_all[flattened_df_all['LAP_ID'].duplicated(keep=False)]
print("Number of duplicated LAP_ID entries:", duplicated_lap_ids.shape[0])
#print("Unique LAP_IDs that are duplicated:", duplicated_lap_ids['LAP_ID'].unique())    
    

fully_duplicated_rows = flattened_df_all[flattened_df_all.duplicated(keep=False)]
print("Number of fully duplicated rows:", fully_duplicated_rows.shape[0])


unique_lap_ids = flattened_df_all['LAP_ID'].value_counts()[flattened_df_all['LAP_ID'].value_counts() == 1].index.tolist()

problematic_df = flattened_df_all[~flattened_df_all['LAP_ID'].isin(unique_lap_ids)]

unique_counts = problematic_df.nunique(dropna=False)
print(unique_counts)


suspicious_columns = unique_counts[unique_counts > 1].index.tolist()
print("Number of suspicious columns:", len(suspicious_columns))



lap_id_counts = flattened_df_all['LAP_ID'].value_counts()
print(lap_id_counts.describe())


sample_duplicated_lap_ids = lap_id_counts.head(5).index
for lap_id in sample_duplicated_lap_ids:
    print("\nDisplaying rows for LAP_ID:", lap_id)
    display(flattened_df_all[flattened_df_all['LAP_ID'] == lap_id])



most_common_differing_columns = []
for lap_id in lap_id_counts.index:
    differing_columns = (flattened_df_all[flattened_df_all['LAP_ID'] == lap_id].nunique() > 1).index.tolist()
    most_common_differing_columns.extend(differing_columns)

column_counts = pd.Series(most_common_differing_columns).value_counts()
top_diff_columns = column_counts.head(20).index.tolist()

print("Top columns with differences:", top_diff_columns)



unique_rows_per_lap_id = flattened_df_all.groupby('LAP_ID').apply(lambda x: x.drop('LAP_ID', axis=1).drop_duplicates().shape[0])
print(unique_rows_per_lap_id.value_counts())


flattened_df_all_unique = flattened_df_all.drop_duplicates(subset='LAP_ID', keep='first')
print(flattened_df_all_unique.shape)


#------------------------------------------------------------------------------
# CHECKS FOR Add Max & Min Speed in each segment 
#------------------------------------------------------------------------------

#--------------------
# Diagnose problem 


segment_cols = [f'SEG{i}_{metric}' for i in range(1, 11) for metric in ['MAX_SPEED', 'MIN_SPEED', 'MAX_SPEED_CURR_DIST', 'MIN_SPEED_CURR_DIST']]


segment_data = flattened_df_all_unique[segment_cols]

missing_data = segment_data.isnull().sum()


# 
stats = segment_data.describe()


# Check for any column having a single unique value 
single_val_cols = segment_data.columns[segment_data.nunique() == 1].tolist()
print("\nColumns with single unique value:")
print(single_val_cols)

     # Found issue,, simple code err. 
#--------------------
# Fix 


flattened_df_all_unique.drop(columns=segment_cols, inplace=True)

# 2. Recalc max and min speeds and distances for each seg from car_df

def compute_speed_stats(group):
    max_idx = group['SPEED_KPH'].idxmax()
    min_idx = group['SPEED_KPH'].idxmin()
    
    return pd.Series({
        'MAX_SPEED': group.loc[max_idx, 'SPEED_KPH'],
        'MIN_SPEED': group.loc[min_idx, 'SPEED_KPH'],
        'MAX_SPEED_CURR_DIST': group.loc[max_idx, 'X_TRACK_DIST'],
        'MIN_SPEED_CURR_DIST': group.loc[min_idx, 'X_TRACK_DIST']
    })

grouped = car_df.groupby('X_TRACK_SEGMENT').apply(compute_speed_stats).reset_index()

#Merge the computed values onto flattened_df_all_unique based on X_TRACK_SEGMENT

for i, row in grouped.iterrows():
    segment = row['X_TRACK_SEGMENT']
    
    # Use LAP_ID to correctly merge the values
    mask = flattened_df_all_unique['LAP_ID'] == segment
    flattened_df_all_unique.loc[mask, f'SEG{segment}_MAX_SPEED'] = row['MAX_SPEED']
    flattened_df_all_unique.loc[mask, f'SEG{segment}_MIN_SPEED'] = row['MIN_SPEED']
    flattened_df_all_unique.loc[mask, f'SEG{segment}_MAX_SPEED_CURR_DIST'] = row['MAX_SPEED_CURR_DIST']
    flattened_df_all_unique.loc[mask, f'SEG{segment}_MIN_SPEED_CURR_DIST'] = row['MIN_SPEED_CURR_DIST']


flattened_df_all_unique.shape
flattened_df_all_unique.columns.to_list()


# Create a mapping from old cols name -> new 
rename_mapping = {}
for col in flattened_df_all_unique.columns:
    if '.0_' in col:
        rename_mapping[col] = col.replace('.0_', '_')


flattened_df_all_unique = flattened_df_all_unique.rename(columns=rename_mapping)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Check point 4
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

output_path = "/Users/Owenbrooks/Desktop/flattened_5_checksreq.csv"
flattened_df_all_unique.to_csv(output_path, index=False)

input_path = "/Users/Owenbrooks/Desktop/flattened_5_checksreq.csv"
flattened_df_all_unique = pd.read_csv(input_path)


flattened_df_all_unique.shape

#------------------------------------------------------------------------------
# Reorder [ house cleaning ]
pd.set_option('display.max_columns', None)

# Extracting column names
cols = flattened_df_all_unique.columns.tolist()

# Sorting columns based on the criteria

# 1. LAP_ID
lap_id_col = [col for col in cols if col == "LAP_ID"]

# 2. Total lap metrics
tot_cols = [col for col in cols if col.startswith("TOT_")]

# 3. Segmented lap aggs
seg_cols = sorted([col for col in cols if col.startswith("SEG")], 
                  key=lambda x: (int(x.split('_')[0][3:]), x.split('_')[1]))

# 4. Time metrics (response variables)
resp_cols = [col for col in cols if "(RESP)" in col]

# Combining all these lists to get the new order
new_order = lap_id_col + tot_cols + seg_cols + resp_cols


flattened_df_all_unique = flattened_df_all_unique[new_order]
flattened_df_all_unique.columns.tolist()
flattened_df_all_unique.set_index('LAP_ID', inplace=True)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Check point 5
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#output_path = "/Users/Owenbrooks/Desktop/flattened_car_df_checksreq.csv"
#flattened_df_all_unique.to_csv(output_path, index=False)

input_path = "/Users/Owenbrooks/Desktop/flattened_car_df_checksreq.csv"
flattened_df_all_unique = pd.read_csv(input_path)


flattened_df_all_unique.shape


#flattened_df_all_unique['TOT_LAP_TRACK_SEGMENT'].value_counts()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# IDentify critical segments for univariate analysis 



# 1. Identify most time gained or lost segments
avg_time_deltas = []
avg_rate_of_changes = []

for i in range(1, 11):
    avg_time_deltas.append(flattened_df_all_unique[f'SEG{i}_TIME_DELTA_(RESP).1'].mean())
    avg_rate_of_changes.append(flattened_df_all_unique[f'SEG{i}_RATE_OF_CHANGE_(RESP).1'].mean())



# 2. Identify segments with most laps invalidated
laps_invalidated_counts = flattened_df_all_unique['TOT_LAP_INVALID_IN_SEGMENT'].value_counts().sort_index()
laps_invalidated = [laps_invalidated_counts.get(i, 0) for i in range(1, 11)]

# 3. Identify segments where cars most frequently first reversed
cars_reversed_counts = flattened_df_all_unique['TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'].value_counts().sort_index()
cars_reversed = [cars_reversed_counts.get(i, 0) for i in range(1, 11)]

# STORE res 
segment_data = pd.DataFrame({
    'Segment': list(range(1, 11)),
    'Avg_Time_Delta': avg_time_deltas,
    'Avg_Rate_of_Change': avg_rate_of_changes,
    'Laps_Invalidated': laps_invalidated,
    'Cars_Reversed': cars_reversed
})


#------------------------------------------------------------------------------
# TODO FIX BELOW for AVG VARIANCE of TIME vars 
# Initialize lists to store variances for each segment
var_time_deltas = []
var_rate_of_changes = []

# Iterate thru seg & calc vars
for i in range(1, 11):
    segment_time_deltas = flattened_df_all_unique[f'SEG{i}_TIME_DELTA_(RESP)'].tolist()
    segment_rate_of_changes = flattened_df_all_unique[f'SEG{i}_RATE_OF_CHANGE_(RESP)'].tolist()

    var_time_deltas.append(np.var(segment_time_deltas))
    var_rate_of_changes.append(np.var(segment_rate_of_changes))


segment_data['Avg_Variance_Time_Delta'] = var_time_deltas
segment_data['Avg_Variance_Rate_of_Change'] = var_rate_of_changes

segment_data.to_csv("/Users/Owenbrooks/Desktop/segment_basic_summary_analysis.csv", index=False)



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Run checks [genera;l]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


flattened_df.shape
flattened_df.columns.tolist()
#-------------------------------------
# Check for infs 

inf_counts = (flattened_df == float('inf')).sum()

for col, count in inf_counts.items():
    if count > 0:
        print(f"{col}: {count} inf values")

# No infs

#-------------------------------------
# Set nas in distance in gear to 0 

cols_to_fill = [col for col in flattened_df.columns if 'TOT_LAP_DISTANCE_IN_GEAR' in col]
flattened_df[cols_to_fill] = flattened_df[cols_to_fill].fillna(0)


#-------------------------------------
# Checking missing values 
na_counts = flattened_df.isna().sum()
cols_with_na = na_counts[na_counts > 0]

print(cols_with_na)

cols_with_na_g10 = na_counts[na_counts > 10]
print(cols_with_na_g10)
# TO DO inspect further for nas 


#------------------------------------------------------------------------------
# Check point
#------------------------------------------------------------------------------


output_path = "/Users/Owenbrooks/Desktop/flattened_checksreq3.csv"
flattened_df.to_csv(output_path, index=False)

flat_check = pd.read_csv(output_path)
flat_check.shape

flat_check.columns.to_list()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# MODELLING: FIrst run, basic 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Read Files [New vers]
#------------------------------------------------------------------------------


path =  "/Users/Owenbrooks/Desktop/car_df_invalid_s.csv"
car_df = pd.read_csv(path)

f1sim_left = pd.read_csv('/Users/Owenbrooks/Desktop/DATA3001/Project/Data/Other Data/f1sim-ref-left.csv')
f1sim_right = pd.read_csv('/Users/Owenbrooks/Desktop/DATA3001/Project/Data/Other Data/f1sim-ref-right.csv')


left_boundary = f1sim_left
right_boundary = f1sim_right

f1sim_line = pd.read_csv('/Users/Owenbrooks/Desktop/DATA3001/Project/Data/Other Data/f1sim-ref-line.csv')
f1sim_turns = pd.read_csv('/Users/Owenbrooks/Desktop/DATA3001/Project/Data/Other Data/f1sim-ref-turns.csv')

ref_line = pd.read_csv('/Users/Owenbrooks/Desktop/DATA3001/Project/Data/Other Data/f1sim-ref-line.csv')


input_path = "/Users/Owenbrooks/Desktop/flattened_checksreq3.csv"
flattened_df = pd.read_csv(input_path)


flattened_df.columns.to_list()
flattened_df.shape
#------------------------------------------------------------------------------
# Fix .1 on resp var names 

flattened_df.columns = [col[:-2] if col.endswith('.1') else col for col in flattened_df.columns]
flattened_df.columns = [col[:-6] if col.endswith('\u2028') else col for col in flattened_df.columns]

# Fix dup cols 

flattened_df = flattened_df.loc[:, ~flattened_df.columns.duplicated()]


car_df.columns.to_list()
car_df['LAP_ID'].nunique()

lap_counts = car_df.groupby('LAP_ID').size()

lap_counts.mean()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# MODEL 1 -> Basic XSBoost Model - VERSION 1 lap time response
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Notes
    #   All features included in X (excluding time features)
    #   Response y = max time for lap (tot lap time)
    #   No cross val
    #   No hyper param tuning 
    #   No filtering out slowest laps 
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


y = flattened_df['SEG10_MAX_CURR_TIME_(RESP)']
X = flattened_df.loc[:, ~flattened_df.columns.str.contains('RESP')]
y.shape
y.columns
y = flattened_df['SEG10_MAX_CURR_TIME_(RESP)'].iloc[:, 0]

# TT Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set params 
param = {
    'max_depth': 10,  
    'eta': 0.01,  # Dec LR for more robust learning
    'subsample': 0.8,  # subsampling for regulairisation
    'colsample_bytree': 0.8,  #  col sampling for regularisation
    'objective': 'reg:squarederror',  
    'eval_metric': 'rmse'
}

num_round = 1000 

# Train 
bst = xgb.train(param, dtrain, num_round, early_stopping_rounds=50, evals=[(dtest, "Test")], verbose_eval=100)

    # Roll 1 
#[0]	Test-rmse:17762.34832
#[100]	Test-rmse:7737.19225
#[200]	Test-rmse:4224.63508
#[300]	Test-rmse:3294.22353
#[400]	Test-rmse:3155.63216
#[500]	Test-rmse:3140.90207
#[600]	Test-rmse:3117.35400
#[700]	Test-rmse:3086.49662
#[800]	Test-rmse:3073.33107
#[900]	Test-rmse:3064.04637
#[999]	Test-rmse:3060.00827

# preds
preds = bst.predict(dtest)


#------------------------------------------------------------------------------
# Evaluation Metrics Model 1 
#------------------------------------------------------------------------------

#  RMSE
rmse = np.sqrt(mean_squared_error(y_test, preds))
#RMSE: 3060.0082597055675

#Note std dev y = 7274.034164



#  top 20 most important features
plt.figure(figsize=(16, 6))
xgb.plot_importance(bst, max_num_features=20, title='Top 20 Feature Importances')
plt.show()



# Shap
shap_values = explainer.shap_values(X)
#shap.summary_plot(shap_values, X, max_display=15, plot_type = 'dot')

shap.summary_plot(shap_values[0], X, plot_type='dot')


# Plot y vs y pred 
X.shape
y.shape


residuals = y_test - preds
plt.scatter(preds, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot Model 1')


plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red') # Line for perfect predictions
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Model 1')
plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Model 2 -> Version 1 - Basic XSBoost Model - SAFETY 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Notes 
    #   All features included in X
        # Note removed distance from Ref line (obv)
    #   Response y = invalid lap (BINARY MODEL, either valid or invalid)
            # NOte a multiclass classification model will also be useful later 
    #   No cross val
    #   No hyper param tuning 
    #   No filtering out slowest laps 
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
flattened_df.columns.to_list()


#  1 for invalid laps, 0 for valid laps
y = (flattened_df['TOT_LAP_INVALID_IN_SEGMENT'] > 0).astype(int) 

Too many grp
X = X.drop(columns=['TOT_LAP_INVALID_IN_SEGMENT'])
y.value_counts()


# TTSplit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Update parameters for classification
param = {
    'max_depth': 10,
    'eta': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',   
    'eval_metric': 'logloss'          
}

# Train
num_round = 1000
bst = xgb.train(param, dtrain, num_round, early_stopping_rounds=50, evals=[(dtest, "Test")], verbose_eval=100)


    # Roll 1 
"""
[0]	Test-logloss:0.68734
[100]	Test-logloss:0.40246
[200]	Test-logloss:0.31704
[300]	Test-logloss:0.29792
[392]	Test-logloss:0.29732
"""

# Predict
preds_prob = bst.predict(dtest)
preds_class = (preds_prob > 0.5).astype(int) 


#------------------------------------------------------------------------------
# Evaluation Metrics Model 2 
#------------------------------------------------------------------------------


# Top 20 Feat importance 
plt.figure(figsize=(16, 6))
xgb.plot_importance(bst, max_num_features=20, title='Top 20 Feature Importances Model 2')
plt.show()

# Shap 
explainer2 = shap.TreeExplainer(bst)
shap_values2 = explainer2.shap_values(X_test) 
shap.summary_plot(shap_values2, X_test, plot_type='dot', title='SHAP Values Plot Model 2')


# AUC 

fpr, tpr, thresholds = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve Model 2')
plt.legend(loc="lower right")
plt.show()


# Classification rep

print(classification_report(y_test, preds_class))

#              precision    recall  f1-score   support
#
#           0       0.90      0.88      0.89        51
#           1       0.89      0.91      0.90        53

#    accuracy                           0.89       104
#   macro avg       0.89      0.89      0.89       104
#weighted avg       0.89      0.89      0.89       104



# Confusion matrixx

print(confusion_matrix(y_test, preds_class))

#[[45  6]
# [ 5 48]]

# Not bad for a first run 


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# MODEL 3 -> Basic XSBoost Model - VERSION 3 lap time response
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Notes
    #   Only SEG features trained on (no tot lap aggs)
    #   Response y = max time for lap (tot lap time)
    #   No cross val
    #   No hyper param tuning 
    #   No filtering out slowest laps 
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


y = flattened_df['SEG10_MAX_CURR_TIME_(RESP)']
X = flattened_df.loc[:, ~flattened_df.columns.str.contains('RESP')]
X = X.loc[:, ~X.columns.str.contains('TOT_LAP')]

X.columns.to_list()



# TT Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set params 
param = {
    'max_depth': 10,  
    'eta': 0.01,  # Dec LR for more robust learning
    'subsample': 0.8,  # subsampling for regularization
    'colsample_bytree': 0.8,  #  col sampling for regularisation
    'objective': 'reg:squarederror',  
    'eval_metric': 'rmse'
}

num_round = 1000 

# Train 
bst = xgb.train(param, dtrain, num_round, early_stopping_rounds=50, evals=[(dtest, "Test")], verbose_eval=100)

    # Roll 1 
"""
[0]	Test-rmse:8254.48128
[100]	Test-rmse:5761.74023
[200]	Test-rmse:5141.26327
[300]	Test-rmse:4959.64374
[400]	Test-rmse:4847.28429
[500]	Test-rmse:4789.12800
[600]	Test-rmse:4768.56343
[700]	Test-rmse:4764.50051
[719]	Test-rmse:4761.42094
"""


# preds
preds = bst.predict(dtest)


#------------------------------------------------------------------------------
# Evaluation Metrics Model 1 
#------------------------------------------------------------------------------

#  RMSE
rmse = np.sqrt(mean_squared_error(y_test, preds))
#RMSE: 4761.3

# Note ver1 with tot lap aggs was 3060.0082597055675, expected 

#Note std dev y = 7274.034164


#  top 20 most important features
plt.figure(figsize=(16, 6))
xgb.plot_importance(bst, max_num_features=20, title='Top 20 Feature Importances')
plt.show()



# Shap
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, max_display=15)


# Plot y vs y pred 
X.shape
y.shape


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# MODEL 4 -> CONSTRAINT OPTIMISATION [ Ver 1 ]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Notes

#Objective:
#Minimize lap time whilst maintaining a valid lap 

#Constraints:


#Subject to:
 #   TOT_LAP_INVALID_IN_SEGMENT = 0
  #  TOT_LAP_CAR_FIRST_REVERSED_AT_DIST = 0 


# 1. Reg Model for lap time prediction (more interpretable, version 1)
# 2. Classification mod for valid lap 
# 3. Global optimisation (bayesian)
   
import cvxpy as cp

y_time = flattened_df['SEG10_MAX_CURR_TIME_(RESP)']
y_valid = ((flattened_df['TOT_LAP_INVALID_IN_SEGMENT'] == 0) & 
           (flattened_df['TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'] == 0)).astype(int)



X = flattened_df.loc[:, ~flattened_df.columns.str.contains('RESP')]
X = X.loc[:, ~X.columns.str.contains('TOT_LAP_INVALID_IN_SEGMENT')]
X = X.loc[:, ~X.columns.str.contains('TOT_LAP_CAR_FIRST_REVERSED_IN_SEG')]
X.columns.to_list()


# Train 1. 
regressor = xgb.XGBRegressor()
regressor.fit(X, y_time)

# Train 2. 
classifier = xgb.XGBClassifier()
classifier.fit(X, y_valid)


#------------------------------------------------------------------------------
# Opt 


from bayes_opt import BayesianOptimization

def objective_function(**kwargs):
    metrics_values = list(kwargs.values())
    predicted_time = regressor.predict([metrics_values])[0]
    valid_prob = classifier.predict_proba([metrics_values])[0][1]  #  Prob valid lap 
    
    # Impose a penalty if valid_prob is below the thresh
    if valid_prob < 0.95:
        return predicted_time + 1000 # TODO make a more intelligent penalty 
    return predicted_time

#-------
# Define bounds 
    # lets take the poss range of vals from the 20 fastest laps + some small threshold 
    
# Nte FIX THIS needs to be 20 smallest times given valid 
top_20 = flattened_df.nsmallest(20, 'SEG10_MAX_CURR_TIME_(RESP)')

# Extract bounds 
pbounds = {}
for column in X.columns:  
    min_val = top_20[column].min()
    max_val = top_20[column].max()
    
    # Add thresh 
    exp_min = min_val - 0.05 * (max_val - min_val)
    exp_max = max_val + 0.05 * (max_val - min_val)
    
    pbounds[column] = (exp_min, exp_max)
    
    

optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    random_state=42,
)
optimizer.maximize(init_points=10, n_iter=50)

print(optimizer.max)

#------------------------------------------------------------------------------
# Plots 
best_params = optimizer.max['params']
best_target = optimizer.max['target']

def plot_convergence(optimizer):
   
    x = list(range(1, len(optimizer.space)+1))
    y = [res['target'] for res in optimizer.res]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, '-o')
    plt.title('Convergence Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Target Value')
    plt.show()

def plot_hyperparameters(optimizer, param_name):
  
    x = list(range(1, len(optimizer.space)+1))
    y = [res['params'][param_name] for res in optimizer.res]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, '-o')
    plt.title(f'{param_name} Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel(param_name)
    plt.show()


plot_convergence(optimizer)
for param in pbounds.keys():
    plot_hyperparameters(optimizer, param)

# No Convergence on any of the vars 

#------------------------------------------------------------------------------
# MODEL 5 -> CONSTRAINT OPTIMISATION [ Ver 2 ]
#------------------------------------------------------------------------------

#  Notes 
     # Drop all tot lap aggs 
     # More iterations 


y_time = flattened_df['SEG10_MAX_CURR_TIME_(RESP)']
y_valid = ((flattened_df['TOT_LAP_INVALID_IN_SEGMENT'] == 0) & 
           (flattened_df['TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'] == 0)).astype(int)



X = flattened_df.loc[:, ~flattened_df.columns.str.contains('RESP')]
X = X.loc[:, ~X.columns.str.contains('TOT_LAP_INVALID_IN_SEGMENT')]
X = X.loc[:, ~X.columns.str.contains('TOT_LAP_CAR_FIRST_REVERSED_IN_SEG')]
X = X.loc[:, ~X.columns.str.contains('TOT_LAP')]
X.columns.to_list()


# Train 1. 
regressor = xgb.XGBRegressor()
regressor.fit(X, y_time)

# Train 2. 
classifier = xgb.XGBClassifier()
classifier.fit(X, y_valid)



#------------------------------------------------------------------------------
# Plots 
best_params = optimizer.max['params']
best_target = optimizer.max['target']

def plot_convergence(optimizer):
   
    x = list(range(1, len(optimizer.space)+1))
    y = [res['target'] for res in optimizer.res]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, '-o')
    plt.title('Convergence Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Target Value')
    plt.show()
    
    plt.savefig("/Users/Owenbrooks/Desktop/DATA3001/Project/[8] Optimisation/Ver 2 [No tot lap aggs]/convergence_plot.png")
    plt.close()


def plot_hyperparameters(optimizer, param_name):
  
    x = list(range(1, len(optimizer.space)+1))
    y = [res['params'][param_name] for res in optimizer.res]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, '-o')
    plt.title(f'{param_name} Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel(param_name)
    plt.show()
    
    plt.savefig(f"/Users/Owenbrooks/Desktop/DATA3001/Project/[8] Optimisation/Ver 2 [No tot lap aggs]/{param}_plot.png")
    plt.close()

plot_convergence(optimizer)
for param in pbounds.keys():
    plot_hyperparameters(optimizer, param)

flattened_df.columns.to_list()


#------------------------------------------------------------------------------
# MODEL 6 -> CONSTRAINT OPTIMISATION [ Ver 3 ]
#------------------------------------------------------------------------------

#  Notes 
     # Drop all tot lap aggs 
     # ONLY MODEL SEG1 
     # More iterations

# Filter the dataset to only consider SEG1 columns for X and the specified y
X_SEG1 = flattened_df.filter(like='SEG1', axis=1)
X_SEG1 = X_SEG1.loc[:, ~X_SEG1.columns.str.contains('(RESP)')]
X_SEG1 = X_SEG1.loc[:, ~X_SEG1.columns.str.contains('SEG10')]

X_SEG1.columns.to_list()

y_SEG1_time = flattened_df['SEG1_MAX_CURR_TIME_(RESP)']
y_SEG1_time.shape
y_SEG1_valid = (flattened_df['TOT_LAP_INVALID_IN_SEGMENT'] == 1).astype(int)  # 0 for valid, 1 for invalid
y_SEG1_valid.value_counts()

print(y_SEG1_time.isna().sum()) 
print(y_SEG1_valid.isna().sum())

nan_indices = y_SEG1_time[y_SEG1_time.isna()].index
X_SEG1 = X_SEG1.drop(nan_indices, axis=0)
y_SEG1_time = y_SEG1_time.drop(nan_indices, axis=0)
y_SEG1_valid = y_SEG1_valid.drop(nan_indices, axis=0)



# Train regressor and classifier on the new X_SEG1 and y_SEG1_time data
regressor.fit(X_SEG1, y_SEG1_time)
classifier.fit(X_SEG1, y_SEG1_valid)

# Extract bounds for the new X_SEG1 data
top_20_SEG1 = flattened_df.nsmallest(20, 'SEG1_MAX_CURR_TIME_(RESP)')
pbounds_SEG1 = {}
for column in X_SEG1.columns:
    min_val = top_20_SEG1[column].min()
    max_val = top_20_SEG1[column].max()
    
    # Add a threshold
    exp_min = min_val - 0.05 * (max_val - min_val)
    exp_max = max_val + 0.05 * (max_val - min_val)
    
    pbounds_SEG1[column] = (exp_min, exp_max)

# Bayesian Optimization for the SEG1 data
optimizer_SEG1 = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds_SEG1,
    random_state=42,
)

optimizer_SEG1.maximize(
    init_points=20, 
    n_iter=250#,     
    #acq='ei',
)

print(optimizer_SEG1.max)

#------------------------------------------------------------------------------
# Plots 

best_params = optimizer_SEG1.max['params']
best_target = optimizer_SEG1.max['target']


# Add smoothing 
def rolling_average(data, window_size=5):
    """Compute a rolling average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_convergence(optimizer, window_size=5):
   
    x = list(range(1, len(optimizer.space)+1))
    y = [res['target'] for res in optimizer.res]
    y_smooth = rolling_average(y, window_size)
    
    # Adjust x for the loss of data points due to rolling average
    x_smooth = x[:len(y_smooth)]
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_smooth, y_smooth, '-o')
    plt.title('Convergence Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Target Value')
    
  
    plt.savefig("/Users/Owenbrooks/Desktop/DATA3001/Project/[8] Optimisation/Ver 3 [SEG1 only]/convergence_smooth.png")
    
    plt.show()
    plt.close()


def plot_hyperparameters(optimizer, param_name, window_size=5):
  
    x = list(range(1, len(optimizer.space)+1))
    y = [res['params'][param_name] for res in optimizer.res]
    y_smooth = rolling_average(y, window_size)
    
    # Adjust x for the loss of data points due to rolling average
    x_smooth = x[:len(y_smooth)]
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_smooth, y_smooth, '-o')
    plt.title(f'{param_name} Over Iterations (Smoothed)')
    plt.xlabel('Iteration')
    plt.ylabel(param_name)
    

    plt.savefig(f"/Users/Owenbrooks/Desktop/DATA3001/Project/[8] Optimisation/Ver 3 [SEG1 only]/{param_name}_smooth.png")
    
    plt.show()
    plt.close()

columns_to_plot = X_SEG1.columns.to_list()

for param in columns_to_plot:
    plot_hyperparameters(optimizer_SEG1, param)



results_df = pd.DataFrame([res['params'] for res in optimizer_SEG1.res])
results_df['target'] = [res['target'] for res in optimizer_SEG1.res]

# Bucket the target vals into bins 
number_of_bins = 10 
labels = range(number_of_bins)
results_df['target_bin'] = pd.cut(results_df['target'], bins=number_of_bins, labels=labels)

# 1. Parallel Coordinates Plot using binned target values
plt.figure(figsize=(15,8))
pd.plotting.parallel_coordinates(results_df, 'target_bin', colormap='viridis')
plt.title("Parallel Coordinates Plot of Hyperparameters")
plt.ylabel("Hyperparameter Value")
plt.xlabel("Hyperparameter")
plt.xticks(rotation=45)
plt.show()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Modelling -> second RUN 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Split down to key driver decision metrics [ REVISED VERSION ]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Based on domain konwledge & PCA 

flattened_df.columns.to_list()


# For segment-level features
seg_driver_decision_features = [
    col for col in flattened_df.columns 
    if any(feature in col for feature in ['AVG_GEAR', 'GEAR_CHANGES', 'WEIGHTED_AVG_GEAR', 'BRAKING_DURATION_MS',
                                          'DISTANCE_BEFORE_BRAKE', 'GEAR_VARIABILITY', 'MAX_DIST_FROM_RL',
                                          'MAX_CURR_TIME_(RESP)', 'MEAN_DIST_FROM_RL', 'MIN_CURR_TIME_(RESP)',
                                          'MIN_STEERING_CURR_DISTANCE', 'RANGE_DIST_FROM_RL', 'RATE_OF_CHANGE_(RESP)',
                                          'TIME_DELTA_(RESP)', 'TOTAL_BRAKING',
                                          'ENGINE_RPM_MIN', 'ENGINE_RPM_MAX', 'ENGINE_RPM_MEAN', 'ENGINE_RPM_STD',
                                          'STEERING_STD', 'MAX_STEERING_CURR_DISTANCE', 'STEERING_MIN', 'STEERING_MAX',
                                          'TOT_LAP_INVALID_IN_SEGMENT'])
    and 'SEG' in col
]


tot_lap_driver_decision_features = [
    col for col in flattened_df.columns 
    if any(feature in col for feature in ['TOT_LAP_AVG_GEAR', 'TOT_LAP_AVG_THROTTLE', 'TOT_LAP_CAR_FIRST_REVERSED_AT_DIST',
                                          'TOT_LAP_DISTANCE_BEFORE_BRAKE', 'TOT_LAP_GEAR_CHANGES', 'TOT_LAP_MAX_DIST_FROM_RL', 
                                          'TOT_LAP_MEAN_DIST_FROM_RL', 'TOT_LAP_MIN_DIST_FROM_RL', 'TOT_LAP_RANGE_DIST_FROM_RL',
                                          'TOT_LAP_TOTAL_BRAKING', 'TOT_LAP_INVALID_IN_SEGMENT', 'TOT_LAP_CAR_FIRST_REVERSED_IN_SEG',
                                          'SEG10_MAX_CURR_TIME_(RESP)']) 

]


segment_level_df = flattened_df[seg_driver_decision_features]
total_lap_df = flattened_df[tot_lap_driver_decision_features]


segment_level_df.shape

segment_level_df.isna().sum()
total_lap_df.isna().sum()


segment_level_df_clean = segment_level_df.dropna()
total_lap_df_clean = total_lap_df.dropna()

total_lap_df_clean.shape
segment_level_df_clean.shape

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Handle outliers
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


mean_lap_time = segment_level_df_clean['SEG10_MAX_CURR_TIME_(RESP)'].mean()
std_lap_time = segment_level_df_clean['SEG10_MAX_CURR_TIME_(RESP)'].std()
threshold_lap_time = mean_lap_time + 1.5 * std_lap_time

# Remove outliers based on the defined threshold
segment_level_df_clean_no_outliers = segment_level_df_clean[segment_level_df_clean['SEG10_MAX_CURR_TIME_(RESP)'] <= threshold_lap_time]
total_lap_df_clean_no_outliers = total_lap_df_clean[total_lap_df_clean['SEG10_MAX_CURR_TIME_(RESP)'] <= threshold_lap_time]

segment_level_df_clean_no_outliers.shape

total_lap_df_clean_no_outliers.shape



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# XGBoost - Total Lap Aggregates (filtered for driver decisisons)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#Responnse = Total lap time 
#------------------------------------------------------------------------------
total_lap_df_clean.columns.to_list()

X = total_lap_df_clean.drop(columns=['SEG10_MAX_CURR_TIME_(RESP)', 'TOT_LAP_INVALID_IN_SEGMENT', 'TOT_LAP_CAR_FIRST_REVERSED_IN_SEG', 'TOT_LAP_CAR_FIRST_REVERSED_AT_DIST'])
y = total_lap_df_clean['SEG10_MAX_CURR_TIME_(RESP)']

# TT Spliy 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train 
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)

# Plot feature im[p]
xgb.plot_importance(model, max_num_features=10)
plt.show()

# SHAP 
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)


shap.summary_plot(shap_values, X_test)
plt.show()

#------------------------------------------------------------------------------
#Responmse = Lap validity 
#------------------------------------------------------------------------------


classification_df = total_lap_df_clean_no_outliers.copy()

# Convert the response to binary for classification
classification_df['BIN_TOT_LAP_INVALID_IN_SEGMENT'] = classification_df['TOT_LAP_INVALID_IN_SEGMENT'].apply(lambda x: 1 if x > 0 else 0)


X = classification_df.drop(columns=['TOT_LAP_INVALID_IN_SEGMENT', 'TOT_LAP_CAR_FIRST_REVERSED_IN_SEG', 'SEG10_MAX_CURR_TIME_(RESP)', 'BIN_TOT_LAP_INVALID_IN_SEGMENT', 'TOT_LAP_CAR_FIRST_REVERSED_AT_DIST'])
X = X.drop(columns=['TOT_LAP_MAX_DIST_FROM_RL', 'TOT_LAP_MEAN_DIST_FROM_RL', 'TOT_LAP_MIN_DIST_FROM_RL', 'TOT_LAP_RANGE_DIST_FROM_RL'])
X.columns.to_list()


y = classification_df['BIN_TOT_LAP_INVALID_IN_SEGMENT']
y.value_counts()

# TT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)



X_train.shape
X_test.shape


model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


xgb.plot_importance(model, max_num_features=10)
plt.show()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
plt.show()


#------------------------------------------------------------------------------

# Predict probs for the test data
y_probs = model.predict_proba(X_test)[:, 1]  # probabilities for the positive class

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# CR
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
# Conf Matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Plot 
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Valid', 'Invalid'], yticklabels=['Valid', 'Invalid'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()


#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# XGBoost - Seg Level Aggs (filtered for driver decisisons)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
#Response = Total lap time [ Ver 1 ] RMSE Way too high 
#------------------------------------------------------------------------------
segment_level_df_clean_no_outliers.columns.to_list()




exclude_columns = ['TIME', 'RESP', 'REVERSED', 'INVALID']
X_segment = segment_level_df_clean_no_outliers.drop(columns=[col for col in segment_level_df_clean.columns if any(ex_str in col for ex_str in exclude_columns)])
X_segment.columns.to_list()

y_segment = segment_level_df_clean_no_outliers['SEG10_MAX_CURR_TIME_(RESP)']

y_segment.describe()

X_train_segment, X_test_segment, y_train_segment, y_test_segment = train_test_split(X_segment, y_segment, test_size=0.2, random_state=42)


model_segment = xgb.XGBRegressor(objective='reg:squarederror')
model_segment.fit(X_train_segment, y_train_segment)

y_pred_segment = model_segment.predict(X_test_segment)
mse = mean_squared_error(y_test_segment, y_pred_segment)
print(f"The Mean Squared Error on the test set is: {mse:.2f}")

# Plot 
xgb.plot_importance(model_segment, max_num_features=10)
plt.show()

explainer_segment = shap.TreeExplainer(model_segment)
shap_values_segment = explainer_segment.shap_values(X_test_segment)

shap.summary_plot(shap_values_segment, X_test_segment, max_display=15)
plt.show()



#------------------------------------------------------------------------------
#Response = Total lap time [ Ver 2 ] 
#------------------------------------------------------------------------------


# GRID SEARCH FOR HYPER PARAM TUNING

X_train_segment.columns.to_list()




param_grid_simple = {
    'n_estimators': [100, 200],  
    'max_depth': [3, 5],         
    'learning_rate': [0.01, 0.1],  
}

# INitilisa
xgb_reg_simple = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search_simple = GridSearchCV(xgb_reg_simple, param_grid_simple, scoring='neg_mean_squared_error', cv=5, verbose=1)

# Fit GridSearchCV
grid_search_simple.fit(X_train_segment, y_train_segment)

# Best est
best_regressor_simple = grid_search_simple.best_estimator_

# Preds
y_pred_segment_simple = best_regressor_simple.predict(X_test_segment)

# Compute new MSE
new_mse_simple = mean_squared_error(y_test_segment, y_pred_segment_simple)
print(f"The MSE on the test set is: {new_mse_simple:.2f}")

# Best parameters
print(f"Best hyperparameters: {grid_search_simple.best_params_}")



import math
rmse = math.sqrt(910245.85)
print(f"(RMSE) on the test set is: {rmse:.2f}")



xgb.plot_importance(model_segment, max_num_features=10)
plt.show()


explainer_segment = shap.TreeExplainer(model_segment)
shap_values_segment = explainer_segment.shap_values(X_test_segment)

shap.summary_plot(shap_values_segment, X_test_segment, max_display=15)
plt.show()



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# XGBoost - Seg Level Aggs -> 10 MODELS, ONE FOR EACH SEG 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
segment_level_df_clean_no_outliers.columns.to_list()

import shap

models = {}
mse_scores = {}
feature_importances = {}
shap_values_dict = {}
# Loop through each segment
for seg_num in range(1, 11):
    # Prepare the response variable
    y_seg = segment_level_df_clean_no_outliers[f'SEG{seg_num}_TIME_DELTA_(RESP)']


    X_seg = segment_level_df_clean_no_outliers.filter(regex=f'^SEG{seg_num}_(?!.*RESP)(?!.*AVG_GEAR)(?!.*RPM)')
    

    X_train_seg, X_test_seg, y_train_seg, y_test_seg = train_test_split(X_seg, y_seg, test_size=0.2, random_state=42)


    model_seg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model_seg.fit(X_train_seg, y_train_seg)
    
    # PReds -> STORE ALL RES BELOW FOR EACH SEG 
    y_pred_seg = model_seg.predict(X_test_seg)

    mse_seg = mean_squared_error(y_test_seg, y_pred_seg)
    mse_scores[f'SEG{seg_num}'] = mse_seg
    

    models[f'SEG{seg_num}'] = model_seg
    

    importance_df = pd.DataFrame({
        'feature': X_train_seg.columns,
        'importance': model_seg.feature_importances_
    }).sort_values(by='importance', ascending=False).head(3)
    
    feature_importances[f'SEG{seg_num}'] = importance_df

    explainer = shap.Explainer(model_seg)
    shap_values = explainer(X_test_seg)
    
    # Get the mean absolute SHAP values for each feature and sort
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': X_test_seg.columns,
        'shap_importance': shap_sum
    }).sort_values(by='shap_importance', ascending=False).head(3)
    
    # Store SHAP values
    shap_values_dict[f'SEG{seg_num}'] = importance_df
    

# ERROR 
#importance_df
#is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
#is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
#is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead
#is_categorical_dtype is deprecated and will be removed in a future version. Use isinstance(dtype, CategoricalDtype) instead


#Restarting kernel...
#------------------------------------------------------------------------------
 



#------------------------------------------------------------------------------
# PRINT RES STAT S

for seg, mse in mse_scores.items():
    print(f"The Mean Squared Error for {seg} on the test set is: {mse:.2f}")

import math
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_log_error, explained_variance_score


for seg, mse in mse_scores.items():
    rmse = math.sqrt(mse)
    print(f"The Root Mean Squared Error (RMSE) for {seg} on the test set is: {rmse:.2f}")


for seg_num in range(1, 11):

    model_seg = models[f'SEG{seg_num}']

 
    X_test_seg = segment_level_df_clean_no_outliers.filter(regex=f'^SEG{seg_num}_(?!.*RESP)(?!.*AVG_GEAR)(?!.*RPM)')
    y_test_seg = segment_level_df_clean_no_outliers[f'SEG{seg_num}_TIME_DELTA_(RESP)']

    y_pred_seg = model_seg.predict(X_test_seg)
    rmse = math.sqrt(mse_scores[f'SEG{seg_num}'])  
    print(f"RMSE for SEG{seg_num} on the test set: {rmse:.2f}")

    # R-squared
    r2 = r2_score(y_test_seg, y_pred_seg)

    n = X_test_seg.shape[0]  # Num of samples
    k = X_test_seg.shape[1]  # Num of independent vars
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    print(f"Adjusted R-squared for SEG{seg_num} on the test set: {adj_r2:.2f}")

    # Explained Variance Score
    evs = explained_variance_score(y_test_seg, y_pred_seg)
    print(f"Explained Variance Score for SEG{seg_num} on the test set: {evs:.2f}")

    print("\n") 



#------------------------------------------------------------------------------
# Feat imp

import seaborn as sns
import matplotlib.colors as mcolors
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20), dpi=90)

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, (seg, importance_df) in enumerate(feature_importances.items()):

    cmap = sns.light_palette("darkred", reverse=False, as_cmap=True)
    norm = mcolors.Normalize(vmin=importance_df['importance'].min(), vmax=importance_df['importance'].max())
    colors = [cmap(norm(value)) for value in importance_df['importance']]
    
    sns.barplot(data=importance_df, x='importance', y='feature', ax=axes[i], palette=colors)
    axes[i].set_title(f'{seg}', fontsize=16, fontweight='bold')
    axes[i].set_yticklabels(axes[i].get_yticklabels(), fontweight='bold')

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# SHAP 


for seg_num in range(1, 11):

    y_seg = segment_level_df_clean_no_outliers[f'SEG{seg_num}_TIME_DELTA_(RESP)']
    X_seg = segment_level_df_clean_no_outliers.filter(regex=f'^SEG{seg_num}_(?!.*RESP)(?!.*AVG_GEAR)(?!.*RPM)')


    X_train_seg, X_test_seg, y_train_seg, y_test_seg = train_test_split(X_seg, y_seg, test_size=0.2, random_state=42)

    model_seg = models[f'SEG{seg_num}']


    explainer = shap.Explainer(model_seg)
    shap_values = explainer(X_test_seg)

    shap.summary_plot(shap_values, X_test_seg, max_display=5)
    plt.title(f'Segment {seg_num} SHAP Values')


    plt.savefig(f'/Users/Owenbrooks/Desktop/DATA3001/Project/[9] Fnal Report/SegLap [Time]/Shaps/shap_plot_segment_{seg_num}.png')
    plt.close()
    
    
#------------------------------------------------------------------------------
# PDP - Would take too long to interpret & not sig val add, leave out 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# XGBoost - Seg Level Aggs -> 10 MODELS, ONE FOR EACH SEG -> VA
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
segment_level_df_clean_no_outliers.columns.to_list()

segment_df_binary_targets = segment_level_df_clean_no_outliers.copy()

# 10 new binary resp cols 
for seg_num in range(1, 11):
    segment_df_binary_targets[f'TOT_LAP_INVALID_IN_SEG{seg_num}'] = (
        segment_df_binary_targets['TOT_LAP_INVALID_IN_SEGMENT'] == seg_num
    ).astype(int)

segment_df_binary_targets[f'TOT_LAP_INVALID_IN_SEG1'].value_counts() 
segment_df_binary_targets[f'TOT_LAP_INVALID_IN_SEG2'].value_counts() 
segment_df_binary_targets[f'TOT_LAP_INVALID_IN_SEG3'].value_counts() 


segment_df_binary_targets[f'TOT_LAP_INVALID_IN_SEG6'].value_counts() 
segment_df_binary_targets[f'TOT_LAP_INVALID_IN_SEG7'].value_counts() 

# Check
segment_level_df_clean_no_outliers['TOT_LAP_INVALID_IN_SEGMENT'].value_counts()


#------------------------------------------------------------------------------
# Train

models_invalid = {}
mse_scores_invalid = {}
feature_importances_invalid = {}
accuracy_scores_invalid = {}
shap_values_dict_invalid = {}
confusion_matrices_invalid = {}  


for seg_num in range(1, 11):

    y_seg_invalid = segment_df_binary_targets[f'TOT_LAP_INVALID_IN_SEG{seg_num}']
    X_seg = segment_df_binary_targets.filter(regex=f'^SEG{seg_num}_(?!.*RESP)(?!.*AVG_GEAR)(?!.*RPM)(?!.*RL)')
    

    X_train_seg, X_test_seg, y_train_seg, y_test_seg = train_test_split(
        X_seg, y_seg_invalid, test_size=0.2, random_state=1
    )


    model_seg_invalid = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_seg_invalid.fit(X_train_seg, y_train_seg)
    
    # Predict
    y_pred_seg_invalid = model_seg_invalid.predict(X_test_seg)

    # Acc
    accuracy_seg = accuracy_score(y_test_seg, y_pred_seg_invalid)
    accuracy_scores_invalid[f'SEG{seg_num}'] = accuracy_seg

    # Store
    models_invalid[f'SEG{seg_num}'] = model_seg_invalid
    
    # Feat imps
    importance_df = pd.DataFrame({
        'feature': X_train_seg.columns,
        'importance': model_seg_invalid.feature_importances_
    }).sort_values(by='importance', ascending=False).head(3)
    
    feature_importances_invalid[f'SEG{seg_num}'] = importance_df
        
    # SHAP
    explainer = shap.Explainer(model_seg_invalid)
    shap_values = explainer(X_test_seg)
    
    # Get the mean absolute SHAP values for each feature and sort
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': X_test_seg.columns,
        'shap_importance': shap_sum
    }).sort_values(by='shap_importance', ascending=False).head(3)
    
    # Store SHAP values
    shap_values_dict_invalid[f'SEG{seg_num}'] = importance_df
    
    # CM 
    cm = confusion_matrix(y_test_seg, y_pred_seg_invalid)
    confusion_matrices_invalid[f'SEG{seg_num}'] = cm

    

#------------------------------------------------------------------------------
# Feat imp
# Create a 5x2 grid for feature imps 
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20), dpi=90)
fig.subplots_adjust(hspace=0.5)

# Flatten the axes array for efficient iteration
axes = axes.flatten()

valid_segments = [1, 2, 3, 4, 5, 8, 9, 10] 

for i, seg_num in enumerate(valid_segments):
    importance_df = feature_importances_invalid[f'SEG{seg_num}']
    
    # Create a color map from dark red to light red
    cmap = sns.light_palette("darkred", reverse=False, as_cmap=True)
    norm = mcolors.Normalize(vmin=importance_df['importance'].min(), vmax=importance_df['importance'].max())
    colors = [cmap(norm(value)) for value in importance_df['importance']]
    
    sns.barplot(data=importance_df, x='importance', y='feature', ax=axes[i], palette=colors)
    axes[i].set_title(f'SEG{seg_num} - Feature Importances', fontsize=16, fontweight='bold')
    axes[i].set_xlabel('Importance', fontsize=12)
    axes[i].set_ylabel('Feature', fontsize=12)
    axes[i].tick_params(axis='both', labelsize=10)
    axes[i].invert_yaxis()

for i in range(8, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------
# Confusion

from imblearn.over_sampling import RandomOverSampler

fig_cm, axes_cm = plt.subplots(nrows=4, ncols=2, figsize=(15, 20), dpi=90)
fig_cm.subplots_adjust(hspace=0.5, wspace=0.5)
axes_cm = axes_cm.flatten()

valid_segments = [1, 2, 3, 4, 5, 8, 9, 10]

for i, seg_num in enumerate(valid_segments):
    cm = confusion_matrices_invalid[f'SEG{seg_num}']
    ax_cm = axes_cm[i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title(f'SEG{seg_num} - Confusion Matrix', fontsize=16, fontweight='bold')
    ax_cm.set_xlabel('Predicted', fontsize=12)
    ax_cm.set_ylabel('Actual', fontsize=12)
    ax_cm.xaxis.set_ticklabels(['0', '1'])
    ax_cm.yaxis.set_ticklabels(['0', '1'])


for i in range(8, len(axes_cm)):
    fig_cm.delaxes(axes_cm[i])


plt.tight_layout()
plt.show()




#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# XGBoost - Seg Level Aggs -> SINGLE MODEL DUE TO DATA IMBALANCE 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
df_binary = segment_level_df_clean_no_outliers.copy()


# Convert TOT_LAP_INVALID_IN_SEGMENT to binary (1 if > 0, else 0)
df_binary['TOT_LAP_INVALID_IN_SEGMENT'] = (df_binary['TOT_LAP_INVALID_IN_SEGMENT'] > 0).astype(int)

#df_binary['TOT_LAP_INVALID_IN_SEGMENT'].value_counts()

X_seg = df_binary.filter(regex=r'^SEG\d_(?!.*RESP)(?!.*AVG_GEAR)(?!.*RPM)(?!.*RL)(?!.*INVALID)')
X_seg.columns.to_list()

y_seg = df_binary['TOT_LAP_INVALID_IN_SEGMENT']


X_train_seg, X_test_seg, y_train_seg, y_test_seg = train_test_split(X_seg, y_seg, test_size=0.2, random_state=1)

model_seg = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1)
model_seg.fit(X_train_seg, y_train_seg)

# Calculate feature importances using SHAP
explainer = shap.Explainer(model_seg)
shap_values = explainer(X_test_seg)


shap.summary_plot(shap_values, X_test_seg, max_display=15)

#------------------------------------------------------------------------------
y_pred_seg = model_seg.predict(X_test_seg)
y_prob_seg = model_seg.predict_proba(X_test_seg)[:, 1]  


cm = confusion_matrix(y_test_seg, y_pred_seg)

# Reportt 
class_report = classification_report(y_test_seg, y_pred_seg)

# AUC & ROC
fpr, tpr, _ = roc_curve(y_test_seg, y_prob_seg)
roc_auc = auc(fpr, tpr)


class_labels = ['0', '1']


cm
#Conf matrix is having some issue, hard code vals 
cm = np.array([[43, 11], [6, 37]])

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot the confusion matrix on the left subplot
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=['Valid', 'Invalid'], yticklabels=['Valid', 'Invalid'], ax=axes[0])
for i in range(len(cm)):
    for j in range(len(cm)):
        axes[0].text(j + 0.5, i + 0.5, str(cm[i, j]), ha='center', va='center', fontsize=18, fontweight='bold', color='black')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')
axes[0].set_title('Confusion Matrix')

# Plot the ROC curve on the right subplot
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()


print('Classification Report:\n', class_report)
print(f'AUC Score: {roc_auc:.2f}')


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# XGBoost - Seg Level Aggs -> SINGLE MODEL FOR SEGS 1 to 3
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
X_seg = segment_level_df_clean_no_outliers.filter(regex=r'^(SEG1|SEG2|SEG3)_(?!.*RESP)(?!.*AVG_GEAR)(?!.*RPM)(?!.*RL)(?!.*INVALID)')

X_seg.columns.to_list()


y_seg = ((segment_level_df_clean_no_outliers['TOT_LAP_INVALID_IN_SEGMENT'] > 0) & (segment_level_df_clean_no_outliers['TOT_LAP_INVALID_IN_SEGMENT'] <= 3)).astype(int)
y_seg[y_seg == 1] = 1
y_seg.value_counts()

#------------------------------------------------------------------------------
# TT SPLIT 
X_train_seg, X_test_seg, y_train_seg, y_test_seg = train_test_split(X_seg, y_seg, test_size=0.2, random_state=1)


model_seg = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1)
model_seg.fit(X_train_seg, y_train_seg)


explainer = shap.Explainer(model_seg)
shap_values = explainer(X_test_seg)


shap.summary_plot(shap_values, X_test_seg, max_display=15)


y_pred_seg = model_seg.predict(X_test_seg)
y_prob_seg = model_seg.predict_proba(X_test_seg)[:, 1]  


import seaborn as sns


fpr, tpr, _ = roc_curve(y_test_seg, y_prob_seg)
roc_auc = auc(fpr, tpr)



cm = confusion_matrix(y_test_seg, y_pred_seg)


class_labels = ['0', '1']


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
for i in range(len(cm)):
    for j in range(len(cm)):
        axes[0].text(j + 0.5, i + 0.5, str(cm[i, j]), ha='center', va='center', fontsize=18, fontweight='bold', color='black')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')
axes[0].set_title('Confusion Matrix')

# Plot the ROC curve on the right subplot
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(loc='lower right')


plt.tight_layout()
plt.show()


class_report = classification_report(y_test_seg, y_pred_seg)

print('Classification Report:\n', class_report)
print(f'AUC Score: {roc_auc:.2f}')

plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# Dists 

feature_importances = model_seg.feature_importances_


top_n_features = 6  

# Get indices of top N features
top_feature_indices = np.argsort(feature_importances)[::-1][:top_n_features]

# Get the feature names of the top N features
top_feature_names = X_train_seg.columns[top_feature_indices]

# Analysefeat dists for class 0 (hiusts )
for feature_name in top_feature_names:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=X_train_seg[y_train_seg == 0][feature_name], bins=30, kde=True, color='blue', label='Valid Laps')
    sns.histplot(data=X_train_seg[y_train_seg == 1][feature_name], bins=30, kde=True, color='red', label='Invalid Laps')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.title(f'Distribution of {feature_name} for Valid and Invalid Laps')
    plt.show()
    

    valid_feature_values = X_train_seg[y_train_seg == 0][feature_name]
    range_of_valid_values = (valid_feature_values.min(), valid_feature_values.max())
    print(f'Range of valid values for {feature_name}: {range_of_valid_values}')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Comparitive visualisations for lpa times & invalid .. 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Fix .1 on resp var names 

flattened_df.columns = [col[:-2] if col.endswith('.1') else col for col in flattened_df.columns]
flattened_df.columns = [col[:-6] if col.endswith('\u2028') else col for col in flattened_df.columns]

# Fix dup cols 

flattened_df = flattened_df.loc[:, ~flattened_df.columns.duplicated()]


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Split down to key driver decision metrics 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

flattened_df.columns.to_list()

focus_cols = flattened_df.loc[:, ~flattened_df.columns.str.contains('TOT_LAP')]

focus_cols.columns.to_list()

# Before filter
focus_cols.shape
#Out[269]: (517, 320)


# Filter 
base_cols = [
    'AVG_THROTTLE',
    'BRAKING_DURATION_MS',
    'DISTANCE_BEFORE_BRAKE',
    'GEAR_CHANGES',
    'STEERING_MAX',
    'STEERING_MEAN',
    'STEERING_MIN',
    'STEERING_STD',
    'TOTAL_BRAKING',
    'MAX_DIST_FROM_RL',
    'WEIGHTED_AVG_GEAR'
]

len(base_cols)  
# 11 predictors 

response_cols = [
    'MIN_CURR_TIME_(RESP)',
    'MAX_CURR_TIME_(RESP)',
    'TIME_DELTA_(RESP)',
    'RATE_OF_CHANGE_(RESP)'
]

selected_cols = []

for i in range(1, 11):
    segment_prefix = f'SEG{i}_'
    selected_cols.extend([segment_prefix + col for col in base_cols])
    selected_cols.extend([segment_prefix + col for col in response_cols])

reduced_df = focus_cols[selected_cols]


reduced_df.shape
#Out[273]: (517, 150)

#320 - 150 = 170 cols removed




#------------------------------------------------------------------------------
# tot lap agg KDE plot revised 2
#------------------------------------------------------------------------------


features_to_plot = [
    col for col in tot_lap_driver_decision_features 
    if col not in [
        'TOT_LAP_MIN_DIST_FROM_RL', 
        'TOT_LAP_CAR_FIRST_REVERSED_IN_SEG',  
        'TOT_LAP_MAX_DIST_FROM_RL',
        'TOT_LAP_INVALID_IN_SEGMENT'
    ]
]


if 'SEG10_MAX_CURR_TIME_(RESP)' in flattened_df.columns:
    features_to_plot.append('SEG10_MAX_CURR_TIME_(RESP)')
    

features_to_plot
    

# KDE PLOT 
n_features = len(features_to_plot)
ncols = 3 
nrows = 3


fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axes = axes.flatten()


for i, feature in enumerate(features_to_plot):
    sns.kdeplot(
        data=total_lap_df_clean[total_lap_df_clean['TOT_LAP_INVALID_IN_SEGMENT'] == 0],
        x=feature,
        label='Valid lap',
        color='blue',
        ax=axes[i]
    )
    sns.kdeplot(
        data=total_lap_df_clean[total_lap_df_clean['TOT_LAP_INVALID_IN_SEGMENT'] > 0],
        x=feature,
        label='Invalid lap',
        color='red',
        ax=axes[i]
    )
    axes[i].set_title(feature, fontsize=15, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')


for i in range(n_features, nrows * ncols):
    axes[i].axis('off')


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=17)
plt.subplots_adjust(bottom=0.05)
plt.show()



#------------------------------------------------------------------------------
# tot lap agg KDE 10% fastest valid [ Revised ]
#------------------------------------------------------------------------------


valid_laps_df = total_lap_df_clean[total_lap_df_clean['TOT_LAP_INVALID_IN_SEGMENT'] == 0]

# 10% fastest laps threshold
top_10_percent_fastest_threshold = valid_laps_df['SEG10_MAX_CURR_TIME_(RESP)'].quantile(0.1)

is_top_10_percent_fastest = valid_laps_df['SEG10_MAX_CURR_TIME_(RESP)'] <= top_10_percent_fastest_threshold


features_to_plot = [
    feature for feature in features_to_plot 
    if feature not in ['SEG10_MAX_CURR_TIME_(RESP)', 'TOT_LAP_CAR_FIRST_REVERSED_AT_DIST']
]

# KDE 
n_features = len(features_to_plot)
ncols = 3 
nrows = int(np.ceil(n_features / ncols)) 

## PLTO 
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):

    sns.kdeplot(
        data=valid_laps_df[~is_top_10_percent_fastest],
        x=feature,
        label='Other 90% Valid Laps',
        color='blue',
        ax=axes[i]
    )
  
    sns.kdeplot(
        data=valid_laps_df[is_top_10_percent_fastest],
        x=feature,
        label='Top 10% Fastest Valid Laps',
        color='red',
        ax=axes[i]
    )
    axes[i].set_title(feature, fontsize=15, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')


for i in range(n_features, nrows * ncols):
    axes[i].axis('off')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=17)
plt.subplots_adjust(bottom=0.05)
plt.show()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Predictors values changing through segments [ Ver 1 ] 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


segment_level_df_clean.columns.to_list()

boundaries = np.array([0., 76.97, 129.15, 173.21, 214.9, 248.99, 286.43, 329.82, 376.17, 434.65, 499.94])
mid_points = (boundaries[:-1] + boundaries[1:]) / 2

base_features = set(
    '_'.join(col.split('_')[1:]) 
    for col in segment_level_df_clean.columns 
    if not col.endswith('LAP_INVALID_IN_SEGMENT') and "CURR_TIME" not in col and "TIME_DELTA" not in col and 'STEERING_CURR_DISTANCE' not in col
)

ncols = 3
n_features = len(base_features)  #
nrows = int(np.ceil(n_features / ncols))  

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
axes = axes.flatten()  

valid_laps = segment_level_df_clean[segment_level_df_clean['TOT_LAP_INVALID_IN_SEGMENT'] == 0]
invalid_laps = segment_level_df_clean[segment_level_df_clean['TOT_LAP_INVALID_IN_SEGMENT'] > 0]

# Plot 
for i, feature in enumerate(base_features):
    ax = axes[i]  
    valid_feature_values = []
    invalid_feature_values = []

    for seg_num in range(1, 11):
        feature_name = f'SEG{seg_num}_{feature}'
     
        valid_mean = valid_laps[feature_name].mean() if feature_name in valid_laps else np.nan
        invalid_mean = invalid_laps[feature_name].mean() if feature_name in invalid_laps else np.nan
        valid_feature_values.append(valid_mean)
        invalid_feature_values.append(invalid_mean)


    ax.plot(mid_points, valid_feature_values, linestyle='-', color='blue', label='Valid Lap')
    ax.plot(mid_points, invalid_feature_values, linestyle='-', color='red', label='Invalid Lap')

    ax.set_title(feature, fontsize=14, fontweight='bold')
    ax.set_xticks(mid_points)
    ax.set_xticklabels([f'SEG{num}' for num in range(1, 11)], rotation=45)
    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.set_xlabel('Segment')
    ax.set_ylabel('Average Value')
    #ax.legend()


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=17)
plt.subplots_adjust(bottom=0.05)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Predictors values changing through segments [ Ver 2 ] - Valid Vs Invalid  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


valid_laps = segment_level_df_clean[segment_level_df_clean['TOT_LAP_INVALID_IN_SEGMENT'] == 0]
invalid_laps = segment_level_df_clean[segment_level_df_clean['TOT_LAP_INVALID_IN_SEGMENT'] > 0]


base_features = set('_'.join(col.split('_')[1:]) for col in segment_level_df_clean.columns if "CURR_TIME" not in col and "TIME_DELTA" not in col)

ncols = 3
n_features = len(base_features)
nrows = int(np.ceil(n_features / ncols))


fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
axes = axes.flatten()  

for i, base_feature in enumerate(base_features):
    ax = axes[i]
    valid_feature_values = []
    invalid_feature_values = []


    for seg_num in range(1, 11):
        valid_segment_feature = f'SEG{seg_num}_{base_feature}'
        invalid_segment_feature = f'SEG{seg_num}_{base_feature}'
        
       
        valid_mean = valid_laps[valid_segment_feature].mean()
        invalid_mean = invalid_laps[invalid_segment_feature].mean()
        valid_feature_values.append(valid_mean)
        invalid_feature_values.append(invalid_mean)


    ax.plot(mid_points, valid_feature_values, linestyle='-', color='blue', label='Valid')
    ax.plot(mid_points, invalid_feature_values, linestyle='-', color='red', label='Invalid')

    ax.set_title(base_feature, fontsize=14, fontweight='bold')
    ax.set_xticks(mid_points)
    ax.set_xticklabels([f'SEG{num}' for num in range(1, 11)], rotation=45)
    ax.set_xlim(boundaries[0], boundaries[-1])
    ax.set_xlabel('Segment')
    ax.set_ylabel('Value')
    ax.legend()

plt.tight_layout()
plt.show()



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# XGBoost - Total Lap Aggregates (filtered for driver decisisons)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


# For total lap aggregate features, we retain features that match the criteria set above for segment-level features
tot_lap_driver_decision_features = [
    col for col in flattened_df.columns 
    if any(feature in col for feature in ['TOT_LAP_AVG_GEAR', 'TOT_LAP_AVG_THROTTLE', 'TOT_LAP_CAR_FIRST_REVERSED_AT_DIST',
                                          'TOT_LAP_DISTANCE_BEFORE_BRAKE', 'TOT_LAP_GEAR_CHANGES', 'TOT_LAP_MAX_DIST_FROM_RL', 
                                          'TOT_LAP_MEAN_DIST_FROM_RL', 'TOT_LAP_MIN_DIST_FROM_RL', 'TOT_LAP_RANGE_DIST_FROM_RL',
                                          'TOT_LAP_TOTAL_BRAKING', 'TOT_LAP_INVALID_IN_SEGMENT', 'TOT_LAP_CAR_FIRST_REVERSED_IN_SEG',
                                          'SEG10_MAX_CURR_TIME_(RESP)'])  # Including only relevant 'TOT_LAP' features

]



total_lap_df = flattened_df[tot_lap_driver_decision_features]
total_lap_df.isna().sum()
total_lap_df_clean = total_lap_df.dropna()

#------------------------------------------------------------------------------
#Responmse = Total lap time 
#------------------------------------------------------------------------------
total_lap_df_clean.columns.to_list()

X = total_lap_df_clean.drop(columns=['SEG10_MAX_CURR_TIME_(RESP)', 'TOT_LAP_INVALID_IN_SEGMENT', 'TOT_LAP_CAR_FIRST_REVERSED_IN_SEG', 'TOT_LAP_CAR_FIRST_REVERSED_AT_DIST'])
y = total_lap_df_clean['SEG10_MAX_CURR_TIME_(RESP)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X_train, y_train)


xgb.plot_importance(model, max_num_features=10)
plt.show()


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
plt.show()

#------------------------------------------------------------------------------
#Responmse = Lap validity 
#------------------------------------------------------------------------------


classification_df = total_lap_df_clean.copy()

classification_df['BIN_TOT_LAP_INVALID_IN_SEGMENT'] = classification_df['TOT_LAP_INVALID_IN_SEGMENT'].apply(lambda x: 1 if x > 0 else 0)

X = classification_df.drop(columns=['TOT_LAP_INVALID_IN_SEGMENT', 'TOT_LAP_CAR_FIRST_REVERSED_IN_SEG', 'SEG10_MAX_CURR_TIME_(RESP)', 'BIN_TOT_LAP_INVALID_IN_SEGMENT', 'TOT_LAP_CAR_FIRST_REVERSED_AT_DIST'])
y = classification_df['BIN_TOT_LAP_INVALID_IN_SEGMENT']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


xgb.plot_importance(model, max_num_features=10)
plt.show()


#------------
y_probs = model.predict_proba(X_test)[:, 1]  # probabilities for the positive class


fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)
cm


disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Number of invalid segs 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

flattened_df.columns.to_list()


flattened_df['TOT_LAP_INVALID_IN_SEGMENT'].value_counts()


df_filtered = flattened_df[flattened_df['TOT_LAP_INVALID_IN_SEGMENT'] != 0]
value_counts_filtered = df_filtered['TOT_LAP_INVALID_IN_SEGMENT'].value_counts().sort_index()

total_counts_filtered = value_counts_filtered.sum()
percentages = (value_counts_filtered / total_counts_filtered) * 100

total_valid = len(flattened_df[flattened_df['TOT_LAP_INVALID_IN_SEGMENT'] == 0])

plt.figure(figsize=(16, 6))


fig, ax = plt.subplots()
bars = ax.bar(value_counts_filtered.index.astype(str), value_counts_filtered.values)

# Annotate perc
for bar, percentage in zip(bars, percentages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval +10, f'{percentage:.1f}%', color='black', 
             ha='center', va='top', fontweight='bold', fontsize=10)


ax.set_xlabel('Segment')
ax.set_ylabel('Number of Invalidated Laps')
plt.ylim(0, 200)

total_invalid = value_counts_filtered.sum()
total_valid = flattened_df['TOT_LAP_INVALID_IN_SEGMENT'].value_counts().loc[0]

plt.show()



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Number of invalid segs & reversed in seg together 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


plt.figure(figsize=(16, 8))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))


invalid_segments = flattened_df[flattened_df['TOT_LAP_INVALID_IN_SEGMENT'] != 0]
invalid_value_counts = invalid_segments['TOT_LAP_INVALID_IN_SEGMENT'].value_counts().sort_index()
invalid_total_counts = invalid_value_counts.sum()
invalid_percentages = (invalid_value_counts / invalid_total_counts) * 100
invalid_bars = ax1.bar(invalid_value_counts.index.astype(str), invalid_value_counts.values, color='blue')

for bar, percentage in zip(invalid_bars, invalid_percentages):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval / 2, f'{percentage:.1f}%', color='white', ha='center', va='center', fontweight='bold')

ax1.set_xlabel('Segment')
ax1.set_ylabel('Number of Invalidated Laps')
ax1.set_title('Invalidated Laps by Segment')
invalid_total_valid = flattened_df['TOT_LAP_INVALID_IN_SEGMENT'].value_counts().loc[0]
ax1.legend([f'Total invalid: {invalid_total_counts}', f'Total valid (0): {invalid_total_valid}'])


reversed_segments = flattened_df[flattened_df['TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'] != 0]
reversed_value_counts = reversed_segments['TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'].value_counts().sort_index()
reversed_total_counts = reversed_value_counts.sum()
reversed_percentages = (reversed_value_counts / reversed_total_counts) * 100
reversed_bars = ax2.bar(reversed_value_counts.index.astype(str), reversed_value_counts.values, color='green')

for bar, percentage in zip(reversed_bars, reversed_percentages):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval / 2, f'{percentage:.1f}%', color='white',
             ha='center', va='center', fontweight='bold', fontsize=13)

ax2.set_xlabel('Segment')
ax2.set_ylabel('Number of Cars First Reversed in Segment')
ax2.set_title('Cars First Reversed by Segment')
reversed_total_valid = flattened_df['TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'].value_counts().loc[0]
ax2.legend([f'Total reversed: {reversed_total_counts}', f'Total not reversed (0): {reversed_total_valid}'])

plt.tight_layout()
plt.show()



#flattened_df['TOT_LAP_CAR_FIRST_REVERSED_IN_SEG'].value_counts()


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Normalised Variance in times per segment, and variance in ROC 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def normalize_variance(variances):
    min_var = min(variances)
    max_var = max(variances)
    return [(var - min_var) / (max_var - min_var) for var in variances]

plt.figure(figsize=(16, 8))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# normalise
time_delta_variances = [flattened_df[f'SEG{seg_num}_TIME_DELTA_(RESP)'].var() for seg_num in range(1, 11)]
time_delta_variances_normalised = normalise_variance(time_delta_variances)


time_delta_bars = ax1.bar(range(1, 11), time_delta_variances_normalized, color='blue')
ax1.set_xlabel('Segment')
ax1.set_ylabel('Normalized Variance')
ax1.set_title('Variance of Times by Segment')
ax1.set_ylim([0, 1.05])

# Put the normalized variance value above each bar
for bar, variance in zip(time_delta_bars, time_delta_variances_normalized):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+.02, f'{variance:.2f}', color='black', ha='center', fontweight='bold', fontsize=13)


rate_of_change_variances = [flattened_df[f'SEG{seg_num}_RATE_OF_CHANGE_(RESP)'].var() for seg_num in range(1, 11)]
rate_of_change_variances_normalized = normalize_variance(rate_of_change_variances)

rate_of_change_bars = ax2.bar(range(1, 11), rate_of_change_variances_normalized, color='green')
ax2.set_xlabel('Segment')
ax2.set_ylabel('Normalized Variance')
ax2.set_title('Variance of Rate of Change by Segment')
ax2.set_ylim([0, 1.05])

# Put the normalized variance value above each bar
for bar, variance in zip(rate_of_change_bars, rate_of_change_variances_normalized):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f'{variance:.2f}', color='black', ha='center', fontweight='bold', fontsize=13)


plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# PCA SEG LEVEL [ Revised Ver ]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



seg_features_without_resp = [col for col in flattened_df.columns if 'SEG' in col and 'RESP' not in col and 'TOT_LAP' not in col]

segment_features_df = flattened_df[seg_features_without_resp]

segment_features_df.columns.to_list()

segment_features_df_clean = segment_features_df.dropna()

# PCA
pca_seg = PCA()
pca_seg.fit(segment_features_df_clean)

# Loadings 
seg_loadings = pca_seg.components_

# print top 5 loadings 
n_components_to_examine = 5 
top_features_per_component = {}
for i in range(n_components_to_examine):
    sorted_indices = np.argsort(-np.abs(seg_loadings[i]))
    top_features = [(segment_features_df_clean.columns[index], seg_loadings[i][index]) for index in sorted_indices[:5]]
    top_features_per_component[f'Component {i+1}'] = top_features


for component, features in top_features_per_component.items():
    print(f"{component} - Top contributing features:")
    for feature, loading in features:
        print(f"{feature}: {loading:.3f}")
    print("\n")



# Explained variance for each PCA component
explained_variance_seg = pca_seg.explained_variance_ratio_
cumulative_explained_variance_seg = np.cumsum(explained_variance_seg)




n_components_to_plot = min(len(explained_variance), 30) 

# "Scree" plot
plt.figure(figsize=(12, 6))
components = range(1, n_components_to_plot + 1)

# Plotting individual explained variance in bars
plt.bar(components, explained_variance[:n_components_to_plot], alpha=0.5, align='center', label='Individual explained variance')

# Plotting cumulative explained variance as a step plot
plt.step(components, cumulative_explained_variance[:n_components_to_plot], where='mid', label='Cumulative explained variance')

# Adding a horizontal line for 95% cumulative explained variance
plt.axhline(y=0.95, color='green', linestyle='--', label='95% Explained Variance')

# Labeling the axes and the plot
plt.xlabel('Principal Components')
plt.ylabel('Explained variance ratio')
plt.title('Explained Variance by PCA Component for Segment Features (No RESP)')
plt.legend(loc='best')

plt.xticks(components)  

plt.grid(True)

plt.tight_layout()
plt.show()



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# PCA TOT LAP LEVEL 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


tot_lap_columns = [col for col in flattened_df.columns if 'TOT_LAP' in col]
tot_lap_df = flattened_df[tot_lap_columns]

tot_lap_df.columns.to_list()
len(tot_lap_columns)
tot_lap_df_clean = tot_lap_df.dropna()

pca_tot_lap = PCA()
pca_tot_lap.fit(tot_lap_df_clean)

#------------------------------------------------------------------------------

explained_variance_tot_lap = pca_tot_lap.explained_variance_ratio_
cumulative_explained_variance_tot_lap = np.cumsum(explained_variance_tot_lap)

# "Scree" plot
plt.figure(figsize=(12, 6))
components_tot_lap = range(1, len(explained_variance_tot_lap) + 1)

plt.bar(components_tot_lap, explained_variance_tot_lap, alpha=0.5, align='center', label='Individual explained variance')
plt.step(components_tot_lap, cumulative_explained_variance_tot_lap, where='mid', label='Cumulative explained variance')

plt.xlabel('Principal Components')
plt.ylabel('Explained variance ratio')
plt.title('Explained Variance by PCA Component for Total Lap Features')
plt.legend(loc='best')
plt.xticks(components_tot_lap)  
plt.grid(True)
plt.tight_layout()
plt.show()




#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Post PCA  LAP LEV

pca_tot_lap = PCA()
pca_tot_lap.fit(tot_lap_df_clean)


loadings = pca_tot_lap.components_


n_components_to_examine = 5  #
top_features_per_component = {}

for i in range(n_components_to_examine):
  
    sorted_idx = np.argsort(np.abs(loadings[i]))[::-1]

    top_features = [(tot_lap_columns[index], loadings[i][index]) for index in sorted_idx]
    top_features_per_component[f'Component {i+1}'] = top_features

for component, features in top_features_per_component.items():
    print(f"{component} - Top contributing features:")
    for feature, loading in features[:10]: 
        print(f"{feature}: {loading:.3f}")
    print("\n")


flattened_df.columns.to_list()















































































































































































































































































































































































































































































































































































































































































































































































































































