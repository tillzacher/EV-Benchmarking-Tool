##### General Functions #####

from scipy.signal import butter, filtfilt, savgol_filter
import numpy as np

def smooth_signals(data, columns, filter_type='moving_average', **kwargs):
    """
    Smooths the specified columns in the dataframe using the specified filter.
    
    Parameters:
        data (pd.DataFrame): The input dataframe containing the signals to be smoothed.
        columns (list): List of column names to be smoothed.
        filter_type (str): The type of filter to use ('moving_average', 'exponential_moving_average', 
                           'lowpass', 'savitzky_golay').
        **kwargs: Additional parameters for the specified filter type.
        
            For 'moving_average':
                - window_size (int): The window size for the moving average.
            
            For 'exponential_moving_average':
                - alpha (float): The smoothing factor (0 < alpha <= 1).
            
            For 'lowpass':
                - cutoff_frequency (float): The cutoff frequency for the lowpass filter.
                - sample_rate (float): The sample rate of the signal.
                - order (int): The order of the Butterworth filter.
            
            For 'savitzky_golay':
                - window_length (int): The length of the filter window (must be odd).
                - polyorder (int): The order of the polynomial to fit.
    
    Returns:
        pd.DataFrame: A new dataframe with the smoothed columns.
    """
    # Create a copy of the dataframe to avoid modifying the original data
    smoothed_data = data.copy()
    
    for col in columns:
        if filter_type == 'moving_average':
            window_size = kwargs.get('window_size', 5)
            smoothed_data[col] = smoothed_data[col].rolling(window=window_size, center=True).mean()
        
        elif filter_type == 'exponential_moving_average':
            alpha = kwargs.get('alpha', 0.3)
            smoothed_data[col] = smoothed_data[col].ewm(alpha=alpha).mean()
        
        elif filter_type == 'lowpass':
            cutoff_frequency = kwargs.get('cutoff_frequency', 0.1)
            sample_rate = kwargs.get('sample_rate', 1.0)  # Default sample rate
            order = kwargs.get('order', 4)
            
            nyquist = 0.5 * sample_rate
            normal_cutoff = cutoff_frequency / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            smoothed_data[col] = filtfilt(b, a, smoothed_data[col])
        
        elif filter_type == 'savitzky_golay':
            window_length = kwargs.get('window_length', 5)
            polyorder = kwargs.get('polyorder', 2)
            smoothed_data[col] = savgol_filter(smoothed_data[col], window_length, polyorder)
        
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
    
    smoothed_data.dropna(subset=columns, inplace=True)

    return smoothed_data

##### Functions for RLC Estimators #####

def calculate_accelerating_force(segment, speed_column='vehicle_speed', torque_column='rear_motor_torque', rpm_column='engine_rpm'):
    """
    Calculates the accelerating force for each row in the segment.

    Parameters:
    - segment (pd.DataFrame): The segmented DataFrame.
    - speed_column (str): Name of the vehicle speed column.
    - torque_column (str): Name of the rear motor torque column.
    - rpm_column (str): Name of the engine RPM column.

    Returns:
    - segment (pd.DataFrame): The DataFrame with an added 'accelerating_force' column.
    """
    # Convert engine RPM to radians per second
    omega = segment[rpm_column].values * 2 * np.pi / 60  # rad/s

    # Calculate power in watts
    power = segment[torque_column].values * omega  # W

    # Calculate accelerating force in newtons
    # Avoid division by zero by setting a small epsilon
    epsilon = 1e-3  # m/s
    vehicle_speed = segment[speed_column].values / 3.6  # m/s

    # Calculate force, set to NaN where vehicle_speed < epsilon
    # Suppress division by zero warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        accelerating_force = np.where(
            vehicle_speed >= epsilon,
            power / vehicle_speed,
            np.nan  # or 0 if you prefer
        )

    # Add the 'accelerating_force' column to the segment
    segment['accelerating_force'] = accelerating_force

    # check if data contains pitch_induced_force_N and if so add it to the accelerating force
    if 'pitch_induced_force_N' in segment.columns:
        segment['applied_drivetrain_force'] = segment['accelerating_force'].copy()
        segment['accelerating_force'] -= segment['pitch_induced_force_N']

        #print('added pitch induced force to accelerating force')

    return segment

def calc_force(aggregated_deceleration_df, vehicle_mass, verbose=False):
    """
    Calculates the mean braking force and its standard deviation for each speed bucket.

    Parameters:
    - aggregated_deceleration_df (pd.DataFrame): The aggregated DataFrame containing deceleration statistics.
    - vehicle_mass (float): The mass of the vehicle in kilograms.
    - verbose (bool): If True, prints detailed information during the computation.

    Returns:
    - pd.DataFrame: The input DataFrame with additional columns for mean braking force and its standard deviation.
    """
    import pandas as pd

    # Copy the input DataFrame to avoid modifying the original
    df = aggregated_deceleration_df.copy()

    if verbose:
        print("Calculating braking force for each bucket...")

    # Ensure that 'mean' and 'std_deviation' columns exist in the DataFrame
    required_columns = ['mean', 'std_deviation']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The DataFrame must contain the following columns: {required_columns}")

    # Convert deceleration from (km/h)/s to m/s²
    df['mean_mps2'] = df['mean'] / 3.6
    df['std_mps2'] = df['std_deviation'] / 3.6

    # Calculate mean braking force and its standard deviation for each bucket
    df['mean_force'] = vehicle_mass * df['mean_mps2']
    df['std_force'] = vehicle_mass * df['std_mps2']

    if verbose:
        for index, row in df.iterrows():
            print(f"\nBucket {row['bucket_nr']} (Speed range: {row['bucket_floor']} - {row['bucket_ceiling']} km/h):")
            print(f"  Mean Deceleration: {row['mean']} (km/h)/s => {row['mean_mps2']} m/s²")
            print(f"  Standard Deviation of Deceleration: {row['std_deviation']} (km/h)/s => {row['std_mps2']} m/s²")
            print(f"  Vehicle Mass: {vehicle_mass} kg")
            print(f"  Mean Braking Force: {row['mean_force']} N")
            print(f"  Standard Deviation of Force: {row['std_force']} N")
    return df


# Function to calculate similarity between two sets of parameters
import numpy as np
def calc_loss(f0, f1, f2, f0_gt, f1_gt, f2_gt, type='rel', sec=(0, 230)):
    if type == 'abs':
        loss = abs(f0 - f0_gt) + abs(f1 - f1_gt) + abs(f2 - f2_gt)
    elif type == 'rel':
        # Handle division by zero by adding a small epsilon or skipping calculation
        loss = (
            (abs(f0 - f0_gt) / f0_gt if f0_gt != 0 else float('inf')) +
            (abs(f1 - f1_gt) / f1_gt if f1_gt != 0 else float('inf')) +
            (abs(f2 - f2_gt) / f2_gt if f2_gt != 0 else float('inf'))
        )
    elif type == 'sec_rmse':
        # use parameters to create two quadratic functions in specified range
        x = np.linspace(sec[0], sec[1], 100)
        y_1 = f0 + f1 * x + f2 * x**2
        y_2 = f0_gt + f1_gt * x + f2_gt * x**2
        # calculate root mean squared error
        loss = np.sqrt(np.mean((y_1 - y_2)**2))

    return loss

def calculate_all_losses(f0, f1, f2, loss_type='rel'):
    # Hardcoded ground truth values
    gt_values = {
        'very_low_suspension': (173.9, 0.005, 0.02871),
        'low_suspension': (188.7, 0.064, 0.025),
        'medium_suspension': (138.4, 1.981, 0.01456),
        'high_suspension': (164.3, 0.698, 0.02649),
        'CoC_values': (192.4, 1.2, 0.01976),
        'EPA_values': (169.7, 1.2, 0.02240)
    }

    gt_ranges = {
        'very_low_suspension': (15, 160),
        'low_suspension': (15, 160),
        'medium_suspension': (15, 95),
        'high_suspension': (15, 160),
        'CoC_values': (5, 230),
        'EPA_values': (5, 230)
    }

    losses = {}
    for setting, (f0_gt, f1_gt, f2_gt) in gt_values.items():
        losses[setting] = calc_loss(f0, f1, f2, f0_gt, f1_gt, f2_gt, type=loss_type, sec=gt_ranges[setting])
    return losses

def compute_drag_coefficients(f1, f2, vehicle_mass, frontal_area, air_density=1.225, gravity=9.81):
    """
    Computes the aerodynamic drag coefficient (Cd) and rolling resistance coefficient (Crr)
    based on the fitted parameters from the quadratic model of braking force.

    Parameters:
    - f1: Linear coefficient from the quadratic model fit.
    - f2: Quadratic coefficient from the quadratic model fit.
    - vehicle_mass: Mass of the vehicle in kilograms.
    - frontal_area: Frontal area of the vehicle in square meters.
    - air_density: Air density in kg/m³ (default is 1.225 kg/m³).
    - gravity: Acceleration due to gravity in m/s² (default is 9.81 m/s²).

    Returns:
    - Cd: Aerodynamic drag coefficient.
    - Crr: Rolling resistance coefficient.
    """
    # Compute Cd
    Cd = (2 * f2) / (air_density * frontal_area)

    # Compute Crr
    Crr = f1 / (vehicle_mass * gravity)

    return Cd, Crr


##### Functions for Powertrain Efficiency Analysis #####

def calc_power_mechanical(data, torque_column='rear_motor_torque', rpm_column='engine_rpm'):
    data['power_mechanical'] = data[torque_column] * data[rpm_column] * 2 * np.pi / 60 / 1000
    return data

def calc_power_electrical(data, current_column='hv_battery_current', voltage_column='hv_battery_voltage'):
    data['power_electrical'] = data[current_column] * data[voltage_column] / 1000
    return data

def calc_efficiency(data, substract_auxiliary_power=True, auxiliary_power_column='dcdc_power_hv'):
    if substract_auxiliary_power:
        data['efficiency'] = data['power_mechanical'] / (-data['power_electrical'] - data[auxiliary_power_column]/1000)
    else:
        data['efficiency'] = data['power_mechanical'] / -data['power_electrical']
    return data
    
def quantize(data, n_bins=50):
    data = data.copy()
    bucket_size_torque = (data['rear_motor_torque'].max() / n_bins)
    bucket_size_rpm = (data['engine_rpm'].max() / n_bins)

    # calculate bucket coordinates
    data['torque_bucket'] = (data['rear_motor_torque'] / bucket_size_torque).astype(int)
    data['rpm_bucket'] = (data['engine_rpm'] / bucket_size_rpm).astype(int)
    return data
    
# Function to calculate the efficiency map grid data
def calculate_efficiency_map(data, n_bins=50):
    # Initialize an array to store the average efficiency values for each bucket
    data = data.copy()
    efficiency_map = np.zeros((n_bins, n_bins))

    # Fill the array with the average efficiency values for each bucket
    for i in range(n_bins):
        for j in range(n_bins):
            # Get the data points for the current bucket
            data_bucket = data[(data['torque_bucket'] == i) & (data['rpm_bucket'] == j)]
            # Calculate the average efficiency value for the current bucket
            if not data_bucket.empty:
                efficiency_map[i, j] = data_bucket['efficiency'].mean()
            else:
                efficiency_map[i, j] = np.nan  # Assign NaN if there are no data points in the
    return efficiency_map



##### Functions for Gear Strategy Finder #####

from scipy.spatial.distance import directed_hausdorff, cdist
from frechetdist import frdist

def compute_unique_distances(x_array, y_array):
    """
    Given arrays x_array, y_array describing a line in 2D,
    compute and return an array of cumulative distances
    (arc length) at each point.
    """
    dx = np.diff(x_array)
    dy = np.diff(y_array)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    distance_along_curve = np.concatenate(([0], np.cumsum(segment_lengths)))
    return distance_along_curve

def compute_arc_length(curve):
    """
    Compute cumulative arc length for a curve of shape (n_points, 2).
    """
    deltas = np.diff(curve, axis=0)
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    return arc_length

def compute_hausdorff_distance(curve_ground, curve_driving):
    """
    Compute the Hausdorff distance between two curves (each shape (n,2)).
    """
    d_forward = directed_hausdorff(curve_ground, curve_driving)[0]
    d_backward = directed_hausdorff(curve_driving, curve_ground)[0]
    return max(d_forward, d_backward)

def compute_frechet_distance(curve_ground, curve_driving):
    """
    Compute the Fréchet distance between two curves (each shape (n,2)).
    """
    return frdist(curve_ground, curve_driving)


def compute_chamfer_distance(curve_ground, curve_driving):
    """
    Compute the Chamfer distance between two curves (each shape (n,2)).
    """
    distances = cdist(curve_ground, curve_driving)
    min_dist_ground = np.min(distances, axis=1)
    min_dist_driving = np.min(distances, axis=0)
    chamfer_distance = np.mean(min_dist_ground) + np.mean(min_dist_driving)
    return chamfer_distance

def compute_parametric_metrics(curve_ground, curve_driving):
    """
    Compute parametric metrics (MSE, RMSE, MAE, Max Error, Area Between Curves)
    by interpolating both curves onto a common arc-length parameter.
    """
    t_ground = compute_arc_length(curve_ground)
    t_driving = compute_arc_length(curve_driving)

    # Normalize t to a common range
    t_min = max(t_ground.min(), t_driving.min())
    t_max = min(t_ground.max(), t_driving.max())
    common_t = np.linspace(t_min, t_max, 500)

    # Interpolate x,y for each curve over common_t
    x_ground = np.interp(common_t, t_ground, curve_ground[:, 0])
    y_ground = np.interp(common_t, t_ground, curve_ground[:, 1])

    x_driving = np.interp(common_t, t_driving, curve_driving[:, 0])
    y_driving = np.interp(common_t, t_driving, curve_driving[:, 1])

    # Distances between corresponding points
    differences = np.hypot(x_ground - x_driving, y_ground - y_driving)

    mse = np.mean(differences ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(differences)
    max_error = np.max(differences)
    area_between_curves = np.trapz(differences, common_t)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Max Error': max_error,
        'Area Between Curves': area_between_curves
    }

##### Functions for Sim Pipeline #####

def drag_force(v_mps, f0_driving_mps, f1_driving_mps, f2_driving_mps,
               f0_lab_mps, f1_lab_mps, f2_lab_mps):
    """
    Compute the drag forces for both 'driving' and 'lab' given velocity in m/s.
    
    Returns
    -------
    (drag_driving, drag_lab) : tuple of float or np.array
    """
    drag_driving = (f0_driving_mps
                    + f1_driving_mps * v_mps
                    + f2_driving_mps * (v_mps ** 2))
    drag_lab = (f0_lab_mps
                + f1_lab_mps * v_mps
                + f2_lab_mps * (v_mps ** 2))
    return drag_driving, drag_lab