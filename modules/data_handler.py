import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import griddata, interp1d
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import numpy as np
import os

##### General functions #####

# Initialize the data cache
data_cache = {}

def load_can_data(filename, verbose=True):
    # Check if the data is already in the cache
    if filename in data_cache:
        if verbose:
            print(f"Using cached data for {filename}")
        #print(f"Yay! Using cached data for {filename}")
        # Return data from cache and skip the rest of the function
        return data_cache[filename]

    # Load the CAN data
    data = pd.read_csv(f'data/{filename}') 
    #print(f"Loaded {filename} from disk")
    data.drop(columns=['Unnamed: 0'], inplace=True)

    # Fix the column names and unify them across different types of vehicles
    if 'axle_torque_nominal' in data.columns:
        ID3 = True
        if verbose:
            print(f'Found ID3 data in {filename}')
            print('Renaming torque, voltage and aux power columns...')
        # rename column axle_torque_nominal to rear_axle_torque
        data.rename(columns={'axle_torque_nominal': 'rear_motor_torque'}, inplace=True)
    else:
        ID3 = False

    if ID3:
        # rename hv_battery_voltage_y to hv_battery_voltage
        data.rename(columns={'hv_battery_voltage_x': 'hv_battery_voltage'}, inplace=True)
        # rename hv_aux_power to dcdc_power_hv
        data.rename(columns={'hv_aux_power': 'dcdc_power_hv'}, inplace=True)

    if ID3:
        if verbose:
            print('Dropping cell voltage columns...')
        # drop all columns that contain 'cell'
        cell_columns = [col for col in data.columns if 'cell_voltage' in col]
        data.drop(columns=cell_columns, inplace=True)
        if verbose:
            print(f'Dropping pack_temp columns...')
        # drop all columns that contain 'pack_temp'
        pack_temp_columns = [col for col in data.columns if 'pack_temp' in col]
        data.drop(columns=pack_temp_columns, inplace=True)

    if ID3:
        if verbose:
            print('Creating new dummy gear_selected_neutral column...')
        # create new dummy column gear_selected_neutral
        # set new column to 8 if gear_selected is 1 and to 1 otherwise
        data['gear_selected_neutral'] = data['gear_selected'].apply(lambda x: 8 if x == 1 else 1)
        if verbose:
            print('Adding Dummy suspension_level column...')
        # create new dummy column suspension_level
        data['suspension_level'] = 'vw'
        if 'brake_pedal_activated' not in data.columns:
            if verbose:
                print('Adding dummy brake_pedal_activated column...')
            data['brake_pedal_activated'] = 0

        

    # check if data contains column hv_battery_voltage and if so rename it to hv_battery_voltage_updt
    if 'hv_battery_voltage_updt' in data.columns:
        data.rename(columns={'hv_battery_voltage_updt': 'hv_battery_voltage'}, inplace=True)
        if verbose:
            print(f'Renamed column hv_battery_voltage_updt to hv_battery_voltage')

    # check if data contains column vehicle_speed_test8 and if so rename it to vehicle_speed
    if 'vehicle_speed_test8' in data.columns:
        data.rename(columns={'vehicle_speed_test8': 'vehicle_speed'}, inplace=True)
        if verbose:
            print(f'Renamed column vehicle_speed_test8 to vehicle_speed')

    if 'gear_selected_Neutral' in data.columns:
        data.rename(columns={'gear_selected_Neutral': 'gear_selected_neutral'}, inplace=True)
        if verbose:
            print(f'Renamed column gear_selected_Neutral to gear_selected_neutral')

    if 'motor_torque_test_3' in data.columns:
        #data['rear_motor_torque'] = data['motor_torque_test_3'] # optionally choose new signal
        # remove the column motor_torque_test_3
        data.drop(columns=['motor_torque_test_3'], inplace=True)
        if verbose:
            print('Found better torque signal: motor_torque_test_3, renamed to rear_motor_torque')
    
    if 'gearbox_speed_input' in data.columns:
        data['engine_rpm'] = data['gearbox_speed_input']
        # remove the column gearbox_speed_input
        data.drop(columns=['gearbox_speed_input'], inplace=True)
        if verbose:
            print('Found better rpm signal: gearbox_speed_input, renamed to engine_rpm')

    data['timestamp'] = pd.to_datetime(data['time'], unit='ms')
    data['seconds'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds()

    # remove the last 4 characters (.csv) and add _IMU.csv
    imu_filename = filename[:-4] + '_IMU.csv'

    # check if IMU file exists
    if os.path.isfile('data/'+ imu_filename):
        if verbose:
            print(f'Found IMU file: {imu_filename}')

        # load IMU data
        imu_data = pd.read_csv('data/' + imu_filename, sep=';', decimal=',')
        imu_data.drop(index=0, inplace=True)

        # convert 'Time (HH:mm:ss.fff)' column to datetime and then to seconds
        imu_data['time'] = pd.to_datetime(imu_data['Time (HH:mm:ss.fff)'], format='%H:%M:%S.%f')

        imu_data['seconds'] = (imu_data['time'] - imu_data['time'].min()).dt.total_seconds()
        imu_data['vehicle_speed_gps'] = imu_data['Speed 3D with hold (m/s)']*3.6

        # fix pitch
        imu_data['pitch'] = imu_data['Pitch (deg)']+ imu_data['Pitch misalignment angle (deg)']

        # fix heading
        imu_data['heading'] = imu_data['Heading (deg)'] + imu_data['Heading misalignment angle (deg)']
        
        # drop columns that are not needed
        columns_to_drop = ['Time (HH:mm:ss.fff)', 'Speed 3D with hold (m/s)', 'Time (GPS s)', 'UTC offset (s)', 
                   'IMU time difference ()','time', 'Pitch (deg)', 'Pitch accuracy (deg)', 
                   'Pitch misalignment angle (deg)', 'Pitch misalignment angle accuracy (deg)', 
                   'Heading misalignment angle (deg)', 'Heading misalignment angle accuracy (deg)', 
                   'Heading (deg)', 'Heading accuracy (deg)']

        # Drop only the columns that exist in the DataFrame
        imu_data.drop(columns=[col for col in columns_to_drop if col in imu_data.columns], inplace=True)

        # resampling to main data frequency
        target_frequency = 1 / (data['seconds'][1]-data['seconds'][0])
        imu_data = resample_imu_data(imu_data, target_frequency)

        # find the best offset using MSE on vehicle speed
        offsets = (-30, 30)
        best_offset = find_best_offset(data, imu_data, offsets)

        # shift the imu data by the best offset
        imu_data['seconds'] += best_offset

        if verbose:
            print(f"Best time offset found: {round(best_offset,2)} seconds, merging...")

        # merge dataframes on secods using a common time range
        data = pd.merge_asof(data.sort_values('seconds'), imu_data.sort_values('seconds'), on='seconds', direction='nearest')

            
    else:
        if verbose:
            print(f'No IMU data for: {filename}')
            imu_data = None

    if verbose:
        print(f'Loaded {filename}, shape: {data.shape}, with columns: {data.columns}')

    duration = data['seconds'].max() - data['seconds'].min()
    duration = datetime.timedelta(seconds=duration)
    if verbose:
        print(f'duration of recording: {duration}')

    # Store the data in the cache before returning
    data_cache[filename] = data

    return data


def load_all_data(verbose=False, list_of_files=None, directory='data/', mark_ends=False, exclude_keywords=None):
    if list_of_files is None:
        list_of_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    # Check both the hardcoded exclusions and the optional exclude_keywords
                    exclude_conditions = ['_IMU', 'mobile', 'WLTP', 'log', '2024_09_21-22_05_11', '_vertices', '2024_10_10', 'exclude', 'sim_efficiency_maps']
                    if exclude_keywords:
                        exclude_conditions.extend(exclude_keywords)

                    if all(x not in full_path for x in exclude_conditions):
                        list_of_files.append(full_path[5:])
        if verbose:
            print(list_of_files)

    master_dataframe = pd.DataFrame()

    for file in list_of_files:
        data = load_can_data(file, verbose=verbose)
        if mark_ends:
            data['file_ends'] = False
            data.loc[-1, 'file_ends'] = True
        master_dataframe = pd.concat([master_dataframe, data])

    if True:
        # calculate total duration of data
        seconds = len(master_dataframe) / 10
        minutes = seconds / 60
        hours = minutes / 60
        print(f'Loaded {len(list_of_files)} files, worth {round(hours, 2)} hours of data')

    return master_dataframe


# function to retrieve a list of csv files in given folder that dont contain _IMU
def get_can_files(folder='data/', exclude_keywords=None):
    import os
    list_of_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                #standard_exclusions = ['_IMU', 'mobile', 'WLTP', 'log', '2024_09_21-22_05_11', '_vertices', '2024_10_10', 'exclude']
                standard_exclusions = ['_IMU', 'mobile', 'WLTP', 'log', '_vertices', '2024_10_10', 'exclude', 'sim_efficiency_maps']
                if exclude_keywords:
                    standard_exclusions.extend(exclude_keywords)
                if all(x not in full_path for x in standard_exclusions):
                    list_of_files.append(full_path[5:])
    return list_of_files

def resample_imu_data(imu_data, target_frequency):
    # Set the new sampling interval for resampling based on the target frequency
    new_interval = 1 / target_frequency
    
    # Generate new time points for resampling
    new_time_points = np.arange(imu_data['seconds'].min(), imu_data['seconds'].max(), new_interval)
    
    # Resample all columns using interpolation
    resampled_data = {}
    for column in imu_data.columns:
        if column == 'seconds':
            resampled_data[column] = new_time_points
        else:
            interp_func = interp1d(imu_data['seconds'], imu_data[column], kind='linear', fill_value='extrapolate')
            resampled_data[column] = interp_func(new_time_points)
    
    # Create new resampled IMU dataframe
    resampled_imu_data = pd.DataFrame(resampled_data)
    
    return resampled_imu_data

def compute_goodness_of_fit(signal_1, signal_2):
    # Compute the mean squared error between two signals
    mse = mean_squared_error(signal_1, signal_2)
    return mse

def find_best_offset(data, imu_data, offsets):
    # Function to calculate the error between two signals given an offset
    def objective(offset):
        shifted_seconds = imu_data['seconds'] + offset
        # Use interpolation to estimate vehicle speed at shifted seconds
        interp_func = interp1d(shifted_seconds, imu_data['vehicle_speed_gps'], kind='linear', fill_value='extrapolate')
        interpolated_vehicle_speed_gps = interp_func(data['seconds'])
        return compute_goodness_of_fit(data['vehicle_speed'], interpolated_vehicle_speed_gps)
    
    # Find the offset that minimizes the goodness of fit error
    best_fit = minimize(objective, x0=0, bounds=[(offsets[0], offsets[1])])
    return best_fit.x[0]

import scipy.io

def load_driving_cycle(cycle_name):
    if cycle_name in ['wltc', 'wltc_city', 'ftp75', 'ftm_ul', 'ftm_city', 'ftm_hw', 'hwfet']:
        cycle_name_path_dict = {
            'wltc': 'data/driving_cycles/WLTC_Class3b.mat',
            'wltc_city': 'data/driving_cycles/WLTC_Class3b_city.mat',
            'ftp75': 'data/driving_cycles/FTP75.mat',
            'ftm_ul': 'data/driving_cycles/ftm_ul_40.mat',
            'ftm_city': 'data/driving_cycles/ftm_city_50.mat',
            'ftm_hw': 'data/driving_cycles/ftm_hw_15.mat',
            'hwfet': 'data/driving_cycles/HWFET_single.mat'
        }

        data = scipy.io.loadmat(cycle_name_path_dict[cycle_name])
        key = list(data.keys())[-1]
        driving_cycle = data[key]
        if cycle_name == 'ftm_ul':
           driving_cycle = driving_cycle[0:int(len(driving_cycle)/40)]
        elif cycle_name == 'ftm_city':
            driving_cycle = driving_cycle[0:int(len(driving_cycle)/50)]
        elif cycle_name == 'ftm_hw':
            driving_cycle = driving_cycle[0:int(len(driving_cycle)/15)]
        sim_data = pd.DataFrame()
        sim_data['time'] = driving_cycle[:,0]
        sim_data['velocity_profile'] = driving_cycle[:,1]
    else:
        sim_data_temp = load_can_data(cycle_name)
        sim_data = pd.DataFrame()
        sim_data['time'] = sim_data_temp['time']
        sim_data['velocity_profile'] = sim_data_temp['vehicle_speed']
        # make time start at 0 and convert to seconds
        sim_data['time'] = (sim_data['time'] - sim_data['time'].min())/1000
        
    return sim_data

def plot_overview(data, figsize=(10, 5), columns=None):
    if columns is None:
        for column in data.columns:
            if column != 'timestamp' and column != 'time' and column != 'seconds':
                plt.plot(data['seconds'], data[column])
                plt.title(column)
                plt.xlabel('time')
                plt.ylabel(column)
                plt.grid()
                # set figsize
                plt.gcf().set_size_inches(figsize)
                plt.show()
                plt.clf()
    else:
        for column in columns:
            plt.plot(data['seconds'], data[column])
            plt.title(column)
            plt.xlabel('time')
            plt.ylabel(column)
            plt.grid()
            # set figsize
            plt.gcf().set_size_inches(figsize)
            plt.show()
            plt.clf()


##### Functions for Grid Search Analysis #####
def parse_and_load_parameters(best_parameters):
    import pandas as pd
    import ast

    # Define the parameter parsing function
    def parse_parameter(value, expected_type):
        if expected_type == 'ignore':
            return None
        if pd.isna(value):
            return None
        if expected_type == 'eval':
            if isinstance(value, str):
                try:
                    # Safely evaluate the string to the actual data type
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return value  # Return the string as is if it can't be evaluated
            else:
                return value
        elif expected_type == 'eval_or_none':
            if pd.isna(value):
                return None
            elif isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return value
            else:
                return value
        elif expected_type == 'int_or_none':
            if pd.isna(value):
                return None
            else:
                return int(float(value))
        elif expected_type == 'int_or_str':
            try:
                return int(float(value))
            except ValueError:
                return value
        elif expected_type == 'int':
            return int(float(value))
        elif expected_type == 'float':
            return float(value)
        elif expected_type == 'bool':
            if isinstance(value, str):
                if value.lower() in ('true', '1'):
                    return True
                elif value.lower() in ('false', '0'):
                    return False
                else:
                    raise ValueError(f"Cannot interpret '{value}' as bool")
            else:
                return bool(value)
        elif expected_type == 'str':
            value = value.replace('\\', '/')
            return str(value)
        else:
            return value

    # Add new parameters for the gear strategy pipeline
    parameters_info = {
        # Common parameters
        'files': 'eval',  # This will be parsed as a list of strings
        'generate_plots': 'bool',
        'verbose': 'bool',

        # Efficiency map specific parameters
        'data_cache': 'ignore',
        'gear': 'int_or_str',  
        'efficiency_limit_lower': 'float',
        'efficiency_limit_upper': 'float',
        'soc_limit_lower': 'float',
        'soc_limit_upper': 'float',
        'remove_neutral_gear': 'bool',
        'smoothing_kwargs': 'eval_or_none',
        'columns_to_smooth': 'eval_or_none',
        'substract_auxiliary_power': 'bool',
        'which_full_load_curve': 'str',
        'twoD_smoothing_kwargs': 'eval_or_none',
        'high_fidelity_interpolation': 'bool',
        'n_quantize_bins': 'int',
        'at_middle_of_bin': 'bool',
        'n_interpolation_bins': 'int',
        'global_offset': 'float',

        # Coastdown specific parameters
        'do_pitch_correction': 'bool',
        'speed_signal': 'str',
        'bucket_size': 'int',
        'vehicle_mass': 'float',
        'rotating_mass_eq': 'float',
        'frontal_area': 'float',
        'steering_angle_limit': 'int_or_none',
        'select_suspension_level': 'int_or_none',
        'select_spoiler_level': 'int_or_none',
        'cut_time': 'int_or_none',
        'filter_for_acceleration': 'bool',

        # Constspeed pipeline specific parameters
        'speed_threshold': 'float',
        'min_n_samples': 'int',
        'min_avg_speed': 'float',

        # Gear strategy pipeline specific parameters
        'gear_change_offset_samples': 'int',
        'outlier_eps': 'float',
        'outlier_min_samples': 'int',
        'close_points_merge_thr': 'float',
        'n_clusters': 'int',
        'attach_endpoints': 'bool',
        'anchor_points': 'bool',
        'spline_order': 'int',
        'num_knots': 'int',
        'cluster_weight': 'float',
        'normal_weight': 'float',
        'knot_distr_method': 'str',

        # Gear classifier parameters (if any remain in code)
        'n_samples_around_event': 'int_or_none',
        'remove_gear_8_events': 'bool',
        'remove_gear_0_events': 'bool',
        'signals': 'eval_or_none',
        'sample_offsets': 'eval_or_none',
        'label_strategy': 'str',
        'test_size': 'float',
        'random_state': 'int',
        'stratify': 'bool',
        'hyperparameters': 'eval'
    }

    # Parse parameters
    parsed_parameters = {}
    for name in parameters_info:
        if name in best_parameters.index:
            value = best_parameters[name]
            expected_type = parameters_info[name]
            parsed_value = parse_parameter(value, expected_type)
            if expected_type != 'ignore':
                parsed_parameters[name] = parsed_value

    return parsed_parameters


##### Functions for Powertrain Efficiency Analysis #####

# Quantization to regular grid average out values with standard deviation calculation
import numpy as np
import pandas as pd

def quantize_data(data, rpm_column, torque_column, n_bins=100, at_middle_of_bin=False):
    n_bins_rpm = n_bins
    n_bins_torque = n_bins
    # Get the min and max values for the columns
    rpm_min = data[rpm_column].min()
    rpm_min = 0
    rpm_max = data[rpm_column].max()
    torque_min = data[torque_column].min()
    torque_min = 0
    torque_max = data[torque_column].max()

    # Create the bins and calculate the bin width
    bins_rpm = np.linspace(rpm_min, rpm_max, n_bins_rpm)
    bins_torque = np.linspace(torque_min, torque_max, n_bins_torque)

    # List to store each row of quantized data
    quantized_data = []

    for i in range(n_bins_rpm - 1):
        for j in range(n_bins_torque - 1):
            rpm_low = bins_rpm[i]
            rpm_high = bins_rpm[i + 1]
            torque_low = bins_torque[j]
            torque_high = bins_torque[j + 1]

            # Filter data within the current bin
            data_bin = data[
                (data[rpm_column] >= rpm_low) & (data[rpm_column] < rpm_high) &
                (data[torque_column] >= torque_low) & (data[torque_column] < torque_high)
            ]

            # Number of data points in the bin
            n_points = data_bin.shape[0]

            if n_points > 0:
                # Calculate mean and standard deviation for each column within the bin
                data_mean = data_bin.select_dtypes(include=[np.number]).mean()
                efficiency_std = data_bin['efficiency'].std()  # Standard deviation for efficiency

                if at_middle_of_bin:
                    data_mean[rpm_column] = (rpm_low + rpm_high) / 2
                    data_mean[torque_column] = (torque_low + torque_high) / 2
                else:
                    data_mean[rpm_column] = rpm_low
                    data_mean[torque_column] = torque_low

                data_mean['n_points'] = n_points
                data_mean['efficiency_std'] = efficiency_std  # Add standard deviation to the result

                # Append the row to the list
                quantized_data.append(data_mean)

    # Convert the list of rows into a DataFrame
    data_quantized = pd.DataFrame(quantized_data)

    return data_quantized

from scipy.interpolate import griddata
import pandas as pd

def interpolate_efficiency_map(
    data_quantized, 
    n_bins=20, 
    rpm_column='engine_rpm', 
    torque_column='rear_motor_torque', 
    efficiency_column='efficiency',
    efficiency_diff_threshold=0.1,
    n_points_limit=5,
    efficiency_std_limit=0.05
):
    """
    Interpolates the efficiency map from quantized data.

    Parameters:
    - data_quantized: DataFrame containing the quantized data with columns for RPM, Torque, Efficiency, n_points, and efficiency_std.
    - n_bins: Number of bins to use for the grid in each dimension (RPM and Torque).
    - rpm_column: Name of the RPM column in the data.
    - torque_column: Name of the Torque column in the data.
    - efficiency_column: Name of the Efficiency column in the data.
    - apply_outlier_filtering: Boolean flag to apply outlier filtering.
    - efficiency_diff_threshold: Threshold for efficiency difference in outlier filtering.
    - n_points_limit: Minimum number of data points for a quantized point to be considered reliable.
    - efficiency_std_limit: Maximum standard deviation of efficiency for a quantized point to be considered reliable.

    Returns:
    - interpolated_data: DataFrame containing the interpolated data, compatible with efficiency_scatterplot function.
    """
    # Normalize RPM and Torque
    rpm_min = data_quantized[rpm_column].min()
    rpm_min = 0
    rpm_max = data_quantized[rpm_column].max()
    rpm_max = 14000
    torque_min = data_quantized[torque_column].min()
    torque_min = 0
    torque_max = data_quantized[torque_column].max()
    torque_max = 330
    
    data_quantized['rpm_norm'] = (data_quantized[rpm_column] - rpm_min) / (rpm_max - rpm_min)
    data_quantized['torque_norm'] = (data_quantized[torque_column] - torque_min) / (torque_max - torque_min)

    # Create normalized grid
    grid_rpm_norm = np.linspace(0, 1, n_bins)
    grid_torque_norm = np.linspace(0, 1, n_bins)
    RPM_norm_grid, Torque_norm_grid = np.meshgrid(grid_rpm_norm, grid_torque_norm)

    # Prepare data points and values for interpolation
    points = np.vstack((data_quantized['rpm_norm'], data_quantized['torque_norm'])).T
    efficiency_values = data_quantized[efficiency_column]
    n_points_values = data_quantized['n_points']
    efficiency_std_values = data_quantized['efficiency_std']

    # Interpolate efficiency
    efficiency_grid = griddata(
        points, efficiency_values, (RPM_norm_grid, Torque_norm_grid), method='linear'
    )

    # Interpolate n_points and efficiency_std
    n_points_grid = griddata(
        points, n_points_values, (RPM_norm_grid, Torque_norm_grid), method='linear'
    )
    efficiency_std_grid = griddata(
        points, efficiency_std_values, (RPM_norm_grid, Torque_norm_grid), method='linear'
    )

    # Fill NaNs with nearest neighbor interpolation
    def fill_nan_with_nearest(grid, points, values):
        nan_mask = np.isnan(grid)
        grid[nan_mask] = griddata(
            points, values, (RPM_norm_grid, Torque_norm_grid), method='nearest'
        )[nan_mask]
        return grid

    efficiency_grid = fill_nan_with_nearest(efficiency_grid, points, efficiency_values)
    n_points_grid = fill_nan_with_nearest(n_points_grid, points, n_points_values)
    efficiency_std_grid = fill_nan_with_nearest(efficiency_std_grid, points, efficiency_std_values)

    # Flatten the grids and un-normalize RPM and Torque
    rpm_norm_flat = RPM_norm_grid.flatten()
    torque_norm_flat = Torque_norm_grid.flatten()
    efficiency_flat = efficiency_grid.flatten()
    n_points_flat = n_points_grid.flatten()
    efficiency_std_flat = efficiency_std_grid.flatten()

    # Un-normalize RPM and Torque
    rpm_values = rpm_norm_flat * (rpm_max - rpm_min) + rpm_min
    torque_values = torque_norm_flat * (torque_max - torque_min) + torque_min

    # Create DataFrame
    interpolated_data = pd.DataFrame({
        rpm_column: rpm_values,
        torque_column: torque_values,
        efficiency_column: efficiency_flat,
        'n_points': n_points_flat,
        'efficiency_std': efficiency_std_flat
    })

    return interpolated_data

def fill_efficiency_map(df):
    import pandas as pd
    import numpy as np
    # Pivot the DataFrame to create a 2D grid
    efficiency_pivot = df.pivot(index='Torque', columns='RPM', values='Efficiency')
    
    # Get the index and columns values
    torque_values = efficiency_pivot.index.values
    rpm_values = efficiency_pivot.columns.values
    
    # Compute maximum values to normalize the scales
    max_Torque = torque_values.max()
    max_RPM = rpm_values.max()
    
    # Create meshgrid of RPM and Torque values
    RPM_grid, Torque_grid = np.meshgrid(rpm_values, torque_values)
    
    # Normalize RPM and Torque
    RPM_normalized = RPM_grid / max_RPM
    Torque_normalized = Torque_grid / max_Torque
    
    # Create a mask where normalized RPM >= normalized Torque (below or on the 45-degree line)
    rpm_gte_torque = RPM_normalized >= Torque_normalized
    
    # Convert the boolean mask to a DataFrame
    mask_rpm_gte_torque = pd.DataFrame(
        rpm_gte_torque, index=efficiency_pivot.index, columns=efficiency_pivot.columns
    )
    
    # Create the complementary mask for RPM < Torque (above the 45-degree line)
    mask_rpm_lt_torque = ~mask_rpm_gte_torque
    
    # Initialize filled DataFrames for both regions
    filled_efficiency_ge = efficiency_pivot.where(mask_rpm_gte_torque)
    filled_efficiency_lt = efficiency_pivot.where(mask_rpm_lt_torque)
    
    # For RPM >= Torque (Below 45-degree line):
    # Fill NaNs to the right in each row (extend values towards higher RPM)
    filled_efficiency_ge = filled_efficiency_ge.ffill(axis=1)
    # Fill NaNs downwards in each column (extend values towards lower Torque)
    filled_efficiency_ge = filled_efficiency_ge.bfill(axis=0)
    
    # For RPM < Torque (Above 45-degree line):
    # Fill NaNs upwards in each column (extend values towards higher Torque)
    filled_efficiency_lt = filled_efficiency_lt.ffill(axis=0)
    # Fill NaNs to the left in each row (extend values towards lower RPM)
    filled_efficiency_lt = filled_efficiency_lt.bfill(axis=1)
    
    # Combine the two filled DataFrames
    efficiency_filled = filled_efficiency_ge.combine_first(filled_efficiency_lt)
    
    # Fill any remaining NaNs using both forward and backward fill methods
    efficiency_filled = efficiency_filled.ffill(axis=1).ffill(axis=0).bfill(axis=1).bfill(axis=0)
    
    # Reset index to return to original DataFrame structure
    filled_df = efficiency_filled.reset_index().melt(
        id_vars='Torque', var_name='RPM', value_name='Efficiency'
    )
    
    # Ensure data types match the original DataFrame
    filled_df['RPM'] = filled_df['RPM'].astype(df['RPM'].dtype)
    filled_df['Torque'] = filled_df['Torque'].astype(df['Torque'].dtype)
    filled_df['Efficiency'] = filled_df['Efficiency'].astype(df['Efficiency'].dtype)
    
    # Sort the DataFrame to match original order (optional)
    filled_df = filled_df.sort_values(by=['RPM', 'Torque']).reset_index(drop=True)
    
    return filled_df


def find_enclosing_fullload_curve(data, rpm_column, torque_column, buffer=0):
    import numpy as np
    import pandas as pd
    from shapely.geometry import Polygon
    from scipy.spatial import ConvexHull

    # Step 1: Prepare Data
    temporary_data = data.copy()
    temporary_data['engine_power'] = (
        temporary_data[rpm_column] * 2 * np.pi / 60
    ) * temporary_data[torque_column] / 1000
    temporary_data = temporary_data[temporary_data['engine_power'] > 0]
    points = temporary_data[[rpm_column, 'engine_power']].values

    # Step 2: Compute Convex Hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Step 3: Recalculate Torque
    engine_torque = (hull_points[:, 1] * 60 * 1000) / (2 * np.pi * hull_points[:, 0])
    hull_vertices_torque_rpm = np.column_stack((hull_points[:, 0], engine_torque))

    # Convert hull vertices to a DataFrame
    hull_vertices_torque_rpm_df = pd.DataFrame(hull_vertices_torque_rpm, columns=['RPM', 'Torque'])

    # Step 4: Close the loop
    hull_vertices_torque_rpm_df = pd.concat([hull_vertices_torque_rpm_df, hull_vertices_torque_rpm_df.iloc[0:1]], ignore_index=True)

    # Step 5: Normalize the final points (for uniform buffering)
    if buffer != 0:
        # Normalize the points
        max_rpm = hull_vertices_torque_rpm_df['RPM'].max()
        max_torque = hull_vertices_torque_rpm_df['Torque'].max()
        normalized_points = hull_vertices_torque_rpm_df.copy()
        normalized_points['RPM'] /= max_rpm
        normalized_points['Torque'] /= max_torque

        # Create polygon and buffer
        polygon = Polygon(normalized_points[['RPM', 'Torque']].values)
        buffer_fraction = buffer / 100.0  # Convert percentage to fraction
        buffered_polygon = polygon.buffer(buffer_fraction)

        # Extract buffered coordinates
        buffered_coords = np.array(buffered_polygon.exterior.coords)

        # Rescale back to original scale
        buffered_coords[:, 0] *= max_rpm
        buffered_coords[:, 1] *= max_torque

        # Update the DataFrame with buffered points
        hull_vertices_torque_rpm_df = pd.DataFrame(buffered_coords, columns=['RPM', 'Torque'])

    return hull_vertices_torque_rpm_df

def dataframe_to_arrays(df):
    import numpy as np
    import pandas as pd
    """
    Transforms a DataFrame with 'Torque', 'RPM', and 'Efficiency' columns
    into three 2D numpy arrays representing the grid.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    RPM_dd (np.ndarray): 2D array of RPM values.
    Torque_dd (np.ndarray): 2D array of Torque values.
    Eff_T2W_dd (np.ndarray): 2D array of Efficiency values.
    """
    # Pivot the DataFrame to create a 2D grid
    eff_pivot = df.pivot(index='Torque', columns='RPM', values='Efficiency')

    # Extract the sorted unique Torque and RPM values
    torque_values = eff_pivot.index.to_numpy()
    rpm_values = eff_pivot.columns.to_numpy()

    # Create meshgrid to obtain RPM_dd and Torque_dd arrays
    RPM_dd, Torque_dd = np.meshgrid(rpm_values, torque_values)

    # Extract the Efficiency values as a numpy array
    Eff_T2W_dd = eff_pivot.to_numpy()

    return RPM_dd, Torque_dd, Eff_T2W_dd


def smooth_2d_data(data, x_col, y_col, z_col, method='idw', **kwargs):
    """
    Smooths 2D data (z values over x and y coordinates) using the specified method.

    Parameters:
        data (pd.DataFrame): The input dataframe containing the data to be smoothed.
        x_col (str): Name of the column representing the x-coordinate.
        y_col (str): Name of the column representing the y-coordinate.
        z_col (str): Name of the column representing the z-values to be smoothed.
        method (str): The smoothing method to use ('idw', 'gaussian_filter', 'griddata', 'regression').
        **kwargs: Additional parameters for the specified smoothing method.

            For 'idw' (Inverse Distance Weighting):
                - power (float): Power parameter for IDW weighting (default 2).
                - num_neighbors (int): Number of nearest neighbors to consider (default 10).
                - outlier_detection (bool): Whether to perform outlier detection (default False).
                - threshold_multiplier (float): Multiplier for std deviation in outlier detection (default 2).

            For 'gaussian_filter':
                - sigma (float): Standard deviation for Gaussian kernel.
                - grid_size (int): Size of the grid to interpolate onto.

            For 'griddata':
                - method (str): Interpolation method ('linear', 'cubic', 'nearest').

            For 'regression':
                - model (sklearn regressor): Regression model instance (e.g., RandomForestRegressor()).
                - model_params (dict): Parameters for the regression model.

    Returns:
        smoothed_data (pd.DataFrame): DataFrame with an additional column '{z_col}_smoothed'.
        mean_abs_change (float): Mean absolute change in z_col due to smoothing.
    """
    import numpy as np
    import pandas as pd
    from scipy.interpolate import griddata
    from scipy.spatial import KDTree
    from scipy.ndimage import gaussian_filter
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor

    smoothed_data = data.copy()
    coords = smoothed_data[[x_col, y_col]].values
    values = smoothed_data[z_col].values

    if method == 'idw':
        power = kwargs.get('power', 2)
        num_neighbors = kwargs.get('num_neighbors', 10)
        outlier_detection = kwargs.get('outlier_detection', False)
        threshold_multiplier = kwargs.get('threshold_multiplier', 2)
    
        # Build KDTree for efficient neighbor search
        tree = KDTree(coords)
        smoothed_values = np.zeros_like(values)
    
        for i, point in enumerate(coords):
            # Find nearest neighbors
            distances, idx = tree.query(point, k=num_neighbors + 1)
            distances = distances[1:]  # Exclude the point itself
            idx = idx[1:]
            neighbor_values = values[idx]
    
            # Handle zero distances
            if np.any(distances == 0):
                smoothed_values[i] = neighbor_values[distances == 0][0]
                continue
            
            weights = 1 / distances ** power
    
            if outlier_detection:
                # Compute mean and standard deviation
                mean_val = np.average(neighbor_values, weights=weights)
                std_val = np.sqrt(np.average((neighbor_values - mean_val) ** 2, weights=weights))
                threshold = threshold_multiplier * std_val
                
                # Identify valid neighbors
                valid = np.abs(neighbor_values - mean_val) <= threshold
                
                if np.any(valid):
                    # Update weights and neighbors if valid neighbors exist
                    weights = weights[valid]
                    neighbor_values = neighbor_values[valid]
                else:
                    # If no valid neighbors, fallback to original value
                    weights = 1 / distances ** power  # Reset weights to all neighbors
                    neighbor_values = values[idx]
    
            # Calculate smoothed value
            smoothed_values[i] = np.sum(weights * neighbor_values) / np.sum(weights)

    elif method == 'gaussian_filter':
        sigma = kwargs.get('sigma', 1)
        grid_size = kwargs.get('grid_size', 100)

        # Create grid
        xi = np.linspace(smoothed_data[x_col].min(), smoothed_data[x_col].max(), grid_size)
        yi = np.linspace(smoothed_data[y_col].min(), smoothed_data[y_col].max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate onto grid
        zi = griddata(coords, values, (xi, yi), method='linear')

        # Apply Gaussian filter
        zi_filtered = gaussian_filter(zi, sigma=sigma)

        # Map back to original points
        smoothed_values = griddata((xi.flatten(), yi.flatten()), zi_filtered.flatten(), coords, method='linear')

    elif method == 'griddata':
        interp_method = kwargs.get('interp_method', 'cubic')

        # Interpolate using griddata
        smoothed_values = griddata(coords, values, coords, method=interp_method)

    elif method == 'regression':
        model = kwargs.get('model', 'random_forest')
        if model == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
        model_params = kwargs.get('model_params', {})
        model.set_params(**model_params)

        # Fit the model
        model.fit(coords, values)
        smoothed_values = model.predict(coords)

    else:
        raise ValueError(f"Unsupported smoothing method: {method}")

    # Handle any NaNs resulting from interpolation
    nan_mask = np.isnan(smoothed_values)
    smoothed_values[nan_mask] = values[nan_mask]

    smoothed_data[f'{z_col}_smoothed'] = smoothed_values
    mean_abs_change = np.mean(np.abs(values - smoothed_values))

    # overwrite the original column
    smoothed_data[z_col] = smoothed_data[f'{z_col}_smoothed']

    return smoothed_data, mean_abs_change



##### Functions for Coastdown Analysis #####

def segment_df_by_drivemode(data, drop_keys=None):
    data = data.copy()
    data['change'] = data['gear_selected_neutral'].diff().ne(0)
    data['group'] = data['change'].cumsum()

    drive_segments = {idx: group.drop(columns=['change', 'group']) for idx, group in data[data['gear_selected_neutral'] == 1].groupby('group')}

    coast_segments = {idx: group.drop(columns=['change', 'group']) for idx, group in data[data['gear_selected_neutral'] == 8].groupby('group')}

    # fix time column
    for key, df in drive_segments.items():
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df['seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    for key, df in coast_segments.items():
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
        df['seconds'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    if drop_keys is not None:
        for key in drop_keys:
            if key in drive_segments:
                del drive_segments[key]
            if key in coast_segments:
                del coast_segments[key]

    return drive_segments, coast_segments

def pitch_correction(data, vehicle_mass, rotating_mass_eq, verbose=True):

    if 'pitch' not in data.columns:
        if verbose:
            print('Pitch angle not found in data, skipping pitch correction')
        return data
    from scipy.integrate import cumulative_trapezoid
    g = 9.81
    pitch_angle = data['pitch']
    
    # Calculate pitch induced force
    pitch_induced_force = vehicle_mass * g * np.sin(np.deg2rad(pitch_angle)) # N
    data['pitch_induced_force_N'] = pitch_induced_force

    # Acceleration due to pitch
    a_pitch = pitch_induced_force / (vehicle_mass + rotating_mass_eq) # m/s^2
    data['pitch_induced_acc'] = a_pitch

    # Time differences (dt)
    dt = np.diff(data['seconds'], prepend=0)
    
    # Integrate to get the velocity change (use cumtrapz for better accuracy)
    delta_v_pitch = cumulative_trapezoid(a_pitch, data['seconds'], initial=0) # m/s
    data['pitch_induced_delta_v'] = delta_v_pitch

    # Corrected vehicle speed
    data['vehicle_speed_pitch_corrected'] = data['vehicle_speed'] + data['pitch_induced_delta_v']

    # Correct the GPS speed as well
    data['vehicle_speed_gps_pitch_corrected'] = data['vehicle_speed_gps'] + data['pitch_induced_delta_v']

    return data

def filter_segments_range(segments_dict, min_value, max_value, column, verbose=True):
    import copy
    segments_dict = copy.deepcopy(segments_dict)
    new_segments_dict = {}
    new_dict_key = 0
    total_removed_data = 0

    for key, df in segments_dict.items():
        #print(f'Filtering segment {key}')
        
        # Identify the indices where the data becomes invalid
        invalid_condition = (df[column] > max_value) | \
                            (df[column] < min_value)
        
        # Count the number of invalid data points
        total_invalid_data_points = invalid_condition.sum()
        #print(f'Found {total_invalid_data_points} invalid data points')
        
        # Track removed data points
        total_removed_data += total_invalid_data_points
        
        # Split the data into segments based on validity changes
        invalid_condition = invalid_condition.astype(int)
        val_change_indices = invalid_condition.diff().fillna(0).ne(0)
        segment_start_indices = val_change_indices[val_change_indices].index.tolist()
        segment_start_indices = [0] + segment_start_indices + [len(df)]

        # Iterate through the segments
        for i in range(len(segment_start_indices) - 1):
            start_idx = segment_start_indices[i]
            end_idx = segment_start_indices[i + 1]

            # Ensure start_idx is within the bounds of invalid_condition
            if start_idx < len(invalid_condition):
                segment = df.iloc[start_idx:end_idx].copy()

                # Check if the segment is valid
                if not invalid_condition.iloc[start_idx]:
                    # Reset 'seconds' column to start from zero
                    if 'seconds' in segment.columns:
                        segment['seconds'] = segment['seconds'] - segment['seconds'].iloc[0]
                    
                    # Store the valid segment in the new dictionary
                    new_segments_dict[f"{key}_segment_{new_dict_key}"] = segment
                    new_dict_key += 1

    if verbose:
        print(f'Filtered out {total_removed_data} data points because they were outside the range [{min_value}, {max_value}], of the column {column}')

    return new_segments_dict, total_removed_data

def filter_segments_target_value(segments_dict, target_value, column, verbose=True):
    segments_dict = segments_dict.copy()
    original_num_data_points = sum([len(df) for df in segments_dict.values()])
    new_segments_dict = {}
    new_dict_key = 0
    for key, df in segments_dict.items():
        # if whole segment has target suspension level, keep it
        if df[column].unique()[0] == target_value:
            new_segments_dict[new_dict_key] = df
            new_dict_key += 1
        # if segment has multiple suspension levels, split it and keep the section with target suspension level
        elif len(df[column].unique()) > 1:
            suspension_level_indices = df[df[column] == target_value].index
            if len(suspension_level_indices) > 0:
                start_index = suspension_level_indices[0]
                end_index = suspension_level_indices[-1]
                new_segments_dict[new_dict_key] = df.loc[start_index:end_index]
                new_dict_key += 1
    new_num_data_points = sum([len(df) for df in new_segments_dict.values()])
    if verbose:
        print(f'Filtered out {original_num_data_points - new_num_data_points} data points because they were not at the target value {target_value} of the column {column}')
    return new_segments_dict, original_num_data_points - new_num_data_points

def cut_segments(segments_dict, start_cut_duration=0, end_cut_duration=0):
    segments_dict = segments_dict.copy()
    total_data_points_original = sum([len(df) for df in segments_dict.values()])
    start_cut = start_cut_duration * 10
    end_cut = end_cut_duration * 10
    for key, df in segments_dict.items():
        segments_dict[key] = df.iloc[start_cut:-end_cut]

    # clean up empty segments
    keys_to_remove = [key for key, df in segments_dict.items() if df.empty]
    for key in keys_to_remove:
        del segments_dict[key]

    total_data_points_cut = sum([len(df) for df in segments_dict.values()])
    total_data_points_removed = total_data_points_original - total_data_points_cut

    return segments_dict, total_data_points_removed

import random

def sample_segments(segments_dict, seed, num_samples):
    """
    Samples random key-value pairs from the input dictionary and resets the keys to be continuous.

    Parameters:
    - segments_dics (dict): The original dictionary with integer keys and DataFrame values.
    - seed (int): The random seed for reproducibility.
    - num_samples (int): The number of key-value pairs to sample.

    Returns:
    - dict: A new dictionary with sampled key-value pairs and reset keys.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Ensure num_samples does not exceed the number of available items
    num_samples = min(num_samples, len(segments_dict))
    
    # Get a list of keys from the original dictionary
    keys = list(segments_dict.keys())
    
    # Randomly sample the specified number of keys
    sampled_keys = random.sample(keys, num_samples)
    
    # Retrieve the corresponding values for the sampled keys
    sampled_values = [segments_dict[key] for key in sampled_keys]
    
    # Create a new dictionary with reset keys starting from 0
    new_coast_segments = {i: value for i, value in enumerate(sampled_values)}
    
    return new_coast_segments


def build_deceleration_df(segments_dict, bucket_size=2, column='vehicle_speed', verbose=False):
    import pandas as pd

    # Initialize a list to collect data
    deceleration_data = []

    # Loop through all DataFrames in the dictionary
    for key, df in segments_dict.items():
        if verbose:
            print(f'Processing key {key}')
            print(f'DataFrame contains {len(df)} rows')

        # Skip if DataFrame is empty
        if df.empty:
            if verbose:
                print(f'DataFrame for key {key} is empty, skipping.')
            continue

        # Check if required columns exist
        required_columns = ['seconds', column]
        if not all(col in df.columns for col in required_columns):
            if verbose:
                print(f"Required columns {required_columns} not found in DataFrame for key {key}, skipping.")
            continue

        # Ensure DataFrame is sorted by time
        df = df.sort_values('seconds')

        start_speed = df[column].iloc[0]
        end_speed = df[column].iloc[-1]
        if verbose:
            print(f'start_speed: {start_speed}, end_speed: {end_speed}')

        # Calculate bucket numbers
        highest_bucket_nr = int(start_speed // bucket_size)
        lowest_bucket_nr = int(end_speed // bucket_size)
        if verbose:
            print(f'highest_bucket_nr: {highest_bucket_nr}, which is {highest_bucket_nr * bucket_size} km/h')
            print(f'lowest_bucket_nr: {lowest_bucket_nr}, which is {lowest_bucket_nr * bucket_size} km/h')

        # Loop through the valid buckets
        for bucket_nr in range(lowest_bucket_nr, highest_bucket_nr + 1):
            if verbose:
                print(f'Processing bucket {bucket_nr}')

            # Select data within the current bucket
            bucket_data = df[
                (df[column] >= bucket_nr * bucket_size) &
                (df[column] < (bucket_nr + 1) * bucket_size)
            ]

            if verbose:
                print(f'Bucket contains {len(bucket_data)} rows')

            # Skip if bucket_data is insufficient
            if len(bucket_data) < 2:
                if verbose:
                    print(f'Not enough data points in bucket {bucket_nr}, skipping.')
                continue

            # Calculate average deceleration
            bucket_ceiling = bucket_data[column].iloc[0]
            bucket_floor = bucket_data[column].iloc[-1]
            bucket_begin = bucket_data['seconds'].iloc[0]
            bucket_end = bucket_data['seconds'].iloc[-1]
            time_diff = bucket_end - bucket_begin

            # Avoid division by zero
            if time_diff == 0:
                if verbose:
                    print(f'Time difference is zero in bucket {bucket_nr}, cannot compute deceleration, skipping.')
                continue

            average_deceleration = (bucket_ceiling - bucket_floor) / time_diff
            if verbose:
                print(f'average_deceleration: {average_deceleration}')

            # Collect the data in the list
            deceleration_data.append({
                'key': key,
                'start_speed': bucket_ceiling,
                'end_speed': bucket_floor,
                'bucket_nr': bucket_nr,
                'average_deceleration': average_deceleration
            })

    # Create the deceleration DataFrame from the list
    deceleration_df = pd.DataFrame(deceleration_data)

    return deceleration_df


def aggregate_buckets(deceleration_df, bucket_size=2, verbose=False):
    import pandas as pd

    # List to collect bucket statistics
    aggregated_data = []

    # Get the unique bucket numbers present in the deceleration DataFrame
    bucket_numbers = sorted(deceleration_df['bucket_nr'].unique())

    if verbose:
        print(f'Aggregating deceleration data over {len(bucket_numbers)} buckets: {bucket_numbers}')

    # Iterate over each unique bucket number
    for bucket_nr in bucket_numbers:
        bucket_floor = bucket_nr * bucket_size
        bucket_ceiling = (bucket_nr + 1) * bucket_size
        bucket_mid = (bucket_floor + bucket_ceiling) / 2

        # Select data corresponding to the current bucket number
        bucket_data = deceleration_df[deceleration_df['bucket_nr'] == bucket_nr]

        if verbose:
            print(f'\nProcessing bucket {bucket_nr} (Speed range: {bucket_floor}-{bucket_ceiling} km/h)')
            print(f'Number of data points in bucket: {len(bucket_data)}')

        # Skip the bucket if there's no data
        if bucket_data.empty:
            if verbose:
                print(f'No data for bucket {bucket_nr}, skipping.')
            continue

        # Calculate statistical metrics for the current bucket
        mean = bucket_data['average_deceleration'].mean()
        std_deviation = bucket_data['average_deceleration'].std()
        count = len(bucket_data)
        min_val = bucket_data['average_deceleration'].min()
        max_val = bucket_data['average_deceleration'].max()
        percentile_25 = bucket_data['average_deceleration'].quantile(0.25)
        percentile_50 = bucket_data['average_deceleration'].quantile(0.50)
        percentile_75 = bucket_data['average_deceleration'].quantile(0.75)

        # Append the statistics to the list
        aggregated_data.append({
            'bucket_nr': bucket_nr,
            'bucket_floor': bucket_floor,
            'bucket_ceiling': bucket_ceiling,
            'bucket_mid': bucket_mid,
            'mean': mean,
            'std_deviation': std_deviation,
            'count': count,
            'min': min_val,
            'max': max_val,
            '25%': percentile_25,
            '50%': percentile_50,
            '75%': percentile_75
        })

    # Create the aggregated DataFrame from the list
    aggregated_deceleration_df = pd.DataFrame(aggregated_data)

    if verbose:
        print('\nAggregation complete.')

    return aggregated_deceleration_df


import numpy as np
from scipy.optimize import curve_fit

def quadratic_model(v, f0, f1, f2):
    return f0 + f1 * v + f2 * v**2

def fit_quadratic_model(df, remove_outlier_threshold=None, speed_range=(15, 160)):

    # Filter data within the specified speed speed_range
    if speed_range is not None:
        if 'bucket_mid' in df.columns:
            df = df[(df['bucket_mid'] >= speed_range[0]) & (df['bucket_mid'] <= speed_range[1])]
        elif 'mean_speed' in df.columns:
            df = df[(df['mean_speed'] >= speed_range[0]) & (df['mean_speed'] <= speed_range[1])]
        else:
            raise ValueError("Speed column not found in DataFrame, please check column names.")

    # Extract vehicle speed and mean braking force
    if 'bucket_mid' in df.columns:
        v = df['bucket_mid'].values
    elif 'mean_speed' in df.columns:
        v = df['mean_speed'].values
    else:
        raise ValueError("Speed column not found in DataFrame, please check column names.")

    if 'mean_force' in df.columns:
        force = df['mean_force'].values
    elif 'mean_acc_force' in df.columns:
        force = df['mean_acc_force'].values
    else:
        raise ValueError("Force column not found in DataFrame, please check column names.")

    # Initial guesses for the parameters
    initial_guess = [1e-3, 1e-3, 1e-3]
    parameter_bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    # Initialize total outliers removed
    total_outliers_removed = 0

    # If remove_outliers is False, proceed as before
    if remove_outlier_threshold is None:
        # Fit the quadratic model with bounds
        params, covariance = curve_fit(
            quadratic_model,
            v,
            force,
            p0=initial_guess,
            bounds=parameter_bounds
        )
    else:
        # Perform iterative outlier removal
        max_iterations = 10
        threshold = remove_outlier_threshold
        indices = np.arange(len(v))  # Indices of data points to include
        for iteration in range(max_iterations):
            # Fit the quadratic model with current data
            params, covariance = curve_fit(
                quadratic_model,
                v[indices],
                force[indices],
                p0=initial_guess,
                bounds=parameter_bounds
            )
            # Compute residuals
            force_pred = quadratic_model(v[indices], *params)
            residuals = force[indices] - force_pred
            # Compute standard deviation of residuals
            std_residuals = np.std(residuals)
            # Identify outliers
            outlier_mask = np.abs(residuals) > threshold * std_residuals
            num_outliers = np.sum(outlier_mask)
            if num_outliers == 0:
                # No more outliers
                break
            else:
                # Remove outliers
                indices = indices[~outlier_mask]
                total_outliers_removed += num_outliers
                #print(f"Iteration {iteration + 1}: Removed {num_outliers} outliers.")
                if len(indices) < 3:
                    # Not enough data points to fit the model
                    #print("Not enough data points after outlier removal.")
                    break
        else:
            #print("Maximum iterations reached.")
            # do nothing
            pass
            

    f0, f1, f2 = params

    # Generate data for plotting the model
    v_model = np.linspace(v.min(), v.max(), 400)
    force_model = quadratic_model(v_model, *params)

    return params, covariance, v_model, force_model, total_outliers_removed


##### Functions for Road Load Constant Approach #####

def segment_df_by_constant_speed(
    data, 
    speed_column, 
    threshold, 
    min_n_samples=10, 
    min_avg_speed=2.0
):
    # Reset index to ensure sequential indexing
    data = data.reset_index(drop=True)

    # Convert speed column to NumPy array for faster operations
    speed = data[speed_column].values
    n = len(speed)

    # Initialize lists to store segment information
    lengths = []
    start_indices = []
    end_indices = []

    # Initialize the starting index
    i = 0

    # Iterate through the speed array using a while loop for efficiency
    while i < n:
        j = i + 1  # Start checking from the next index
        # Find the end of the segment where speed deviation is within the threshold
        while j < n and abs(speed[j] - speed[i]) < threshold:
            j += 1
        # Record the segment if it has more than one sample
        if j - i > 1:
            lengths.append(j - i)
            start_indices.append(i)
            end_indices.append(j)
        i = j  # Move to the next potential segment

    # Combine segment information into a list of tuples and sort by length descending
    segments = sorted(
        zip(lengths, start_indices, end_indices), 
        key=lambda x: x[0], 
        reverse=True
    )

    # Initialize a boolean array to keep track of occupied indices
    occupied = np.zeros(n, dtype=bool)

    # Initialize the dictionary to hold final segments
    constant_speed_segments = {}
    segment_idx = 0

    # Iterate through sorted segments and assign non-overlapping segments
    for length, start, end in segments:
        if length >= min_n_samples and not occupied[start:end].any():
            # Extract the segment
            segment = data.iloc[start:end].copy()
            # Calculate average speed
            avg_speed = segment[speed_column].mean()
            # Check if average speed meets the minimum requirement
            if abs(avg_speed) >= min_avg_speed:
                # Convert 'time' from milliseconds to datetime and calculate elapsed seconds
                segment['timestamp'] = pd.to_datetime(segment['time'], unit='ms')
                segment['seconds'] = (
                    segment['timestamp'] - segment['timestamp'].min()
                ).dt.total_seconds()
                # Add to the segments dictionary
                constant_speed_segments[segment_idx] = segment
                segment_idx += 1
                # Mark indices as occupied
                occupied[start:end] = True

    return constant_speed_segments


def filter_segments_by_negative_derivative(segments, speed_signal='vehicle_speed'):
    """
    Filters segments by calculating the numerical derivative of the speed signal for each segment.
    Keeps only segments where the speed derivative is negative at all points.

    Parameters:
    - segments (dict): Dictionary of coast segments, where each value is a DataFrame with a 'seconds' column.
    - speed_signal (str): Name of the column representing the speed signal.

    Returns:
    - dict: Filtered dictionary of segments with only negative derivatives.
    """
    derivatives = {}

    for key in segments.keys():
        # Copy segment and calculate derivative
        segment = segments[key].copy()
        segment['speed_derivative'] = segment[speed_signal].diff() / segment['seconds'].diff()
        segment.dropna(inplace=True)  # Drop NaN values resulting from differentiation

        # Store the derivative in the dictionary
        derivatives[key] = segment

    # Filter out segments with any non-negative derivative
    filtered_segments = {
        key: segment for key, segment in derivatives.items()
        if (segment['speed_derivative'] < 0).all()
    }

    return filtered_segments

def filter_segments_by_derivative_limit(segments_dict, speed_signal='vehicle_speed', limit_lower=-0.3, limit_upper=0):
    filtered_segments = {}

    for key, segment in segments_dict.items():
        # Reset the index to ensure sequential integer indexing
        segment = segment.reset_index(drop=True)

        # Calculate the speed derivative
        segment['speed_derivative'] = segment[speed_signal].diff()

        # Find the indices where the derivative is within the limits
        valid_indices = (segment['speed_derivative'] > limit_lower) & (segment['speed_derivative'] < limit_upper)

        # Initialize start index and a flag for tracking valid segments
        start_idx = None

        for i in range(len(valid_indices)):
            if valid_indices.iloc[i]:  # Use .iloc for index-based access
                # Start a new segment if not already tracking one
                if start_idx is None:
                    start_idx = i
            else:
                # If the current index is invalid and a segment is being tracked, finalize it
                if start_idx is not None:
                    if i - start_idx >= 5:  # Only keep segments of at least 5 samples
                        filtered_segments[f"{key}_segment_{start_idx}_{i-1}"] = segment.iloc[start_idx:i]
                    start_idx = None

        # Handle the case where the valid range extends to the end of the segment
        if start_idx is not None and len(valid_indices) - start_idx >= 5:
            filtered_segments[f"{key}_segment_{start_idx}_{len(valid_indices)-1}"] = segment.iloc[start_idx:]

    # Reset the key names to start from 0
    filtered_segments = {f"segment_{i}": seg for i, seg in enumerate(filtered_segments.values())}

    return filtered_segments


##### Functions for Gear Strategy Finder #####

from scipy.interpolate import splprep, splev, PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt

def recreate_line_function(unique_distances, x_unique, y_unique):
    """
    Re-create the line_function using PchipInterpolator, based on
    unique_distances, x_unique, and y_unique.
    """
    interpolator_x = PchipInterpolator(unique_distances, x_unique)
    interpolator_y = PchipInterpolator(unique_distances, y_unique)

    def line_function(input_distances):
        eval_x = interpolator_x(input_distances)
        eval_y = interpolator_y(input_distances)
        return eval_x, eval_y

    return line_function

def fit_parametric_line(cluster_centers, smooth_factor=0.5, anchor_points=False, fit_type="spline", degree=3, plot=False, title="Fitted Line"):
    """
    Fit a parametric line or regression model to cluster centers.

    Parameters:
    - cluster_centers: list of tuples, sorted cluster center coordinates.
    - smooth_factor: float, controls the spline smoothness (lower = tighter fit, higher = smoother fit).
    - anchor_points: bool, if True, force the line to intersect the first and last cluster centers.
    - fit_type: str, type of fit ("spline", "polynomial", or "pchip").
    - degree: int, degree of the polynomial (if fit_type="polynomial").
    - plot: bool, if True, plot the fitted line and cluster centers.
    - title: str, title of the plot.

    Returns:
    - fitted_x: np.ndarray, x-coordinates of the fitted line.
    - fitted_y: np.ndarray, y-coordinates of the fitted line.
    - line_function: callable, function to evaluate the fitted line.
    - unique_distances (only for 'pchip'): np.ndarray, unique distances along the curve.
    - x_unique (only for 'pchip'): np.ndarray, unique x-values corresponding to unique distances.
    - y_unique (only for 'pchip'): np.ndarray, unique y-values corresponding to unique distances.
    """
    cluster_centers = np.array(cluster_centers)
    x = cluster_centers[:, 0]
    y = cluster_centers[:, 1]

    # Fit using the specified method
    if fit_type == "spline":
        # Optionally anchor the first and last points
        if anchor_points:
            tck, u = splprep([x, y], s=smooth_factor, k=min(3, len(x) - 1), t=[0, 1])
        else:
            tck, u = splprep([x, y], s=smooth_factor, k=min(3, len(x) - 1))
        fitted_x, fitted_y = splev(np.linspace(0, 1, 500), tck)

        # Define a function to evaluate the fitted spline
        def line_function(input_x):
            t = np.interp(input_x, fitted_x, np.linspace(0, 1, len(fitted_x)))
            eval_x, eval_y = splev(t, tck)
            return eval_y

    elif fit_type == "polynomial":
        # Fit polynomial regression to the data
        poly_coeffs = np.polyfit(x, y, degree)
        poly_func = np.poly1d(poly_coeffs)

        # Generate fitted values
        fitted_x = np.linspace(x.min(), x.max(), 500)
        fitted_y = poly_func(fitted_x)

        # Define a function to evaluate the polynomial
        def line_function(input_x):
            return poly_func(input_x)

    elif fit_type == "pchip":
        # Calculate cumulative distances (arc length)
        cumulative_distances = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        
        # Remove duplicate distances
        unique_distances, unique_indices = np.unique(cumulative_distances, return_index=True)
        x_unique = x[unique_indices]
        y_unique = y[unique_indices]
        
        # Use Piecewise Cubic Hermite Interpolation (PCHIP)
        interpolator_x = PchipInterpolator(unique_distances, x_unique)
        interpolator_y = PchipInterpolator(unique_distances, y_unique)
        
        # Generate fitted values
        fitted_distances = np.linspace(unique_distances.min(), unique_distances.max(), 500)
        fitted_x = interpolator_x(fitted_distances)
        fitted_y = interpolator_y(fitted_distances)
        
        # Define a function to evaluate the PCHIP interpolation
        def line_function(input_distances):
            eval_x = interpolator_x(input_distances)
            eval_y = interpolator_y(input_distances)
            return eval_x, eval_y

    else:
        raise ValueError("fit_type must be 'spline', 'polynomial', or 'pchip'.")

    # Plot the results
    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color="blue", label="Cluster Centers")
        plt.plot(fitted_x, fitted_y, color="red", label=f"Fitted Line ({fit_type})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()

    if fit_type == "pchip":
        return fitted_x, fitted_y, line_function, unique_distances, x_unique, y_unique
    else:
        return fitted_x, fitted_y, line_function, None, None, None


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline
from sklearn.cluster import KMeans


def fit_parametric_spline(
    cluster_centers,
    normal_points,
    anchor_points=False,
    k: int = 3,
    num_knots: int = 8,
    cluster_weight: float = 1e6,
    normal_weight: float = 1.0,
    knot_method: str = "equidistant",
):
    """Return a weighted least-squares spline and its sampled curve.

    Parameters are unchanged except that all plotting options are removed.
    Returns
    -------
    fitted_x : ndarray
    fitted_y : ndarray
    spline   : LSQUnivariateSpline
    knots    : ndarray (interior knots)
    """

    cluster_centers = np.asarray(cluster_centers)
    normal_points = np.asarray(normal_points)

    # sort cluster centers by x
    idx = np.argsort(cluster_centers[:, 0])
    cluster_centers = cluster_centers[idx]
    x_c, y_c = cluster_centers.T

    x_n = normal_points[:, 0] if normal_points.size else np.empty(0)
    y_n = normal_points[:, 1] if normal_points.size else np.empty(0)

    # concatenate data
    x_all = np.concatenate([x_c, x_n])
    y_all = np.concatenate([y_c, y_n])
    idx = np.argsort(x_all)
    x_all, y_all = x_all[idx], y_all[idx]

    # weights
    w = np.concatenate([
        np.full(len(x_c), cluster_weight),
        np.full(len(x_n), normal_weight),
    ])[idx]

    if anchor_points and len(x_c) > 1:
        first_mask = (x_all == x_c[0]) & (y_all == y_c[0])
        last_mask = (x_all == x_c[-1]) & (y_all == y_c[-1])
        if first_mask.any():
            w[first_mask.argmax()] = cluster_weight * 1e12
        if last_mask.any():
            w[last_mask.argmax()] = cluster_weight * 1e12

    if num_knots <= 2:
        raise ValueError("num_knots must be greater than 2 for a cubic spline.")

    interior = num_knots - 2
    x_min, x_max = x_all[0], x_all[-1]

    if knot_method == "equidistant":
        knots = np.linspace(x_min, x_max, num_knots)[1:-1]
    elif knot_method == "quantile":
        q = np.linspace(0, 100, num_knots)[1:-1]
        knots = np.percentile(x_all, q)
    elif knot_method == "cluster":
        if interior <= 0:
            raise ValueError("num_knots must be at least 3 to have at least one interior knot.")
        kmeans = KMeans(n_clusters=interior, random_state=42)
        kmeans.fit(x_all.reshape(-1, 1))
        knots = np.sort(np.clip(kmeans.cluster_centers_.ravel(), x_min + 1e-12, x_max - 1e-12))
    else:
        raise ValueError("knot_method must be 'equidistant', 'quantile', or 'cluster'.")

    spline = LSQUnivariateSpline(x_all, y_all, knots, w=w, k=k)

    fitted_x = np.linspace(x_all.min(), x_all.max(), 500)
    fitted_y = spline(fitted_x)

    return fitted_x, fitted_y, spline, knots




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def perform_kmeans_clustering(
    df, x_column, y_column, k, use_centroid=True, plot=False, attach_endpoints=False, title="K-Means Clustering and Greedy Sorting with Endpoints", y_label="Y", figsize=(10, 6), filename=None
):
    """
    Perform k-means clustering, compute cluster centers (centroids or medians), and sort both 
    cluster centers and datapoints in a greedy manner starting from the closest to the origin.

    Parameters:
    - df: pandas DataFrame, the input data.
    - x_column: str, the name of the x-axis column.
    - y_column: str, the name of the y-axis column.
    - k: int, the number of clusters.
    - use_centroid: bool, True to use centroids, False to use medians for cluster centers.
    - plot: bool, True to plot the clustered points with color-coded clusters.
    - attach_endpoints: bool, True to attach endpoints. Endpoints are determined by normalizing 
      coordinates and selecting the point closest and farthest from the origin in the normalized space.
    - title: str, the title for the plot.

    Returns:
    - cluster_centers: list of tuples, sorted cluster centers (including artificial endpoints if enabled).
    - sorted_points: list of tuples, datapoints sorted in a greedy distance-based manner.
    """

    # Extract relevant columns as a NumPy array
    X = df[[x_column, y_column]].values

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    # Compute cluster centers
    if use_centroid:
        cluster_centers = kmeans.cluster_centers_
    else:
        cluster_centers = np.array([np.median(X[labels == i], axis=0) for i in range(k)])

    def greedy_sort(points):
        sorted_points = []
        remaining_points = points.tolist()

        # Start with the point closest to the origin
        current_point = min(remaining_points, key=lambda p: np.linalg.norm(p))
        sorted_points.append(current_point)
        remaining_points.remove(current_point)

        # Iteratively find the next closest point
        while remaining_points:
            next_point = min(remaining_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(current_point)))
            sorted_points.append(next_point)
            remaining_points.remove(next_point)
            current_point = next_point

        return np.array(sorted_points)

    # Sort cluster centers
    sorted_centers = greedy_sort(cluster_centers)

    # Sort all datapoints
    sorted_points = greedy_sort(X)

    # need to attach endpoints based on normalized coordinates
    if attach_endpoints:

        # Normalize using min-max normalization instead of standardization
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1e-9  # avoid division by zero

        X_normalized = (X - X_min) / X_range

        distances = np.linalg.norm(X_normalized, axis=1)
        min_idx = np.argmin(distances)  # closest to (0,0) in normalized space
        max_idx = np.argmax(distances)  # farthest from (0,0) in normalized space

        first_point = X[min_idx]
        last_point = X[max_idx]

        # Add them to the cluster centers
        sorted_centers = np.vstack([sorted_centers, first_point, last_point])

        # Re-sort the combined centers
        sorted_centers = greedy_sort(sorted_centers)

    # Plot the clusters if required
    if plot:
        plt.figure(figsize=figsize)
        # Use LaTeX for text rendering
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'sans-serif',
            'font.sans-serif': 'Arial',
            'font.size': 12,
            'text.latex.preamble': r'\usepackage{siunitx}'
        })
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.7, label="Data Points")
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color="red", s=100, label="Cluster Centers")
        if attach_endpoints:
            # Plot the chosen endpoints (first_point and last_point)
            plt.scatter(
                [first_point[0], last_point[0]],
                [first_point[1], last_point[1]],
                color="orange", s=150, label="Endpoints"
            )
        #plt.xlabel(x_column)
        plt.xlabel('Vehicle Speed in \si{\kilo\meter\per\hour}')
        if y_label is not None:
            plt.ylabel(y_label)
        if title is not None:
            plt.title(title)
        plt.legend()
        # disable spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        
        plt.grid()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
            print(f"Plot saved to: {filename}")
        plt.show()

    return sorted_centers.tolist(), sorted_points.tolist()

from sklearn.cluster import DBSCAN
import numpy as np

def mark_outliers_with_dbscan(df, x_column, y_column, eps=0.05, min_samples=5, verbose=False):
    """
    Perform DBSCAN clustering to detect and accumulate outliers. After detecting outliers, 
    normalize the data and identify two endpoints in normalized space (closest and farthest 
    from the origin). If these endpoints were marked as outliers, re-classify them as non-outliers.
    
    Parameters:
    - df: pandas DataFrame containing the data.
    - x_column: Name of the column to use as the x-axis.
    - y_column: Name of the column to use as the y-axis.
    - eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    - verbose: bool, if True, print additional debugging information.
    
    Returns:
    - df: DataFrame with updated 'is_outlier' column and cluster labels, and endpoints guaranteed not outliers.
    """
    df = df.reset_index(drop=True)
    if verbose:
        print(f"Clustering data with DBSCAN using eps={eps}, min_samples={min_samples}... on columns {x_column} and {y_column}")

    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Extract the data for clustering
    X = df[[x_column, y_column]].values

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)

    # Add the cluster labels to the DataFrame
    df['cluster_label'] = dbscan.labels_

    # Initialize 'is_outlier' column if not present
    if 'is_outlier' not in df:
        df['is_outlier'] = False

    # Mark points with cluster_label = -1 as outliers
    new_outliers = (df['cluster_label'] == -1)
    df['is_outlier'] = df['is_outlier'] | new_outliers

    if verbose:
        print(f"Found {new_outliers.sum()} new outliers. Total outliers so far: {df['is_outlier'].sum()}.")

    # Normalize the data and find endpoints
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1e-9

    X_normalized = (X - X_min) / X_range
    distances = np.linalg.norm(X_normalized, axis=1)

    # Find endpoints: closest and farthest from origin in normalized space
    min_idx = np.argmin(distances)
    max_idx = np.argmax(distances)

    # If endpoints are marked as outliers, override them
    # Mark them as non-outliers
    if df.at[min_idx, 'is_outlier']:
        if verbose:
            print(f"Endpoint at index {min_idx} was an outlier, reclassifying as non-outlier.")
        df.at[min_idx, 'is_outlier'] = False

    if df.at[max_idx, 'is_outlier']:
        if verbose:
            print(f"Endpoint at index {max_idx} was an outlier, reclassifying as non-outlier.")
        df.at[max_idx, 'is_outlier'] = False

    return df

def merge_close_points(df, x_column, y_column, threshold=0.05):
    """
    Merge points in a DataFrame that are closer than a given threshold in the normalized space,
    keeping all original columns and appropriately merging their values.

    Parameters:
    - df: pandas DataFrame containing the data.
    - x_column: Name of the column to use as the x-axis.
    - y_column: Name of the column to use as the y-axis.
    - threshold: The maximum distance between two points for them to be merged.

    Returns:
    - df_merged: DataFrame with merged points, including all original columns.
    """
    import pandas as pd
    import numpy as np

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    # Initialize a list to store the merged rows
    merged_rows = []
    # Keep track of indices that have already been processed
    processed_indices = set()

    # Convert the DataFrame to numpy arrays for efficient computation
    x_values = df_copy[x_column].values
    y_values = df_copy[y_column].values
    indices = df_copy.index.values

    # Iterate over the DataFrame
    for idx in indices:
        # Skip if the index has already been processed
        if idx in processed_indices:
            continue

        # Get the current point
        x1 = x_values[df_copy.index.get_loc(idx)]
        y1 = y_values[df_copy.index.get_loc(idx)]

        # Compute distances to all other points
        distances = np.sqrt((x_values - x1)**2 + (y_values - y1)**2)

        # Find indices of points within the threshold and not yet processed
        close_point_indices = indices[(distances < threshold) & (~np.isin(indices, list(processed_indices)))]

        # Extract the close points
        close_points = df_copy.loc[close_point_indices]

        # Initialize a dictionary to hold merged values
        merged_row = {}

        # Iterate over all columns to compute merged values
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns, calculate the mean
                merged_value = close_points[col].mean()
            else:
                # For non-numeric columns, take the mode (most frequent value)
                merged_value = close_points[col].mode().iloc[0]
            merged_row[col] = merged_value

        # Append the merged row to the list
        merged_rows.append(merged_row)

        # Mark the close points as processed
        processed_indices.update(close_point_indices)

    # Create the merged DataFrame from the list of merged rows
    df_merged = pd.DataFrame(merged_rows)

    return df_merged




##### Legacy GSC Functions #####

import os
import random

def load_gear_change_data(list_of_files=None, n_samples_around_event=100, remove_gear_eight=True, remove_gear_zero=True, verbose=False):
    # Build list of files if not provided
    if list_of_files is None:
        list_of_files = []
        for root, dirs, files in os.walk('data/'):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    if all(x not in full_path for x in ['_IMU', 'mobile', 'WLTP', 'log', '2024_09_21-22_05_11', '_vertices', '2024_10_10']):
                        list_of_files.append(full_path[5:])
        if verbose:
            print(list_of_files)

    # Initialize dictionaries to store shift events and counters for total shifts
    shift_ups = {}
    shift_downs = {}
    non_shifts = {}
    total_shift_ups = 0
    total_shift_downs = 0
    total_non_shifts = 0

    n = n_samples_around_event  # Number of rows before and after the event to extract

    # Process each file
    for filename in list_of_files:
        if verbose:
            print(f'Processing {filename}...')
        
        data = load_can_data(filename, verbose=verbose)
        
        # Ensure the dataframe is sorted by time if not already
        data = data.sort_values(by='seconds').reset_index(drop=True)
        
        # Create shift up and shift down columns
        data['shift_up'] = data['gear_selected'].diff() == 1
        data['shift_down'] = data['gear_selected'].diff() == -1
        
        # Update total counts
        total_shift_ups += data['shift_up'].sum()
        total_shift_downs += data['shift_down'].sum()
        
        # Collect shift up events
        for i in data[data['shift_up']].index:
            if i - n < 0 or i + n >= len(data):
                continue  # Skip if window goes out of bounds
            event_df = data.loc[i-n:i+n].copy()
            event_df['seconds'] -= data.loc[i, 'seconds']
            shift_ups[f'{filename}_shift_up_{i}'] = event_df
        
        # Collect shift down events
        for i in data[data['shift_down']].index:
            if i - n < 0 or i + n >= len(data):
                continue  # Skip if window goes out of bounds
            event_df = data.loc[i-n:i+n].copy()
            event_df['seconds'] -= data.loc[i, 'seconds']
            shift_downs[f'{filename}_shift_down_{i}'] = event_df
        
        # Determine the number of non-shift events to sample
        required_non_shifts = (total_shift_ups + total_shift_downs) / 2
        
        # Initialize a set to keep track of sampled indices to avoid duplicates
        sampled_indices = set()
        
        while len(non_shifts) < required_non_shifts:
            # Randomly select a row index
            rand_index = random.randint(n, len(data) - n - 1)
            
            if rand_index in sampled_indices:
                continue  # Skip if already sampled
            
            # Check if this window overlaps with any shift events
            window_start = rand_index - n
            window_end = rand_index + n
            
            # Check for shift events within the window
            window_shifts = data.loc[window_start:window_end, ['shift_up', 'shift_down']]
            if window_shifts['shift_up'].any() or window_shifts['shift_down'].any():
                continue  # Shift event found in window, skip
            
            # If no shift events, extract the window
            event_df = data.loc[window_start:window_end].copy()
            event_df['seconds'] -= data.loc[rand_index, 'seconds']
            non_shifts[f'{filename}_non_shift_{rand_index}'] = event_df
            total_non_shifts += 1
            sampled_indices.add(rand_index)
            
            # Optional: Add a maximum number of attempts to prevent infinite loops
            if len(sampled_indices) > (len(data) / 2):
                if verbose:
                    print(f"Reached maximum sampling attempts for {filename}.")
                break

    
    if remove_gear_eight:
        # loop through all shift up events and shift down events and non shift events and remove ones where 8 is present in gear_selected
        keys_to_delete = []
        for key, df in shift_ups.items():
            if 8 in df['gear_selected'].values:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del shift_ups[key]

        keys_to_delete = []
        for key, df in shift_downs.items():
            if 8 in df['gear_selected'].values:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del shift_downs[key]

        keys_to_delete = []
        for key, df in non_shifts.items():
            if 8 in df['gear_selected'].values:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del non_shifts[key]

    if remove_gear_zero:
        # loop through all shift up events and shift down events and non shift events and remove ones where 0 is present in gear_selected
        keys_to_delete = []
        for key, df in shift_ups.items():
            if 0 in df['gear_selected'].values:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del shift_ups[key]

        keys_to_delete = []
        for key, df in shift_downs.items():
            if 0 in df['gear_selected'].values:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del shift_downs[key]

        keys_to_delete = []
        for key, df in non_shifts.items():
            if 0 in df['gear_selected'].values:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del non_shifts[key]

    # Output the total number of shifts and non-shifts found if verbose
    if verbose:
        print(f'Total number of shifts up: {total_shift_ups}')
        print(f'Total number of shifts down: {total_shift_downs}')
        print(f'Total number of non-shifts: {total_non_shifts}')
    
    return shift_ups, shift_downs, non_shifts



def build_shift_info_df(shifts_dict, signals, time_offsets, label_strategy='multiclass', shift_type=None):
    """
    Build a DataFrame containing shift events and selected features for training/testing classifiers.

    Parameters:
    shifts_dict (dict): Dictionary containing shift events (keys as event names, values as DataFrames).
    signals (list of str): List of signal names to use for features.
    time_offsets (list of int): List of time offsets to sample for each signal.
    label_strategy (str): 'multiclass' for shift type label, 'binary' for gear_selected label (+5 samples).
    shift_type (str, optional): Shift type to use for multiclass labeling ('up', 'down', 'non').

    Returns:
    pd.DataFrame: A DataFrame containing the extracted features and labels.
    """
    shift_info_df = pd.DataFrame()

    for key, shift_df in shifts_dict.items():
        zero_time_index = shift_df.index[shift_df['seconds'] == 0][0]

        # Create a dictionary to collect feature data
        feature_data = {}
        
        # Extract features for each signal and time offset
        for signal in signals:
            for offset in time_offsets:
                column_name = f"{signal}_{offset}"
                target_index = zero_time_index + offset
                feature_data[column_name] = shift_df.loc[target_index, signal]

        # Determine the label based on label_strategy
        if label_strategy == 'multiclass':
            feature_data['label'] = shift_type
        elif label_strategy == 'binary':
            target_index = zero_time_index + 5
            feature_data['label'] = int(shift_df.loc[target_index, 'gear_selected'])

        # Append feature data to the dataframe
        shift_info_df = pd.concat([shift_info_df, pd.DataFrame(feature_data, index=[key])])

    # Reset index
    shift_info_df = shift_info_df.reset_index(drop=True)

    return shift_info_df

from sklearn.model_selection import train_test_split

def perform_train_test_split(data, test_size=0.2, random_state=None, stratify=False, shuffle=True):
    """
    Split the dataset into training and testing sets.

    Parameters:
    X (pd.DataFrame): Features.
    y (pd.Series): Labels.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int, optional): Controls the shuffling applied to the data before splitting.
    stratify (array-like, optional): If not None, data is split in a stratified fashion.
    shuffle (bool): Whether or not to shuffle the data before splitting.

    Returns:
    X_train, X_test, y_train, y_test: Split datasets.
    """
    X = data.drop('label', axis=1)
    y = data['label']
    if stratify:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=shuffle)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)


import importlib

def fit_classifier(X_train, y_train, classifier_name, hyperparameters=None, random_state=None):
    """
    Initialize and fit a classifier with the given parameters.

    Parameters:
    classifier_name (str): The abbreviation of the classifier (e.g., 'DT', 'RF').
    hyperparameters (dict, optional): Dictionary containing hyperparameters for the classifier.
    random_state (int, optional): Random state for reproducibility.

    Returns:
    clf: Fitted classifier.
    """
    if hyperparameters is None:
        hyperparameters = {}

    # Map abbreviations to classifier names and modules
    classifier_mapping = {
        'DT': ('sklearn.tree', 'DecisionTreeClassifier'),
        'RF': ('sklearn.ensemble', 'RandomForestClassifier'),
        'GB': ('sklearn.ensemble', 'GradientBoostingClassifier'),
        'AB': ('sklearn.ensemble', 'AdaBoostClassifier'),
        'Bag': ('sklearn.ensemble', 'BaggingClassifier'),
        'ET': ('sklearn.ensemble', 'ExtraTreesClassifier'),
        'SVC': ('sklearn.svm', 'SVC'),
        'NuSVC': ('sklearn.svm', 'NuSVC'),
        'LSVC': ('sklearn.svm', 'LinearSVC'),
        'kNN': ('sklearn.neighbors', 'KNeighborsClassifier'),
        'LR': ('sklearn.linear_model', 'LogisticRegression'),
        'SGD': ('sklearn.linear_model', 'SGDClassifier'),
        'GNB': ('sklearn.naive_bayes', 'GaussianNB'),
        'MNB': ('sklearn.naive_bayes', 'MultinomialNB'),
        'BNB': ('sklearn.naive_bayes', 'BernoulliNB'),
        'LDA': ('sklearn.discriminant_analysis', 'LinearDiscriminantAnalysis'),
        'QDA': ('sklearn.discriminant_analysis', 'QuadraticDiscriminantAnalysis'),
        'MLP': ('sklearn.neural_network', 'MLPClassifier')
    }

    if classifier_name not in classifier_mapping:
        raise ValueError(f"Classifier '{classifier_name}' is not recognized. Please check the name and try again.")

    # Get module name and classifier class name
    module_name, class_name = classifier_mapping[classifier_name]

    # Import the classifier dynamically
    module = importlib.import_module(module_name)
    classifier_class = getattr(module, class_name)

    # Initialize and fit the classifier
    if classifier_name in ['kNN', 'GNB', 'MNB', 'BNB', 'LDA', 'QDA']:
        clf = classifier_class(**hyperparameters)
    else:
        clf = classifier_class(random_state=random_state, **hyperparameters)
    clf.fit(X_train, y_train)

    return clf


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

def evaluate_classification(clf, X_test, y_test, generate_plots=True):
    """
    Evaluate the classifier with accuracy, recall, precision, and other metrics.
    Optionally generate a confusion matrix heatmap and decision tree plot.

    Parameters:
    clf (ClassifierMixin): Fitted classifier.
    X_test (pd.DataFrame): Test feature set.
    y_test (pd.Series): Test label set.
    generate_plots (bool): Whether to generate plots (confusion matrix and decision tree).

    Returns:
    dict: Dictionary containing accuracy, recall, precision, f1 score, confusion matrix, and classification report.
    """
    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    # Convert class labels to strings to avoid errors
    class_names = [str(cls) for cls in clf.classes_]

    # Generate classification report
    clf_report = classification_report(y_test, y_pred, target_names=class_names)

    # Generate figures if requested
    if generate_plots:
        # Plot the confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Plot the Decision Tree (only if it's a DecisionTreeClassifier)
        if isinstance(clf, DecisionTreeClassifier):
            plt.figure(figsize=(20, 10))
            plot_tree(clf, filled=True, feature_names=X_test.columns, class_names=class_names, rounded=True, fontsize=12)
            plt.show()

        # Plot feature importances (only if available) and sort by importance
        if hasattr(clf, 'feature_importances_'):
            feature_importances = clf.feature_importances_
            sorted_indices = feature_importances.argsort()[::-1]
            sorted_features = X_test.columns[sorted_indices]
            sorted_importances = feature_importances[sorted_indices]

            plt.figure(figsize=(10, 10))
            sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importances')
            plt.show()

    # Return evaluation metrics
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': clf_report
    }

def calculate_shared_colorscale(results_df, hyperparameter_pairs, target_metric, aggregation='mean'):
    """
    Calculates the global minimum and maximum of the aggregated target metric
    for a set of hyperparameter pairs.

    Parameters:
    - results_df: DataFrame containing the results of hyperparameter tuning.
    - hyperparameter_pairs: List of tuples, each containing two hyperparameters (hp1, hp2).
    - target_metric: The metric to aggregate and analyze.
    - aggregation: Aggregation method to apply (default is 'mean').

    Returns:
    - vmin, vmax: Shared minimum and maximum values of the aggregated target metric.
    """
    aggregated_values = []

    for hp1, hp2 in hyperparameter_pairs:
        # Prepare data
        data = results_df[[hp1, hp2, target_metric]].dropna(subset=[target_metric])

        # Pivot table with specified aggregation
        pivot_df = data.pivot_table(index=hp1, columns=hp2, values=target_metric, aggfunc=aggregation)

        # Collect the aggregated values (flattened)
        aggregated_values.extend(pivot_df.values.flatten())

    # Remove NaN values and calculate min and max
    aggregated_values = np.array(aggregated_values)
    aggregated_values = aggregated_values[~np.isnan(aggregated_values)]

    vmin, vmax = aggregated_values.min(), aggregated_values.max()
    return vmin, vmax


##### Functions for Simulation Pipeline #####

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

def preprocess_efficiency_map(em_df, eff_col):
    """
    Prepare the data from an efficiency map for 2D interpolation.
    
    Parameters
    ----------
    em_df : pd.DataFrame
        The raw efficiency map containing columns ['rear_motor_torque', 'engine_rpm', eff_col].
    eff_col : str
        Name of the column that holds efficiency values to be interpolated.

    Returns
    -------
    lin_interp : LinearNDInterpolator
    nn_interp : NearestNDInterpolator
    torque_min, torque_max : float
    rpm_min, rpm_max : float
    """
    torque_min = em_df['rear_motor_torque'].min()
    torque_max = em_df['rear_motor_torque'].max()
    rpm_min = em_df['engine_rpm'].min()
    rpm_max = em_df['engine_rpm'].max()

    t_norm = (em_df['rear_motor_torque'].values - torque_min) / (torque_max - torque_min)
    r_norm = (em_df['engine_rpm'].values - rpm_min) / (rpm_max - rpm_min)
    points = np.column_stack([t_norm, r_norm])

    values = em_df[eff_col].values
    lin_interp = LinearNDInterpolator(points, values)
    nn_interp = NearestNDInterpolator(points, values)

    return lin_interp, nn_interp, torque_min, torque_max, rpm_min, rpm_max

def batch_interpolate(lin_interp, nn_interp, torque_array, rpm_array,
                      torque_min, torque_max, rpm_min, rpm_max):
    """
    Interpolate an array of torque and rpm points using a mixture of
    linear and nearest-neighbor interpolation.

    Parameters
    ----------
    lin_interp, nn_interp : callable
        Interpolation functions created by preprocess_efficiency_map.
    torque_array : np.ndarray
    rpm_array : np.ndarray
    torque_min, torque_max, rpm_min, rpm_max : float
        Normalization bounds.

    Returns
    -------
    interpolated : np.ndarray
        Interpolated efficiency values (or any other property).
    """
    tq_norm = (torque_array - torque_min) / (torque_max - torque_min)
    rpm_norm = (rpm_array - rpm_min) / (rpm_max - rpm_min)

    interpolated = lin_interp(tq_norm, rpm_norm)
    nan_mask = np.isnan(interpolated)
    interpolated[nan_mask] = nn_interp(tq_norm[nan_mask], rpm_norm[nan_mask])
    return interpolated