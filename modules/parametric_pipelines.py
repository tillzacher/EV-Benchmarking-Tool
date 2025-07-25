##### RLC Coastdown Estimator #####
import time
import numpy as np

from modules.data_handler import (
    get_can_files,
    load_can_data,
    segment_df_by_drivemode,
    pitch_correction,
    build_deceleration_df,
    aggregate_buckets,
    fit_quadratic_model,
    filter_segments_range,
    filter_segments_target_value,
    cut_segments,
    sample_segments,
    filter_segments_by_derivative_limit
)
from modules.calculator import (
    compute_drag_coefficients,
    smooth_signals,
    calc_force,
    calculate_all_losses
)
from modules.plotter import (
    plot_segments,
    plot_braking_force
)


def coastdown_pipeline(
    data_cache,
    files,
    do_pitch_correction,
    speed_signal,
    bucket_size,
    vehicle_mass=2300,
    rotating_mass_eq=50,
    frontal_area=2.3,
    smoothing_kwargs=None,
    columns_to_smooth=None,
    steering_angle_limit=None,
    select_suspension_level=None,
    deriv_lower_limit=None,
    deriv_upper_limit=None,
    cut_time=None,
    seed=None,
    target_n_segments=None,
    outlier_threshold=None,
    loss_type='rel',
    generate_plots=False,
    verbose=False,
    filename_prefix='coastdown',
    figsize_1=(10, 5),
    figsize_2=(10, 5),
    speed_range=None
):

    def print_removed_points_info(removed_points, all_segments, label, verbose=False):
        if verbose:
            print(f'Removed {removed_points} data points due to {label} filtering.')
            remaining_data = sum(len(df) for df in all_segments.values())
            print(f"Remaining data points after {label} filtering: {remaining_data}")
    
    start_time = time.time()

    # Gather file list if 'files' is a folder
    if not isinstance(files, list):
        files = get_can_files(folder=files)
        if verbose:
            print(f"Using all data files from folder: {files}")

    all_coast_segments = {}
    coast_segment_counter = 0

    # 1. Read & segment data from each file
    for file in files:
        if verbose:
            print(f"\nProcessing file: {file}")

        # Load DataFrame
        if data_cache is None:
            data = load_can_data(file, verbose=verbose)
        else:
            data = data_cache[file].copy()

        # Must have 'gear_selected_neutral' column
        if 'gear_selected_neutral' not in data.columns:
            if verbose:
                msg = f"Column 'gear_selected_neutral' not found in {file}. Skipping."
                print(msg)
            continue

        # Must have 'suspension_level'
        if 'suspension_level' not in data.columns:
            if verbose:
                msg = f"Column 'suspension_level' not found in {file}. Skipping."
                print(msg)
            continue

        # Segment data by drive mode
        if '2024_09_10-21_57_48.csv' in file:
            # Removing an obvious error segment with 'drop_keys=[22]'
            drive_segments, coast_segments = segment_df_by_drivemode(
                data, drop_keys=[22]
            )
        else:
            drive_segments, coast_segments = segment_df_by_drivemode(data)

        # Gather all coast segments in a single dict
        for _, df_seg in coast_segments.items():
            unique_key = f"file_{file}_segment_{coast_segment_counter}"
            all_coast_segments[unique_key] = df_seg
            coast_segment_counter += 1

    if verbose:
        n_coast = len(all_coast_segments)
        print(f"\nReading of Files complete. Identified {n_coast} coastdown segments.")

    # Optional plot of raw segments
    def plot_raw_segments(all_segments, filename_prefix, figsize, verbose=False):
        if verbose:
            print('Raw coastdown segments:')
            print(f"Number of segments: {len(all_segments)}")
            print(f"Number of data points: {sum(len(df) for df in all_segments.values())}")

        plot_segments(
            all_segments,
            column='vehicle_speed',
            filename=f'{filename_prefix}_segments_raw',
            figsize=figsize
        )
    if generate_plots:
        plot_raw_segments(all_coast_segments, filename_prefix, figsize_1, verbose)

    # 2. Pitch correction (if requested)
    if do_pitch_correction:
        if verbose:
            print("Performing pitch correction...")
        for key in all_coast_segments:
            df_seg = all_coast_segments[key]
            if 'pitch' in df_seg.columns:
                all_coast_segments[key] = pitch_correction(
                    df_seg,
                    vehicle_mass,
                    rotating_mass_eq,
                    verbose=verbose
                )
    else:
        if verbose:
            print("Skipping pitch correction...")

    # 3. Ensure speed_signal is present or fall back to 'vehicle_speed'

    def remove_if_speed_signal_absent(
        all_segments,
        speed_signal,
        columns_to_smooth,
        verbose=False
    ):
        speed_signal_missing_anywhere = any(speed_signal not in df.columns for df in all_segments.values())
        if speed_signal_missing_anywhere:
            if verbose:
                print(f"Speed signal {speed_signal} not found in data. Falling back to 'vehicle_speed'.")
            if columns_to_smooth and speed_signal in columns_to_smooth:
                columns_to_smooth.remove(speed_signal)
                columns_to_smooth.append('vehicle_speed')
                if verbose:
                    print(f"Removing {speed_signal} from columns_to_smooth and adding 'vehicle_speed'.")
            return 'vehicle_speed', columns_to_smooth
        else:
            if verbose:
                print(f"Using speed signal {speed_signal}.")
            return speed_signal, columns_to_smooth
    speed_signal, columns_to_smooth = remove_if_speed_signal_absent(
        all_coast_segments,
        speed_signal,
        columns_to_smooth,
        verbose=verbose
    )

    # 4. Optional time-domain smoothing
    if smoothing_kwargs and columns_to_smooth:
        # If 'vehicle_speed' is in columns_to_smooth, replace with the actual speed_signal
        if 'vehicle_speed' in columns_to_smooth:
            columns_to_smooth.remove('vehicle_speed')
            columns_to_smooth.append(speed_signal)
            if verbose:
                msg = f"Removing 'vehicle_speed' from columns_to_smooth, adding {speed_signal}."
                print(msg)

        if verbose:
            msg = f"Smoothing signals with {smoothing_kwargs.get('filter_type', '(unknown)')} filter..."
            print(msg)

        for key in all_coast_segments:
            df_seg = all_coast_segments[key]
            all_coast_segments[key] = smooth_signals(
                df_seg,
                columns=columns_to_smooth,
                **smoothing_kwargs
            )

    # 5. Filter by steering angle
    if steering_angle_limit is not None:
        if verbose:
            msg = f"Filtering coastdown segments by ±{steering_angle_limit} steering angle limit..."
            print(msg)
        all_coast_segments, metric_rm_steering_angle = filter_segments_range(
            all_coast_segments,
            -steering_angle_limit,
            steering_angle_limit,
            column='steering_wheel_angle',
            verbose=verbose
        )
        print_removed_points_info(metric_rm_steering_angle, all_coast_segments, 'steering angle', verbose)
    else:
        metric_rm_steering_angle = 0
        if verbose:
            print("Skipping steering angle filtering...")

    # 6. Filter out any segments where the brake pedal is activated
    all_coast_segments, metric_rm_brake_pedal = filter_segments_target_value(
        all_coast_segments, 0, column='brake_pedal_activated', verbose=verbose
    )
    print_removed_points_info(metric_rm_brake_pedal, all_coast_segments, 'brake pedal', verbose)

    # 7. Filter by suspension level (if specified)
    if select_suspension_level is not None:
        if verbose:
            msg = f"Filtering coastdown segments by suspension level == {select_suspension_level}..."
            print(msg)
        all_coast_segments, metric_rm_suspension_level = filter_segments_target_value(
            all_coast_segments,
            select_suspension_level,
            column='suspension_level',
            verbose=verbose
        )
        print_removed_points_info(metric_rm_suspension_level, all_coast_segments, 'suspension level', verbose)
    else:
        metric_rm_suspension_level = 0
        if verbose:
            print("Skipping suspension level filtering...")

    # 8. Filter by speed derivative bounds (if provided)
    if deriv_lower_limit is not None and deriv_upper_limit is not None:
        all_coast_segments = filter_segments_by_derivative_limit(
            all_coast_segments,
            speed_signal,
            deriv_lower_limit,
            deriv_upper_limit
        )

    # 9. Cut start/end times of each segment (if requested)
    if cut_time is not None:
        if verbose:
            msg = f"Cutting {cut_time} seconds from start/end of each segment..."
            print(msg)
        all_coast_segments, metric_rm_cut_time = cut_segments(
            all_coast_segments,
            start_cut_duration=cut_time,
            end_cut_duration=cut_time
        )
        print_removed_points_info(metric_rm_cut_time, all_coast_segments, 'time window', verbose)
    else:
        metric_rm_cut_time = 0
        if verbose:
            print("Skipping time window filtering...")

    # 10. Optionally sample the segments (if seed & target_n_segments are set)
    metric_rm_sampling = 0
    if seed and target_n_segments:
        n_prev_data = sum(len(df) for df in all_coast_segments.values())
        n_prev_segments = len(all_coast_segments)

        all_coast_segments = sample_segments(all_coast_segments, seed, target_n_segments)

        n_post_data = sum(len(df) for df in all_coast_segments.values())
        metric_rm_sampling = n_prev_data - n_post_data

    # If select_suspension_level == 2, remove data with speed >= 110 km/h
    if select_suspension_level == 2:
        for key in list(all_coast_segments.keys()):
            df_seg = all_coast_segments[key]
            df_seg = df_seg[df_seg[speed_signal] < 110]
            all_coast_segments[key] = df_seg

        # Discard empty segments
        all_coast_segments = {
            k: v for k, v in all_coast_segments.items() if len(v) > 0
        }

    # Optional plot of processed segments
    def plot_processed_segments(all_segments, filename_prefix, figsize, speed_signal, verbose=False):
        if verbose:
            print('Processed coastdown segments:')
            print(f"Number of segments: {len(all_segments)}")
            print(f"Number of data points: {sum(len(df) for df in all_segments.values())}")

        plot_segments(
            all_segments,
            column=speed_signal,
            filename=f'{filename_prefix}_segments_processed',
            figsize=figsize
        )
    if generate_plots:
        plot_processed_segments(all_coast_segments, filename_prefix, figsize_1, speed_signal, verbose)

    # 11. Build deceleration DataFrame, aggregate, compute force
    deceleration_df = build_deceleration_df(all_coast_segments, column=speed_signal, bucket_size=bucket_size, verbose=False)
    aggregated_df = aggregate_buckets(deceleration_df, bucket_size=bucket_size, verbose=False)
    force_df = calc_force(aggregated_df, vehicle_mass=(vehicle_mass + rotating_mass_eq), verbose=False)

    # Print mean standard deviation of the deceleration-based force
    print('Mean std dev:', force_df['std_force'].mean())

    # Choose a default fit range if select_suspension_level in {2, 'vw'}, else (15,160)
    if select_suspension_level == 2:
        fit_range = (15, 95)
    elif select_suspension_level == 'vw':
        fit_range = (5, 145)
    else:
        fit_range = (15, 160)

    # 12. Fit the quadratic model (f0, f1, f2)
    params, covariance, v_model, force_model, metric_outliers = fit_quadratic_model(
        force_df,
        remove_outlier_threshold=outlier_threshold,
        speed_range=fit_range
    )
    f0, f1, f2 = params

    if any(val is None for val in (f0, f1, f2)):
        raise ValueError("Invalid inputs to calculate_all_losses: f0, f1, and f2 must not be None.")

    # 13. Compute losses under various conditions
    losses = calculate_all_losses(f0, f1, f2, loss_type=loss_type)
    loss_very_low_suspension = losses['very_low_suspension']
    loss_low_suspension = losses['low_suspension']
    loss_medium_suspension = losses['medium_suspension']
    loss_high_suspension = losses['high_suspension']
    loss_CoC_values = losses['CoC_values']
    loss_EPA_values = losses['EPA_values']

    lowest_loss_name = min(losses, key=losses.get)

    # 14. Generate final braking-force plots (if requested)
    if generate_plots:
        # Decide on speed_range for plotting based on the 'vw' gear or else
        param_range = fit_range if select_suspension_level in [2, 'vw'] else speed_range

        plot_braking_force(
            force_df,
            v_model=v_model,
            force_model=force_model,
            filename=f'{filename_prefix}_braking_force',
            plot_parabolas=False,
            figsize=figsize_2,
            params=params,
            speed_range=param_range
        )

        # If 'vw', supply 'vw=True' to the second plot
        is_vw = bool(select_suspension_level == 'vw')
        plot_braking_force(
            force_df,
            v_model=v_model,
            force_model=force_model,
            filename=f'{filename_prefix}_braking_force_parabolas',
            plot_parabolas=True,
            figsize=figsize_2,
            params=params,
            speed_range=param_range,
            vw=is_vw
        )

    # 15. Compute aerodynamic & rolling resistance coefficients, finalize output
    c_air, c_rr = compute_drag_coefficients(f1, f2, vehicle_mass=vehicle_mass, frontal_area=frontal_area)
    total_removed = (
        (steering_angle_limit is not None)        # Just a boolean check,
        + metric_rm_steering_angle                # but these lines are
        + metric_rm_brake_pedal                   # for clarity
        + metric_rm_suspension_level
        + metric_rm_cut_time
        + metric_rm_sampling
    )
    total_removed = metric_rm_steering_angle + metric_rm_brake_pedal + metric_rm_suspension_level + metric_rm_cut_time + metric_rm_sampling

    elapsed_time = time.time() - start_time

    return (
        f0, f1, f2,
        c_air, c_rr,
        metric_rm_steering_angle,
        metric_rm_brake_pedal,
        metric_rm_suspension_level,
        metric_rm_cut_time,
        metric_rm_sampling,
        total_removed,
        metric_outliers,
        loss_very_low_suspension,
        loss_low_suspension,
        loss_medium_suspension,
        loss_high_suspension,
        loss_CoC_values,
        loss_EPA_values,
        lowest_loss_name,
        elapsed_time,
        force_df,
        v_model,
        force_model
)


##### RLC Constspeed Estimator #####
import pandas as pd
import matplotlib.pyplot as plt

from modules.data_handler import (
    get_can_files,
    load_can_data,
    segment_df_by_constant_speed,
    filter_segments_range,
    filter_segments_target_value,
    cut_segments,
    fit_quadratic_model
)

from modules.calculator import (
    smooth_signals,
    calculate_all_losses,
    calculate_accelerating_force,
    compute_drag_coefficients
)

from modules.plotter import (
    plot_segments,
    plot_braking_force,
    plot_ellipse_scatter
)

def constspeed_pipeline(
    data_cache,
    files,
    speed_signal,
    speed_threshold,
    min_n_samples,
    min_avg_speed,
    do_pitch_correction,
    cut_time=None,
    smoothing_kwargs=None,
    columns_to_smooth=None,
    steering_angle_limit=None,
    select_suspension_level=None,
    vehicle_mass=2300,
    frontal_area=2.3,
    outlier_threshold=None,
    loss_type='rel',
    generate_plots=False,
    verbose=False,
    filename_prefix='constspeed',
    figsize_1=(10, 5),
    figsize_2=(10, 5)
):
    start_time = time.time()
    all_segments = {}
    segment_counter = 0

    # 1. Gather file paths, load data
    # If 'files' is not a list, assume it's a folder
    if not isinstance(files, list):
        files = get_can_files(folder=files)
        if verbose:
            print(f"Using all data files from folder: {files}")

    # Process each file
    for file in files:
        if verbose:
            print(f"\nProcessing file: {file}")

        # Load data
        if data_cache is None:
            data = load_can_data(file, verbose=verbose)
        else:
            data = data_cache[file].copy()

        # Check for pitch and suspension columns
        if 'suspension_level' not in data.columns:
            if verbose:
                print(f"Column 'suspension_level' not found in {file}. Skipping this file.")
            continue

        if 'pitch' not in data.columns:
            if verbose:
                print("No 'pitch' column found; defaulting to 0.")
            data['pitch'] = 0

        # Optionally apply pitch correction
        if do_pitch_correction:
            data['pitch_induced_force_N'] = vehicle_mass * 9.81 * np.sin(np.deg2rad(data['pitch']))
            if verbose:
                print("Performing pitch correction...")
        else:
            data['pitch_induced_force_N'] = 0
            if verbose:
                print("Skipping pitch correction...")

        # Segment data by constant speed
        constant_speed_segments = segment_df_by_constant_speed(
            data,
            speed_column=speed_signal,
            threshold=speed_threshold,
            min_n_samples=min_n_samples,
            min_avg_speed=min_avg_speed
        )

        # Combine segments into all_segments with unique keys
        for _, df_seg in constant_speed_segments.items():
            unique_key = f"file_{file}_segment_{segment_counter}"
            all_segments[unique_key] = df_seg
            segment_counter += 1

    if verbose:
        n_seg = len(all_segments)
        print(f"\nReading of Files complete. Identified {n_seg} coastdown segments.")

    # 2. Optional plotting of raw segments
    if generate_plots:
        print("Raw constant speed segments:")
        print(f"Number of segments: {len(all_segments)}")
        n_data_points = sum(len(df) for df in all_segments.values())
        print(f"Number of data points: {n_data_points}")
        plot_segments(
            all_segments,
            column=speed_signal,
            filename=f'{filename_prefix}_segments_raw',
            figsize=figsize_1
        )

    # 3. Compute accelerating force for each segment
    for idx, segment in all_segments.items():
        all_segments[idx] = calculate_accelerating_force(segment, speed_column=speed_signal)

    # 4. Optional time-domain smoothing
    if smoothing_kwargs is not None and columns_to_smooth is not None:
        # If 'vehicle_speed' is in columns_to_smooth, replace with the actual speed_signal
        if 'vehicle_speed' in columns_to_smooth:
            columns_to_smooth.remove('vehicle_speed')
            columns_to_smooth.append(speed_signal)
            if verbose:
                print(f"Replaced 'vehicle_speed' with '{speed_signal}' for smoothing.")
        if verbose:
            print(f"Smoothing signals with {smoothing_kwargs.get('filter_type', '(unknown)')} filter...")

        for key, df_seg in all_segments.items():
            all_segments[key] = smooth_signals(df_seg, columns=columns_to_smooth, **smoothing_kwargs)

    # 5. Filter segments by steering angle
    if steering_angle_limit is not None:
        first_key = next(iter(all_segments), None)
        if first_key and 'steering_wheel_angle' in all_segments[first_key].columns:
            if verbose:
                msg = f"Filtering coastdown segments by steering angle limit of {steering_angle_limit}..."
                print(msg)
            all_segments, metric_rm_steering_angle = filter_segments_range(
                all_segments,
                lower=-steering_angle_limit,
                upper=steering_angle_limit,
                column='steering_wheel_angle',
                verbose=verbose
            )
            if verbose:
                print(f"Removed {metric_rm_steering_angle} data points due to steering angle filtering.")
                n_data = sum(len(df) for df in all_segments.values())
                print(f"Remaining data points after steering angle filtering: {n_data}")
        else:
            if verbose:
                print("Skipping steering angle filtering (column missing).")
            metric_rm_steering_angle = 0
    else:
        if verbose:
            print("Skipping steering angle filtering (limit not provided).")
        metric_rm_steering_angle = 0

    # 6. Filter segments by suspension level
    if select_suspension_level is not None:
        if verbose:
            msg = f"Filtering coastdown segments by suspension level {select_suspension_level}..."
            print(msg)
        all_segments, metric_rm_suspension_level = filter_segments_target_value(
            all_segments,
            select_suspension_level,
            column='suspension_level',
            verbose=verbose
        )
        if verbose:
            print(f"Removed {metric_rm_suspension_level} data points due to suspension level filtering.")
            n_data = sum(len(df) for df in all_segments.values())
            print(f"Remaining data points after suspension level filtering: {n_data}")
    else:
        if verbose:
            print("Skipping suspension level filtering...")
        metric_rm_suspension_level = 0

    # 7. Cut time window from start and end of segments
    if cut_time is not None:
        if verbose:
            msg = f"Cutting time window of {cut_time} seconds from start and end of segments..."
            print(msg)
        all_segments, metric_rm_cut_time = cut_segments(
            all_segments,
            start_cut_duration=cut_time,
            end_cut_duration=cut_time
        )
        if verbose:
            print(f"Removed {metric_rm_cut_time} data points due to time window filtering.")
            n_data = sum(len(df) for df in all_segments.values())
            print(f"Remaining data points after time window filtering: {n_data}")
    else:
        if verbose:
            print("Skipping time window filtering...")
        metric_rm_cut_time = 0

    # 8. Optional plotting of processed segments
    if generate_plots:
        print("Processed constant speed segments:")
        print(f"Number of segments: {len(all_segments)}")
        n_data_points = sum(len(df) for df in all_segments.values())
        print(f"Number of data points: {n_data_points}")
        plot_segments(
            all_segments,
            column=speed_signal,
            filename=f'{filename_prefix}_segments_processed',
            figsize=figsize_1
        )

    # 9. Compute per-segment statistics
    stats_list = []
    for seg_id, seg_data in all_segments.items():
        acc_force = seg_data['accelerating_force']
        speed_col = seg_data['vehicle_speed']  # 'vehicle_speed' is used in the original code

        stats = {
            'segment_id': seg_id,
            'mean_acc_force': acc_force.mean(),
            'median_acc_force': acc_force.median(),
            'std_acc_force': acc_force.std(),
            'mean_speed': speed_col.mean(),
            'median_speed': speed_col.median(),
            'std_speed': speed_col.std()
        }
        if stats['mean_speed'] > min_avg_speed:
            stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    stats_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    stats_df.dropna(inplace=True)

    # Remove points with mean accelerating force < 5
    stats_df = stats_df[stats_df['mean_acc_force'] > 5]

    # Optional ellipse plot
    if generate_plots and not stats_df.empty:
        plot_ellipse_scatter(stats_df)

    # 10. Fit quadratic model
    params, covariance, v_model, force_model, metric_outliers = fit_quadratic_model(
        stats_df,
        remove_outlier_threshold=outlier_threshold
    )
    f0, f1, f2 = params

    if f0 is None or f1 is None or f2 is None:
        raise ValueError("Invalid inputs to calculate_all_losses: f0, f1, f2 must not be None.")

    # 11. Calculate various losses
    losses = calculate_all_losses(f0, f1, f2, loss_type=loss_type)
    loss_very_low_suspension = losses['very_low_suspension']
    loss_low_suspension = losses['low_suspension']
    loss_medium_suspension = losses['medium_suspension']
    loss_high_suspension = losses['high_suspension']
    loss_CoC_values = losses['CoC_values']
    loss_EPA_values = losses['EPA_values']
    lowest_loss_name = min(losses, key=losses.get)

    # 12. Optional plotting of braking force model
    if generate_plots:
        stats_df.sort_values('median_speed', inplace=True)
        plot_braking_force(
            stats_df,
            v_model=v_model,
            force_model=force_model,
            figsize=figsize_2,
            filename=f'{filename_prefix}_braking_force_fit',
            plot_parabolas=False
        )
        plot_braking_force(
            stats_df,
            v_model=v_model,
            force_model=force_model,
            figsize=figsize_2,
            filename=f'{filename_prefix}_braking_force_fit_parabs',
            plot_parabolas=True
        )

    # 13. Compute aerodynamic & rolling resistance coefficients
    c_air, c_rr = compute_drag_coefficients(
        f1, f2,
        vehicle_mass=vehicle_mass,
        frontal_area=frontal_area
    )

    total_removed = metric_rm_steering_angle + metric_rm_suspension_level + metric_rm_cut_time
    elapsed_time = time.time() - start_time

    # 14. Return results as in original code
    return (
        f0, f1, f2,
        c_air, c_rr,
        metric_rm_steering_angle,
        metric_rm_suspension_level,
        metric_rm_cut_time,
        total_removed,
        metric_outliers,
        loss_very_low_suspension,
        loss_low_suspension,
        loss_medium_suspension,
        loss_high_suspension,
        loss_CoC_values,
        loss_EPA_values,
        lowest_loss_name,
        elapsed_time
)

##### Efficiency Map Estimator #####

import gc
from shapely.geometry import Polygon, Point
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from modules.data_handler import (
    get_can_files,
    load_can_data,
    find_enclosing_fullload_curve,
    quantize_data,
    interpolate_efficiency_map,
    smooth_2d_data
)
from modules.calculator import (
    calc_efficiency,
    calc_power_mechanical,
    calc_power_electrical,
    smooth_signals
)
from modules.plotter import (
    efficiency_scatterplot,
    efficiency_contourplot,
    plot_clustered_dynamo_data,
    plot_clustered_driving_data
)


def efficiencymap_pipeline(
    data_cache,
    files,
    gear,
    efficiency_limit_lower=0.0,
    efficiency_limit_upper=1.0,
    soc_limit_lower=0.0,
    soc_limit_upper=1.0,
    remove_neutral_gear=True,
    smoothing_kwargs=None,
    columns_to_smooth=None,
    substract_auxiliary_power=True,
    which_full_load_curve='driving_data',
    twoD_smoothing_kwargs=None,
    high_fidelity_interpolation=False,
    n_quantize_bins=15,
    at_middle_of_bin=False,
    n_interpolation_bins=15,
    global_offset=0,
    generate_plots=False,
    verbose=False,
    filename_prefix='efficiency_map',
    figsize_1=(10, 8),
    figsize_2=(10, 8),
    figsize_3=(10, 8),
    marker_size=10
):
    
    start_time = time.time()

    # 1. Identify if gear='vw' (single gear) or multiple gears
    if gear == 'vw':
        car_has_multiple = False
        files = [f for f in files if 'ID3' in f]  # keep only ID3 files
    else:
        car_has_multiple = True
        files = [f for f in files if 'ID3' not in f]

    # 2. Loop through all files and load data
    combined_data = pd.DataFrame()
    for file in files:
        sub_aux_tmp = substract_auxiliary_power
        if verbose:
            print(f'Processing {file}...')
        # Load from cache or read from disk
        if data_cache is None:
            data = load_can_data(file, verbose=verbose)
        else:
            data = data_cache[file].copy()

        if 'rear_motor_torque' not in data.columns:
            if verbose:
                print(f"Skipping file {file}: 'rear_motor_torque' not found.")
            continue

        # Optional time-domain smoothing
        if smoothing_kwargs and columns_to_smooth:
            if 'dcdc_power_hv' not in data.columns and 'dcdc_power_hv' in columns_to_smooth:
                columns_to_smooth.remove('dcdc_power_hv')
            data = smooth_signals(data, columns=columns_to_smooth, **smoothing_kwargs)

        # Compute mechanical/electrical power, efficiency
        data = calc_power_mechanical(data)
        data = calc_power_electrical(data)
        if sub_aux_tmp and 'dcdc_power_hv' not in data.columns:
            if verbose:
                print("Sub. auxiliary power set True but 'dcdc_power_hv' missing. Setting to False.")
            sub_aux_tmp = False
        data = calc_efficiency(data, substract_auxiliary_power=sub_aux_tmp)

        # Merge into combined DataFrame
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    # 3. Load dynamometer data
    # Distinguish between multiple gear or single gear
    if car_has_multiple:
        try:
            data_dynamo = pd.read_csv(f'data/path_to_ground_truth_dynamo_data_{gear}.csv')

            do_clustering = True
        except FileNotFoundError:
            print(f"File not found: data/path_to_ground_truth_dynamo_data_{gear}.csv")
            data_dynamo = pd.DataFrame()  # Fallback to empty DataFrame
    else:
        try:
            data_dynamo = pd.read_csv('data/path_to_ground_truth_dynamo_data.csv')
            do_clustering = True
        except FileNotFoundError:
            print("File not found: data/path_to_ground_truth_dynamo_data.csv")
            # Fallback to empty DataFrame
            data_dynamo = pd.DataFrame()

    # 4. Normalize data
    torque_max_driving = combined_data['rear_motor_torque'].max()
    rpm_max_driving = combined_data['engine_rpm'].max()
    torque_max_dynamo = data_dynamo['rear_motor_torque'].max()
    rpm_max_dynamo = data_dynamo['engine_rpm'].max()
    torque_max = max(torque_max_driving, torque_max_dynamo)
    rpm_max = max(rpm_max_driving, rpm_max_dynamo)
    data_dynamo['rear_motor_torque_norm'] = data_dynamo['rear_motor_torque'] / torque_max
    data_dynamo['engine_rpm_norm'] = data_dynamo['engine_rpm'] / rpm_max
    combined_data['rear_motor_torque_norm'] = combined_data['rear_motor_torque'] / torque_max
    combined_data['engine_rpm_norm'] = combined_data['engine_rpm'] / rpm_max

    if verbose:
        print('Finished loading data')

    # 5. Filter for correct gear if multiple gears
    if car_has_multiple:
        if verbose:
            print(f'Filtering for gear {gear}...')
        combined_data = combined_data[combined_data['gear_selected'] == gear]

    # Filter for valid torque region (non-negative)
    combined_data = combined_data[combined_data['rear_motor_torque'] >= 0]

    # Filter by efficiency range
    combined_data = combined_data[
        (combined_data['efficiency'] >= efficiency_limit_lower) &
        (combined_data['efficiency'] <= efficiency_limit_upper)
    ]

    # Filter by SoC range
    combined_data = combined_data[
        (combined_data['hv_soc'] >= soc_limit_lower) &
        (combined_data['hv_soc'] <= soc_limit_upper)
    ]

    # Remove neutral gear if requested
    if remove_neutral_gear and 'gear_selected_neutral' in combined_data.columns:
        combined_data = combined_data[
            (combined_data['gear_selected_neutral'] != 0) &
            (combined_data['gear_selected_neutral'] != 8)
        ]
    elif remove_neutral_gear and 'gear_selected_neutral' not in combined_data.columns and verbose:
        print("Skipping removing neutral gear: 'gear_selected_neutral' not in columns.")

    if verbose:
        print('Finished Preprocessing')

    # 6. Load/locate full-load curve
    if which_full_load_curve == 'driving_data':
        full_load_curve = find_enclosing_fullload_curve(
            combined_data, 'engine_rpm', 'rear_motor_torque'
        )
    elif which_full_load_curve == 'dynamo_data':
        if car_has_multiple:
            try:
                full_load_curve = pd.read_csv('data/dummy_path')
            except FileNotFoundError:
                print("Warning: No full-load curve data specified, falling back to enclosing curve.")
                full_load_curve = find_enclosing_fullload_curve(
                    combined_data, 'engine_rpm', 'rear_motor_torque'
                )
        else:
            try:
                full_load_curve = pd.read_csv('data/dummy_path')
            except FileNotFoundError:
                print("Warning: No full-load curve data specified, falling back to enclosing curve.")
                full_load_curve = find_enclosing_fullload_curve(
                    combined_data, 'engine_rpm', 'rear_motor_torque'
                )
    elif which_full_load_curve == 'adjusted':
        if car_has_multiple:
            try:
                full_load_curve = pd.read_csv(f'data/dummy_path')
            except FileNotFoundError:
                print("Warning: No adjusted full-load curve data specified, falling back to enclosing curve.")
                full_load_curve = find_enclosing_fullload_curve(
                    combined_data, 'engine_rpm', 'rear_motor_torque'
                )
        else:
            try:
                full_load_curve = pd.read_csv('data/dummy_path')
            except FileNotFoundError:
                print("Warning: No adjusted full-load curve data specified, falling back to enclosing curve.")
                full_load_curve = find_enclosing_fullload_curve(
                    combined_data, 'engine_rpm', 'rear_motor_torque'
                )
    elif which_full_load_curve == 'overlap':
        if car_has_multiple:
            try:
                full_load_curve = pd.read_csv(f'data/dummy_path')
            except FileNotFoundError:
                print("Warning: No overlap full-load curve data specified, falling back to enclosing curve.")
                full_load_curve = find_enclosing_fullload_curve(
                    combined_data, 'engine_rpm', 'rear_motor_torque'
                )
        else:
            try:
                full_load_curve = pd.read_csv('data/dummy_path')
            except FileNotFoundError:
                print("Warning: No overlap full-load curve data specified, falling back to enclosing curve.")
                full_load_curve = find_enclosing_fullload_curve(
                    combined_data, 'engine_rpm', 'rear_motor_torque'
                )

    # Optional scatter plot of preprocessed data
    if generate_plots:
        efficiency_scatterplot(
            combined_data,
            title=None,
            add_convex_hull=True,
            colormap=None,
            figsize=figsize_1,
            full_load_curve=full_load_curve,
            filename=f'{filename_prefix}_preprocessed',
            marker_size=marker_size
        )

    # 8. Global offset, optional 2D smoothing
    combined_data['efficiency'] += global_offset

    if twoD_smoothing_kwargs is None:
        mean_abs_change = 0
    else:
        # Apply smoothing in (rpm, torque, efficiency) space
        combined_data, mean_abs_change = smooth_2d_data(
            data=combined_data,
            x_col='rear_motor_torque_norm',
            y_col='engine_rpm_norm',
            z_col='efficiency',
            **twoD_smoothing_kwargs
        )
        if verbose:
            print(f"Mean absolute change in efficiency after 2D smoothing: {mean_abs_change:.4f}")

    # Optional scatter plot after smoothing
    if generate_plots:
        efficiency_scatterplot(
            combined_data,
            title=None,
            add_convex_hull=True,
            colormap=None,
            figsize=figsize_1,
            full_load_curve=full_load_curve,
            filename=f'{filename_prefix}_smoothed',
            marker_size=marker_size
        )

    # 8. Dynamo data clustering to obtain comparison points
    if do_clustering:
        X = data_dynamo[['rear_motor_torque_norm', 'engine_rpm_norm']].values
        eps = 0.01
        min_samples = 5
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X)

        data_dynamo['cluster'] = clusters
        # Remove noise points
        if verbose:
            n_noise = data_dynamo[data_dynamo["cluster"] == -1].shape[0]
            print(f'Number of noise points: {n_noise}')

        clustered_data = data_dynamo[data_dynamo['cluster'] != -1]
        # Build cluster stats
        cluster_stats = clustered_data.groupby('cluster').agg({
            'rear_motor_torque': ['mean', 'std'],
            'engine_rpm': ['mean', 'std'],
            'efficiency': ['mean', 'std'],
            'cluster': 'count'
        })
        cluster_stats.columns = [
            'rear_motor_torque', 'torque_std',
            'engine_rpm', 'rpm_std',
            'efficiency_mean', 'efficiency_std',
            'n_points'
        ]

        # Add normalized columns
        cluster_stats['rear_motor_torque_norm'] = cluster_stats['rear_motor_torque'] / torque_max
        cluster_stats['engine_rpm_norm'] = cluster_stats['engine_rpm'] / rpm_max
    else:
        cluster_stats = data_dynamo.copy()
        cluster_stats['cluster'] = cluster_stats.index
        cluster_stats['efficiency_mean'] = cluster_stats['efficiency']

    cluster_stats = cluster_stats.reset_index()  # Make 'cluster' a column if not already

    # Optional plot of clustered dynamo data
    if generate_plots:
        plot_clustered_dynamo_data(
            data_dynamo=data_dynamo,
            cluster_stats=cluster_stats,
            figsize=figsize_3,
            filename=(
                f'figures/{filename_prefix}_dynamo_cluster.pdf'
                if filename_prefix else None
            ),
            marker_size=marker_size
        )

    # Normalize cluster centers
    cluster_stats['rear_motor_torque_norm'] = cluster_stats['rear_motor_torque'] / torque_max
    cluster_stats['engine_rpm_norm'] = cluster_stats['engine_rpm'] / rpm_max

    # 9. Assign each on-road data point to nearest cluster center
    cluster_centers = cluster_stats[['rear_motor_torque_norm', 'engine_rpm_norm']].values
    cluster_labels = cluster_stats['cluster'].values
    cluster_tree = cKDTree(cluster_centers)

    data_points = combined_data[['rear_motor_torque_norm', 'engine_rpm_norm']].values
    distances, indices = cluster_tree.query(data_points)

    combined_data['cluster'] = cluster_labels[indices]

    # Aggregate efficiency statistics from driving data by cluster
    driving_cluster_stats = combined_data.groupby('cluster').agg({
        'efficiency': ['mean', 'std', 'count']
    })
    driving_cluster_stats.columns = ['efficiency_mean_driving', 'efficiency_std_driving', 'n_points_driving']
    driving_cluster_stats = driving_cluster_stats.reset_index()

    # Merge cluster stats
    merged_cluster_stats = pd.merge(
        cluster_stats, driving_cluster_stats,
        on='cluster', how='left'
    )
    merged_cluster_stats['efficiency_diff'] = (
        merged_cluster_stats['efficiency_mean_driving'] - merged_cluster_stats['efficiency_mean']
    )

    # Optional plot of combined_data with cluster assignment
    if generate_plots:
        plot_clustered_driving_data(
            combined_data=combined_data,
            cluster_stats=cluster_stats,
            data_dynamo=data_dynamo,
            figsize=figsize_3,
            filename=(
                f'figures/{filename_prefix}_driving_cluster.pdf'
                if filename_prefix else None
            ),
            marker_size=marker_size
        )

    # 10. Check points contained in the full-load curve polygon
    vertices = full_load_curve[['RPM', 'Torque']].values
    # Ensure polygon is closed
    if not np.array_equal(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    polygon = Polygon(vertices)

    points = merged_cluster_stats[['engine_rpm', 'rear_motor_torque']].values
    contained = [polygon.contains(Point(x, y)) for x, y in points]
    merged_cluster_stats['contained'] = contained

    # Compute mean absolute diff & RMSE for contained points
    contained_data = merged_cluster_stats[merged_cluster_stats['contained']]
    mean_abs_diff = np.mean(np.abs(contained_data['efficiency_diff']))
    rmse = np.sqrt(np.mean(contained_data['efficiency_diff'] ** 2))

    if verbose:
        print(f'Mean Absolute Difference: {100 * mean_abs_diff:.2f}%')
        print(f'RMSE: {100 * rmse:.2f}%')

    # 11. Generate contour plots for reference and difference
    if generate_plots:
        if car_has_multiple:
            if gear == 1:
                try:
                    full_load_curve_2 = pd.read_csv('data/dummy_path')
                except FileNotFoundError:
                    print("Warning: No gear 1 full-load curve data specified, falling back to enclosing curve.")
                    full_load_curve_2 = find_enclosing_fullload_curve(
                        combined_data, 'engine_rpm', 'rear_motor_torque'
                    )
            else:
                try:
                    full_load_curve_2 = pd.read_csv(f'data/dummy_path')
                except FileNotFoundError:
                    print(f"Warning: No gear {gear} full-load curve data specified, falling back to enclosing curve.")
                    full_load_curve_2 = find_enclosing_fullload_curve(
                        combined_data, 'engine_rpm', 'rear_motor_torque'
                    )
        else:
            try:
                full_load_curve_2 = pd.read_csv('data/dummy_path')
            except FileNotFoundError:
                print("Warning: No full-load curve data specified, falling back to enclosing curve.")
                full_load_curve_2 = find_enclosing_fullload_curve(
                    combined_data, 'engine_rpm', 'rear_motor_torque'
                )

        efficiency_contourplot(
            merged_cluster_stats,
            full_load_curve,
            rpm_column='engine_rpm',
            torque_column='rear_motor_torque',
            efficiency_column='efficiency_mean',
            full_load_curve_rpm_column='RPM',
            full_load_curve_torque_column='Torque',
            title=None,
            colormap=None,
            figsize=figsize_1,
            filename=f'{filename_prefix}_dynamo_contourplot',
            add_lab_measurement_points=cluster_stats,
            full_load_curve_2=full_load_curve_2
        )

        efficiency_contourplot(
            merged_cluster_stats,
            full_load_curve,
            rpm_column='engine_rpm',
            torque_column='rear_motor_torque',
            efficiency_column='efficiency_mean_driving',
            full_load_curve_rpm_column='RPM',
            full_load_curve_torque_column='Torque',
            title=None,
            colormap=None,
            figsize=figsize_1,
            filename=f'{filename_prefix}_driving_contourplot',
        )

        efficiency_contourplot(
            merged_cluster_stats,
            full_load_curve,
            rpm_column='engine_rpm',
            torque_column='rear_motor_torque',
            efficiency_column='efficiency_diff',
            full_load_curve_rpm_column='RPM',
            full_load_curve_torque_column='Torque',
            title=None,
            colormap='coolwarm',
            figsize=figsize_1,
            filename=f'{filename_prefix}_diff_contourplot',
            contour_levels=[-0.2, -0.15, -0.1, -0.05, -0.025, 0, 0.025, 0.05, 0.1, 0.15, 0.2]
        )

        # Save final cluster stats if prefix provided
        if filename_prefix:
            merged_cluster_stats.to_csv(
                f'data/sim_efficiency_maps/{filename_prefix}_merged_cluster_stats.csv',
                index=False
            )

    # 12. Perform High-fidelity interpolation section (optional)
    if high_fidelity_interpolation:
        data_quantized = quantize_data(
            combined_data,
            rpm_column='engine_rpm',
            torque_column='rear_motor_torque',
            n_bins=n_quantize_bins,
            at_middle_of_bin=at_middle_of_bin
        )

        interpolated_data = interpolate_efficiency_map(
            data_quantized, n_bins=n_interpolation_bins
        )

        efficiency_contourplot(
            interpolated_data,
            full_load_curve,
            rpm_column='engine_rpm',
            torque_column='rear_motor_torque',
            efficiency_column='efficiency',
            full_load_curve_rpm_column='RPM',
            full_load_curve_torque_column='Torque',
            title=None,
            colormap=None,
            figsize=figsize_3,
            filename=f'{filename_prefix}_interpolated_contourplot'
        )

        if generate_plots:
            # Scatterplot calls for quantized data
            efficiency_scatterplot(
                data_quantized,
                title=None,
                colormap=None,
                figsize=figsize_2,
                full_load_curve=full_load_curve,
                filename=f'{filename_prefix}_quantized',
                marker_size=marker_size
            )
            efficiency_scatterplot(
                data_quantized,
                title=None,
                colormap=None,
                figsize=figsize_2,
                full_load_curve=full_load_curve,
                color_column='efficiency_std',
                filename=f'{filename_prefix}_quantized_std',
                marker_size=marker_size
            )
            efficiency_scatterplot(
                data_quantized,
                title=None,
                colormap=None,
                figsize=figsize_2,
                full_load_curve=full_load_curve,
                color_column='n_points',
                filename=f'{filename_prefix}_quantized_n_points',
                marker_size=marker_size
            )

    # 13. Garbage collection and final output
    elapsed_time = time.time() - start_time

    # Free up memory
    del data_cache, files
    for var_name in [
        'combined_data',
        'data_dynamo',
        'sub_aux_tmp',
        'polygon',
        'points',
        'cluster_stats',
        'cluster_centers',
        'cluster_labels',
        'cluster_tree',
        'dbscan',
        'clustered_data',
        'X',
        'distances',
        'indices',
        'driving_cluster_stats',
        'merged_cluster_stats'
    ]:
        if var_name in locals():
            del locals()[var_name]

    gc.collect()

    return mean_abs_diff, rmse, elapsed_time, mean_abs_change


##### Gear Strategy Finder #####
import pickle

from modules.data_handler import (
    mark_outliers_with_dbscan,
    get_can_files,
    load_can_data,
    merge_close_points,
    perform_kmeans_clustering,
    fit_parametric_spline,
    recreate_line_function
)
from modules.plotter import (
    plot_gear_domains_with_lines,
    plot_mode_comparison,
    plot_gear_shift_data,
    plot_data_with_outliers,
    plot_parametric_spline,
    plot_combined_parametric_splines
)
from modules.calculator import (
    compute_unique_distances,
    compute_arc_length,
    compute_parametric_metrics,
    compute_hausdorff_distance,
    compute_frechet_distance,
    compute_chamfer_distance
)

def gearstrategy_pipeline(
    data_cache=None,
    normal_files='data/gear_strategy/normal',
    sport_files='data/gear_strategy/sport',
    smoothing_kwargs=None,
    columns_to_smooth=None,
    gear_change_offset_samples=0,
    outlier_eps=0.1,
    outlier_min_samples=1,
    close_points_merge_thr=0,
    n_clusters=10,
    attach_endpoints=True,
    anchor_points=False,
    spline_order=1,
    num_knots=5,
    cluster_weight=1,
    normal_weight=None,
    knot_distr_method='cluster',
    verbose=False,
    generate_comparison_plots=True,
    generate_plots=False,
    filename_prefix=None,
    figsize_1=(10, 8),
    figsize_2=(10, 8),
    figsize_3=(10, 8),
    choose_files=2
):
    # 1. Unpack cached data, and re-inject `line_function` references
    if data_cache is None:
        with open('data/path_to_results_normal.pkl', 'rb') as f:
            results_normal_loaded = pickle.load(f)
        with open('data/path_to_results_sport.pkl', 'rb') as f:
            results_sport_loaded = pickle.load(f)

        # Recreate 'line_function' for results_normal_loaded because not pickalable
        for axis2 in results_normal_loaded:
            unique_distances = results_normal_loaded[axis2]['unique_distances']
            x_unique = results_normal_loaded[axis2]['x_unique']
            y_unique = results_normal_loaded[axis2]['y_unique']
            results_normal_loaded[axis2]['line_function'] = recreate_line_function(
                unique_distances,
                x_unique,
                y_unique
            )

        # Recreate 'line_function' for results_sport_loaded because not pickalable
        for axis2 in results_sport_loaded:
            unique_distances = results_sport_loaded[axis2]['unique_distances']
            x_unique = results_sport_loaded[axis2]['x_unique']
            y_unique = results_sport_loaded[axis2]['y_unique']
            results_sport_loaded[axis2]['line_function'] = recreate_line_function(
                unique_distances,
                x_unique,
                y_unique
            )

        # Load the gear change data
        normal_files = get_can_files(normal_files)
        normal_data = load_can_data(normal_files[0], verbose=False)

        sport_files = get_can_files(sport_files)
        sport_data = load_can_data(sport_files[0], verbose=False)


    else:
        # Skip loading by obtaining from data_cache
        results_normal_loaded = data_cache['results_normal_loaded']
        results_sport_loaded = data_cache['results_sport_loaded']
        normal_data = data_cache['normal_data']
        sport_data = data_cache['sport_data']

        # Recreate 'line_function'
        for axis2 in results_normal_loaded:
            unique_distances = results_normal_loaded[axis2]['unique_distances']
            x_unique = results_normal_loaded[axis2]['x_unique']
            y_unique = results_normal_loaded[axis2]['y_unique']
            results_normal_loaded[axis2]['line_function'] = recreate_line_function(
                unique_distances, x_unique, y_unique
            )
        for axis2 in results_sport_loaded:
            unique_distances = results_sport_loaded[axis2]['unique_distances']
            x_unique = results_sport_loaded[axis2]['x_unique']
            y_unique = results_sport_loaded[axis2]['y_unique']
            results_sport_loaded[axis2]['line_function'] = recreate_line_function(
                unique_distances, x_unique, y_unique
            )

    # 2. Generate Normal Weight (if None), optionally smooth time data
    if normal_weight is None:
        normal_weight = 1 - cluster_weight

    if smoothing_kwargs is not None and columns_to_smooth is not None:
        normal_data = smooth_signals(normal_data, columns=columns_to_smooth, **smoothing_kwargs)
        sport_data = smooth_signals(sport_data, columns=columns_to_smooth, **smoothing_kwargs)

    # 3. Process gear change data (create gear_change column, apply offset, etc.)
    normal_data['gear_change'] = None
    gear_up_indices = normal_data.loc[
        (normal_data['gear_selected'].shift(1) == 1) &
        (normal_data['gear_selected'] == 2)
    ].index + gear_change_offset_samples

    normal_data.loc[gear_up_indices[gear_up_indices < len(normal_data)], 'gear_change'] = 'up'

    gear_down_indices = normal_data.loc[
        (normal_data['gear_selected'].shift(1) == 2) &
        (normal_data['gear_selected'] == 1)
    ].index + gear_change_offset_samples

    normal_data.loc[gear_down_indices[gear_down_indices < len(normal_data)], 'gear_change'] = 'down'

    sport_data['gear_change'] = None
    gear_up_indices = sport_data.loc[
        (sport_data['gear_selected'].shift(1) == 1) &
        (sport_data['gear_selected'] == 2)
    ].index + gear_change_offset_samples

    sport_data.loc[gear_up_indices[gear_up_indices < len(sport_data)], 'gear_change'] = 'up'

    gear_down_indices = sport_data.loc[
        (sport_data['gear_selected'].shift(1) == 2) &
        (sport_data['gear_selected'] == 1)
    ].index + gear_change_offset_samples

    sport_data.loc[gear_down_indices[gear_down_indices < len(sport_data)], 'gear_change'] = 'down'

    # Collect only the 'up' gear shifts
    gear_shifts_normal = normal_data.loc[normal_data['gear_change'] == 'up'].copy()
    gear_shifts_sport = sport_data.loc[sport_data['gear_change'] == 'up'].copy()

    # 4. Compute mechanical power columns, drop extraneous columns / invalid rows
    gear_shifts_normal['power_mechanical'] = (
        gear_shifts_normal['rear_motor_torque'] * gear_shifts_normal['engine_rpm'] / 60 * 2 * np.pi / 1000
    )
    gear_shifts_sport['power_mechanical'] = (
        gear_shifts_sport['rear_motor_torque'] * gear_shifts_sport['engine_rpm'] / 60 * 2 * np.pi / 1000
    )

    gear_shifts_normal = gear_shifts_normal[['vehicle_speed', 'engine_rpm', 'rear_motor_torque', 'accelerator_pedal', 'power_mechanical']]
    gear_shifts_sport = gear_shifts_sport[['vehicle_speed', 'engine_rpm', 'rear_motor_torque', 'accelerator_pedal', 'power_mechanical']]

    gear_shifts_normal = gear_shifts_normal[gear_shifts_normal['rear_motor_torque'] >= -10]
    gear_shifts_sport = gear_shifts_sport[gear_shifts_sport['rear_motor_torque'] >= -10]

    # 5. Plot raw gear shift data (if requested)
    if generate_plots:
        plot_gear_shift_data(
            data_normal=gear_shifts_normal,
            data_sport=gear_shifts_sport,
            x_column='vehicle_speed',
            y_column='rear_motor_torque',
            x_label='Vehicle Speed in \si{\kilo\meter\per\hour}',
            y_label='Torque in \si{\\newton\meter}',
            xlim=[70, 130],
            ylim=[0, 250],
            title=None,
            filename=f'figures/{filename_prefix}_gear_shifts_torque.pdf' if filename_prefix else None,
            figsize=figsize_3
        )

        plot_gear_shift_data(
            data_normal=gear_shifts_normal,
            data_sport=gear_shifts_sport,
            x_column='vehicle_speed',
            y_column='accelerator_pedal',
            x_label='Vehicle Speed in \si{\kilo\meter\per\hour}',
            y_label='Accelerator Pedal Position in \si{\percent}',
            xlim=[70, 130],
            ylim=[0, 100],
            title=None,
            filename=f'figures/{filename_prefix}_gear_shifts_accelerator_pedal.pdf' if filename_prefix else None,
            figsize=figsize_3
        )

        plot_gear_shift_data(
            data_normal=gear_shifts_normal,
            data_sport=gear_shifts_sport,
            x_column='vehicle_speed',
            y_column='power_mechanical',
            x_label='Vehicle Speed in \si{\kilo\meter\per\hour}',
            y_label='Power in \si{\kilo\watt}',
            xlim=[70, 130],
            ylim=[0, 330],
            title=None,
            filename=f'figures/{filename_prefix}_gear_shifts_power_mechanical.pdf' if filename_prefix else None,
            figsize=figsize_3
        )

    # 6. Normalize columns
    min_max_values = {
        'engine_rpm': (8000, 15000),
        'rear_motor_torque': (0, 250),
        'accelerator_pedal': (0, 100),
        'power_mechanical': (0, 330),
        'vehicle_speed': (60, 130)
    }

    def normalize_columns(df, columns_min_max):
        for col, (min_val, max_val) in columns_min_max.items():
            df[col + '_norm'] = (df[col] - min_val) / (max_val - min_val)

    normalize_columns(gear_shifts_normal, min_max_values)
    normalize_columns(gear_shifts_sport, min_max_values)

    # 7. Mark outliers with DBSCAN for each relevant dimension
    eps_torque = outlier_eps
    min_samples_torque = outlier_min_samples
    eps_acc = outlier_eps
    min_samples_acc = outlier_min_samples
    eps_power = outlier_eps
    min_samples_power = outlier_min_samples

    # rear_motor_torque
    gear_shifts_normal = mark_outliers_with_dbscan(
        gear_shifts_normal, 'vehicle_speed_norm', 'rear_motor_torque_norm',
        eps_torque, min_samples_torque, verbose=verbose
    )
    gear_shifts_sport = mark_outliers_with_dbscan(
        gear_shifts_sport, 'vehicle_speed_norm', 'rear_motor_torque_norm',
        eps_torque, min_samples_torque, verbose=verbose
    )

    # accelerator_pedal
    gear_shifts_normal = mark_outliers_with_dbscan(
        gear_shifts_normal, 'vehicle_speed_norm', 'accelerator_pedal_norm',
        eps_acc, min_samples_acc, verbose=verbose
    )
    gear_shifts_sport = mark_outliers_with_dbscan(
        gear_shifts_sport, 'vehicle_speed_norm', 'accelerator_pedal_norm',
        eps_acc, min_samples_acc, verbose=verbose
    )

    # power_mechanical
    gear_shifts_normal = mark_outliers_with_dbscan(
        gear_shifts_normal, 'vehicle_speed_norm', 'power_mechanical_norm',
        eps_power, min_samples_power, verbose=verbose
    )
    gear_shifts_sport = mark_outliers_with_dbscan(
        gear_shifts_sport, 'vehicle_speed_norm', 'power_mechanical_norm',
        eps_power, min_samples_power, verbose=verbose
    )

    # 8. Optional plotting of outliers
    if generate_plots:
        for y_column in ['rear_motor_torque_norm', 'accelerator_pedal_norm', 'power_mechanical_norm']:
            if y_column == 'rear_motor_torque_norm':
                y_label = 'Rear Motor Torque Normalized in -'
            elif y_column == 'accelerator_pedal_norm':
                y_label = 'Accelerator Pedal Normalized in -'
            elif y_column == 'power_mechanical_norm':
                y_label = 'Power Mechanical Normalized in -'
            else:
                y_label = y_column.replace('_', ' ').title()

            plot_data_with_outliers(
                df=gear_shifts_normal,
                x_column='vehicle_speed_norm',
                y_column=y_column,
                x_label='Vehicle Speed Normalized in -',
                y_label=y_label,
                title='normal_mode_with_outliers',
                figsize=figsize_1,
                filename=(
                    f'figures/{filename_prefix}_{y_column}_normal_mode_with_outliers.pdf'
                    if filename_prefix else None
                ),
            )

            plot_data_with_outliers(
                df=gear_shifts_sport,
                x_column='vehicle_speed_norm',
                y_column=y_column,
                x_label='Vehicle Speed Normalized in -',
                y_label=y_label,
                title='sport_mode_with_outliers',
                figsize=figsize_1,
                filename=(
                    f'figures/{filename_prefix}_{y_column}_sport_mode_with_outliers.pdf'
                    if filename_prefix else None
                ),
            )

    # Drop outliers
    gear_shifts_normal = gear_shifts_normal[~gear_shifts_normal['is_outlier']]
    gear_shifts_sport = gear_shifts_sport[~gear_shifts_sport['is_outlier']]

    # 9. Merge close points
    gear_shifts_normal = merge_close_points(
        gear_shifts_normal, x_column='vehicle_speed_norm', y_column='rear_motor_torque_norm', threshold=close_points_merge_thr
    )
    gear_shifts_sport = merge_close_points(
        gear_shifts_sport, x_column='vehicle_speed_norm', y_column='rear_motor_torque_norm', threshold=close_points_merge_thr
    )

    gear_shifts_normal = merge_close_points(
        gear_shifts_normal, x_column='vehicle_speed_norm', y_column='accelerator_pedal_norm', threshold=close_points_merge_thr
    )
    gear_shifts_sport = merge_close_points(
        gear_shifts_sport, x_column='vehicle_speed_norm', y_column='accelerator_pedal_norm', threshold=close_points_merge_thr
    )

    gear_shifts_normal = merge_close_points(
        gear_shifts_normal, x_column='vehicle_speed_norm', y_column='power_mechanical_norm', threshold=close_points_merge_thr
    )
    gear_shifts_sport = merge_close_points(
        gear_shifts_sport, x_column='vehicle_speed_norm', y_column='power_mechanical_norm', threshold=close_points_merge_thr
    )

    # 10. Plot preprocessed gear shifts
    if generate_plots:
        plot_gear_shift_data(
            data_normal=gear_shifts_normal,
            data_sport=gear_shifts_sport,
            x_column='vehicle_speed',
            y_column='rear_motor_torque',
            x_label='Vehicle Speed in \si{\kilo\meter\per\hour}',
            y_label='Torque in \si{\\newton\meter}',
            xlim=[70, 130],
            ylim=[0, 250],
            title=None,
            filename=f'figures/{filename_prefix}_gear_shifts_torque_preprocessed.pdf' if filename_prefix else None,
            figsize=figsize_3
        )

        plot_gear_shift_data(
            data_normal=gear_shifts_normal,
            data_sport=gear_shifts_sport,
            x_column='vehicle_speed',
            y_column='accelerator_pedal',
            x_label='Vehicle Speed in \si{\kilo\meter\per\hour}',
            y_label='Accelerator Pedal in \si{\percent}',
            xlim=[70, 130],
            ylim=[0, 100],
            title=None,
            filename=f'figures/{filename_prefix}_gear_shifts_accelerator_pedal_preprocessed.pdf' if filename_prefix else None,
            figsize=figsize_3
        )

        plot_gear_shift_data(
            data_normal=gear_shifts_normal,
            data_sport=gear_shifts_sport,
            x_column='vehicle_speed',
            y_column='power_mechanical',
            x_label='Vehicle Speed in \si{\kilo\meter\per\hour}',
            y_label='Power in \si{\kilo\watt}',
            xlim=[70, 130],
            ylim=[0, 330],
            title=None,
            filename=f'figures/{filename_prefix}_gear_shifts_power_mechanical_preprocessed.pdf' if filename_prefix else None,
            figsize=figsize_3
        )

    # 11. Perform k-means clustering & spline fitting for Normal and Sport
    x_column = 'vehicle_speed'
    axis2_options = ['rear_motor_torque', 'power_mechanical', 'accelerator_pedal']

    results_normal_driving_data = {}
    results_sport_driving_data = {}
    results_normal_driving_data_for_saving = {}
    results_sport_driving_data_for_saving = {}

    n_samples = min(len(gear_shifts_normal), len(gear_shifts_sport))
    if n_clusters > n_samples:
        n_clusters = n_samples

    for y_column in axis2_options:
        if y_column == 'rear_motor_torque':
            y_label = 'Torque in \si{\\newton\meter}'
        elif y_column == 'power_mechanical':
            y_label = 'Power in \si{\\kilo\watt}'
        elif y_column == 'accelerator_pedal':
            y_label = 'Accelerator Pedal in \si{\percent}'
        else:
            y_label = y_column.replace('_', ' ').title()

        # Normal mode
        cluster_centers_normal, sorted_points_normal = perform_kmeans_clustering(
            gear_shifts_normal,
            x_column,
            y_column,
            n_clusters,
            plot=generate_plots,
            attach_endpoints=attach_endpoints,
            title=None,  # No explicit title
            y_label=y_label,
            figsize=figsize_3,
            filename=f'figures/{filename_prefix}_{y_column}_normal_mode.pdf' if filename_prefix else None
        )

        # Sport mode
        cluster_centers_sport, sorted_points_sport = perform_kmeans_clustering(
            gear_shifts_sport,
            x_column,
            y_column,
            n_clusters,
            plot=generate_plots,
            attach_endpoints=attach_endpoints,
            title=None,
            y_label=y_label,
            figsize=figsize_3,
            filename=f'figures/{filename_prefix}_{y_column}_sport_mode.pdf' if filename_prefix else None
        )

        # Sort cluster centers by x
        cluster_centers_normal = np.array(cluster_centers_normal)
        cluster_centers_normal = cluster_centers_normal[np.argsort(cluster_centers_normal[:, 0])]

        cluster_centers_sport = np.array(cluster_centers_sport)
        cluster_centers_sport = cluster_centers_sport[np.argsort(cluster_centers_sport[:, 0])]
        
        # --- spline fitting and plotting ----------------------------------
        # Normal
        fitted_x_normal, fitted_y_normal, spline_normal, knots_normal = fit_parametric_spline(
            cluster_centers_normal,
            normal_points=sorted_points_normal,
            anchor_points=anchor_points,
            k=spline_order,
            num_knots=num_knots,
            cluster_weight=cluster_weight,
            normal_weight=normal_weight,
            knot_method=knot_distr_method,
        )

        if generate_plots:
            plot_parametric_spline(
                cluster_centers_normal,
                sorted_points_normal,
                spline_normal,
                knots_normal,
                fitted_x_normal,
                fitted_y_normal,
                y_label=y_label,
                figsize=figsize_3,
                filename=(
                    f"figures/{filename_prefix}_{y_column}_normal_fitted_spline.pdf"
                    if filename_prefix else None
                ),
            )

        # Sport
        fitted_x_sport, fitted_y_sport, spline_sport, knots_sport = fit_parametric_spline(
            cluster_centers_sport,
            normal_points=sorted_points_sport,
            anchor_points=anchor_points,
            k=spline_order,
            num_knots=num_knots,
            cluster_weight=cluster_weight,
            normal_weight=normal_weight,
            knot_method=knot_distr_method,
        )

        if generate_plots:
            plot_parametric_spline(
                cluster_centers_sport,
                sorted_points_sport,
                spline_sport,
                knots_sport,
                fitted_x_sport,
                fitted_y_sport,
                y_label=y_label,
                figsize=figsize_3,
                filename=(
                    f"figures/{filename_prefix}_{y_column}_sport_fitted_spline.pdf"
                    if filename_prefix else None
                ),
            )
            
        if generate_plots:
            plot_combined_parametric_splines(
                cluster_centers_normal=cluster_centers_normal,
                normal_points_normal=sorted_points_normal,
                spline_normal=spline_normal,
                fitted_x_normal=fitted_x_normal,
                fitted_y_normal=fitted_y_normal,
                cluster_centers_sport=cluster_centers_sport,
                normal_points_sport=sorted_points_sport,
                spline_sport=spline_sport,
                fitted_x_sport=fitted_x_sport,
                fitted_y_sport=fitted_y_sport,
                y_label=y_label,
                figsize=figsize_3,
                filename=(
                    f"figures/{filename_prefix}_{y_column}_combined_fitted_spline.pdf"
                    if filename_prefix else None
                ),
            )

        # --- collect results ----------------------------------------------
        results_normal_driving_data[y_column] = {
            "fitted_x": fitted_x_normal,
            "fitted_y": fitted_y_normal,
            "line_function": spline_normal,   # callable
        }
        results_sport_driving_data[y_column] = {
            "fitted_x": fitted_x_sport,
            "fitted_y": fitted_y_sport,
            "line_function": spline_sport,
        }

        dist_normal = compute_unique_distances(fitted_x_normal, fitted_y_normal)
        results_normal_driving_data_for_saving[y_column] = {
            "unique_distances": dist_normal,
            "x_unique": fitted_x_normal,
            "y_unique": fitted_y_normal,
        }

        dist_sport = compute_unique_distances(fitted_x_sport, fitted_y_sport)
        results_sport_driving_data_for_saving[y_column] = {
            "unique_distances": dist_sport,
            "x_unique": fitted_x_sport,
            "y_unique": fitted_y_sport,
        }

    # 12. Save final results as pickle
    with open('data/path_to_results_normal_driving_data.pkl', 'wb') as f:
        pickle.dump(results_normal_driving_data_for_saving, f)
    with open('data/path_to_results_sport_driving_data.pkl', 'wb') as f:
        pickle.dump(results_sport_driving_data_for_saving, f)

    # 13. Optional: comparison plots
    if generate_comparison_plots:
        for y_column in ['rear_motor_torque', 'power_mechanical']: # add more if desired
            # Normal
            plot_mode_comparison(
                y_column=y_column,
                mode_label='Normal',
                results_loaded=results_normal_loaded,
                results_driving_data=results_normal_driving_data,
                filename_prefix=filename_prefix,
                figsize=figsize_1,
                xlim=[70, 130],
                ylim=[0, 200]
            )
            # Sport
            plot_mode_comparison(
                y_column=y_column,
                mode_label='Sport',
                results_loaded=results_sport_loaded,
                results_driving_data=results_sport_driving_data,
                filename_prefix=filename_prefix,
                figsize=figsize_1,
                xlim=[70, 130],
                ylim=[0, 200]
            )

        plot_gear_domains_with_lines(
            results_normal_driving_data=results_normal_driving_data,
            results_sport_driving_data=results_sport_driving_data,
            results_normal_dynamo_data=results_normal_loaded,
            results_sport_dynamo_data=results_sport_loaded,
            gear1_csv='data/dummy_path',
            gear2_csv='data/dummy_path',
            transmission_ratio_gear_1=15.56,
            transmission_ratio_gear_2=8.16,
            engine_to_wheel_ratio_gear_1=44,
            engine_to_wheel_ratio_gear_2=23,
            filename=f'{filename_prefix}_gear_domains_with_lines.pdf' if filename_prefix else None,
            figsize=figsize_2,
            add_scatters=True
        )

    # 14. Compute distance metrics (Hausdorff, Frechet, etc.) for Normal & Sport
    # --- Normal Mode ---
    curve_ground_normal_rmt = np.column_stack((
        results_normal_loaded['rear_motor_torque']['fitted_x'],
        results_normal_loaded['rear_motor_torque']['fitted_y']
    ))
    curve_driving_normal_rmt = np.column_stack((
        results_normal_driving_data['rear_motor_torque']['fitted_x'],
        results_normal_driving_data['rear_motor_torque']['fitted_y']
    ))

    hd_rmt_normal = compute_hausdorff_distance(curve_ground_normal_rmt, curve_driving_normal_rmt)
    fd_rmt_normal = compute_frechet_distance(curve_ground_normal_rmt, curve_driving_normal_rmt)
    dtw_rmt_normal = None  # Not computed
    cd_rmt_normal = compute_chamfer_distance(curve_ground_normal_rmt, curve_driving_normal_rmt)
    param_metrics_rmt_normal = compute_parametric_metrics(curve_ground_normal_rmt, curve_driving_normal_rmt)
    prmse_rmt_normal = param_metrics_rmt_normal['RMSE']

    curve_ground_normal_pm = np.column_stack((
        results_normal_loaded['power_mechanical']['fitted_x'],
        results_normal_loaded['power_mechanical']['fitted_y']
    ))
    curve_driving_normal_pm = np.column_stack((
        results_normal_driving_data['power_mechanical']['fitted_x'],
        results_normal_driving_data['power_mechanical']['fitted_y']
    ))

    hd_pm_normal = compute_hausdorff_distance(curve_ground_normal_pm, curve_driving_normal_pm)
    fd_pm_normal = compute_frechet_distance(curve_ground_normal_pm, curve_driving_normal_pm)
    dtw_pm_normal = None
    cd_pm_normal = compute_chamfer_distance(curve_ground_normal_pm, curve_driving_normal_pm)
    param_metrics_pm_normal = compute_parametric_metrics(curve_ground_normal_pm, curve_driving_normal_pm)
    prmse_pm_normal = param_metrics_pm_normal['RMSE']

    curve_ground_normal_ap = np.column_stack((
        results_normal_loaded['accelerator_pedal']['fitted_x'],
        results_normal_loaded['accelerator_pedal']['fitted_y']
    ))
    curve_driving_normal_ap = np.column_stack((
        results_normal_driving_data['accelerator_pedal']['fitted_x'],
        results_normal_driving_data['accelerator_pedal']['fitted_y']
    ))

    hd_ap_normal = compute_hausdorff_distance(curve_ground_normal_ap, curve_driving_normal_ap)
    fd_ap_normal = compute_frechet_distance(curve_ground_normal_ap, curve_driving_normal_ap)
    dtw_ap_normal = None
    cd_ap_normal = compute_chamfer_distance(curve_ground_normal_ap, curve_driving_normal_ap)
    param_metrics_ap_normal = compute_parametric_metrics(curve_ground_normal_ap, curve_driving_normal_ap)
    prmse_ap_normal = param_metrics_ap_normal['RMSE']

    prmse_normal = (prmse_rmt_normal + prmse_pm_normal + prmse_ap_normal) / 3

    # --- Sport Mode ---
    curve_ground_sport_rmt = np.column_stack((
        results_sport_loaded['rear_motor_torque']['fitted_x'],
        results_sport_loaded['rear_motor_torque']['fitted_y']
    ))
    curve_driving_sport_rmt = np.column_stack((
        results_sport_driving_data['rear_motor_torque']['fitted_x'],
        results_sport_driving_data['rear_motor_torque']['fitted_y']
    ))

    hd_rmt_sport = compute_hausdorff_distance(curve_ground_sport_rmt, curve_driving_sport_rmt)
    fd_rmt_sport = compute_frechet_distance(curve_ground_sport_rmt, curve_driving_sport_rmt)
    dtw_rmt_sport = None
    cd_rmt_sport = compute_chamfer_distance(curve_ground_sport_rmt, curve_driving_sport_rmt)
    param_metrics_rmt_sport = compute_parametric_metrics(curve_ground_sport_rmt, curve_driving_sport_rmt)
    prmse_rmt_sport = param_metrics_rmt_sport['RMSE']

    curve_ground_sport_pm = np.column_stack((
        results_sport_loaded['power_mechanical']['fitted_x'],
        results_sport_loaded['power_mechanical']['fitted_y']
    ))
    curve_driving_sport_pm = np.column_stack((
        results_sport_driving_data['power_mechanical']['fitted_x'],
        results_sport_driving_data['power_mechanical']['fitted_y']
    ))

    hd_pm_sport = compute_hausdorff_distance(curve_ground_sport_pm, curve_driving_sport_pm)
    fd_pm_sport = compute_frechet_distance(curve_ground_sport_pm, curve_driving_sport_pm)
    dtw_pm_sport = None
    cd_pm_sport = compute_chamfer_distance(curve_ground_sport_pm, curve_driving_sport_pm)
    param_metrics_pm_sport = compute_parametric_metrics(curve_ground_sport_pm, curve_driving_sport_pm)
    prmse_pm_sport = param_metrics_pm_sport['RMSE']

    curve_ground_sport_ap = np.column_stack((
        results_sport_loaded['accelerator_pedal']['fitted_x'],
        results_sport_loaded['accelerator_pedal']['fitted_y']
    ))
    curve_driving_sport_ap = np.column_stack((
        results_sport_driving_data['accelerator_pedal']['fitted_x'],
        results_sport_driving_data['accelerator_pedal']['fitted_y']
    ))

    hd_ap_sport = compute_hausdorff_distance(curve_ground_sport_ap, curve_driving_sport_ap)
    fd_ap_sport = compute_frechet_distance(curve_ground_sport_ap, curve_driving_sport_ap)
    dtw_ap_sport = None
    cd_ap_sport = compute_chamfer_distance(curve_ground_sport_ap, curve_driving_sport_ap)
    param_metrics_ap_sport = compute_parametric_metrics(curve_ground_sport_ap, curve_driving_sport_ap)
    prmse_ap_sport = param_metrics_ap_sport['RMSE']

    prmse_sport = (prmse_rmt_sport + prmse_pm_sport + prmse_ap_sport) / 3
    prmse_global = (prmse_normal + prmse_sport) / 2

    if verbose:
        print("Normal Mode:\n")
        print("Rear Motor Torque (RMT):")
        print(f"- Hausdorff Distance: {hd_rmt_normal:.2f}")
        print(f"- Frechet Distance: {fd_rmt_normal:.2f}")
        print(f"- Chamfer Distance: {cd_rmt_normal:.2f}")
        print(f"- Parametric RMSE: {prmse_rmt_normal:.2f}\n")

        print("Power Mechanical (PM):")
        print(f"- Hausdorff Distance: {hd_pm_normal:.2f}")
        print(f"- Frechet Distance: {fd_pm_normal:.2f}")
        print(f"- Chamfer Distance: {cd_pm_normal:.2f}")
        print(f"- Parametric RMSE: {prmse_pm_normal:.2f}\n")

        print("Accelerator Pedal (AP):")
        print(f"- Hausdorff Distance: {hd_ap_normal:.2f}")
        print(f"- Frechet Distance: {fd_ap_normal:.2f}")
        print(f"- Chamfer Distance: {cd_ap_normal:.2f}")
        print(f"- Parametric RMSE: {prmse_ap_normal:.2f}\n")

        print(f"Mean RMSE (Normal Mode): {prmse_normal:.2f}\n")

        print("Sport Mode:\n")
        print("Rear Motor Torque (RMT):")
        print(f"- Hausdorff Distance: {hd_rmt_sport:.2f}")
        print(f"- Frechet Distance: {fd_rmt_sport:.2f}")
        print(f"- Chamfer Distance: {cd_rmt_sport:.2f}")
        print(f"- Parametric RMSE: {prmse_rmt_sport:.2f}\n")

        print("Power Mechanical (PM):")
        print(f"- Hausdorff Distance: {hd_pm_sport:.2f}")
        print(f"- Frechet Distance: {fd_pm_sport:.2f}")
        print(f"- Chamfer Distance: {cd_pm_sport:.2f}")
        print(f"- Parametric RMSE: {prmse_pm_sport:.2f}\n")

        print("Accelerator Pedal (AP):")
        print(f"- Hausdorff Distance: {hd_ap_sport:.2f}")
        print(f"- Frechet Distance: {fd_ap_sport:.2f}")
        print(f"- Chamfer Distance: {cd_ap_sport:.2f}")
        print(f"- Parametric RMSE: {prmse_ap_sport:.2f}\n")

        print(f"Mean RMSE (Sport Mode): {prmse_sport:.2f}\n")
        print(f"Global Mean RMSE: {prmse_global:.2f}")

    return (
        hd_rmt_normal, fd_rmt_normal, dtw_rmt_normal, cd_rmt_normal, prmse_rmt_normal,
        hd_pm_normal, fd_pm_normal, dtw_pm_normal, cd_pm_normal, prmse_pm_normal,
        hd_ap_normal, fd_ap_normal, dtw_ap_normal, cd_ap_normal, prmse_ap_normal,
        prmse_normal,
        hd_rmt_sport, fd_rmt_sport, dtw_rmt_sport, cd_rmt_sport, prmse_rmt_sport,
        hd_pm_sport, fd_pm_sport, dtw_pm_sport, cd_pm_sport, prmse_pm_sport,
        hd_ap_sport, fd_ap_sport, dtw_ap_sport, cd_ap_sport, prmse_ap_sport,
        prmse_sport, prmse_global
)


##### Sim Pipeline #####
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point, Polygon

from modules.calculator import drag_force
from modules.data_handler import recreate_line_function, preprocess_efficiency_map, batch_interpolate
from modules.plotter import plot_single_signals, plot_signal_pairs


# Gear-check logic (Up-Shift / Down-Shift)
def create_closed_boundary(line_function, unique_distances,
                           support_point=(0, 330), num_samples=200):
    """
    Create a polygon boundary for a gear-change threshold decision criterion.
    """
    min_d = unique_distances[0]
    max_d = unique_distances[-1]

    # Sample the line
    dvals = np.linspace(min_d, max_d, num_samples)
    xvals, yvals = line_function(dvals)

    # Find minimal torque point (lowest y)
    min_torque_idx = np.argmin(yvals)
    min_torque_speed = xvals[min_torque_idx]

    # Construct the polygon boundary
    boundary_x = np.concatenate([
        [-10],                # extended "left" or "origin"
        [min_torque_speed],
        xvals,
        [support_point[0]],
        [-10]
    ])
    boundary_y = np.concatenate([
        [-1000],
        [-1000],
        yvals,
        [support_point[1]],
        [-1000]
    ])

    return boundary_x, boundary_y


def is_point_inside(px, py, boundary_x, boundary_y):
    """
    Check if a point (px, py) is within a polygon defined by boundary_x, boundary_y.
    """
    polygon = Polygon(zip(boundary_x, boundary_y))
    return polygon.contains(Point(px, py))


def gear_check_up_shift(
    torque,
    vehicle_speed,
    drive_mode,    # 'sport' or 'normal' or 'range'
    boundary_type, # 'lab' or 'driving'
    generate_plot=False
):
    """
    Decide if a gear up-shift should occur (returns gear=1 or 2) based on the
    torque, vehicle_speed, drive_mode, and boundary_type.

    If drive_mode is 'range', applies a fixed threshold (20 km/h).
    Otherwise, loads the correct threshold line from a .pkl file, creates
    a closed boundary, and checks if (vehicle_speed, torque) is inside or
    outside that boundary.
    """
    # Quick special handling for 'range' drive mode
    if drive_mode == 'range':
        return 2 if vehicle_speed > 20 else 1

    # Otherwise, pick the appropriate .pkl file
    paths = {
        ('normal', 'lab'):      'data/path_to_results_normal.pkl',
        ('normal', 'driving'):  'data/path_to_results_normal_driving_data.pkl',
        ('sport',  'lab'):      'data/path_to_results_sport.pkl',
        ('sport',  'driving'):  'data/path_to_results_sport_driving_data.pkl'
    }

    file_path = paths.get((drive_mode, boundary_type))
    if file_path is None:
        raise ValueError(f"Unsupported (drive_mode={drive_mode}, boundary_type={boundary_type})")

    with open(file_path, 'rb') as f:
        results = pickle.load(f)

    # Key in the results dict for torque-based threshold
    KEY = 'rear_motor_torque'
    data = results[KEY]
    unique_distances = data['unique_distances']
    x_unique = data['x_unique']  # vehicle speed
    y_unique = data['y_unique']  # torque

    # Rebuild the line function
    line_func = recreate_line_function(unique_distances, x_unique, y_unique)

    # Build the closed boundary polygon
    bx, by = create_closed_boundary(line_func, unique_distances, support_point=(-10, 330))

    inside = is_point_inside(vehicle_speed, torque, bx, by)

    # Optional debug plot
    if generate_plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        dvals = np.linspace(unique_distances[0], unique_distances[-1], 200)
        xvals, yvals = line_func(dvals)
        color = 'b' if drive_mode == 'normal' else 'r'

        ax.plot(xvals, yvals, color, label=f'{drive_mode.capitalize()} Line')
        ax.fill(bx, by, color=color, alpha=0.15, label=f'{drive_mode.capitalize()} Boundary')
        ax.plot(vehicle_speed, torque, 'ko', label=f'Test Point ({vehicle_speed}, {torque})')

        ax.set_xlabel("Vehicle Speed in km/h")
        ax.set_ylabel("Rear Motor Torque in Nm")
        ax.set_title(f"Drive Mode: {drive_mode.capitalize()}, Boundary: {boundary_type.capitalize()}")
        ax.set_xlim([-20, 130])
        ax.set_ylim([-1000, 330])
        ax.grid()
        ax.legend()
        plt.show()

    # If inside the boundary => gear remains 1, else up-shift to 2
    return 1 if inside else 2


def gear_check_down_shift(
    torque,
    vehicle_speed,
    drive_mode,
    boundary_type,
    generate_plot=False
):
    """
    Decide if a gear down-shift should occur (returns gear=1 or 2).

    For 'range' mode, a simple speed threshold is used.
    Otherwise, for normal or sport, a fixed 54 km/h check is used.
    """
    # 'range' mode quick logic
    if drive_mode == 'range':
        return 1 if vehicle_speed < 5 else 2

    # For normal/sport: 54 km/h threshold
    return 1 if vehicle_speed < 54 else 2


# Timestep-based simulation function
def simulate(df, drive_mode='normal', use_lab_gsc=True):
    """
    Compute gear selection and resulting torque/power/efficiency for BOTH 'driving'
    and 'lab' in a single pass.

    The "driving" scenario is stored in `gear_selected_sim`,
    while the "lab" scenario is stored in `gear_lab`.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)

    # Arrays to hold final gear choices
    gear_driving_array = np.zeros(n, dtype=int)
    gear_lab_array = np.zeros(n, dtype=int)

    current_gear_driving = 1
    current_gear_lab = 1

    # Prepare scenario-based torque, efficiency, rpm, etc.
    final_torque_driving = np.zeros(n)
    final_eff_driving = np.zeros(n)
    final_rpm_driving = np.zeros(n)
    final_mech_pow_drv = np.zeros(n)
    final_elec_pow_drv = np.zeros(n)

    final_torque_lab = np.zeros(n)
    final_eff_lab = np.zeros(n)
    final_rpm_lab = np.zeros(n)
    final_mech_pow_lab = np.zeros(n)
    final_elec_pow_lab = np.zeros(n)

    # Decide which boundary_type to use for "lab" gear checks
    boundary_type_driving = 'driving'
    boundary_type_lab = 'lab' if use_lab_gsc else 'driving'

    for i in tqdm(range(n)):
        row = df.iloc[i]
        speed_kmh = row['velocity_profile']

        # DRIVING gear decision
        if current_gear_driving == 1:
            torque_g1_drv = row['engine_torque_gear_1_driving']
            new_gear_drv = gear_check_up_shift(
                torque=torque_g1_drv,
                vehicle_speed=speed_kmh,
                drive_mode=drive_mode,
                boundary_type=boundary_type_driving
            )
        else:
            torque_g2_drv = row['engine_torque_gear_2_driving']
            new_gear_drv = gear_check_down_shift(
                torque=torque_g2_drv,
                vehicle_speed=speed_kmh,
                drive_mode=drive_mode,
                boundary_type=boundary_type_driving
            )
        current_gear_driving = new_gear_drv

        # LAB gear decision
        if current_gear_lab == 1:
            torque_g1_lab = row['engine_torque_gear_1_lab']
            new_gear_lab = gear_check_up_shift(
                torque=torque_g1_lab,
                vehicle_speed=speed_kmh,
                drive_mode=drive_mode,
                boundary_type=boundary_type_lab
            )
        else:
            torque_g2_lab = row['engine_torque_gear_2_lab']
            new_gear_lab = gear_check_down_shift(
                torque=torque_g2_lab,
                vehicle_speed=speed_kmh,
                drive_mode=drive_mode,
                boundary_type=boundary_type_lab
            )
        current_gear_lab = new_gear_lab

        # Store gear for each scenario
        gear_driving_array[i] = current_gear_driving
        gear_lab_array[i] = current_gear_lab

        # Fill scenario-based columns
        if current_gear_driving == 1:
            final_torque_driving[i] = row['engine_torque_gear_1_driving']
            final_eff_driving[i] = row['efficiency_g1_driving']
            final_rpm_driving[i] = row['rpm_gear_1']
            final_mech_pow_drv[i] = row['mechanical_power_g1_driving']
            final_elec_pow_drv[i] = row['electrical_power_g1_driving']
        else:
            final_torque_driving[i] = row['engine_torque_gear_2_driving']
            final_eff_driving[i] = row['efficiency_g2_driving']
            final_rpm_driving[i] = row['rpm_gear_2']
            final_mech_pow_drv[i] = row['mechanical_power_g2_driving']
            final_elec_pow_drv[i] = row['electrical_power_g2_driving']

        if current_gear_lab == 1:
            final_torque_lab[i] = row['engine_torque_gear_1_lab']
            final_eff_lab[i] = row['efficiency_g1_lab']
            final_rpm_lab[i] = row['rpm_gear_1']
            final_mech_pow_lab[i] = row['mechanical_power_g1_lab']
            final_elec_pow_lab[i] = row['electrical_power_g1_lab']
        else:
            final_torque_lab[i] = row['engine_torque_gear_2_lab']
            final_eff_lab[i] = row['efficiency_g2_lab']
            final_rpm_lab[i] = row['rpm_gear_2']
            final_mech_pow_lab[i] = row['mechanical_power_g2_lab']
            final_elec_pow_lab[i] = row['electrical_power_g2_lab']

    # Save columns back
    df['gear_selected_sim'] = gear_driving_array
    df['gear_lab'] = gear_lab_array

    df['engine_torque_driving'] = final_torque_driving
    df['efficiency_driving'] = final_eff_driving
    df['rpm_driving'] = final_rpm_driving
    df['mechanical_power_driving'] = final_mech_pow_drv
    df['electrical_power_driving'] = final_elec_pow_drv

    df['engine_torque_lab'] = final_torque_lab
    df['efficiency_lab'] = final_eff_lab
    df['rpm_lab'] = final_rpm_lab
    df['mechanical_power_lab'] = final_mech_pow_lab
    df['electrical_power_lab'] = final_elec_pow_lab

    # No-regen powers & cumulative energies
    df['electrical_power_driving_no_regen'] = df['electrical_power_driving'].where(df['force_inertial'] > 0, 0)
    df['electrical_power_lab_no_regen']     = df['electrical_power_lab'].where(df['force_inertial'] > 0, 0)

    df['cumulative_energy_driving'] = df['electrical_power_driving_no_regen'].cumsum() / 3600000 / 10
    df['cumulative_energy_lab']     = df['electrical_power_lab_no_regen'].cumsum() / 3600000 / 10

    return df


# Main simulation pipeline
def simulation_pipeline(
    sim_data,
    drive_mode,
    filename_prefix,
    figsize,
    start_idx=0,
    end_idx=-1,
    pitch_corrected=True,
    generate_tex=False,
    generate_figures=False,
    save_figures=False,
    vary_only=None
):
    """
    Simulates a drive cycle with two scenarios in parallel: "driving" and "lab".
    The 'vary_only' parameter determines which of the three major parameters
    (RLC, GSC, EM) differ between lab & driving, and which are forced to on-road.

    vary_only can be:
      - None:  all three differ (RLC, GSC, EM are lab vs. driving)
      - 'rlc': only RLC differs; GSC & EM are forced to "driving" for lab
      - 'gsc': only GSC differs; RLC & EM are forced to "driving" for lab
      - 'em':  only EM differs; RLC & GSC are forced to "driving" for lab
    """

    # Load RLC values
    with open('data/results_pickles/rlc_values.pkl', 'rb') as f:
        rlc_values = pickle.load(f)

    # Decide which ones to allow lab vs. driving
    use_lab_rlc = (vary_only is None) or (vary_only == 'rlc')
    use_lab_gsc = (vary_only is None) or (vary_only == 'gsc')
    use_lab_em  = (vary_only is None) or (vary_only == 'em')

    # Select RLC for driving vs. lab
    if drive_mode == 'normal':
        if pitch_corrected:
            f0_driving, f1_driving, f2_driving = rlc_values['pitch_corrected_till'][1]
        else:
            f0_driving, f1_driving, f2_driving = rlc_values['drivemode_normal_till'][1]
        f0_lab, f1_lab, f2_lab = rlc_values['drivemode_normal_linus'][1]

    elif drive_mode == 'sport':
        f0_driving, f1_driving, f2_driving = rlc_values['low_suspension_till'][1]
        f0_lab, f1_lab, f2_lab = rlc_values['low_suspension_linus'][1]

    elif drive_mode == 'range':
        f0_driving, f1_driving, f2_driving = rlc_values['low_suspension_till'][1]
        f0_lab, f1_lab, f2_lab = rlc_values['low_suspension_linus'][1]

    if not use_lab_rlc:
        # Force RLC: lab = driving
        f0_lab = f0_driving
        f1_lab = f1_driving
        f2_lab = f2_driving

    # Basic kinematics
    sim_data = sim_data.copy()
    sim_data['velocity_profile_ms'] = sim_data['velocity_profile'] / 3.6

    i_1 = 15.56
    i_2 = 8.16
    wheel_radius = 0.35  # m
    mass_vehicle = 2282  # kg
    mass_rotating_eq = 0.015 * mass_vehicle  # kg

    sim_data['rpm_gear_1'] = (sim_data['velocity_profile_ms'] * i_1 * 60) / (wheel_radius * 2 * np.pi)
    sim_data['rpm_gear_2'] = (sim_data['velocity_profile_ms'] * i_2 * 60) / (wheel_radius * 2 * np.pi)

    # Approximate acceleration & inertial force
    sim_data['acceleration'] = np.gradient(sim_data['velocity_profile_ms'], sim_data['time'])
    sim_data['acceleration'] = sim_data['acceleration'].rolling(5, center=True).mean()  # smoothing
    sim_data['force_inertial'] = (mass_vehicle + mass_rotating_eq) * sim_data['acceleration']

    # Convert RLC from "per km/h" to "per m/s"
    f0_driving_mps = f0_driving
    f1_driving_mps = f1_driving * 3.6
    f2_driving_mps = f2_driving * (3.6 ** 2)

    f0_lab_mps = f0_lab
    f1_lab_mps = f1_lab * 3.6
    f2_lab_mps = f2_lab * (3.6 ** 2)

    # Drag forces
    fd_d, fd_l = drag_force(
        sim_data['velocity_profile_ms'],
        f0_driving_mps, f1_driving_mps, f2_driving_mps,
        f0_lab_mps, f1_lab_mps, f2_lab_mps
    )
    sim_data['force_drag_driving'] = fd_d
    sim_data['force_drag_lab']     = fd_l

    sim_data['force_traction_driving'] = sim_data['force_drag_driving'] + sim_data['force_inertial']
    sim_data['force_traction_lab']     = sim_data['force_drag_lab'] + sim_data['force_inertial']

    sim_data['traction_torque_driving'] = sim_data['force_traction_driving'] * wheel_radius
    sim_data['traction_torque_lab']     = sim_data['force_traction_lab'] * wheel_radius

    sim_data['engine_torque_gear_1_driving'] = sim_data['traction_torque_driving'] / i_1
    sim_data['engine_torque_gear_2_driving'] = sim_data['traction_torque_driving'] / i_2
    sim_data['engine_torque_gear_1_lab']     = sim_data['traction_torque_lab'] / i_1
    sim_data['engine_torque_gear_2_lab']     = sim_data['traction_torque_lab'] / i_2

    sim_data.dropna(inplace=True)

    # Preprocess Efficiency Maps for gear1 & gear2
    em_g1 = pd.read_csv('data/sim_efficiency_maps/em_results_g1_merged_cluster_stats.csv')
    em_g2 = pd.read_csv('data/sim_efficiency_maps/em_results_g2_merged_cluster_stats.csv')

    g1_drive_interp, g1_drive_nn, g1_tmin, g1_tmax, g1_rmin, g1_rmax = preprocess_efficiency_map(
        em_g1, 'efficiency_mean_driving'
    )
    g2_drive_interp, g2_drive_nn, g2_tmin, g2_tmax, g2_rmin, g2_rmax = preprocess_efficiency_map(
        em_g2, 'efficiency_mean_driving'
    )

    g1_dyn_interp, g1_dyn_nn, _, _, _, _ = preprocess_efficiency_map(em_g1, 'efficiency_mean')
    g2_dyn_interp, g2_dyn_nn, _, _, _, _ = preprocess_efficiency_map(em_g2, 'efficiency_mean')

    # Driving scenario efficiency
    sim_data['efficiency_g1_driving'] = batch_interpolate(
        g1_drive_interp, g1_drive_nn,
        sim_data['engine_torque_gear_1_driving'].values, sim_data['rpm_gear_1'].values,
        g1_tmin, g1_tmax, g1_rmin, g1_rmax
    )
    sim_data['efficiency_g2_driving'] = batch_interpolate(
        g2_drive_interp, g2_drive_nn,
        sim_data['engine_torque_gear_2_driving'].values, sim_data['rpm_gear_2'].values,
        g2_tmin, g2_tmax, g2_rmin, g2_rmax
    )

    # Lab scenario efficiency
    sim_data['efficiency_g1_lab'] = batch_interpolate(
        g1_dyn_interp, g1_dyn_nn,
        sim_data['engine_torque_gear_1_lab'].values, sim_data['rpm_gear_1'].values,
        g1_tmin, g1_tmax, g1_rmin, g1_rmax
    )
    sim_data['efficiency_g2_lab'] = batch_interpolate(
        g2_dyn_interp, g2_dyn_nn,
        sim_data['engine_torque_gear_2_lab'].values, sim_data['rpm_gear_2'].values,
        g2_tmin, g2_tmax, g2_rmin, g2_rmax
    )

    # If variation of EM not wanted, unify only the "efficiency" columns for lab
    if not use_lab_em:
        sim_data['efficiency_g1_lab'] = sim_data['efficiency_g1_driving']
        sim_data['efficiency_g2_lab'] = sim_data['efficiency_g2_driving']

    # Compute mechanical powers from torque & rpm
    sim_data['mechanical_power_g1_driving'] = (
        sim_data['engine_torque_gear_1_driving'] * sim_data['rpm_gear_1'] * 2 * np.pi / 60
    )
    sim_data['mechanical_power_g2_driving'] = (
        sim_data['engine_torque_gear_2_driving'] * sim_data['rpm_gear_2'] * 2 * np.pi / 60
    )
    sim_data['mechanical_power_g1_lab'] = (
        sim_data['engine_torque_gear_1_lab'] * sim_data['rpm_gear_1'] * 2 * np.pi / 60
    )
    sim_data['mechanical_power_g2_lab'] = (
        sim_data['engine_torque_gear_2_lab'] * sim_data['rpm_gear_2'] * 2 * np.pi / 60
    )

    # Compute electrical powers = mechanical / efficiency
    sim_data['electrical_power_g1_driving'] = (
        sim_data['mechanical_power_g1_driving'] / sim_data['efficiency_g1_driving']
    )
    sim_data['electrical_power_g2_driving'] = (
        sim_data['mechanical_power_g2_driving'] / sim_data['efficiency_g2_driving']
    )
    sim_data['electrical_power_g1_lab'] = (
        sim_data['mechanical_power_g1_lab'] / sim_data['efficiency_g1_lab']
    )
    sim_data['electrical_power_g2_lab'] = (
        sim_data['mechanical_power_g2_lab'] / sim_data['efficiency_g2_lab']
    )


    # Run gear-selection simulation (where GSC may or may not differ)
    sim_data = simulate(sim_data, drive_mode=drive_mode, use_lab_gsc=use_lab_gsc)

    # Plotting results if requested
    plot_data = sim_data.iloc[start_idx:end_idx].copy()
    gear_changes = plot_data['gear_selected_sim'].diff().fillna(0).astype(int)
    gear_change_times = plot_data.loc[gear_changes != 0, 'time']

    single_signals = ['velocity_profile']
    signal_pairs = [
        ('gear_selected_sim', 'gear_lab', 'Gear'),
        ('rpm_driving', 'rpm_lab', 'Simulated Engine Speed in \\si{RPM}'),
        ('force_drag_driving', 'force_drag_lab', 'Simulated Drag Force in \\si{\\newton}'),
        ('engine_torque_driving', 'engine_torque_lab', 'Simulated Engine Torque in \\si{\\newton\\meter}'),
        ('efficiency_driving', 'efficiency_lab', 'Simulated Efficiency in \\si{\\percent}'),
        ('mechanical_power_driving', 'mechanical_power_lab', 'Simulated Mechanical Power in \\si{\\kilo\\watt}'),
        ('electrical_power_driving', 'electrical_power_lab', 'Simulated Electrical Power in \\si{\\kilo\\watt}'),
        ('cumulative_energy_driving', 'cumulative_energy_lab', 'Cumulative Energy Consumption in \\si{\\kilo\\watt\\hour}')
    ]

    if generate_figures:
        plot_single_signals(plot_data, single_signals, gear_change_times, figsize, filename_prefix, save_figures)
        plot_signal_pairs(plot_data, signal_pairs, gear_change_times, figsize, filename_prefix, save_figures)

    # Final results
    cumulative_energy_driving = plot_data['cumulative_energy_driving'].iloc[-1]
    cumulative_energy_lab = plot_data['cumulative_energy_lab'].iloc[-1]
    percentage_difference = ((cumulative_energy_lab - cumulative_energy_driving)
                             / cumulative_energy_driving * 100)

    print(f"Energy consumption driving: {cumulative_energy_driving:.2f} kWh")
    print(f"Energy consumption lab: {cumulative_energy_lab:.2f} kWh")
    print(f"Percentage difference: {percentage_difference:.2f}%")

    return cumulative_energy_driving, cumulative_energy_lab, percentage_difference

##### Pipeline Abstraction Layer  (Make them callable from the same function) #####
from modules.data_handler import parse_and_load_parameters, load_can_data, get_can_files
from modules.plotter import plot_histogram

def run_pipeline_with_best_parameters(
    best_combination,
    algorithm,
    generate_plots=True,
    verbose=False,
    test_drive_kwargs=None,
    data_cache=None
):
    """
    Runs the specified pipeline function with a set of "best" hyperparameters,
    optionally loading data into a shared cache if required.

    Parameters
    ----------
    best_combination : pd.Series or None
        A row of hyperparameters indicating the best combination. If None,
        the function will simply print a message and return.
    algorithm : str
        Name of the pipeline (e.g., 'coastdown', 'efficiencymap', 'gearstrategy', 'constspeed').
    generate_plots : bool, default=True
        Whether to generate and save relevant plots.
    verbose : bool, default=False
        Whether to print verbose logs.
    test_drive_kwargs : dict or None
        Not currently used within this function, but reserved for future expansions.
    data_cache : dict or None
        A dictionary that can hold pre-loaded data. If None, data is loaded as needed.

    Returns
    -------
    None
    """
    if best_combination is None:
        print("No best combination found, doing nothing.")
        return

    # Map pipeline names to their respective functions
    pipeline_functions = {
        'coastdown': coastdown_pipeline,
        'efficiencymap': efficiencymap_pipeline,
        'gearstrategy': gearstrategy_pipeline,
        'constspeed': constspeed_pipeline
    }

    # Select the pipeline function for the given algorithm
    pipeline_function = pipeline_functions.get(algorithm)
    if pipeline_function is None:
        print(f"Algorithm '{algorithm}' not recognized.")
        return

    # Parse hyperparameters from best_combination
    parsed_parameters = parse_and_load_parameters(best_combination)

    # For certain pipelines, a data_cache is required
    if algorithm in ['efficiencymap', 'coastdown', 'constspeed']:
        if data_cache is None:
            files = parsed_parameters.get('files')
            if files == 'all':
                files_to_load = get_can_files()
            else:
                files_to_load = files

            data_cache = {}
            for filename in files_to_load:
                data = load_can_data(filename, verbose=verbose)
                data_cache[filename] = data

        # Insert data_cache into parsed parameters so the pipeline can use it
        parsed_parameters['data_cache'] = data_cache

    # Execute the pipeline with the chosen hyperparameters
    results = pipeline_function(
        **parsed_parameters,
        generate_plots=generate_plots,
        verbose=verbose
    )

    # Print a summary of the pipeline run
    print("\nPipeline run with best parameters:")
    if algorithm == 'efficiencymap':
        # Unpack the returned tuple
        mean_abs_diff, rmse, elapsed_time, mean_abs_change = results
        print(f"Mean Absolute Difference: {round(100 * mean_abs_diff, 2)}%")
        print(f"RMSE: {round(100 * rmse, 2)}%")
        print(f"Elapsed Time: {elapsed_time}")
        print(f"Mean absolute change in efficiency with outlier detection: {mean_abs_change}")

    elif algorithm == 'coastdown':
        (
            f0, f1, f2, c_air, c_rr,
            metric_rm_steering_angle, metric_rm_brake_pedal,
            metric_rm_suspension_level, metric_rm_cut_time,
            metric_rm_sampling, total_removed, metric_outliers,
            loss_very_low_suspension, loss_low_suspension,
            loss_medium_suspension, loss_high_suspension,
            loss_CoC_values, loss_EPA_values, lowest_loss_name,
            elapsed_time
        ) = results

        print("Aerodynamic Drag Coefficient:", c_air)
        print("Rolling Resistance Coefficient:", c_rr)
        losses = {
            'very_low_suspension': loss_very_low_suspension,
            'low_suspension': loss_low_suspension,
            'medium_suspension': loss_medium_suspension,
            'high_suspension': loss_high_suspension,
            'CoC_values': loss_CoC_values,
            'EPA_values': loss_EPA_values
        }
        min_loss = min(losses, key=losses.get)
        print(f'Lowest loss: {min_loss} with value {losses[min_loss]}')

    elif algorithm == 'gearstrategy':
        (
            hd_rmt_normal, fd_rmt_normal, dtw_rmt_normal, cd_rmt_normal, prmse_rmt_normal,
            hd_pm_normal, fd_pm_normal, dtw_pm_normal, cd_pm_normal, prmse_pm_normal,
            hd_ap_normal, fd_ap_normal, dtw_ap_normal, cd_ap_normal, prmse_ap_normal,
            prmse_normal, hd_rmt_sport, fd_rmt_sport, dtw_rmt_sport, cd_rmt_sport,
            prmse_rmt_sport, hd_pm_sport, fd_pm_sport, dtw_pm_sport, cd_pm_sport,
            prmse_pm_sport, hd_ap_sport, fd_ap_sport, dtw_ap_sport, cd_ap_sport,
            prmse_ap_sport, prmse_sport, prmse_global
        ) = results

        print("Normal Mode:")
        print(f'Parametric RMSE Rear Motor Torque: {prmse_rmt_normal}')
        print(f'Parametric RMSE Power Mechanical: {prmse_pm_normal}')
        print(f'Parametric RMSE Accelerator Pedal: {prmse_ap_normal}')
        print(f'Mean RMSE (Normal Mode): {prmse_normal}')
        print("\nSport Mode:")
        print(f'Parametric RMSE Rear Motor Torque: {prmse_rmt_sport}')
        print(f'Parametric RMSE Power Mechanical: {prmse_pm_sport}')
        print(f'Parametric RMSE Accelerator Pedal: {prmse_ap_sport}')
        print(f'Mean RMSE (Sport Mode): {prmse_sport}')

    elif algorithm == 'constspeed':
        (
            f0, f1, f2, c_air, c_rr,
            metric_rm_steering_angle, metric_rm_suspension_level,
            metric_rm_cut_time, total_removed, metric_outliers,
            loss_very_low_suspension, loss_low_suspension,
            loss_medium_suspension, loss_high_suspension,
            loss_CoC_values, loss_EPA_values, lowest_loss_name,
            elapsed_time
        ) = results

        print("Aerodynamic Drag Coefficient:", c_air)
        print("Rolling Resistance Coefficient:", c_rr)
        losses = {
            'very_low_suspension': loss_very_low_suspension,
            'low_suspension': loss_low_suspension,
            'medium_suspension': loss_medium_suspension,
            'high_suspension': loss_high_suspension,
            'CoC_values': loss_CoC_values,
            'EPA_values': loss_EPA_values
        }
        min_loss = min(losses, key=losses.get)
        print(f'Lowest loss: {min_loss} with value {losses[min_loss]}')


def evaluate_results(
    results_df,
    target_metric,
    algorithm,
    higher_is_better=None
):
    """
    Examines a DataFrame of hyperparameter results, identifies the best combination
    by the given metric, prints summary statistics, and returns the varied parameters
    plus the best combination's hyperparameters.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing columns of hyperparameters and metrics.
    target_metric : str
        Name of the column to use for determining the best combination.
    algorithm : str
        Name of the pipeline or algorithm (e.g., 'coastdown', 'constspeed', 'efficiencymap', 'gearstrategy').
    higher_is_better : bool or None
        If True, the best combination is the one with the maximum target_metric.
        If False, the best combination is the one with the minimum target_metric.
        If None, no best combination is selected or reported.

    Returns
    -------
    varied_parameters : list
        The hyperparameters that have more than one unique value in the search space.
    best_params : pd.Series or None
        The hyperparameters for the best combination, or None if higher_is_better is None.
    """
    comment = results_df['comment'].iloc[0]
    print(f'The selected search space includes {len(results_df)} hyperparameter combinations with the comment "{comment}"')

    if higher_is_better is None:
        print('The target metric has no specified direction.')
        best_combination = None
    elif higher_is_better:
        best_combination = results_df.loc[results_df[target_metric].idxmax()]
        print(f'Best combination according to {target_metric}: with value {best_combination[target_metric]}')
        for key, value in best_combination.items():
            print(f'{key}: {value}')
    else:
        best_combination = results_df.loc[results_df[target_metric].idxmin()]
        print(f'Best combination according to {target_metric}: with value {best_combination[target_metric]}')
        for key, value in best_combination.items():
            print(f'{key}: {value}')

    # Drop columns that are not hyperparameters for clarity
    if algorithm == 'coastdown':
        hyperparams_df = results_df.drop(
            columns=[
                'comment', 'generate_plots', 'verbose',
                'f0', 'f1', 'f2', 'c_air', 'c_rr',
                'metric_rm_steering_angle', 'metric_rm_brake_pedal',
                'metric_rm_suspension_level', 'metric_rm_spoiler_level',
                'metric_rm_time_window', 'total_removed',
                'loss_very_low_suspension', 'loss_low_suspension',
                'loss_medium_suspension', 'loss_high_suspension',
                'loss_CoC_values', 'loss_EPA_values', 'lowest_loss_name',
                'metric_rm_cut_time', 'metric_outliers', 'elapsed_time',
                'error', 'metric_rm_sampling'
            ],
            errors='ignore'
        )
        best_params = best_combination.drop(
            [
                'comment', 'generate_plots', 'verbose',
                'f0', 'f1', 'f2', 'c_air', 'c_rr',
                'metric_rm_steering_angle', 'metric_rm_brake_pedal',
                'metric_rm_suspension_level', 'metric_rm_spoiler_level',
                'metric_rm_time_window', 'total_removed',
                'loss_very_low_suspension', 'loss_low_suspension',
                'loss_medium_suspension', 'loss_high_suspension',
                'loss_CoC_values', 'loss_EPA_values', 'lowest_loss_name',
                'metric_rm_cut_time', 'metric_outliers', 'elapsed_time'
            ],
            errors='ignore'
        )

    elif algorithm == 'constspeed':
        hyperparams_df = results_df.drop(
            columns=[
                'comment', 'generate_plots', 'verbose',
                'f0', 'f1', 'f2', 'c_air', 'c_rr',
                'metric_rm_steering_angle', 'metric_rm_suspension_level',
                'metric_rm_cut_time', 'total_removed', 'metric_outliers',
                'loss_very_low_suspension', 'loss_low_suspension',
                'loss_medium_suspension', 'loss_high_suspension',
                'loss_CoC_values', 'loss_EPA_values', 'lowest_loss_name',
                'elapsed_time'
            ],
            errors='ignore'
        )
        best_params = best_combination.drop(
            [
                'comment', 'generate_plots', 'verbose',
                'f0', 'f1', 'f2', 'c_air', 'c_rr',
                'metric_rm_steering_angle', 'metric_rm_suspension_level',
                'metric_rm_cut_time', 'total_removed', 'metric_outliers',
                'loss_very_low_suspension', 'loss_low_suspension',
                'loss_medium_suspension', 'loss_high_suspension',
                'loss_CoC_values', 'loss_EPA_values', 'lowest_loss_name',
                'elapsed_time'
            ],
            errors='ignore'
        )

    elif algorithm == 'efficiencymap':
        hyperparams_df = results_df.drop(
            columns=[
                'comment', 'generate_plots', 'verbose',
                'mean_abs_diff', 'rmse', 'elapsed_time', 'error',
                'mean_abs_change'
            ],
            errors='ignore'
        )
        best_params = best_combination.drop(
            [
                'comment', 'generate_plots', 'verbose',
                'mean_abs_diff', 'rmse', 'elapsed_time', 'error',
                'mean_abs_change'
            ],
            errors='ignore'
        )

    elif algorithm == 'gearstrategy':
        hyperparams_df = results_df.drop(
            columns=[
                'comment', 'generate_plots', 'verbose',
                'hd_rmt_normal', 'fd_rmt_normal', 'dtw_rmt_normal',
                'cd_rmt_normal', 'prmse_rmt_normal', 'hd_pm_normal',
                'fd_pm_normal', 'dtw_pm_normal', 'cd_pm_normal',
                'prmse_pm_normal', 'hd_ap_normal', 'fd_ap_normal',
                'dtw_ap_normal', 'cd_ap_normal', 'prmse_ap_normal',
                'prmse_normal', 'hd_rmt_sport', 'fd_rmt_sport',
                'dtw_rmt_sport', 'cd_rmt_sport', 'prmse_rmt_sport',
                'hd_pm_sport', 'fd_pm_sport', 'dtw_pm_sport', 'cd_pm_sport',
                'prmse_pm_sport', 'hd_ap_sport', 'fd_ap_sport',
                'dtw_ap_sport', 'cd_ap_sport', 'prmse_ap_sport',
                'prmse_sport', 'prmse_global', 'hd_global', 'dtw_global',
                'cd_global', 'fd_global'
            ],
            errors='ignore'
        )
        best_params = best_combination.drop(
            [
                'comment', 'generate_plots', 'verbose',
                'hd_rmt_normal', 'fd_rmt_normal', 'dtw_rmt_normal',
                'cd_rmt_normal', 'prmse_rmt_normal', 'hd_pm_normal',
                'fd_pm_normal', 'dtw_pm_normal', 'cd_pm_normal',
                'prmse_pm_normal', 'hd_ap_normal', 'fd_ap_normal',
                'dtw_ap_normal', 'cd_ap_normal', 'prmse_ap_normal',
                'prmse_normal', 'hd_rmt_sport', 'fd_rmt_sport',
                'dtw_rmt_sport', 'cd_rmt_sport', 'prmse_rmt_sport',
                'hd_pm_sport', 'fd_pm_sport', 'dtw_pm_sport', 'cd_pm_sport',
                'prmse_pm_sport', 'hd_ap_sport', 'fd_ap_sport',
                'dtw_ap_sport', 'cd_ap_sport', 'prmse_ap_sport',
                'prmse_sport', 'prmse_global', 'hd_global', 'dtw_global',
                'cd_global', 'fd_global'
            ],
            errors='ignore'
        )
    else:
        # In case an unsupported algorithm is passed
        hyperparams_df = results_df
        best_params = best_combination

    # Identify hyperparameters that actually vary
    nunique_values = hyperparams_df.nunique(dropna=False)
    varied_parameters = nunique_values[nunique_values > 1].index.tolist()
    print(f'The following hyperparameters were varied in the search: {varied_parameters}')

    # Plot distribution of the target metric
    plot_histogram(results_df, target_metric, bins=50, figsize=(8, 4))

    return varied_parameters, best_params