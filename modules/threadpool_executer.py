import os
import csv
import time
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from functools import partial
from collections import Counter

def run_pipeline_generic(data_cache, pipeline_function, params, param_keys):
    """
    Run a pipeline function with the specified parameters and return the results as a dictionary.
    """
    # Define the output labels for each pipeline inside the function
    output_labels_dict = {
        'efficiencymap_pipeline': [
            "mean_abs_diff", "rmse", "elapsed_time", "mean_abs_change"
        ],
        'coastdown_pipeline': [
            "f0", "f1", "f2",
            "c_air", "c_rr",
            "metric_rm_steering_angle",
            "metric_rm_brake_pedal",
            "metric_rm_suspension_level",
            "metric_rm_cut_time",
            "metric_rm_sampling",
            "total_removed",
            "metric_outliers",
            "loss_very_low_suspension",
            "loss_low_suspension",
            "loss_medium_suspension",
            "loss_high_suspension",
            "loss_CoC_values",
            "loss_EPA_values",
            "lowest_loss_name",
            "elapsed_time"
        ],
        'constspeed_pipeline': [
            "f0", "f1", "f2",
            "c_air", "c_rr",
            "metric_rm_steering_angle",
            "metric_rm_suspension_level",
            "metric_rm_cut_time",
            "total_removed",
            "metric_outliers",
            "loss_very_low_suspension",
            "loss_low_suspension",
            "loss_medium_suspension",
            "loss_high_suspension",
            "loss_CoC_values",
            "loss_EPA_values",
            "lowest_loss_name",
            "elapsed_time"
        ],
        'gearstrategy_pipeline': [
            'hd_rmt_normal',
            'fd_rmt_normal',
            'dtw_rmt_normal',
            'cd_rmt_normal',
            'prmse_rmt_normal',
            'hd_pm_normal',
            'fd_pm_normal',
            'dtw_pm_normal',
            'cd_pm_normal',
            'prmse_pm_normal',
            'hd_ap_normal',
            'fd_ap_normal',
            'dtw_ap_normal',
            'cd_ap_normal',
            'prmse_ap_normal',
            'prmse_normal',
            'hd_rmt_sport',
            'fd_rmt_sport',
            'dtw_rmt_sport',
            'cd_rmt_sport',
            'prmse_rmt_sport',
            'hd_pm_sport',
            'fd_pm_sport',
            'dtw_pm_sport',
            'cd_pm_sport',
            'prmse_pm_sport',
            'hd_ap_sport',
            'fd_ap_sport',
            'dtw_ap_sport',
            'cd_ap_sport',
            'prmse_ap_sport',
            'prmse_sport',
            'prmse_global'
        ]
    }

    # Make a copy of params to avoid modifying the original dict
    params_copy = params.copy()

    # Remove keys that are not needed for the pipeline function
    params_for_pipeline = {k: v for k, v in params_copy.items() if k != 'comment'}

    # Extract parameters in the correct order for the pipeline function
    extracted_params = [params_for_pipeline[key] for key in params_for_pipeline]

    try:
        # Call the pipeline function with the extracted parameters
        result_tuple = pipeline_function(data_cache, *extracted_params)

        # Get the correct output labels for the pipeline function using its name
        pipeline_name = pipeline_function.__name__
        output_labels = output_labels_dict.get(pipeline_name, [f"output_{i}" for i in range(len(result_tuple))])

        # Prepare the result dictionary
        result = params_copy.copy()
        # Handle expected results from pipeline function
        result.update({
            k: v for k, v in zip(output_labels, result_tuple)
        })

    except Exception as e:
        # Handle exceptions and record the error message
        result = params_copy.copy()
        # Fill with None for expected outputs if there's an error
        output_labels = output_labels_dict.get(pipeline_function.__name__, [f"output_{i}" for i in range(15)])
        result.update({
            k: None for k in output_labels
        })
        result['error'] = str(e)
    return result


def execute_pipeline(
    pipeline_function, 
    param_combinations, 
    param_keys, 
    log_path, 
    num_workers=6, 
    data_cache=None, 
    batch_size=100
):
    """
    Execute a pipeline function in parallel using a ProcessPoolExecutor and write results in batches.
    Sends notifications upon completion and optionally during progress at specified intervals.
    Ensures consistent column order across runs.
    
    Args:
        pipeline_function: The function to execute for each parameter combination.
        param_combinations: List of parameter combinations to evaluate.
        param_keys: List of parameter keys corresponding to the combinations.
        log_path: Path to the CSV file for logging results.
        num_workers: Number of parallel workers to use.
        data_cache: Optional shared data cache.
        batch_size: Number of results to write at once to minimize file IO overlap.
    """

    # Helper function to format time in H:M:S
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} hours, {minutes} minutes"

    # Create a list of dictionaries with parameter names as keys and parameter values as values
    param_list = [
        {key: value for key, value in zip(param_keys, params)}
        for params in param_combinations
    ]

    # Prepare the partial function
    partial_run_pipeline_generic = partial(run_pipeline_generic, data_cache)

    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Determine the column order
    if os.path.exists(log_path):
        # Read the existing column order from the file
        with open(log_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            existing_header = next(reader)
        fixed_fieldnames = existing_header
    else:
        # Start with param_keys and add placeholders for result metrics
        fixed_fieldnames = param_keys.copy()

    # Function to ensure consistent fieldnames
    def ensure_consistent_fieldnames(results):
        for key in results.keys():
            if key not in fixed_fieldnames:
                fixed_fieldnames.append(key)

    # Shared multiprocessing lock to synchronize file writes
    write_lock = multiprocessing.Lock()
    results_buffer = []

    # Tracking success and errors
    success_count = 0
    error_count = 0
    error_messages = []

    # Progress reporting setup
    total_tasks = len(param_combinations)
    last_reported_progress = 0
    progress_intervals = []
    if prog_rep_intervall and prog_rep_intervall > 0:
        progress_intervals = [
            int((i / prog_rep_intervall) * total_tasks) 
            for i in range(1, prog_rep_intervall + 1)
        ]

    # Start timing
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(partial_run_pipeline_generic, pipeline_function, params, param_keys): params
            for params in param_list
        }

        for idx, future in enumerate(
            tqdm(concurrent.futures.as_completed(futures), total=total_tasks, desc="Processing combinations")
        ):
            try:
                result = future.result()
                ensure_consistent_fieldnames(result)
                results_buffer.append(result)

                # Write results in batches
                if len(results_buffer) >= batch_size:
                    with write_lock:
                        file_exists = os.path.exists(log_path)
                        with open(log_path, mode='a', newline='') as file:
                            writer = csv.DictWriter(file, fieldnames=fixed_fieldnames)
                            if not file_exists:
                                writer.writeheader()
                            writer.writerows(results_buffer)
                        results_buffer.clear()

                success_count += 1

            except Exception as e:
                error_count += 1
                error_messages.append(str(e))

            # Check if it's time to send a progress report
            completed_tasks = idx + 1
            if progress_intervals and completed_tasks >= progress_intervals[last_reported_progress]:
                last_reported_progress += 1

                # Calculate progress percentage
                progress_percentage = (completed_tasks / total_tasks) * 100

                # Calculate elapsed time and estimated remaining time
                time_so_far = time.time() - start_time
                fraction_done = completed_tasks / total_tasks
                # Avoid divide-by-zero if fraction_done is extremely small
                if fraction_done > 0:
                    estimated_total_time = time_so_far / fraction_done
                else:
                    estimated_total_time = 0
                time_remaining = estimated_total_time - time_so_far

                # Format times
                time_so_far_str = format_time(time_so_far)
                time_remaining_str = format_time(time_remaining)


    # Write any remaining results in the buffer
    if results_buffer:
        with write_lock:
            file_exists = os.path.exists(log_path)
            with open(log_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fixed_fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(results_buffer)

    # Calculate execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_string = format_time(elapsed_time)

    print("Pipeline execution completed and results saved.")