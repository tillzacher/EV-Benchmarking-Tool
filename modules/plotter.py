import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import re


##### Functions for RLC Estimators #####

def plot_rlc_models(prefix, rlc_values, gt_ranges, figsize_main, figsize_others):
    """
    Plots quadratic RLC models, with dashed extrapolated regions, and saves plots as PDFs.

    Parameters:
    prefix (str): Filename prefix for saving plots.
    rlc_values (dict): Dictionary with RLC values and coefficients.
    gt_ranges (dict): Dictionary with valid speed ranges for each RLC.
    figsize_main (tuple): Figure size for the combined plot (Linus, Till, EPA, CoC).
    figsize_others (tuple): Figure size for individual suspension-level plots (Very Low, Low, Medium).

    Returns:
    None
    """
    def quadratic_model(speed, f0, f1, f2):
        return f0 + f1 * speed + f2 * speed**2

    def plot_model(ax, label, coefficients, valid_range, color, style='solid'):
        """
        Plots a quadratic model on the given axis with solid and dashed extrapolated regions.
        """
        f0, f1, f2 = coefficients

        # Define the valid and extrapolated speed ranges
        speed_valid = np.linspace(valid_range[0], valid_range[1], 500)
        speed_extrapolated = np.linspace(0, 170, 600)

        # Plot valid region
        ax.plot(speed_valid, quadratic_model(speed_valid, f0, f1, f2), label=label, color=color, linestyle=style)

        # Plot extrapolated regions (dashed and thinner)
        ax.plot(speed_extrapolated, quadratic_model(speed_extrapolated, f0, f1, f2),
                color=color, linestyle='dashed', alpha=0.7)

    # Combined plot (Linus, Till, EPA, CoC)
    plt.figure(figsize=figsize_main)
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })
    ax = plt.gca()
    # On-road test with pitch correction: TUM Blue Light (#5E94D4)
    # plot_model(ax, "Drive Mode Normal On-Road Test, Pitch Corrected", rlc_values['pitch_corrected_till'][1],
    #            gt_ranges['drivemode_normal'], '#5E94D4')
    # On-road test: TUM Blue Dark (#072140)
    plot_model(ax, "Normal mode \\textbar{} open-road test", rlc_values['drivemode_normal_till'][1],
               gt_ranges['drivemode_normal'], '#5E94D4')
    # Lab test: TUM Green (#9FBA36)
    plot_model(ax, "Normal mode \\textbar{} closed-road test", rlc_values['drivemode_normal_linus'][1],
               gt_ranges['drivemode_normal'], '#9FBA36') 
    # EPA: TUM Red (#EA7237)
    plot_model(ax, "EPA data", rlc_values['EPA_values'][1],
               gt_ranges['drivemode_normal'], '#EA7237')
    # CoC: TUM Orange (#F7B11E)
    plot_model(ax, "CoC data", rlc_values['CoC_values'][1],
               gt_ranges['drivemode_normal'], '#F7B11E')
    
    
    ax.set_xlabel("Speed in \si{\kilo\meter\per\hour}")
    ax.set_ylabel("Force in \si{\\newton}")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend()
    plt.grid(True)
    plt.savefig(f"figures/{prefix}_combined_models.pdf")
    print(f"Combined plot saved to: figures/{prefix}_combined_models.pdf")
    plt.show()

    # Suspension-level plots
    # For these plots, the on-road test will use TUM Blue Dark and lab test TUM Green.
    suspension_levels = ['very_low_suspension', 'low_suspension', 'medium_suspension']
    for level in suspension_levels:
        plt.figure(figsize=figsize_others)
        ax = plt.gca()
        # On-road test: TUM Blue Dark (#072140)
        plot_model(ax, "On-Road Test", rlc_values[f'{level}_till'][1],
                   gt_ranges[level], '#072140')
        # Lab test: TUM Green (#9FBA36)
        plot_model(ax, "Lab Test", rlc_values[f'{level}_linus'][1],
                   gt_ranges[level], '#9FBA36')
        ax.set_xlabel('Speed in \si{\kilo\meter\per\hour}')
        ax.set_ylabel('Force in \si{\\newton}')
        # Remove spines for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.legend(loc='upper left')
        # Set axis limits as needed
        ax.set_ylim(150, 999)
        ax.set_xlim(1, 170)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"figures/{prefix}_{level}_models.pdf")
        plt.show()

def plot_braking_force_mod(df_1, df_2=None, figsize=(10, 5), filename='braking_force_plot'):
    """
    Plots mean braking forces (with standard deviation) for up to two DataFrames.
    - df_1: first force DataFrame
    - df_2: second force DataFrame (optional)
    - figsize: figure size tuple
    - filename: output PDF name
    """

    import matplotlib.pyplot as plt
    df_1 = df_1.copy()
    if df_2 is not None:
        df_2 = df_2.copy()

    # Ensure 'bucket_mid' exists
    for df in [df_1, df_2] if df_2 is not None else [df_1]:
        if df is None:
            continue
        if 'bucket_mid' not in df.columns:
            if 'bucket_floor' in df.columns and 'bucket_ceiling' in df.columns:
                df['bucket_mid'] = (df['bucket_floor'] + df['bucket_ceiling']) / 2
            elif 'mean_speed' in df.columns:
                df['bucket_mid'] = df['mean_speed']
            else:
                raise ValueError("No 'bucket_mid' or suitable columns to build it.")

    # Ensure required columns exist in both DataFrames
    required_cols = ['bucket_mid', 'mean_force', 'std_force']
    def ensure_columns(df):
        """
        Check for required columns. If missing, attempt to rename from 
        the 'constspeed' pipeline columns.
        """
        data_is_from_constspeed_pipeline = False
        for col in required_cols:
            if col not in df.columns:
                # Attempt renames for constspeed pipeline
                if col == 'mean_force':
                    if 'mean_acc_force' in df.columns:
                        df['mean_force'] = df['mean_acc_force']
                        data_is_from_constspeed_pipeline = True
                    else:
                        raise ValueError(f"No column '{col}' or 'mean_acc_force' found.")
                elif col == 'std_force':
                    if 'std_acc_force' in df.columns:
                        df['std_force'] = df['std_acc_force']
                    else:
                        raise ValueError(f"No column '{col}' or 'std_acc_force' found.")
                else:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")
        return data_is_from_constspeed_pipeline

    df1_constspeed = ensure_columns(df_1)
    df2_constspeed = ensure_columns(df_2) if df_2 is not None else False

    # Apply styling via rc_context
    with plt.rc_context({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    }):
        plt.figure(figsize=figsize)

        # --- Plot DF_1 ---
        if not df1_constspeed:
            # Plot mean braking force (line) + std (fill)
            plt.plot(df_1['bucket_mid'], df_1['mean_force'],
                     label='Mean Force No Pitch Correction', color='#EA7237')
            plt.fill_between(
                df_1['bucket_mid'],
                df_1['mean_force'] - df_1['std_force'],
                df_1['mean_force'] + df_1['std_force'],
                color='#EA7237', alpha=0.2, label='Standard Deviation No Pitch Correction \SI{\pm 1}{\sigma}'
            )
        else:
            # constspeed pipeline -> scatter
            plt.scatter(df_1['bucket_mid'], df_1['mean_force'],
                        label='Mean Force', color='#3070B3')

        # --- Plot DF_2 (if supplied) ---
        if df_2 is not None:
            if not df2_constspeed:
                plt.plot(df_2['bucket_mid'], df_2['mean_force'],
                         label='Mean Force Pitch Corrected', color='#3070B3')
                plt.fill_between(
                    df_2['bucket_mid'],
                    df_2['mean_force'] - df_2['std_force'],
                    df_2['mean_force'] + df_2['std_force'],
                    color='#9ABCE4', alpha=0.4, label='Standard Deviation Pitch Corrected $\SI{\pm 1}{\sigma}$'
                )
            else:
                plt.scatter(df_2['bucket_mid'], df_2['mean_force'],
                            label='Mean Force (DF2)', color='#EA7237')

        # Labels and legend
        plt.xlabel('Vehicle Speed in \si{\kilo\meter\per\hour}')
        plt.ylabel('Braking Force in \si{\\newton}')
        plt.legend()

        # Grid styling and axes handling
        plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        # Tight layout
        plt.tight_layout()

        # Save figure
        if filename is not None:
            plt.savefig(f'figures/{filename}.pdf', format='pdf')
            print(f"Plot saved to: figures/{filename}.pdf")

        plt.show()

def generate_latex_table(filename_prefix, caption, f0, f1, f2, 
                         loss_very_low_suspension, loss_low_suspension, 
                         loss_medium_suspension, loss_high_suspension, 
                         loss_CoC_values, loss_EPA_values):
    """
    Generates a LaTeX table with custom styling and saves it to a file.

    Parameters:
        filename_prefix (str): The prefix for the filename.
        caption (str): Caption for the table.
        f0, f1, f2 (float): Values to include in the table, rounded to 4 decimal places.
        loss_very_low_suspension, loss_low_suspension, loss_medium_suspension (float): Loss values, rounded to 2 decimal places.
        loss_CoC_values, loss_EPA_values (float): Additional loss values.
    """
    # Gather all loss values into a list
    loss_values = [
        loss_very_low_suspension, loss_low_suspension, 
        loss_medium_suspension, loss_CoC_values, loss_EPA_values
    ]
    
    # Identify the minimum value
    min_loss = min(loss_values)
    
    # Format the values, bolding the minimum value
    formatted_losses = [
        f"\\textbf{{{value:.2f}}}" if value == min_loss else f"{value:.2f}" 
        for value in loss_values
    ]
    
    # Assign formatted losses to variables
    very_low, low, medium, coc, epa = formatted_losses
    
    # Create the LaTeX table
    latex_table = f"""
        \\begin{{table}}[!htpb]
            \\centering
            \\caption{{{caption}}}
            \\begin{{tabular}}{{ccccccccc}}
            \\toprule
            \\multicolumn{{3}}{{c}}{{\\textbf{{\\glspl{{rlc}}}}}} & \\multicolumn{{5}}{{c}}{{\\textbf{{\\glspl{{rmse}}}}}} \\\\
            \\midrule
            $f_0$ & $f_1$ & $f_2$ & Very Low Susp. & Low Susp. & Medium Susp. & CoC & EPA \\\\
            \\midrule
            {f0:.4f} & {f1:.4f} & {f2:.4f} & {very_low} & {low} & {medium} & {coc} & {epa} \\\\
            \\bottomrule
            \\end{{tabular}}
            \\label{{tab:{filename_prefix}_results}}
        \\end{{table}}
    """

    # Define the output path
    output_path = f"source/results_tables/{filename_prefix}_results.tex"
    
    # Save the table to a file
    with open(output_path, "w") as file:
        file.write(latex_table)
    print(f"LaTeX table saved to {output_path}")

def update_coastdown_results_line(
    file_path,
    suspension_setting,
    f0,
    f1,
    f2,
    loss_very_low,
    loss_low,
    loss_medium,
    loss_high,
    loss_coc,
    loss_epa):
    
    # Read in the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    losses = [loss_very_low, loss_low, loss_medium, loss_coc, loss_epa]
    min_val = min(losses)
    
    # Function to bold the minimum value
    def bold_if_min(val):
        return f"\\textbf{{{val:.2f}}}" if abs(val - min_val) < 1e-12 else f"{val:.2f}"
    
    new_line = (
        f"{suspension_setting} & {f0:.4f} & {f1:.4f} & {f2:.4f} & "
        f"{bold_if_min(loss_very_low)} & {bold_if_min(loss_low)} & {bold_if_min(loss_medium)} & "
        f"{bold_if_min(loss_coc)} & {bold_if_min(loss_epa)} \\\\"
    )
    
    # Create a regex pattern to match the line
    pattern = re.compile(rf'^{suspension_setting}\s*&')
    new_lines = []
    found_line = False
    
    # Update the line if it exists, otherwise append it
    for line in lines:
        if pattern.match(line.strip()):
            new_lines.append(new_line + "\n")
            found_line = True
        else:
            new_lines.append(line)
    
    if not found_line:
        new_lines.append(new_line + "\n")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

from matplotlib.patches import Ellipse

def plot_ellipse_scatter(stats_df):
    """
    Plot medians of acceleration force vs. medians of vehicle speeds,
    adding ellipses scaled by standard deviations of speed and force.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Must contain columns:
          ['median_speed', 'median_acc_force', 'std_speed', 'std_acc_force'].
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stats_df['median_speed'], stats_df['median_acc_force'], color='blue', zorder=2)

    # Ellipses represent 2 * stddev in each dimension
    for i in range(len(stats_df)):
        ellipse = Ellipse(
            (stats_df['median_speed'].iloc[i], stats_df['median_acc_force'].iloc[i]),
            width=2 * stats_df['std_speed'].iloc[i],
            height=2 * stats_df['std_acc_force'].iloc[i],
            edgecolor='none',
            facecolor='blue',
            alpha=0.6
        )
        ax.add_patch(ellipse)

    # Remove Spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xlabel('Median Vehicle Speed in km/h)')
    ax.set_ylabel('Median Accelerating Force in N')
    ax.set_title('Median Accelerating Force vs Median Vehicle Speeds')
    ax.grid(True)
    plt.show()


##### Functions for Drivetrain Efficiency Analysis plots #####

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import Point, Polygon
import matplotlib.colors

# Function to plot convex hull
def plot_convex_hull(x, y, ax, **kwargs):
    points = np.column_stack([x, y])
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], **kwargs)

def efficiency_scatterplot(
    data, 
    title='Torque vs RPM', 
    add_convex_hull=True,
    full_load_curve=None,
    colormap=None, 
    figsize=(10, 10), 
    filename='efficiency_scatterplot', 
    columns=['engine_rpm', 'rear_motor_torque', 'efficiency'],
    color_column='efficiency',
    marker_size=20,
    rasterize=True
):
    """
    Function to plot a scatter plot of torque vs RPM, with optional color mapping for efficiency, standard deviation, or number of points.
    Saves the figure as a PDF in the 'figures' folder.

    Parameters:
    - data: DataFrame containing 'engine_rpm', 'rear_motor_torque', and 'efficiency'.
    - title: String, title of the plot.
    - add_convex_hull: Boolean, whether to add the convex hull or not.
    - colormap: String, the colormap for efficiency values.
    - figsize: Tuple, figure size.
    - filename: String, the name of the file to save the figure as a PDF.
    - color_column: String, column name to use for color encoding ('efficiency', 'efficiency_std', or 'n_points').
    """
    # Apply the same styling as in the reference function
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle log scaling for n_points if selected
    color_data = data[color_column]
    if color_column == 'n_points':
        color_data = np.log1p(color_data)  # Log transformation with log1p to handle zero values smoothly
        
    if colormap is None:
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "custom",
            ["navy", "dodgerblue", "blue", "seagreen", "green", "gold", "orange", "darkorange", "darkred"]
        )

    # Create the scatter plot
    scatter = ax.scatter(
        data[columns[0]],               # Engine RPM
        data[columns[1]],               # Rear Motor Torque
        c=color_data,                   # Color column for encoding, transformed if 'n_points'
        cmap=colormap,                  # Colormap
        s=marker_size,                           # Size of points
        vmin=0.4 if color_column == 'efficiency' else None,  # Set minimum color limit for efficiency only
        rasterized=rasterize,  # Rasterize for performance if needed
    )
    
    # Set the plot title and labels
    if title is not None:
        ax.set_title(title)
    #ax.set_xlabel('Engine RPM \\(\\frac{1}{min}\\)')
    #ax.set_ylabel('Rear Motor Torque (Nm)')
    ax.set_xlabel('Engine Speed in \si{RPM}')
    ax.set_ylabel('Torque in \si{\\newton\meter}')
    #ax.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
    ax.grid(axis='both', linestyle='-', alpha=0.7, zorder=0)

    # Set axis limits
    ax.set_ylim(0, data[columns[1]].max() * 1.1)

    # Style the axes (hide top, right, and left spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if full_load_curve is None:
        # Add convex hull if toggled on
        if add_convex_hull:
            plot_convex_hull(data[columns[0]], data[columns[1]], ax, color='black', lw=2)
    else:
        ax.plot(full_load_curve['RPM'], full_load_curve['Torque'], color='black', lw=2)

    # Add a colorbar with an appropriate label
    cbar = plt.colorbar(scatter, ax=ax)
    cbar_label = 'Efficiency in -' if color_column == 'efficiency' else (
        'Standard Deviation of Efficiency' if color_column == 'efficiency_std' else 'Log of Number of Points'
    )
    cbar.set_label(cbar_label)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the figure as a PDF in the 'figures' folder
    plt.savefig(f'figures/{filename}.pdf', format='pdf',dpi=600)
    print(f'Plot saved to: figures/{filename}.pdf')
    
    # Show the plot
    plt.show()
    
def efficiency_contourplot(
    df,
    full_load_curve,
    rpm_column='engine_rpm',
    torque_column='rear_motor_torque',
    efficiency_column='efficiency',
    full_load_curve_rpm_column='RPM',
    full_load_curve_torque_column='Torque',
    contour_levels=None,
    title='Efficiency Map with Contours',
    colormap=None,
    figsize=(10, 10),
    filename='efficiency_contourplot',
    xlabel='Engine Speed in \si{RPM}',
    ylabel='Torque in \si{\\newton\meter}',
    plot_full_load_curve=True,
    rpm_max=14500,    # Max RPM for normalization
    torque_max=350,    # Max torque for normalization
    add_lab_measurement_points=None,
    full_load_curve_2=None
):        
    # Define default contour levels if not provided
    if contour_levels is None:
        using_default_levels = True
        contour_levels = [1, 0.95, 0.94, 0.93, 0.92, 0.90, 0.88, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4]
        contour_levels.sort()
    else:
        using_default_levels = False

    # Apply consistent styling
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })

    # Use custom colormap if colormap is None
    if colormap is None:
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "custom",
            ["navy", "dodgerblue", "blue", "seagreen", "green", "gold", "orange", "darkorange", "darkred"]
        )

    # Normalize RPM and Torque in both data and full load curve
    df_norm = df.copy()
    df_norm['rpm_norm'] = df[rpm_column] / rpm_max
    df_norm['torque_norm'] = df[torque_column] / torque_max

    full_load_curve_norm = full_load_curve.copy()
    full_load_curve_norm['RPM_norm'] = full_load_curve[full_load_curve_rpm_column] / rpm_max
    full_load_curve_norm['Torque_norm'] = full_load_curve[full_load_curve_torque_column] / torque_max

    # If a second full load curve is provided, normalize it as well.
    if full_load_curve_2 is not None:
        full_load_curve_2_norm = full_load_curve_2.copy()
        full_load_curve_2_norm['RPM_norm'] = full_load_curve_2[full_load_curve_rpm_column] / rpm_max
        full_load_curve_2_norm['Torque_norm'] = full_load_curve_2[full_load_curve_torque_column] / torque_max

    # Create the plot in normalized coordinates
    fig, ax = plt.subplots(figsize=figsize)

    # Remove rows with NaN in the efficiency column
    df_norm = df_norm.dropna(subset=[efficiency_column])

    # Contour plot using normalized values (this is drawn for the first envelope)
    contourf_plot = ax.tricontourf(
        df_norm['rpm_norm'],
        df_norm['torque_norm'],
        df_norm[efficiency_column],
        levels=contour_levels,
        cmap=colormap,
        zorder=1
    )

    # Add contour lines for the first envelope
    contour_lines = ax.tricontour(
        df_norm['rpm_norm'],
        df_norm['torque_norm'],
        df_norm[efficiency_column],
        levels=contour_levels,
        colors='black',
        linewidths=0.5,
        zorder=2
    )

    # Add labels to contour lines (they will be filtered later)
    labels = ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%0.2f')

    # Create a Path object from the normalized full load curve (first envelope)
    vertices = full_load_curve_norm[['RPM_norm', 'Torque_norm']].values
    if not np.array_equal(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])
    path = MplPath(vertices)
    patch = PathPatch(path, transform=ax.transData)

    # Set the clip path for contour elements (first envelope)
    for collection in contourf_plot.collections:
        collection.set_clip_path(patch)
    for collection in contour_lines.collections:
        collection.set_clip_path(patch)

    # If a second full load curve is provided, plot its efficiency map with lowered opacity
    if full_load_curve_2 is not None:
        # Create Path and patch for second envelope
        vertices2 = full_load_curve_2_norm[['RPM_norm', 'Torque_norm']].values
        if not np.array_equal(vertices2[0], vertices2[-1]):
            vertices2 = np.vstack([vertices2, vertices2[0]])
        path2 = MplPath(vertices2)
        patch2 = PathPatch(path2, transform=ax.transData)
        
        # Plot contourf for the second envelope with lowered opacity and a lower zorder so the first envelope stands out
        contourf_plot_2 = ax.tricontourf(
            df_norm['rpm_norm'],
            df_norm['torque_norm'],
            df_norm[efficiency_column],
            levels=contour_levels,
            cmap=colormap,
            alpha=0.3,
            zorder=0
        )
        for collection in contourf_plot_2.collections:
            collection.set_clip_path(patch2)
            
        # Plot contour lines for the second envelope with lowered opacity
        contour_lines_2 = ax.tricontour(
            df_norm['rpm_norm'],
            df_norm['torque_norm'],
            df_norm[efficiency_column],
            levels=contour_levels,
            colors='black',
            linewidths=0.5,
            alpha=0.3,
            zorder=0
        )
        for collection in contour_lines_2.collections:
            collection.set_clip_path(patch2)

    # Remove contour labels outside the envelope(s)
    full_load_polygon_1 = Polygon(vertices)
    if full_load_curve_2 is not None:
        full_load_polygon_2 = Polygon(vertices2)
        for label in labels:
            x, y = label.get_position()
            point = Point(x, y)
            if not (full_load_polygon_1.contains(point) or full_load_polygon_2.contains(point)):
                label.set_visible(False)
    else:
        for label in labels:
            x, y = label.get_position()
            point = Point(x, y)
            if not full_load_polygon_1.contains(point):
                label.set_visible(False)

    # Plot min and max efficiency within the envelope(s)
    if full_load_curve_2 is not None:
        df_inside_polygon = df_norm[
            df_norm.apply(
                lambda row: full_load_polygon_1.contains(Point(row['rpm_norm'], row['torque_norm'])) or 
                            full_load_polygon_2.contains(Point(row['rpm_norm'], row['torque_norm'])), axis=1
            )
        ]
    else:
        df_inside_polygon = df_norm[
            df_norm.apply(lambda row: full_load_polygon_1.contains(Point(row['rpm_norm'], row['torque_norm'])), axis=1)
        ]

    if not df_inside_polygon.empty:
        min_efficiency_idx = df_inside_polygon[efficiency_column].idxmin()
        max_efficiency_idx = df_inside_polygon[efficiency_column].idxmax()
        if using_default_levels:
            ax.scatter(
                df_norm.loc[min_efficiency_idx, 'rpm_norm'],
                df_norm.loc[min_efficiency_idx, 'torque_norm'],
                color='blue', marker='o', s=100, label='Min Efficiency', zorder=4
            )
            ax.scatter(
                df_norm.loc[max_efficiency_idx, 'rpm_norm'],
                df_norm.loc[max_efficiency_idx, 'torque_norm'],
                color='red', marker='x', s=100, label='Max Efficiency', zorder=4
            )
        else:
            ax.scatter(
                df_norm.loc[min_efficiency_idx, 'rpm_norm'],
                df_norm.loc[min_efficiency_idx, 'torque_norm'],
                color='blue', marker='o', s=100, label='Max Underestimation', zorder=4
            )
            ax.scatter(
                df_norm.loc[max_efficiency_idx, 'rpm_norm'],
                df_norm.loc[max_efficiency_idx, 'torque_norm'],
                color='red', marker='x', s=100, label='Max Overestimation', zorder=4
            )

    # Plot the full load curve (first envelope) if toggled
    if plot_full_load_curve:
        ax.plot(
            full_load_curve_norm['RPM_norm'],
            full_load_curve_norm['Torque_norm'],
            color='black',
            linewidth=2,
            label='Full Operating Envelope',
            zorder=3
        )

    # Plot the second full load curve if provided (with lowered opacity)
    if full_load_curve_2 is not None:
        ax.plot(
            full_load_curve_2_norm['RPM_norm'],
            full_load_curve_2_norm['Torque_norm'],
            color='black',
            linewidth=2,
            #label='Full Operating Envelope 2',
            zorder=3,
            alpha=0.5
        )

    # Plot the lab measurement points if provided
    if add_lab_measurement_points is not None:
        ax.scatter(
            add_lab_measurement_points['engine_rpm_norm'],
            add_lab_measurement_points['rear_motor_torque_norm'],
            color='white',
            marker='x',
            s=50,
            alpha=0.8,
            label='Lab Measurement Points',
            zorder=5
        )

    # Set axis labels and title
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set axis ticks to original domain values
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(rpm_max * x)}" for x in np.linspace(0, 1, 6)])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(torque_max * y)}" for y in np.linspace(0, 1, 6)])

    # Disable spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add a colorbar
    cbar = plt.colorbar(contourf_plot, ax=ax)
    if using_default_levels:
        cbar.set_label('Efficiency in -')
    else:
        cbar.set_label('Efficiency Difference in -')

    # Add grid and legend (legend always top right)
    ax.grid(True, linestyle='-', alpha=0.7, zorder=6)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.pdf', format='pdf')
    print(f'Plot saved to: figures/{filename}.pdf')
    plt.show()

def plot_clustered_dynamo_data(
    data_dynamo,
    cluster_stats,
    figsize,
    filename=None,
    marker_size=10
):
    """
    Plot the dynamometer data (colored by efficiency) and overlay cluster centers.

    Parameters
    ----------
    data_dynamo : pd.DataFrame
        Dynamo data containing 'engine_rpm', 'rear_motor_torque', 'efficiency'.
    cluster_stats : pd.DataFrame
        Cluster center positions (columns: engine_rpm, rear_motor_torque).
    figsize : tuple
        Size of the figure.
    filename : str or None
        If provided, saves the figure at this path.
    marker_size : int
        Size of scatter markers.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=data_dynamo,
        x='engine_rpm',
        y='rear_motor_torque',
        hue='efficiency',
        palette='viridis',
        alpha=0.6,
        edgecolor=None,
        legend=False,
        s=marker_size
    )
    plt.scatter(
        cluster_stats['engine_rpm'],
        cluster_stats['rear_motor_torque'],
        marker='x',
        color='red',
        s=100,
        label='Cluster Centers'
    )

    # Disable spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Axis limits and labels
    plt.xlim(0, data_dynamo['engine_rpm'].max() * 1.1)
    plt.ylim(0, data_dynamo['rear_motor_torque'].max() * 1.1)
    plt.grid()
    plt.ylabel('Rear Motor Torque in Nm')
    plt.xlabel('Engine Speed in RPM')
    plt.legend()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_clustered_driving_data(
    combined_data,
    cluster_stats,
    data_dynamo,
    figsize,
    filename=None,
    marker_size=10
):
    """
    Plot the driving data (colored by cluster) and overlay dynamometer cluster centers.

    Parameters
    ----------
    combined_data : pd.DataFrame
        Preprocessed driving data with assigned 'cluster' column.
    cluster_stats : pd.DataFrame
        Contains cluster center positions under columns 'engine_rpm', 'rear_motor_torque'.
    data_dynamo : pd.DataFrame
        Used for setting axis limits based on max rpm and torque.
    figsize : tuple
        Figure size (width, height).
    filename : str or None
        If provided, saves the figure at this path.
    marker_size : int
        Size of scatter markers.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=combined_data,
        x='engine_rpm',
        y='rear_motor_torque',
        hue='cluster',
        palette='tab10',
        alpha=0.6,
        edgecolor=None,
        legend=False,
        s=marker_size
    )

    plt.scatter(
        cluster_stats['engine_rpm'],
        cluster_stats['rear_motor_torque'],
        marker='x',
        color='red',
        s=100,
        label='Cluster Centers'
    )

    # Disable spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Axis limits and labels
    plt.xlim(0, data_dynamo['engine_rpm'].max() * 1.1)
    plt.ylim(0, data_dynamo['rear_motor_torque'].max() * 1.1)
    plt.grid()
    plt.ylabel('Rear Motor Torque in Nm')
    plt.xlabel('Engine Speed in RPM')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


##### Functions for Coastdown Analysis plots #####

def plot_segments(segments_dict, column='vehicle_speed', figsize=(10, 5), filename='segments_plot'):
    """
    Function to plot vehicle segments over time.
    Saves the figure as a PDF in the 'figures' folder.

    Parameters:
    - segments_dict: Dictionary of DataFrames, where keys are segment labels, and values are DataFrames containing time-series data.
    - column: String, the column to plot from each segment DataFrame.
    - figsize: Tuple, figure size.
    - filename: String, the name of the file to save the figure as a PDF.
    """
    # Apply the same styling as in the reference function
    plt.figure(figsize=figsize)
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })

    # Plot each segment's data
    for key in segments_dict:
        df = segments_dict[key]
        #plt.plot(df['time'] - df['time'].min(), df[column], label=key)
        plt.plot(df['seconds'] - df['seconds'].min(), df[column], label=key)

    # set y range
    plt.ylim(0, 220)
    # Set plot title and labels
    #plt.title(column)
    plt.xlabel('Time in \si{\second}')
    plt.ylabel('Vehicle Speed in \si{\kilo\meter\per\hour}')

    # Grid styling and axes handling (consistent with other functions)
    plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # Set figure size
    plt.gcf().set_size_inches(figsize)

    # Tight layout for better spacing
    plt.tight_layout()

    if filename is not None:
        # Save the figure as a PDF in the 'figures' folder
        plt.savefig(f'figures/{filename}.pdf', format='pdf')
        print(f"Plot saved to: figures/{filename}.pdf")

    # Show the plot
    plt.show()

def plot_braking_force(df, v_model=None, force_model=None, figsize=(10, 5), filename='braking_force_plot', plot_parabolas=False, params=None, speed_range=None, vw=False):
    import matplotlib.pyplot as plt
    import numpy as np
    
    df = df.copy()

    # Ensure 'bucket_mid' column exists
    if 'bucket_mid' not in df.columns:
        if 'bucket_floor' in df.columns and 'bucket_ceiling' in df.columns:
            df['bucket_mid'] = (df['bucket_floor'] + df['bucket_ceiling']) / 2
        else:
            if 'mean_speed' in df.columns:
                df['bucket_mid'] = df['mean_speed']

    # Create the background force model from params
    if params is not None:
        f0, f1, f2 = params
        # find the range of the data
        v_range = np.linspace(df['bucket_mid'].min(), df['bucket_mid'].max(), 200)
        extrapolated_force_model = f0 + f1 * v_range + f2 * v_range ** 2

    # Ensure required columns exist
    required_columns = ['bucket_mid', 'mean_force', 'std_force']
    data_is_from_constspeed_pipeline = False
    for col in required_columns:
        if col not in df.columns:
            print(f'Column "{col}" not found in DataFrame. Looking for alternative...')
            if col == 'mean_force':
                if 'mean_acc_force' in df.columns:
                    df['mean_force'] = df['mean_acc_force']
                    data_is_from_constspeed_pipeline = True
                else:
                    raise ValueError(f"No Column for '{col}' not found in DataFrame.")
            elif col == 'std_force':
                if 'std_acc_force' in df.columns:
                    df['std_force'] = df['std_acc_force']
                else:
                    raise ValueError(f"No Column for '{col}' not found in DataFrame.")

    if vw:
        gt_values = {
            'lab_test': (121.7, 1.28, 0.02122),
            'previous_on_road_method': (127.3, 0.83, 0.02529),
            'coc': (110, 0.855, 0.02445)
        }

        gt_ranges = {
            'lab_test': (10, 150),
            'previous_on_road_method': (10, 150),
            'coc': (10, 150)
        }
    
    else:
        # Hardcoded ground truth values
        gt_values = {
            'very_low_suspension': (173.9, 0.005, 0.02871),
            'low_suspension': (188.7, 0.064, 0.025),
            'medium_suspension': (138.4, 1.981, 0.01456),
        }

        gt_ranges = {
            'very_low_suspension': (15, 160),
            'low_suspension': (15, 160),
            'medium_suspension': (15, 95)
        }

    # Calculate y-axis limits based on parabolas and fitted model
    y_min, y_max = float('inf'), float('-inf')

    # If plot_parabolas is True, calculate min and max from the parabolas
    if plot_parabolas:
        v_range = np.linspace(df['bucket_mid'].min(), df['bucket_mid'].max(), 200)
        for coeffs in gt_values.values():
            f0, f1, f2 = coeffs
            force_values = f0 + f1 * v_range + f2 * v_range ** 2
            y_min = min(y_min, force_values.min())
            y_max = max(y_max, force_values.max())

    # Include the fitted model if provided
    if force_model is not None:
        y_min = min(y_min, force_model.min())
        y_max = max(y_max, force_model.max())

    # Define line styles and colors for the optional parabolas
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    line_styles = ['-.','-.','-.','-.' ]
    parabola_colors = ['#FED702', '#B55CA5', '#9FBA36']

    # Apply the same styling as in the reference function
    with plt.rc_context({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    }):
        plt.figure(figsize=figsize)

        if not data_is_from_constspeed_pipeline:
            # Plot mean braking force as a line using the specified blue color (#3070B3)
            plt.plot(df['bucket_mid'], df['mean_force'], label='Mean Braking Force', color='#3070B3')
            # Plot uncertainty band using fill_between with the specified light blue color (#9ABCE4)
            plt.fill_between(
                df['bucket_mid'],
                df['mean_force'] - df['std_force'],
                df['mean_force'] + df['std_force'],
                color='#9ABCE4', alpha=0.2, label='Standard Deviation  $\SI{\pm 1}{\sigma}$'
            )
        else:
            # Plot mean braking forces as a scatterplot using the specified blue color (#3070B3)
            plt.scatter(df['bucket_mid'], df['mean_force'], label='Constant Speed Segment', color='#3070B3')

        # Plot the fitted quadratic model if data is provided using the specified red color (#EA7237)
        if v_model is not None and force_model is not None:
            plt.plot(v_model, force_model, label='Quadratic Fit', color='#EA7237', linewidth=2)

        # Plot the extrapolated force model if params are provided using the specified red color (#EA7237)
        if params is not None:
            plt.plot(v_range, extrapolated_force_model, label='Extrapolated Force Model', color='#EA7237', linestyle='--', linewidth=1)
            # Update y-axis limits based on the extrapolated model
            y_min = min(y_min, extrapolated_force_model.min())
            y_max = max(y_max, extrapolated_force_model.max())

        # Determine x limits either from speed_range or from v_model
        if speed_range is not None:
            x_min = speed_range[0]
            x_max = speed_range[1]
            plt.xlim(x_min, x_max)
            # Create force model for the specified range
            v_range_lim = np.linspace(x_min, x_max, 200)
            force_values_lim = f0 + f1 * v_range_lim + f2 * v_range_lim ** 2
            y_min = min(y_min, force_values_lim.min())
            y_max = max(y_max, force_values_lim.max())
            y_min = y_min * 0.9
            y_max = y_max * 1.1
        else:
            x_min = v_model.min()
            x_max = v_model.max()
            plt.xlim(0, x_max * 1.1)
            y_min = force_model.min()
            y_max = force_model.max()
            y_min = y_min * 0.9
            y_max = y_max * 1.1

        # Plot parabolas with specified coefficients if toggled on, using the TUM colors
        if plot_parabolas:
            for i, (key, coeffs) in enumerate(gt_values.items()):
                f0, f1, f2 = coeffs
                force_values = f0 + f1 * v_range + f2 * v_range ** 2
                plt.plot(v_range, force_values, label=key.replace('_', ' ').title(),
                         linestyle=line_styles[i % len(line_styles)],
                         color=parabola_colors[i % len(parabola_colors)],
                         linewidth=2)

        # Set labels, legend, and y-axis limits
        plt.xlabel('Vehicle Speed in \si{\kilo\meter\per\hour}')
        plt.ylabel('Braking Force in \si{\\newton}')
        plt.ylim(y_min, y_max)
        plt.legend()

        # Grid styling and axes handling
        plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        # Tight layout for better spacing
        plt.tight_layout()

        if filename is not None:
            # Save the figure as a PDF in the 'figures' folder
            plt.savefig(f'figures/{filename}.pdf', format='pdf')
            print(f"Plot saved to: figures/{filename}.pdf")

        # Show the plot
        plt.show()

##### Functions for Gear Strategy Finder #####

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import time
from modules.data_handler import load_can_data    
import plotly.graph_objs as go


def plot_gear_domains_with_lines(
    results_normal_driving_data,
    results_sport_driving_data,
    results_normal_dynamo_data,
    results_sport_dynamo_data,
    gear1_csv='data/dummy_path',
    gear2_csv='data/dummy_path',
    transmission_ratio_gear_1=15.56,
    transmission_ratio_gear_2=8.16,
    engine_to_wheel_ratio_gear_1=44,
    engine_to_wheel_ratio_gear_2=23,
    filename='gear_strategy_test_drive_plan.pdf',
    figsize=(12, 8),
    add_scatters=False,
):

    # Load the full-load gear polygons
    full_load_curve_1 = pd.read_csv(gear1_csv)
    full_load_curve_2 = pd.read_csv(gear2_csv)

    # Convert RPM to rad/s
    full_load_curve_1['omega'] = full_load_curve_1['RPM'] * 2 * np.pi / 60.0
    full_load_curve_2['omega'] = full_load_curve_2['RPM'] * 2 * np.pi / 60.0

    # Convert to vehicle speed in m/s -> then to km/h
    full_load_curve_1['m/s'] = full_load_curve_1['omega'] / engine_to_wheel_ratio_gear_1
    full_load_curve_2['m/s'] = full_load_curve_2['omega'] / engine_to_wheel_ratio_gear_2
    full_load_curve_1['km/h'] = full_load_curve_1['m/s'] * 3.6
    full_load_curve_2['km/h'] = full_load_curve_2['m/s'] * 3.6

    # Compute wheel torque (motor torque * gear ratio)
    full_load_curve_1['Wheel_Torque'] = full_load_curve_1['Torque'] * transmission_ratio_gear_1
    full_load_curve_2['Wheel_Torque'] = full_load_curve_2['Torque'] * transmission_ratio_gear_2

    # Optionally close the loop in the polygons
    full_load_curve_1 = pd.concat([full_load_curve_1, full_load_curve_1.iloc[0:1]], ignore_index=True)
    full_load_curve_2 = pd.concat([full_load_curve_2, full_load_curve_2.iloc[0:1]], ignore_index=True)

    # Prepare shift-threshold lines from Normal & Sport (Real + Dyno), in [km/h, wheel_torque]
    normal_mode_speed = results_normal_driving_data['rear_motor_torque']['fitted_x']  # already [km/h]
    normal_mode_motor_torque = results_normal_driving_data['rear_motor_torque']['fitted_y']  # [Nm]
    normal_mode_wheel_torque = normal_mode_motor_torque * transmission_ratio_gear_1

    normal_mode_speed_dyno = results_normal_dynamo_data['rear_motor_torque']['fitted_x']  # [km/h]
    normal_mode_motor_torque_dyno = results_normal_dynamo_data['rear_motor_torque']['fitted_y']  # [Nm]
    normal_mode_wheel_torque_dyno = normal_mode_motor_torque_dyno * transmission_ratio_gear_1

    sport_mode_speed = results_sport_driving_data['rear_motor_torque']['fitted_x']   # [km/h]
    sport_mode_motor_torque = results_sport_driving_data['rear_motor_torque']['fitted_y']   # [Nm]
    sport_mode_wheel_torque = sport_mode_motor_torque * transmission_ratio_gear_1

    sport_mode_speed_dyno = results_sport_dynamo_data['rear_motor_torque']['fitted_x']   # [km/h]
    sport_mode_motor_torque_dyno = results_sport_dynamo_data['rear_motor_torque']['fitted_y']   # [Nm]
    sport_mode_wheel_torque_dyno = sport_mode_motor_torque_dyno * transmission_ratio_gear_1

    # Create figure
    plt.figure(figsize=figsize)

    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })

    # Plot the gear 1 & gear 2 polygons
    plt.plot(
        full_load_curve_1['km/h'], full_load_curve_1['Wheel_Torque'], color='#5E94D4'
    )
    plt.plot(
        full_load_curve_2['km/h'], full_load_curve_2['Wheel_Torque'], color='#EA7237'
    )
    plt.fill_between(
        full_load_curve_1['km/h'], full_load_curve_1['Wheel_Torque'],
        alpha=0.1, color='#5E94D4', label='Gear One Operating Domain'
    )
    plt.fill_between(
        full_load_curve_2['km/h'], full_load_curve_2['Wheel_Torque'],
        alpha=0.1, color='#EA7237', label='Gear Two Operating Domain'
    )

    # Plot Normal mode shift-threshold lines (Real + Dyno)
    plt.plot(
        normal_mode_speed, normal_mode_wheel_torque,
        label='Normal Mode Shift Threshold (On-Road)', color='#072140', linestyle='--'
    )
    plt.plot(
        normal_mode_speed_dyno, normal_mode_wheel_torque_dyno,
        label='Normal Mode Shift Threshold (Dyno)', color='#9FBA36', linestyle='--'
    )

    # Plot Sport mode shift-threshold lines (Real + Dyno)
    plt.plot(
        sport_mode_speed, sport_mode_wheel_torque,
        label='Sport Mode Shift Threshold (On-Road)', color='#072140'
    )
    plt.plot(
        sport_mode_speed_dyno, sport_mode_wheel_torque_dyno,
        label='Sport Mode Shift Threshold (Dyno)', color='#9FBA36'
    )

    def compute_wheel_torque(row):
        if row['gear_selected'] == 1:
            return row['rear_motor_torque'] * transmission_ratio_gear_1 * 2
        else:
            return row['rear_motor_torque'] * transmission_ratio_gear_2 * 2

    if add_scatters:
        normal_data = load_can_data('20250206_Schaltstrategie/Normal/2025_02_06-18_58_57.csv')
        # Compute wheel torque
        normal_data['wheel_torque'] = normal_data.apply(compute_wheel_torque, axis=1)

        gear_up_indices = normal_data.loc[(normal_data['gear_selected'].shift(1) == 1) & (normal_data['gear_selected'] == 2)].index
        normal_data.loc[gear_up_indices[gear_up_indices < len(normal_data)], 'gear_change'] = 'up'
        gear_down_indices = normal_data.loc[(normal_data['gear_selected'].shift(1) == 2) & (normal_data['gear_selected'] == 1)].index
        normal_data.loc[gear_down_indices[gear_down_indices < len(normal_data)], 'gear_change'] = 'down'

        up_shifts_normal = normal_data.loc[normal_data['gear_change'] == 'up']
        down_shifts_normal = normal_data.loc[normal_data['gear_change'] == 'down']

        plt.scatter(
            up_shifts_normal['vehicle_speed'], up_shifts_normal['wheel_torque'],
            marker='^', color='#B55CA5', edgecolor='#B55CA5', s=40, zorder=0,
            label='Normal Up', alpha=0.3
        )
        plt.scatter(
            down_shifts_normal['vehicle_speed'], down_shifts_normal['wheel_torque'],
            marker='v', color='#B55CA5', edgecolor='#B55CA5', s=40, zorder=0,
            label='Normal Down', alpha=0.3
        )

        sport_data = load_can_data('20250206_Schaltstrategie/Sport/2025_02_06-17_38_41.csv')
        # Compute wheel torque
        sport_data['wheel_torque'] = sport_data.apply(compute_wheel_torque, axis=1)

        gear_up_indices = sport_data.loc[(sport_data['gear_selected'].shift(1) == 1) & (sport_data['gear_selected'] == 2)].index
        sport_data.loc[gear_up_indices[gear_up_indices < len(sport_data)], 'gear_change'] = 'up'

        gear_down_indices = sport_data.loc[(sport_data['gear_selected'].shift(1) == 2) & (sport_data['gear_selected'] == 1)].index
        sport_data.loc[gear_down_indices[gear_down_indices < len(sport_data)], 'gear_change'] = 'down'

        up_shifts_sport = sport_data.loc[sport_data['gear_change'] == 'up']
        down_shifts_sport = sport_data.loc[sport_data['gear_change'] == 'down']

        plt.scatter(
            up_shifts_sport['vehicle_speed'], up_shifts_sport['wheel_torque'],
            marker='^', color='#FED702', edgecolor='#FED702', s=40, zorder=0,
            label='Sport Up', alpha=0.3
        )
        plt.scatter(
            down_shifts_sport['vehicle_speed'], down_shifts_sport['wheel_torque'],
            marker='v', color='#FED702', edgecolor='#FED702', s=40, zorder=0,
            label='Sport Down', alpha=0.3
        )

        # Add vertical line at 54 km/h from 0 to 5200Nm
        plt.vlines(x=54, ymin=0, ymax=5200, color='#072140', linestyle='-.', label='Downshift Threshold')
        

    plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.ylim(0, 5900)
    plt.xlabel('Vehicle Speed in \si{\kilo\meter\per\hour}')
    plt.ylabel('Wheel Torque in \si{\\newton\meter}')
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(f'figures/{filename}', format='pdf')
    plt.show()

def visualize_classifier_animation(filename, clf, signals, time_offsets, output_file, window_size=400, movement_speed=3, frames=None, fps=60, dpi=200, bitrate=4000, random_offset=0):
    """
    Create an animation to visualize classifier predictions.

    Parameters:
    filename (str): File name of the CAN data to visualize.
    clf (ClassifierMixin): Trained classifier.
    signals (list of str): List of columns to use for feature extraction and visualization.
    time_offsets (list of int): List of time offsets to apply to each signal for feature extraction.
    output_file (str): File name for saving the animation.
    window_size (int): Size of the sliding window.
    movement_speed (int): Number of indices by which the sliding window moves.
    frames (int): Number of frames in the animation.
    fps (int): Frames per second for the animation.
    dpi (int): DPI for saving the animation.
    bitrate (int): Bitrate for saving the animation.
    random_offset (int): Random offset to apply to the starting index.

    Returns:
    None
    """
    # Load CAN data
    data = load_can_data(filename, verbose=False)

    if frames is None:
        frames = int((len(data) // movement_speed) * 0.9)

    # Ensure only 5 columns are used, issue warning if more than 5 provided
    if len(signals) > 5:
        print(f"Warning: More than 5 signals provided. Using the first 5 signals: {signals[:5]}")
        signals = signals[:5]

    # Ensure 'gear_selected' is always plotted in subplot 4 if present
    if 'gear_selected' in signals:
        signals.remove('gear_selected')
        signals.insert(3, 'gear_selected')

    # Determine the global min and max for each signal to set y-axis limits
    signal_ranges = {signal: (data[signal].min(), data[signal].max()) for signal in signals}

    # Set up the figure and subplots
    fig, axs = plt.subplots(3, 2, figsize=(16, 10))
    axs = axs.flatten()

    # Create the static axes for all subplots
    def setup_static_axis(ax_static, window_size):
        ax_static.set_xlim(-window_size / 10, 0)
        ax_static.set_xticks(np.arange(int(-window_size / 10), 1, 5))
        ax_static.set_xlabel("Time in s")
        ax_static.grid(True)

    static_axes = [axs[i].twiny() for i in range(6)]
    for ax_static in static_axes:
        setup_static_axis(ax_static, window_size)

    # Store predictions to plot as a line
    predictions = []

    # Helper function to apply the sliding window
    def apply_sliding_window(column, frame):
        start_idx = frame * movement_speed + random_offset
        end_idx = start_idx + window_size
        return data[column].iloc[start_idx:end_idx], data['seconds'].iloc[start_idx:end_idx]

    # Update plot function for the animation
    def update_plot(ax, column, frame, color, ax_static):
        ax.clear()
        y_min, y_max = signal_ranges[column]  # Set y-axis limits based on global min and max
        ax.set_ylim(y_min, y_max)
        y_data, x_data = apply_sliding_window(column, frame)

        # Calculate the index for the given offset from the end for each offset
        for offset in time_offsets:
            index_for_offset = min(window_size - 1 + offset, len(y_data) - 1) if len(y_data) > 0 else 0
            if 0 <= index_for_offset < len(y_data):
                y_data_offset = y_data.iloc[index_for_offset]
                ax.scatter(x_data.iloc[index_for_offset], y_data_offset, color=color, marker='o')
                ax.text(x_data.iloc[index_for_offset], y_data_offset, f'{y_data_offset:.2f}', color=color)

        ax.plot(x_data, y_data, color=color)
        ax.set_ylabel(f'{column}')
        ax.set_xlim(x_data.min(), x_data.max())
        ax_static.set_xlim(-window_size / 10, 0)
        ax_static.set_xticks(np.arange(int(-window_size / 10), 1, 5))
        ax_static.grid(True)

    # Update function for the animation
    def update(frame):
        start_time = time.time()
        
        # Update each subplot
        update_plot(axs[0], signals[0], frame, 'blue', static_axes[0])
        update_plot(axs[1], signals[1], frame, 'orange', static_axes[1])
        update_plot(axs[2], signals[2], frame, 'green', static_axes[2])
        update_plot(axs[3], signals[3], frame, 'purple', static_axes[3])
        update_plot(axs[4], signals[4], frame, 'red', static_axes[4])
        
        # Collect feature values for the classifier
        features = []
        for col in signals:
            y_data, _ = apply_sliding_window(col, frame)
            if len(y_data) < abs(max(time_offsets)):  # Handling cases with fewer than required samples
                continue  # Skip this frame if can't get the required offsets

            feature_values = [y_data.iloc[min(window_size - 1 + offset, len(y_data) - 1)] for offset in time_offsets]
            features.extend(feature_values)

        # Only proceed if there are enough features to make a prediction
        if len(features) == len(signals) * len(time_offsets):
            # Convert to DataFrame for prediction
            # Use the same columns that were used to train the classifier
            feature_df = pd.DataFrame([features], columns=clf.feature_names_in_)

            # Make prediction
            prediction = clf.predict(feature_df)[0]

            # Handle label types and set colors accordingly
            if isinstance(prediction, (int, np.integer)):
                # Binary classification case (labels like 1, 2)
                prediction_value = prediction
                if prediction == 1:
                    box_color = 'blue'
                elif prediction == 2:
                    box_color = 'red'
                else:
                    box_color = 'green'
            else:
                # Multiclass classification case (labels like 'up', 'down', 'non')
                if prediction == 'up':
                    prediction_value = 1
                    box_color = 'blue'
                elif prediction == 'down':
                    prediction_value = -1
                    box_color = 'red'
                else:
                    prediction_value = 0
                    box_color = 'green'

            # Store the prediction value for line plot
            predictions.append(prediction_value)
        else:
            # If not enough features, store "stay" as the default value
            prediction_value = 0
            box_color = 'green'
            predictions.append(prediction_value)

        # Keep only the latest `window_size` predictions
        predictions[:] = predictions[-window_size:]

        # Update the last subplot with the prediction as a line
        axs[5].clear()
        axs[5].plot(predictions, color='black', linestyle='-')
        axs[5].set_ylim(-1.5 if not isinstance(prediction_value, (int, np.integer)) else 0.5, 
                        1.5 if not isinstance(prediction_value, (int, np.integer)) else 2.5)
        axs[5].set_title('Gear Shift Prediction')
        axs[5].set_xlabel('Frame')
        axs[5].set_ylabel('Prediction')
        
        if isinstance(prediction_value, (int, np.integer)):
            axs[5].set_yticks([1, 2])
            axs[5].set_yticklabels(['Gear 1', 'Gear 2'])
        else:
            axs[5].set_yticks([-1, 0, 1])
            axs[5].set_yticklabels(['Down', 'Stay', 'Up'])
        
        # Add a text box with the current prediction
        axs[5].text(len(predictions) - 1, prediction_value, f'{prediction}',
                    fontsize=12, ha='center', va='bottom', bbox=dict(facecolor=box_color, alpha=0.5))
        
        # Set title for the whole figure
        fig.suptitle(f'File: {filename}, index: {frame}, prediction: {prediction}', fontsize=16)

        # Calculate and print progress indicator
        elapsed_time = time.time() - start_time
        print(f'Frame {frame + 1}/{frames} processed in {1000 * elapsed_time:.2f} ms, estimated time remaining:{(frames-frame)*elapsed_time/60:.2f} mins', end='\r')

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

    # Save the animation to a file
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Till Zacher'), bitrate=bitrate)
    # Check if the 'animations' directory exists, create it if not
    import os
    if not os.path.exists('animations'):
        os.makedirs('animations')
    anim.save('animations/' + output_file, writer=writer, dpi=dpi)
    print(f'Animation saved as {output_file}')
    plt.close(fig)

def plot_3d_scatter(dataframe, axes_columns, shift_column='label', width=800, height=800):
    """
    Creates an interactive 3D scatterplot using Plotly.

    Parameters:
    - dataframe: DataFrame containing the data.
    - axes_columns: List of three columns to be used for x, y, and z axes.
    - shift_column: The column used to determine the color of the points (default is 'shift_type').
    - width: The width of the plot window (default is 800).
    - height: The height of the plot window (default is 800).
    """

    # Check that exactly three columns are provided for the axes
    if len(axes_columns) != 3:
        raise ValueError('Exactly three columns must be provided for the axes.')

    x_col, y_col, z_col = axes_columns

    # Map the shift type to colors
    color_map = {'up': 'blue', 'down': 'red', 'non': 'green', 1: 'blue', 2: 'red'}
    dataframe['color'] = dataframe[shift_column].map(color_map)

    # Create the 3D scatterplot
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=dataframe[x_col],
        y=dataframe[y_col],
        z=dataframe[z_col],
        mode='markers',
        marker=dict(
            size=5,
            color=dataframe['color'],  # Color based on shift_type
            opacity=0.8
        ),
        text=dataframe[shift_column],  # Hover text to display shift_type
        name='Shifts'
    ))

    # Update the layout with adjustable width and height
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        title="3D Scatterplot of Shifts",
        width=width,
        height=height
    )

    fig.show()

def plot_data_with_outliers(
    df,
    x_column,
    y_column,
    x_label,
    y_label,
    title='',
    figsize=(10, 8),
    filename=None
):
    """
    Plot data with outliers marked in red and inliers in blue.
    """
    plt.figure(figsize=figsize)

    # Use LaTeX for text rendering
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })

    inliers = df[~df['is_outlier']]
    outliers = df[df['is_outlier']]

    plt.scatter(inliers[x_column], inliers[y_column], c='blue', label='Inliers', marker='o', alpha=0.7)
    plt.scatter(outliers[x_column], outliers[y_column], c='red', label='Outliers', marker='x', alpha=0.7)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(title)
    plt.legend()
    plt.grid(True)

    # Force [0,1] range for both axes, as in the original code
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        print(f'Saved plot as {filename}')

    plt.show()


def plot_gear_shift_data(
    data_normal,
    data_sport,
    x_column,
    y_column,
    x_label,
    y_label,
    title=None,
    xlim=None,
    ylim=None,
    filename=None,
    figsize=(10, 8),
):
    """
    Plot gear shift data for Normal vs. Sport mode, marking them with different colors.
    """
    colors = {
        'normal': np.array([0, 101, 189]) / 255,
        'sport': np.array([227, 114, 34]) / 255,
    }
    plt.figure(figsize=figsize)

    # Use LaTeX for text rendering
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })

    if not data_normal.empty:
        # plt.plot(
        #     data_normal[x_column],
        #     data_normal[y_column],
        #     'x',
        #     label='Drivemode Normal',
        #     color=colors['normal'],
        #     linewidth=2.5
        # )
        plt.scatter(
            data_normal[x_column],
            data_normal[y_column],
            label='Normal',
            color=colors['normal'],
            s=50,
            alpha=0.7
        )

    if not data_sport.empty:
        # plt.plot(
        #     data_sport[x_column],
        #     data_sport[y_column],
        #     'x',
        #     label='Drivemode Sport',
        #     color=colors['sport'],
        #     linewidth=2.5
        # )
        plt.scatter(
            data_sport[x_column],
            data_sport[y_column],
            label='Sport',
            color=colors['sport'],
            s=50,
            alpha=0.7
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)

    plt.legend(loc='lower right')
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.grid(True)
    # Remove spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        print(f'Saved plot as {filename}')

    plt.show()

def plot_mode_comparison(
    y_column,
    mode_label,
    results_loaded,
    results_driving_data,
    filename_prefix,
    figsize,
    xlim,
    ylim
):
    """
    Plot the comparison of lab-test curve vs. on-road data curve for a specific
    y_column (e.g. 'rear_motor_torque') and specific mode ('Normal' or 'Sport').
    """
    plt.figure(figsize=figsize)

    # Use LaTeX for text rendering
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })

    # "Ground truth" from dynamometer (lab test)
    plt.plot(
        results_loaded[y_column]['fitted_x'],
        results_loaded[y_column]['fitted_y'],
        label=f'Lab Test {mode_label}',
        color='#9FBA36',  # TUM green
    )

    # On-road data
    plt.plot(
        results_driving_data[y_column]['fitted_x'],
        results_driving_data[y_column]['fitted_y'],
        label=f'On-Road Test {mode_label}',
        color='#072140'  # TUM dark blue
    )

    plt.xlabel('Vehicle Speed in \si{\kilo\meter\per\hour}')
    if y_column == 'rear_motor_torque':
        plt.ylabel('Torque in \si{\\newton\meter}')
    elif y_column == 'power_mechanical':
        plt.ylabel('Power in \si{\kilo\watt}')
    elif y_column == 'accelerator_pedal':
        plt.ylabel('Accelerator Pedal in \si{\percent}')

    plt.legend()
    plt.grid(True)

    # Disable spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.xlim(xlim)
    if y_column == 'rear_motor_torque':
        plt.ylim(ylim)

    if filename_prefix is not None:
        filename = f'figures/{filename_prefix}_{y_column}_{mode_label.lower()}_comparison.pdf'
        plt.savefig(filename)
        print(filename)
    else:
        print('No filename_prefix provided for saving plots')

    plt.show()
    
    

def plot_parametric_spline(
    cluster_centers,
    normal_points,
    spline,
    knots,
    fitted_x,
    fitted_y,
    y_label="Y",
    title="Weighted LSQ Spline",
    figsize=(10, 6),
    filename=None,
):
    """Draw the spline together with data. Looks identical to the original plot."""

    cluster_centers = np.asarray(cluster_centers)
    normal_points = np.asarray(normal_points)
    x_c, y_c = cluster_centers.T
    x_n = normal_points[:, 0] if normal_points.size else np.empty(0)
    y_n = normal_points[:, 1] if normal_points.size else np.empty(0)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
        "font.size": 12,
        "text.latex.preamble": r"\usepackage{siunitx}",
    })

    plt.figure(figsize=figsize)
    plt.scatter(x_c, y_c, color="blue", s=100, label="Cluster Medioids", zorder=3)
    if x_n.size:
        plt.scatter(x_n, y_n, color="gray", alpha=0.7, label="Gear Change Points", zorder=2)
    plt.plot(fitted_x, fitted_y, color="red", label="Weighted LSQ Spline", zorder=4)
    plt.scatter(knots, spline(knots), color="green", marker="x", s=100, label="Knots", zorder=5)

    plt.xlabel("Vehicle Speed in \\si{\\kilo\\meter\\per\\hour}")
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.legend()
    plt.grid()

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Plot saved to: {filename}")

    plt.show()
    
COLORS = {
    "normal": np.array([0, 101, 189]) / 255.0,  # BMW i blue
    "sport": np.array([227, 114, 34]) / 255.0,  # BMW i orange
}
    
def plot_combined_parametric_splines(
    *,
    cluster_centers_normal,
    normal_points_normal,
    spline_normal,
    fitted_x_normal,
    fitted_y_normal,
    cluster_centers_sport,
    normal_points_sport,
    spline_sport,
    fitted_x_sport,
    fitted_y_sport,
    y_label: str = "Y",
    xlim=[70, 130],
    ylim=[0, 250],
    figsize: tuple = (10, 6),
    filename: str | None = None,
):
    """Plot both driving modes in one figure.

    * Splines: blue for normal, orange for sport.
    * All scatter points (normal + sport) are grey.
    * No knot markers.
    """

    cn = np.asarray(cluster_centers_normal)
    cs = np.asarray(cluster_centers_sport)
    pn = np.asarray(normal_points_normal)
    ps = np.asarray(normal_points_sport)

    # scatter points
    x_scatter = np.concatenate([pn[:, 0] if pn.size else np.empty(0), ps[:, 0] if ps.size else np.empty(0)])
    y_scatter = np.concatenate([pn[:, 1] if pn.size else np.empty(0), ps[:, 1] if ps.size else np.empty(0)])

    # plt.rcParams.update(
    #     {
    #         "text.usetex": True,
    #         "font.family": "sans-serif",
    #         "font.sans-serif": "Arial",
    #         "font.size": 12,
    #         "text.latex.preamble": r"\\usepackage{siunitx}",
    #     }
    # )

    plt.figure(figsize=figsize)

    # grey scatter of all gear‑change points
    if x_scatter.size:
        plt.scatter(x_scatter, y_scatter, color="gray", alpha=0.4, label="Gear Changes", zorder=1)

    # normal spline
    plt.plot(
        fitted_x_normal,
        fitted_y_normal,
        color=COLORS["normal"],
        label="Normal Spline",
        lw=2,
        zorder=3,
    )

    # sport spline
    plt.plot(
        fitted_x_sport,
        fitted_y_sport,
        color=COLORS["sport"],
        label="Sport Spline",
        lw=2,
        zorder=4,
    )

    plt.xlabel("Vehicle Speed in \\si{\\kilo\\meter\\per\\hour}")
    plt.ylabel(y_label)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.legend()
    plt.grid()

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
        print(f"Plot saved to: {filename}")

    plt.show()


##### Functions for Parameter Space Analysis plots #####

import plotly.graph_objs as go
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ast
import textwrap


def threeDplot(results_df, hyperparameters, target_metric, figsize=(800,800), higher_is_better=False):
    """
    Plots a 3D surface plot of rmse_diff vs. two hyperparameters.

    Parameters:
    - results_df: DataFrame containing the results of hyperparameter tuning.
    - hyperparameters: List of two hyperparameters to plot.
    """
    import plotly.graph_objs as go
    import numpy as np
    import pandas as pd

    if len(hyperparameters) != 2:
        raise ValueError('Exactly two hyperparameters must be provided.')

    axis1, axis2 = hyperparameters

    # Prepare data
    data = results_df[[axis1, axis2, target_metric]].copy()

    # Shorten long string representations in 'smoothing_kwargs' if applicable
    if 'smoothing_kwargs' in hyperparameters:
        data['smoothing_kwargs'] = data['smoothing_kwargs'].apply(shorten_smoothing_kwargs)

    # Drop rows where rmse_diff is NaN
    data = data.dropna(subset=[target_metric])

    # Check data types of hyperparameters
    axis1_is_numeric = pd.api.types.is_numeric_dtype(data[axis1])
    axis2_is_numeric = pd.api.types.is_numeric_dtype(data[axis2])

    # If hyperparameters are not numeric, convert them to strings
    if not axis1_is_numeric:
        data[axis1] = data[axis1].astype(str)
    else:
        data[axis1] = pd.to_numeric(data[axis1], errors='coerce')
    if not axis2_is_numeric:
        data[axis2] = data[axis2].astype(str)
    else:
        data[axis2] = pd.to_numeric(data[axis2], errors='coerce')

    # Drop rows where hyperparameters are NaN or empty strings
    data = data.dropna(subset=[axis1, axis2])
    data = data[(data[axis1] != '') & (data[axis2] != '')]

    # Pivot the DataFrame to create a grid for plotting
    pivot = data.pivot_table(index=axis1, columns=axis2, values=target_metric, aggfunc='mean')

    # Check if pivot table is empty
    if pivot.empty:
        print("No data to plot.")
        return

    # Create meshgrid for plotting
    X_values = pivot.columns.values
    Y_values = pivot.index.values

    # Map categorical values to numerical indices
    if not axis2_is_numeric:
        X_indices = np.arange(len(X_values))
        X_map = {val: idx for idx, val in enumerate(X_values)}
    else:
        X_indices = X_values.astype(float)
        X_map = {val: val for val in X_values}

    if not axis1_is_numeric:
        Y_indices = np.arange(len(Y_values))
        Y_map = {val: idx for idx, val in enumerate(Y_values)}
    else:
        Y_indices = Y_values.astype(float)
        Y_map = {val: val for val in Y_values}

    X_grid, Y_grid = np.meshgrid(X_indices, Y_indices)
    Z = pivot.values

    # Create the figure
    fig = go.Figure(data=[go.Surface(z=Z, x=X_grid, y=Y_grid, colorscale='Viridis')])

    # Depending on whether higher is better or not add the min or max point, if higher_is_better == None, do nothing
    if not higher_is_better:
        min_target = np.nanmin(Z)
        min_coords = np.unravel_index(np.nanargmin(Z), Z.shape)
        min_x = X_indices[min_coords[1]]
        min_y = Y_indices[min_coords[0]]
        fig.add_trace(go.Scatter3d(
            x=[min_x],
            y=[min_y],
            z=[min_target],
            mode='markers',
            marker=dict(size=5, color='red'),
            name=f'Minimum {target_metric}'
        ))
    elif higher_is_better:
        max_target = np.nanmax(Z)
        max_coords = np.unravel_index(np.nanargmax(Z), Z.shape)
        max_x = X_indices[max_coords[1]]
        max_y = Y_indices[max_coords[0]]
        fig.add_trace(go.Scatter3d(
            x=[max_x],
            y=[max_y],
            z=[max_target],
            mode='markers',
            marker=dict(size=5, color='red'),
            name=f'Maximum {target_metric}'
        ))
    else:
        pass


    # Format the plot
    fig.update_layout(
        title=f'{target_metric} vs. {axis1} and {axis2}',
        autosize=False,
        width=1200,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
        scene=dict(
            xaxis_title=str(axis2),
            yaxis_title=str(axis1),
            zaxis_title=f'{target_metric}',
            xaxis=dict(
                tickmode='array',
                tickvals=X_indices,
                ticktext=X_values
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=Y_indices,
                ticktext=Y_values
            ),
            aspectmode='auto'
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.show()


def plot_hyperparameter_heatmaps(
    results_df, hyperparameters, target_metric, figsize=(12, 12),
    higher_is_better=False, truncate_names=True, use_identical_scale=True,
    aggregation='mean', skip_upper_triangle=False, filename=None,
    rotate_y_ticks=False
):
    """
    Generates a grid of heatmaps for pairs of hyperparameters to visualize their impact on a target metric.

    Parameters:
    - results_df: DataFrame containing the results of hyperparameter tuning, including the target metric.
    - hyperparameters: List of hyperparameters to be considered for the heatmap.
    - target_metric: The metric to plot in the heatmaps.
    - figsize: Size of the overall figure.
    - higher_is_better: Boolean indicating if higher metric values are better (reverses color map).
    - truncate_names: Boolean to toggle truncation of string values longer than 8 characters.
    - use_identical_scale: Boolean to toggle identical color scale for all heatmaps.
      If True, all heatmaps share a single colorbar (manually placed) on the side.
    - aggregation: The aggregation function to apply when pivoting (e.g. 'mean', 'median', 'count').
    - skip_upper_triangle: If True, hides all subplots in the upper triangle (including the diagonal),
      so you only see the lower-triangle subplots.
    - filename: If provided, saves the figure as a PDF.
    - rotate_y_ticks: If True, rotates y-axis tick labels for each heatmap by 45 degrees;
      if False, leaves them unrotated.
    """

    def truncate_string(s):
        return s[:8] if truncate_names and isinstance(s, str) and len(s) > 8 else s

    def convert_string_to_dict(s):
        try:
            return ast.literal_eval(s)
        except:
            return s

    # First pass: collect aggregated pivot tables to find the global min/max if needed
    pivot_tables = {}
    num_params = len(hyperparameters)

    for i, hp1 in enumerate(hyperparameters):
        for j, hp2 in enumerate(hyperparameters):
            # Skip upper triangle (including diagonal) if needed
            if skip_upper_triangle and j >= i:
                continue
            # If not skipping upper triangle, hide diagonal if i == j
            if not skip_upper_triangle and i == j:
                continue

            # Ensure columns exist
            if (
                hp1 not in results_df.columns or 
                hp2 not in results_df.columns or
                target_metric not in results_df.columns
            ):
                continue

            data = results_df[[hp1, hp2, target_metric]].copy()
            data.dropna(subset=[target_metric], inplace=True)


            data[hp1] = data[hp1].where(~data[hp1].isnull(), other="None")
            data[hp2] = data[hp2].where(~data[hp2].isnull(), other="None")

            # Apply any custom transformation
            if hp1 == 'smoothing_kwargs':
                data[hp1] = data[hp1].apply(shorten_smoothing_kwargs)
            if hp2 == 'smoothing_kwargs':
                data[hp2] = data[hp2].apply(shorten_smoothing_kwargs)

            if hp1 == 'speed_signal':
                data[hp1] = data[hp1].apply(shorten_speed_signal_names)
            if hp2 == 'speed_signal':
                data[hp2] = data[hp2].apply(shorten_speed_signal_names)

            if hp1 == 'which_full_load_curve':
                data[hp1] = data[hp1].apply(shorten_full_load_curve)
            if hp2 == 'which_full_load_curve':
                data[hp2] = data[hp2].apply(shorten_full_load_curve)

            if hp1 == 'twoD_smoothing_kwargs':
                data[hp1] = data[hp1].apply(convert_string_to_dict)
                data[hp1] = data[hp1].apply(shorten_twoD_smoothing_kwargs)
            if hp2 == 'twoD_smoothing_kwargs':
                data[hp2] = data[hp2].apply(convert_string_to_dict)
                data[hp2] = data[hp2].apply(shorten_twoD_smoothing_kwargs)

            # Convert to string + truncate
            data[hp1] = data[hp1].apply(truncate_string)
            data[hp2] = data[hp2].apply(truncate_string)

            data.dropna(subset=[hp1, hp2], inplace=True)
            if data.empty:
                continue

            pivot_df = data.pivot_table(
                index=hp1,
                columns=hp2,
                values=target_metric,
                aggfunc=aggregation
            ).dropna(axis=0, how='all').dropna(axis=1, how='all')

            if not pivot_df.empty:
                pivot_tables[(i, j)] = pivot_df

    # Determine global vmin, vmax across all pivot tables AFTER aggregation if use_identical_scale is True
    global_vmin = None
    global_vmax = None
    if use_identical_scale and pivot_tables:
        all_vals = []
        for df in pivot_tables.values():
            all_vals.extend(df.values.flatten())
        global_vmin = min(all_vals)
        global_vmax = max(all_vals)

    # Second pass: create figure and plot
    fig, axes = plt.subplots(num_params, num_params, figsize=(5*num_params, 5*num_params))

    # Use LaTeX for text rendering
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })

    cmap = 'turbo' if higher_is_better else 'turbo_r'

    for i, hp1 in enumerate(hyperparameters):
        for j, hp2 in enumerate(hyperparameters):
            ax = axes[i, j]

            # Hide subplots if skipping upper triangle
            if skip_upper_triangle and j >= i:
                ax.set_visible(False)
                continue
            # Or hide diagonal if not skipping upper triangle
            if not skip_upper_triangle and i == j:
                ax.set_visible(False)
                continue

            pivot_df = pivot_tables.get((i, j))
            if pivot_df is None or pivot_df.empty:
                ax.set_visible(False)
                continue

            # Plot
            sns.heatmap(
                pivot_df,
                ax=ax,
                cmap=cmap,
                cbar=not use_identical_scale,  # no individual colorbar if identical scale
                vmin=global_vmin if use_identical_scale else None,
                vmax=global_vmax if use_identical_scale else None
            )

            # Format x/y labels: underscores -> spaces, capitalized words
            y_label = hp1.replace('_', ' ').title()
            x_label = hp2.replace('_', ' ').title()

            if j == 0:
                ax.set_ylabel(y_label)
            else:
                ax.set_ylabel('')

            if i == num_params - 1:
                ax.set_xlabel(x_label)
            else:
                ax.set_xlabel('')

            # Force the y ticks not to rotate if rotate_y_ticks=False
            if not rotate_y_ticks:
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Single colorbar if identical scale
    if use_identical_scale and pivot_tables:
        import matplotlib as mpl

        sm = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=global_vmin, vmax=global_vmax),
            cmap=cmap
        )
        sm.set_array([])

        plt.tight_layout(pad=1.5, w_pad=10.0, h_pad=5.0)

        fig.subplots_adjust(right=0.8)

        cbar_ax = fig.add_axes([0.7, 0.02, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)

        # Colorbar label
        if target_metric == 'loss_low_suspension':
            cbar_label = 'RMSE Low Suspension in \si{\\newton}'
        else:
            cbar_label = target_metric.replace('_', ' ').title()
        cbar.ax.set_ylabel(cbar_label, rotation=90, labelpad=15)

    else:
        # If not using identical scale, just do a normal tight_layout
        plt.tight_layout()

    # Saving
    fig.set_size_inches(figsize)
    if filename is not None:
        # Save the figure as a PDF with adjusted bounding box and padding
        plt.savefig(f'figures/{filename}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_single_pair_heatmap(
    results_df, hyperparameter1, hyperparameter2, target_metric,
    figsize=(8, 6), higher_is_better=False, truncate_names=True,
    aggregation='mean', keyword='', annotate_boxes=True,
    vmin=None, vmax=None, rotate_xlabels=False
):
    """
    Plots a single heatmap for a pair of hyperparameters to visualize their impact on a target metric.

    Parameters:
    - results_df: DataFrame containing the results of hyperparameter tuning.
    - hyperparameter1: First hyperparameter for the y-axis.
    - hyperparameter2: Second hyperparameter for the x-axis.
    - target_metric: The metric to plot in the heatmap.
    - figsize: Tuple specifying the size of the figure.
    - higher_is_better: Boolean indicating if higher metric values are better (reverses color map).
    - truncate_names: Boolean to toggle truncation of string values longer than 8 characters.
    - aggregation: Aggregation method for the pivot table (default is 'mean').
    - keyword: String keyword to append to the filename for saving the plot.
    - annotate_boxes: Boolean to toggle display of numeric values inside heatmap cells.
    - vmin, vmax: Optional shared minimum and maximum values for the color scale.
    """
    # Helper function to truncate strings
    def truncate_string(s):
        return s[:8] if truncate_names and isinstance(s, str) and len(s) > 8 else s
    
    def replace_cols_to_smooth(s):
        mapping = {
            frozenset(['hv_battery_current', 'hv_battery_voltage', 'rear_motor_torque', 'engine_rpm', 'dcdc_power_hv']): "All Columns",
            frozenset(['dcdc_power_hv', 'hv_battery_current', 'hv_battery_voltage']): "Electrical Columns",
            frozenset(['rear_motor_torque', 'engine_rpm']): "Mechanical Columns",
        }
        try:
            # Convert the string representation of a list into a Python list
            parsed_list = frozenset(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            return s  # Return original string if parsing fails

        # Return the mapped value if it exists; otherwise, return the original string
        return mapping.get(parsed_list, s)

    # Enable LaTeX rendering and use sans-serif font for consistency
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.latex.preamble'] = r'''
    \usepackage{helvet}
    \usepackage{sansmath}
    \sansmath
    \renewcommand{\familydefault}{\sfdefault}
    '''

    # Set color map based on higher_is_better
    cmap = 'turbo' if higher_is_better else 'turbo_r'

    # Prepare data
    data = results_df[[hyperparameter1, hyperparameter2, target_metric]].copy()
    data = data.dropna(subset=[target_metric])

    if hyperparameter1 == 'smoothing_kwargs':
        data[hyperparameter1] = data[hyperparameter1].apply(shorten_smoothing_kwargs)
    if hyperparameter2 == 'smoothing_kwargs':
        data[hyperparameter2] = data[hyperparameter2].apply(shorten_smoothing_kwargs)

    if hyperparameter1 == 'columns_to_smooth':
        data[hyperparameter1] = data[hyperparameter1].apply(replace_cols_to_smooth)
    if hyperparameter2 == 'columns_to_smooth':
        data[hyperparameter2] = data[hyperparameter2].apply(replace_cols_to_smooth)

    

    # Truncate hyperparameter names if needed
    data[hyperparameter1] = data[hyperparameter1].apply(truncate_string)
    data[hyperparameter2] = data[hyperparameter2].apply(truncate_string)

    # Drop rows where hyperparameters are NaN
    data = data.dropna(subset=[hyperparameter1, hyperparameter2])

    # Check if data is empty
    if data.empty:
        print("No valid data to plot.")
        return

    # Pivot the DataFrame for the current pair of hyperparameters
    pivot_df = data.pivot_table(
        index=hyperparameter1,
        columns=hyperparameter2,
        values=target_metric,
        aggfunc=aggregation
    )
    pivot_df = pivot_df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    if pivot_df.empty:
        print("Pivot table is empty. No valid data to plot.")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Use LaTeX for text rendering
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })

    sns.heatmap(
        pivot_df, ax=ax, cmap=cmap, cbar=True,
        annot=annotate_boxes, fmt=".2f" if annotate_boxes else '',
        linewidths=0.5, vmin=vmin, vmax=vmax  # Shared color scale
    )

    # Set labels with LaTeX rendering
    ax.set_xlabel(rf'{hyperparameter2.replace("_", " ").title()}')
    ax.set_ylabel(rf'{hyperparameter1.replace("_", " ").title()}')

    # Set colorbar label
    if target_metric == 'loss_low_suspension':
        cbar_label = 'RMSE Low Suspension in \si{\percent}'
    elif target_metric == 'rmse':
        cbar_label = 'RMSE in -'
    else:
        cbar_label = target_metric.replace('_', ' ').title()
    ax.collections[0].colorbar.set_label(cbar_label, rotation=90, labelpad=15)

    # Rotate x-axis labels if specified
    if rotate_xlabels:
        plt.xticks(rotation=90)

    # Linebreaks for x-axis labels
    ax.set_xticklabels([textwrap.fill(label.get_text(), 12) for label in ax.get_xticklabels()])

    plt.tight_layout(pad=3.0)

    # Save the plot
    filename = f"figures/pairplot_{hyperparameter1}_{hyperparameter2}_{keyword}.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.show()

def plot_anova_bar(anova_results, color='#0065bd', name='anova_plot', figsize=(10,3)):
    # Configure LaTeX rendering and fonts
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.latex.preamble'] = r'''
    \usepackage{helvet}
    \usepackage{sansmath}
    \sansmath
    \renewcommand{\familydefault}{\sfdefault}
    '''
    
    # Prepare the DataFrame
    df_anova = anova_results.reset_index().rename(columns={'index': 'Parameter'})
    df_anova = df_anova[df_anova['Parameter'] != 'Residual']
    df_anova['Parameter'] = df_anova['Parameter'].apply(lambda x: x.replace('_', ' ').title())
    df_anova = df_anova.sort_values('sum_sq', ascending=False)

    # Replace parameter name "Cluster Weight" with "Fit Emphasis" if present
    df_anova['Parameter'] = df_anova['Parameter'].apply(lambda x: 'Fit Emphasis' if x == 'Cluster Weight' else x)

    df_anova['Parameter'] = df_anova['Parameter'].apply(lambda x: 'Number of Knots' if x == 'Num Knots' else x)

    df_anova['Parameter'] = df_anova['Parameter'].apply(lambda x: 'Number of Clusters' if x == 'N Clusters' else x)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=df_anova,
        x='sum_sq',
        y='Parameter',
        ax=ax,
        color=color,
        edgecolor='black'
    )

    # Annotate each bar with F-value and p-value
    for i, row in df_anova.iterrows():
        param = row['Parameter']
        sum_sq = row['sum_sq']
        F_val = row['F']
        p_val = row['PR(>F)']
        
        # Format p-value
        if p_val < 1e-4:
            p_text = r'p$<$1e-4'
        else:
            p_text = f"p={p_val:.4f}"
        
        ax.text(
            sum_sq + (0.01 * df_anova['sum_sq'].max()), 
            param, 
            f"F={F_val:.2f}, {p_text}",
            va='center',
            fontsize=10
        )

    # Labels and styling
    ax.set_xlabel(r'Sum of Squares', fontsize=12)
    ax.set_ylabel(r'Design Parameter', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save and show
    plt.savefig(f'figures/{name}.pdf', bbox_inches='tight')
    print(f"Plot saved to: figures/{name}.pdf")
    plt.show()



def plot_distribution_against_hyperparameter(
    results_df, hyperparameters, target_metric, figsize=(12, 8), axis_limits=None, filename=None, rotate_labels=True, add_x_label=True, add_y_label=True, opacity=0.1, max_points=10000
):
    if len(hyperparameters) != 1:
        raise ValueError("Exactly one hyperparameter must be provided.")

    hp = hyperparameters[0]

    # Copy the relevant data
    data = results_df[[hp, target_metric]].copy()

    # Drop rows with NaN in target_metric
    data = data.dropna(subset=[target_metric])

    # Replace NaN in hyperparameter column with 'None'
    data[hp] = data[hp].fillna("None")

    def convert_string_to_dict(s):
        try:
            return ast.literal_eval(s)
        except:
            return s

    def replace_cols_to_smooth(s):
        mapping = {
            frozenset(['hv_battery_current', 'hv_battery_voltage', 'rear_motor_torque', 'engine_rpm', 'dcdc_power_hv']): "All Columns",
            frozenset(['dcdc_power_hv', 'hv_battery_current', 'hv_battery_voltage']): "Electrical Columns",
            frozenset(['rear_motor_torque', 'engine_rpm']): "Mechanical Columns",
        }
        try:
            parsed_list = frozenset(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            return s  # Return original string if parsing fails
        return mapping.get(parsed_list, s)

    # Handle special hyperparameter cases
    if hp == "smoothing_kwargs":
        data[hp] = data[hp].apply(shorten_smoothing_kwargs)

    if hp == "columns_to_smooth":
        data[hp] = data[hp].apply(replace_cols_to_smooth)

    if hp == "twoD_smoothing_kwargs":
        data[hp] = data[hp].apply(convert_string_to_dict)
        data[hp] = data[hp].apply(shorten_twoD_smoothing_kwargs)

    # Capitalize axis names
    def format_axis_name(name):
        if name.lower() == "rmse":
            return "RMSE in -"
        elif name ==  "loss_low_suspension":
            return "RMSE Low Suspension in \si{\\newton}"
        elif name == "prmse_global":
            return "PRMSE Global in -"
        else:
            return " ".join([word.capitalize() for word in name.replace("_", " ").split()])

    # Wrap long labels to improve readability
    if not pd.api.types.is_numeric_dtype(data[hp]):
        data[hp] = data[hp].astype(str)
        data[hp] = data[hp].apply(lambda x: "\n".join(textwrap.wrap(x, width=15)))


    unique_vals = data[hp].unique().tolist()

    # Check if "None" is present
    none_exists = "None" in unique_vals
    if none_exists:
        unique_vals.remove("None")

    def try_parse_float(s):
        try:
            return float(s)
        except:
            return None

    # Attempt to parse all non-"None" values as floats
    parse_success = True
    float_vals = []
    for val in unique_vals:
        parsed = try_parse_float(val)
        if parsed is None:
            parse_success = False
            break
        float_vals.append((parsed, val))  # (numeric_value, original_str)

    if parse_success:
        # Sort by the numeric value
        float_vals.sort(key=lambda x: x[0])
        # Reconstruct the sorted list using the original strings
        sorted_vals = [orig_str for (_, orig_str) in float_vals]
    else:
        # Fallback: alphabetical
        sorted_vals = sorted(unique_vals)

    # Put "None" back at the front if it existed
    if none_exists:
        sorted_vals = ["None"] + sorted_vals

    # Make sure define an ordered categorical so Seaborn respects the order
    data[hp] = pd.Categorical(data[hp], categories=sorted_vals, ordered=True)

    plt.figure(figsize=figsize)

    # Use LaTeX for text rendering
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })
    
    if max_points is not None and len(data) > max_points:
        print(f'Detected more than {max_points} data points.')
        print(f'Downsampling number of displayed data points from {len(data)} to {max_points} points.')
        data_strip = data.sample(n=max_points, random_state=42)
    else:
        data_strip = data

    # Stripplot with smaller markers to show individual data points
    sns.stripplot(
        x=hp,
        y=target_metric,
        data=data_strip,
        color=".25",
        alpha=opacity,
        size=3,
        jitter=True,
        zorder=1,
    )

    # Boxplot to show mean and spread of target_metric for each category
    sns.boxplot(
        x=hp,
        y=target_metric,
        data=data,
        boxprops=dict(alpha=0.7),
        zorder=2,
        # remove outlier markers
        flierprops=dict(marker='o', markersize=2, markerfacecolor='k', linestyle='none', alpha=0.1),
    )

    # Format axis labels
    if type(add_x_label) == bool and add_x_label:
        plt.xlabel(format_axis_name(hp))
    elif type(add_x_label) == str:
        plt.xlabel(add_x_label)
    else:
        plt.xlabel("")

    if type(add_y_label) == bool and add_y_label:
        plt.ylabel(format_axis_name(target_metric))
    elif type(add_y_label) == str:
        plt.ylabel(add_y_label)
    else:
        plt.ylabel("")

    # Rotate x-axis labels for readability
    if rotate_labels:
        plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout(pad=3.0)

    # Apply axis limits if provided
    if axis_limits:
        if "x" in axis_limits:
            plt.xlim(axis_limits["x"])
        if "y" in axis_limits:
            plt.ylim(axis_limits["y"])

    # Compute statistics per category
    groupstats = data.groupby(hp, observed=False)[target_metric].agg(["median", "var", "mean", "min", "max"])
    groupstats["abs_median"] = groupstats["median"].abs()

    # Compute IQR for each category to find the slimmest box
    quantiles = data.groupby(hp, observed=False)[target_metric].quantile([0.25, 0.75]).unstack()
    quantiles.columns = ["Q1", "Q3"]
    quantiles["IQR"] = quantiles["Q3"] - quantiles["Q1"]

    # Identify categories for special highlights
    lowest_median_cat = groupstats["median"].idxmin()
    lowest_variance_cat = groupstats["var"].idxmin()
    lowest_abs_cat = groupstats["abs_median"].idxmin()
    slimmest_box_cat = quantiles["IQR"].idxmin()  # category with the smallest IQR

    # Get the current axis
    ax = plt.gca()
    # Extract the category labels from the plot
    categories = [tick.get_text() for tick in ax.get_xticklabels()]

    # remove left upper and right spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Enable only horizontal grid lines
    ax.yaxis.grid(True)  # Turn on horizontal grid
    ax.xaxis.grid(False)  # Turn off vertical grid

    # Print the identified categories for debugging
    print(f"Lowest median category: {lowest_median_cat}")
    print(f"Lowest absolute median category: {lowest_abs_cat}")
    print(f"Lowest variance category: {lowest_variance_cat}")
    print(f"Lowest IQR category: {slimmest_box_cat}")

    # Ensure there's room at the top for annotations
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.3)

    if filename is not None:
        path = f'figures/{filename}.pdf'
        plt.savefig(path)
        print(f'Figure saved to {path}')

    plt.show()


def plot_histogram(results_df, target_column, bins=20, figsize=(10, 6), filename=None):
    """
    Plots a histogram with a fitted kernel density estimate for the target column.

    Parameters:
    - results_df: DataFrame containing the target column to be plotted.
    - target_column: Column to plot the histogram for.
    - bins: Number of bins for the histogram.
    - figsize: Size of the figure.
    - filename: Filename to save the plot as a PDF, without extension.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Apply the same styling as in the reference function
    with plt.rc_context({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12
    }):
        # Create a figure
        plt.figure(figsize=figsize)

        # Plot the histogram with KDE
        sns.histplot(results_df[target_column], bins=bins, kde=True, color='blue', alpha=0.7)

        # Set the title and labels
        plt.title(f'Histogram for {target_column}')
        plt.xlabel(target_column)
        plt.ylabel('Frequency')

        # Add grid to the y-axis for clarity
        plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)

        # Remove unnecessary spines for better aesthetics
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Tight layout for better spacing
        plt.tight_layout()

        # Save the plot if a filename is provided
        if filename:
            plt.savefig(f'figures/{filename}.pdf', format='pdf')

        # Show the plot
        plt.show()


def plot_parameter_importance_analysis(results_df, hyperparameters, target, figsize=(10, 6), filename=None):
    """
    Function to analyze the importance of hyperparameters on the accuracy by fitting a random forest regressor to the hyperparameter grid search results with hyperparameters as features and accuracy as the target variable.

    Parameters:
    - results_df: DataFrame containing the results of hyperparameter tuning and model performance, specifically the results dataframe from the hyperparameter_tuning.ipynb notebook.
    - hyperparameters: List of strings representing the hyperparameters to consider.
    - target: String, the name of the column in results_df that contains the target variable (e.g., 'val_accuracy', 'train_accuracy' or 'val_accuracy_post', etc.).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 5))
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12
    })

    # encode categorical variables
    for column in hyperparameters:
        if results_df[column].dtype == 'object':
            le = LabelEncoder()
            results_df[column] = le.fit_transform(results_df[column])
    
    # define the features and the target
    X = results_df[hyperparameters]
    y = results_df[target]

    # handle missing values, fill X with median and handle y
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    # Specific handling for 'max_depth' if it exists in X
    if 'max_depth' in X.columns:
        X['max_depth'] = X['max_depth'].fillna(-1)

    # train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # get feature importances
    importances = model.feature_importances_
    feature_names = X.columns

    # sort the feature importances in descending order
    sorted_indices = np.argsort(importances)[::-1]

    # plotting
    plt.figure(figsize=figsize)
    plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.bar(range(X.shape[1]), importances[sorted_indices], align='center', zorder=3)
    plt.xticks(range(X.shape[1]), feature_names[sorted_indices], rotation=90)
    plt.title(f'Regressor based Estimation on {target} sensitivity to Parameters')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(f'figures/{filename}.pdf', format='pdf')
    plt.show()


##### Functions for Simulation Pipeline #####

def plot_single_signals(plot_data, single_signals, gear_change_times, figsize, filename_prefix, save_figures):
    """
    Plot the single signals (e.g., velocity profile), optionally highlighting gear-change events.

    Parameters
    ----------
    plot_data : pd.DataFrame
        The subset of simulation data to plot.
    single_signals : list
        Column names for single-signal plotting.
    gear_change_times : pd.Series
        Times where gear changes occur.
    figsize : tuple
        Figure size.
    filename_prefix : str
        Prefix for saving figures.
    save_figures : bool
        If True, saves figures as PDF.
    """

    # Use LaTeX rendering and sans-serif font
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })


    path = 'figures/'
    for single_signal in single_signals:
        plt.figure(figsize=figsize)
        plt.plot(plot_data['time'], plot_data[single_signal], label=single_signal, color='#3070B3')

        # Mark gear-change events
        for change_time in gear_change_times:
            plt.axvline(change_time, color='gray', linestyle='--', linewidth=1)

        plt.xlabel('Time in \si{\second}')
        if single_signal == 'velocity_profile':
            plt.ylabel('Vehicle Speed in \si{\kilo\meter\per\hour}')
        else:
            plt.ylabel(single_signal.replace('_', ' ').capitalize())

        plt.grid()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        if save_figures:
            filename = f'{filename_prefix}_{single_signal}.pdf'
            plt.savefig(f'{path}{filename}', bbox_inches='tight')
            print(f'Saved plot to {path}{filename}')
        plt.show()


def plot_signal_pairs(plot_data, signal_pairs, gear_change_times, figsize, filename_prefix, save_figures):
    """
    Plot signal pairs (driving vs. lab) to compare across time.
    Optionally draws vertical lines for gear-change events.

    Parameters
    ----------
    plot_data : pd.DataFrame
        Subset of the simulation data.
    signal_pairs : list of tuples
        Each tuple: (driving_signal, lab_signal, y_label).
    gear_change_times : pd.Series
        Times where gear changes occur.
    figsize : tuple
        Figure size.
    filename_prefix : str
        Prefix for saving figures.
    save_figures : bool
        If True, saves figures as PDF.
    """

    # Use LaTeX rendering and sans-serif font
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 12,
        'text.latex.preamble': r'\usepackage{siunitx}'
    })

    path = 'figures/'
    for driving_signal, lab_signal, y_label in signal_pairs:
        plt.figure(figsize=figsize)

        # If signals are "power" -> convert to kW
        if 'power' in driving_signal:
            plot_data[driving_signal] = plot_data[driving_signal] / 1000
            plot_data[lab_signal] = plot_data[lab_signal] / 1000

        plt.plot(
            plot_data['time'],
            plot_data[driving_signal],
            label='Vehicle Parameters from On-Road Tests',
            color='#072140'
        )
        plt.plot(
            plot_data['time'],
            plot_data[lab_signal],
            label='Vehicle Parameters from Lab Tests',
            color='#9FBA36'
        )

        # Mark gear-change events
        for change_time in gear_change_times:
            plt.axvline(change_time, color='gray', linestyle='--', linewidth=1)

        plt.xlabel('Time in \si{\second}')
        if len(y_label) > 20:
            y_label = y_label[:20] + '\n' + y_label[20:]
        plt.ylabel(y_label)
        plt.legend()
        plt.grid()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)

        if save_figures:
            filename = f'{filename_prefix}_{driving_signal}_comparison.pdf'
            plt.savefig(f'{path}{filename}', bbox_inches='tight')
            print(f'Saved plot to {path}{filename}')
        plt.show()



#### helper function
import ast

def shorten_smoothing_kwargs(s):
    """
    Shortens the string representation of smoothing_kwargs to a concise label.
    """
    try:
        # Safely evaluate the string to a dictionary
        d = ast.literal_eval(s)
        filter_type = d.get('filter_type', '')
        if filter_type == 'moving_average':
            window_size = d.get('window_size', '')
            return f"MA{window_size}"
        elif filter_type == 'savitzky_golay':
            window_length = d.get('window_length', '')
            polyorder = d.get('polyorder', '')
            return f"SG{window_length}_{polyorder}"
        elif filter_type == 'lowpass':
            cutoff_frequency = d.get('cutoff_frequency', '')
            sample_rate = d.get('sample_rate', '')
            order = d.get('order', '')
            return f"LP{cutoff_frequency}_{sample_rate}_{order}"
        elif filter_type == 'exponential_moving_average':
            alpha = d.get('alpha', '')
            return f"EMA{alpha}"
        else:
            return s  # Return the original string if filter_type is unknown
    except Exception as e:
        #print(f"Failed to shorten: {s}, error: {e}")
        return s  # Return the original string if evaluation fails
    
def shorten_speed_signal_names(s):
    """
    Shortens the speed signal names to a concise label.
    """
    try:
        if s == 'vehicle_speed':
            return 'vs'
        elif s == 'vehicle_speed_pitch_corrected':
            return 'vs_c'
        elif s == 'vehicle_speed_gps_pitch_corrected':
            return 'vs_gps_c'
        elif s == 'vehicle_speed_gps':
            return 'vs_gps'
        else:
            return s  # Return the original string if it doesn't match any known pattern
    except Exception as e:
        #print(f"Failed to shorten: {s}, error: {e}")
        return s  # Return the original string if any exception occurs
    
def shorten_full_load_curve(s):
    """
    Shortens the which_full_load_curve entries to concise labels.
    """
    mapping = {
        'driving_data': 'drive',
        'dynamo_data': 'dyno',
        'adjusted': 'adj',
        'overlap': 'ovlp'
    }
    return mapping.get(s, s)  # Default to the original value if no match is found
    

def shorten_twoD_smoothing_kwargs(s):
    """
    Shortens the string representation of twoD_smoothing_kwargs to a concise label.
    """

    if not isinstance(s, dict):
        return str(s)  # Return as string if it's not a dictionary or None
    try:
        method = s.get('method', 'unknown')
        if method == 'idw':
            power = s.get('power', '')
            neighbors = s.get('num_neighbors', '')
            return f"IDW{power}_{neighbors}"
        elif method == 'gaussian_filter':
            sigma = s.get('sigma', '')
            grid_size = s.get('grid_size', '')
            return f"GF{sigma}_{grid_size}"
        elif method == 'griddata':
            interp_method = s.get('interp_method', '')
            return f"GD_{interp_method}"
        elif method == 'regression':  # Assume regression is always random forest
            rf_params = s.get('model_params', {})
            n_estimators = rf_params.get('n_estimators', '')
            max_depth = rf_params.get('max_depth', '')
            random_state = rf_params.get('random_state', '')
            return f"RF{n_estimators}_{max_depth}_{random_state}"
        else:
            return method  # Return the method if it's unknown
    except Exception as e:
        #print(f"Failed to shorten: {s}, error: {e}")
        return str(s)  # Return the original representation if evaluation fails
    