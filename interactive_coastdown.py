import matplotlib
matplotlib.use("Agg")  # No GUI on macOS
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX rendering
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Set default font

import matplotlib.pyplot as plt

def no_op_show(*args, **kwargs):
    """Disable the actual window popup."""
    pass

plt.show = no_op_show
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import base64
import json
from io import BytesIO
import traceback
from modules.parametric_pipelines import coastdown_pipeline
from modules.data_handler import get_can_files, load_can_data


def mplfig_to_base64(fig):
    """
    Convert a Matplotlib figure to a base64-encoded PNG string.
    """
    # Force the figure to render completely
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')

path = 'data/complete_coastdown_data/coastdown_good_imu'
files = get_can_files(folder=path)
data_cache = {}
for f in files:
    data_cache[f] = load_can_data(f, verbose=False)

SPEED_SIGNALS = [
    'vehicle_speed',
    'vehicle_speed_pitch_corrected',
    'vehicle_speed_gps',
    'vehicle_speed_gps_pitch_corrected'
]
BUCKET_SIZES = [1, 2, 4, 6, 8, 10, 14, 18]

SMOOTHING_OPTIONS = [
    {'filter_type': 'moving_average', 'window_size': 5},
    {'filter_type': 'moving_average', 'window_size': 20},
    {'filter_type': 'moving_average', 'window_size': 50},
    {'filter_type': 'exponential_moving_average', 'alpha': 0.01},
    {'filter_type': 'exponential_moving_average', 'alpha': 0.05},
    {'filter_type': 'exponential_moving_average', 'alpha': 0.1},
    {'filter_type': 'exponential_moving_average', 'alpha': 0.2},
    {'filter_type': 'exponential_moving_average', 'alpha': 0.4},
    {'filter_type': 'savitzky_golay', 'window_length': 50,  'polyorder': 3},
    {'filter_type': 'savitzky_golay', 'window_length': 100, 'polyorder': 3},
    {'filter_type': 'savitzky_golay', 'window_length': 150, 'polyorder': 3},
    {'filter_type': 'savitzky_golay', 'window_length': 50,  'polyorder': 5},
    {'filter_type': 'savitzky_golay', 'window_length': 100, 'polyorder': 5},
    {'filter_type': 'savitzky_golay', 'window_length': 150, 'polyorder': 5},
    {'filter_type': 'savitzky_golay', 'window_length': 50,  'polyorder': 1},
    {'filter_type': 'savitzky_golay', 'window_length': 100, 'polyorder': 1},
    {'filter_type': 'savitzky_golay', 'window_length': 150, 'polyorder': 1},
    None
]
STEERING_ANGLE_LIMITS = [1, 2, 5, 10, 20, 40]
DERIV_LOWER_LIMITS    = [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0]
DERIV_UPPER_LIMITS    = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04, 0.05]
CUT_TIMES             = [0.01, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
OUTLIER_THRESHOLDS    = [None, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
SELECT_SUSPENSION_LEVELS = [None, 1, 1.5, 2, 2.5]

DO_PITCH_CORRECTION     = True
VEHICLE_MASS            = 2300
ROTATING_MASS_EQ        = 50
FRONTAL_AREA            = 2.33
COLUMNS_TO_SMOOTH       = ['accelerator_pedal', 'steering_wheel_angle', 'vehicle_speed']
SEED                    = 42
TARGET_N_SEGMENTS       = None
LOSS_TYPE               = 'sec_rmse'
GENERATE_PLOTS          = True
VERBOSE                 = False


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive Coastdown"),

    # Left Column: Parameter Controls
    html.Div([
        # Speed Signal Dropdown
        html.Label("Speed Signal"),
        dcc.Dropdown(
            id="speed-signal",
            options=[{"label": s, "value": s} for s in SPEED_SIGNALS],
            value="vehicle_speed",
            clearable=False
        ),

        # Select Suspension Level Dropdown
        html.Label("Select Suspension Level"),
        dcc.Dropdown(
            id="select-suspension-level",
            options=[
                {"label": "None", "value": "None"},
                {"label": "1", "value": 1},
                {"label": "1.5", "value": 1.5},
                {"label": "2", "value": 2},
                {"label": "2.5", "value": 2.5}
            ],
            value="None",
            clearable=False
        ),

        # Bucket Size Dropdown
        html.Label("Bucket Size"),
        dcc.Dropdown(
            id="bucket-size",
            options=[{"label": str(b), "value": b} for b in BUCKET_SIZES],
            value=2,
            clearable=False
        ),

        # Smoothing Kwargs Dropdown
        html.Label("Smoothing Kwargs"),
        dcc.Dropdown(
            id="smoothing-dropdown",
            options=[
                {
                    "label": f"{idx+1}: {json.dumps(opt)}",
                    "value": json.dumps(opt)
                }
                for idx, opt in enumerate(SMOOTHING_OPTIONS) if opt is not None
            ] + [{"label": "None", "value": "None"}],
            value=json.dumps({'filter_type': 'moving_average', 'window_size': 5}),
            clearable=False
        ),

        # Steering Angle Limit Dropdown
        html.Label("Steering Angle Limit"),
        dcc.Dropdown(
            id="steering-angle-limit",
            options=[
                {"label": "None", "value": "None"}
            ] + [{"label": str(s), "value": s} for s in STEERING_ANGLE_LIMITS],
            value="None",
            clearable=False
        ),

        # Derivative Lower Limit Dropdown
        html.Label("Derivative Lower Limit"),
        dcc.Dropdown(
            id="deriv-lower-limit",
            options=[
                {"label": "None", "value": "None"}
            ] + [{"label": str(d), "value": d} for d in DERIV_LOWER_LIMITS],
            value="None",
            clearable=False
        ),

        # Derivative Upper Limit Dropdown
        html.Label("Derivative Upper Limit"),
        dcc.Dropdown(
            id="deriv-upper-limit",
            options=[
                {"label": "None", "value": "None"}
            ] + [{"label": str(d), "value": d} for d in DERIV_UPPER_LIMITS],
            value="None",
            clearable=False
        ),

        # Cut Time Dropdown
        html.Label("Cut Time (seconds)"),
        dcc.Dropdown(
            id="cut-time",
            options=[{"label": str(c), "value": c} for c in CUT_TIMES],
            value=2,
            clearable=False
        ),

        # Outlier Threshold Dropdown
        html.Label("Outlier Threshold"),
        dcc.Dropdown(
            id="outlier-threshold",
            options=[
                {"label": "None",  "value": "None"},
                {"label": "0.5",   "value": 0.5},
                {"label": "1.0",   "value": 1.0},
                {"label": "1.5",   "value": 1.5},
                {"label": "2.0",   "value": 2.0},
                {"label": "2.5",   "value": 2.5},
                {"label": "3.0",   "value": 3.0},
                {"label": "3.5",   "value": 3.5},
                {"label": "4.0",   "value": 4.0},
                {"label": "4.5",   "value": 4.5},
                {"label": "5.0",   "value": 5.0}
            ],
            value="None",
            clearable=False
        )
    ], style={
        "width": "20%",
        "float": "left",
        "overflowY": "auto",
        "height": "100vh",
        "padding": "10px"
    }),

    # Right Column: Figures + Stats
    html.Div([
        html.H4("Only 2nd & 3rd Matplotlib Figures"),
        html.Div([
            # Stack figures vertically by setting width to 100%
            html.Img(id='fig2-img', style={'width': '100%', 'margin': '10px 0'}),
            html.Img(id='fig3-img', style={'width': '100%', 'margin': '10px 0'})
        ], style={'width': '100%'}),

        # Added "Stats" Section at the Bottom
        html.H4("Stats"),
        html.Div(id='stats-div', style={
            'display': 'flex',
            'flexDirection': 'row',
            'justifyContent': 'space-around',
            'flexWrap': 'wrap',
            'marginTop': '20px'
        })
    ], style={
        "width": "75%",
        "float": "right",
        "padding": "10px",
        "overflowY": "auto",
        "height": "100vh"
    }),

    # Clear floats to maintain layout integrity
    html.Div(style={'clear': 'both'})
])


@app.callback(
    [
        Output('fig2-img', 'src'),
        Output('fig3-img', 'src'),
        Output('stats-div', 'children')
    ],
    [
        Input('speed-signal', 'value'),
        Input('bucket-size', 'value'),
        Input('smoothing-dropdown', 'value'),
        Input('steering-angle-limit', 'value'),
        Input('deriv-lower-limit', 'value'),
        Input('deriv-upper-limit', 'value'),
        Input('cut-time', 'value'),
        Input('outlier-threshold', 'value'),
        Input('select-suspension-level', 'value')
    ]
)
def run_pipeline_callback(
    speed_signal,
    bucket_size,
    smoothing_kwargs_json,
    steering_angle_limit,
    deriv_lower_limit,
    deriv_upper_limit,
    cut_time,
    outlier_threshold_str,
    select_suspension_level_str  # Receive suspension level as string
):
    """Runs pipeline with given parameters, displays figures vertically, and shows stats horizontally."""
    try:
        if outlier_threshold_str == "None":
            outlier_threshold = None
        else:
            try:
                outlier_threshold = float(outlier_threshold_str)
            except ValueError:
                outlier_threshold = None
        # Convert to JSPN
        if smoothing_kwargs_json and smoothing_kwargs_json != "None":
            try:
                smoothing_kwargs = json.loads(smoothing_kwargs_json)
            except json.JSONDecodeError:
                smoothing_kwargs = None  # Fallback
        else:
            smoothing_kwargs = None

        # Convert "None" -> actual None for select_suspension_level
        if select_suspension_level_str == "None":
            select_suspension_level = None
        else:
            try:
                select_suspension_level = float(select_suspension_level_str)
            except ValueError:
                select_suspension_level = None  # Fallback


        if steering_angle_limit == "None":
            steering_angle_limit = None
        else:
            try:
                steering_angle_limit = float(steering_angle_limit)
            except ValueError:
                steering_angle_limit = None  # Fallback


        if deriv_lower_limit == "None":
            deriv_lower_limit = None
        else:
            try:
                deriv_lower_limit = float(deriv_lower_limit)
            except ValueError:
                deriv_lower_limit = None  # Fallback

        if deriv_upper_limit == "None":
            deriv_upper_limit = None
        else:
            try:
                deriv_upper_limit = float(deriv_upper_limit)
            except ValueError:
                deriv_upper_limit = None  # Fallback


        for fignum in plt.get_fignums():
            plt.close(fignum)

        results = coastdown_pipeline(
            data_cache=data_cache,
            files=files,
            do_pitch_correction=DO_PITCH_CORRECTION,
            speed_signal=speed_signal,
            bucket_size=bucket_size,
            vehicle_mass=VEHICLE_MASS,
            rotating_mass_eq=ROTATING_MASS_EQ,
            frontal_area=FRONTAL_AREA,
            smoothing_kwargs=smoothing_kwargs,
            columns_to_smooth=COLUMNS_TO_SMOOTH,
            steering_angle_limit=steering_angle_limit,
            select_suspension_level=select_suspension_level,
            deriv_lower_limit=deriv_lower_limit,
            deriv_upper_limit=deriv_upper_limit,
            cut_time=cut_time,
            seed=SEED,
            target_n_segments=TARGET_N_SEGMENTS,
            outlier_threshold=outlier_threshold,
            loss_type=LOSS_TYPE,
            generate_plots=GENERATE_PLOTS,
            verbose=VERBOSE
        )

        # Unpack pipeline returns (f0, f1, f2, etc.)
        expected_length = 20
        if len(results) < expected_length:
            raise ValueError(f"Expected at least {expected_length} return values from pipeline, got {len(results)}")

        (
            f0, f1, f2, c_air, c_rr,
            metric_rm_steering_angle, metric_rm_brake_pedal, metric_rm_suspension_level,
            metric_rm_cut_time, metric_rm_sampling, total_removed, metric_outliers,
            loss_very_low_suspension, loss_low_suspension, loss_medium_suspension, loss_high_suspension,
            loss_CoC_values, loss_EPA_values, lowest_loss_name, elapsed_time
        ) = results

        fignums = plt.get_fignums()
        fig2_src, fig3_src = "", ""

        if len(fignums) > 1:
            fig2 = plt.figure(fignums[1])
            fig2_src = mplfig_to_base64(fig2)

        if len(fignums) > 2:
            fig3 = plt.figure(fignums[2])
            fig3_src = mplfig_to_base64(fig3)

        try:
            sec_rmse = min(
                loss_very_low_suspension,
                loss_low_suspension,
                loss_medium_suspension,
                loss_high_suspension
            )
        except TypeError:
            sec_rmse = "N/A" 

        # Build stats text for horizontal display
        stats_text = [
            html.Div(f"f0: {f0:.3f}", style={'margin': '5px'}),
            html.Div(f"f1: {f1:.3f}", style={'margin': '5px'}),
            html.Div(f"f2: {f2:.3f}", style={'margin': '5px'}),
            html.Div(f"c_air: {c_air:.3f}", style={'margin': '5px'}),
            html.Div(f"c_rr: {c_rr:.3f}", style={'margin': '5px'}),
            html.Div(f"sec_rmse: {sec_rmse if isinstance(sec_rmse, str) else f'{sec_rmse:.3f}'}", style={'margin': '5px'}),
            html.Div(f"lowest_loss_name: {lowest_loss_name}", style={'margin': '5px'}),
            html.Div(f"elapsed_time: {elapsed_time:.2f} s", style={'margin': '5px'}),
            html.Div(f"total_removed: {total_removed} data points", style={'margin': '5px'})
        ]

        for fignum in plt.get_fignums():
            plt.close(fignum)


        return fig2_src, fig3_src, stats_text

    except Exception as e:
        tb = traceback.format_exc()
        return "", "", [html.Pre(f"Error: {e}\n\nTraceback:\n{tb}")]


if __name__ == "__main__":
    app.run_server(debug=True)