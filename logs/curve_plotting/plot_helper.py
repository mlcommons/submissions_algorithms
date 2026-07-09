import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
import argparse

# Configure styling params
def configure_styling(custom_params=None):
    params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    }
    if custom_params:
        params.update(custom_params)
    plt.rcParams.update(params)

# Define registry of algorithms and their configuration mapping
ALGO_CONFIGS = {
    'sfadamw': {
        'submissions': {
            'schedule_free_adamw': {
                'color': '#1F77B4',     # Classic Blue
                'linestyle': '-',       # Solid
                'label': 'PyTorch v1',
                'alpha': 0.9
            },
            'schedule_free_adamw_v2': {
                'color': '#0B3C5D',     # Deep Navy
                'linestyle': '--',      # Dashed
                'label': 'PyTorch v2',
                'alpha': 0.9
            },
            'schedule_free_adamw_jax': {
                'color': '#FF7F0E',     # Safety Orange
                'linestyle': '-',       # Solid
                'label': 'JAX v1',
                'alpha': 0.9
            },
            'schedule_free_adamw_jax_v2': {
                'color': '#D9531E',     # Vibrant Rust
                'linestyle': '--',      # Dashed
                'label': 'JAX v2',
                'alpha': 0.9
            }
        },
        'sub_dir': 'sfadamw'
    },
    'muon': {
        'submissions': {
            'muon_torch': {
                'color': '#1F77B4',
                'linestyle': '-',
                'label': 'PyTorch v1',
                'alpha': 0.9
            },
            'muon_torch_jax_hps': {
                'color': '#0B3C5D',
                'linestyle': '--',
                'label': 'PyTorch v2 (JAX HPS)',
                'alpha': 0.9
            },
            'muon': {
                'color': '#FF7F0E',
                'linestyle': '-',
                'label': 'JAX v1',
                'alpha': 0.9
            }
        },
        'sub_dir': 'muon'
    },
    'ademamix': {
        'submissions': {
            'ademamix': {
                'color': '#1F77B4',
                'linestyle': '-',
                'label': 'PyTorch',
                'alpha': 0.9
            }
        },
        'sub_dir': 'ademamix'
    },
    'cautious_nadamw': {
        'submissions': {
            'cautious_nadamw': {
                'color': '#FF7F0E',
                'linestyle': '-',
                'label': 'JAX',
                'alpha': 0.9
            }
        },
        'sub_dir': 'cautious_nadamw'
    },
    'lion': {
        'submissions': {
            'lion': {
                'color': '#1F77B4',
                'linestyle': '-',
                'label': 'PyTorch',
                'alpha': 0.9
            }
        },
        'sub_dir': 'lion'
    },
    'nadamw': {
        'submissions': {
            'nadamw': {
                'color': '#FF7F0E',
                'linestyle': '-',
                'label': 'JAX v1',
                'alpha': 0.9
            },
            'nadamw_baselinev05': {
                'color': '#D9531E',
                'linestyle': '--',
                'label': 'JAX Baseline v0.5',
                'alpha': 0.9
            }
        },
        'sub_dir': 'nadamw'
    }
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Publication-grade plotting script for self-tuning benchmarks.")
    parser.add_argument('--algo', type=str, choices=list(ALGO_CONFIGS.keys()) + ['all'], default='all',
                        help="Algorithm name to plot (choices: sfadamw, muon, ademamix, cautious_nadamw, lion, nadamw, all)")
    parser.add_argument('--zoom', type=str, choices=['percentile', 'log'], default='log',
                        help="Type of zoom/scaling to use for the y-axis: 'percentile' (trims initial spikes with a linear scale) or 'log' (default, uses a logarithmic scale with the full data range).")
    return parser.parse_args()

def find_workloads(base_log_dir, submissions):
    """Finds all workloads associated with the given submissions."""
    workloads = set()
    for sub in submissions.keys():
        path = os.path.join(base_log_dir, sub, 'study_*', '*')
        dirs = glob.glob(path)
        for d in dirs:
            if os.path.isdir(d):
                dirname = os.path.basename(d)
                # Normalize workload name by removing framework suffixes
                base_name = dirname.replace('_pytorch', '').replace('_jax', '')
                workloads.add(base_name)
    return workloads

def get_target_metric_and_value(base_log_dir, workload, submissions):
    """Extracts target metric and validation target value from metadata."""
    for sub in submissions.keys():
        pattern = os.path.join(base_log_dir, sub, 'study_*', f"{workload}*", 'trial_*', 'meta_data_0.json')
        files = glob.glob(pattern)
        if files:
            try:
                with open(files[0], 'r') as f:
                    data = json.load(f)
                    target_metric = data.get('workload.target_metric_name')
                    target_value = data.get('workload.validation_target_value')
                    return target_metric, target_value
            except Exception as e:
                print(f"Error reading {files[0]}: {e}")
                continue
    return None, None

def load_workload_data(base_log_dir, workload, submissions, csv_col_name):
    """Loads and filters CSV data for all submissions of a workload."""
    workload_curves = {}
    has_data = False
    all_metric_values = []
    
    for sub, style in submissions.items():
        pattern = os.path.join(base_log_dir, sub, 'study_*', f"{workload}*", 'trial_*', 'eval_measurements.csv')
        files = glob.glob(pattern)
        
        if not files:
            continue
            
        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if csv_col_name in df.columns:
                    df = df.dropna(subset=[csv_col_name, 'accumulated_submission_time', 'global_step'])
                    if not df.empty:
                        dfs.append(df)
                        all_metric_values.extend(df[csv_col_name].tolist())
            except Exception:
                pass
                
        if dfs:
            workload_curves[sub] = (dfs, style)
            has_data = True
            
    return workload_curves, has_data, all_metric_values

def calculate_y_limits(all_metric_values, target_value, zoom, higher_is_better):
    """Calculates y-axis limits based on the zoom strategy and metric type."""
    if not all_metric_values:
        return 0.0, 1.0

    sorted_vals = sorted(all_metric_values)
    n = len(sorted_vals)
    
    if zoom == 'percentile':
        if higher_is_better:
            pct_5 = sorted_vals[int(n * 0.05)]
            ymin = max(0.0, pct_5 * 0.95) if pct_5 > 0.1 else 0.0
            
            ymax = sorted_vals[-1]
            if target_value is not None:
                ymax = max(ymax, target_value)
            ymax = ymax * 1.05
            if all(v <= 1.0 for v in all_metric_values):
                ymax = min(1.0, ymax)
        else:
            min_val = sorted_vals[0]
            ymin = min_val * 0.95
            if target_value is not None:
                ymin = min(ymin, target_value * 0.9)
            ymin = max(0.0, ymin)
            
            pct_90 = sorted_vals[int(n * 0.90)]
            ymax = pct_90
            if target_value is not None:
                ymax = max(ymax, target_value * 1.5)
            if ymax <= ymin:
                ymax = ymin * 2.0 if ymin > 0 else 1.0
    else: # zoom == 'log'
        min_val = sorted_vals[0]
        max_val = sorted_vals[-1]
        
        if higher_is_better:
            ymin = max(0.0, min_val * 0.95)
            ymax = max_val * 1.05
            if target_value is not None:
                ymin = min(ymin, target_value * 0.95)
                ymax = max(ymax, target_value * 1.05)
            if all(v <= 1.0 for v in all_metric_values):
                ymax = min(1.0, ymax)
        else:
            ymin = min_val * 0.95
            if target_value is not None:
                ymin = min(ymin, target_value * 0.9)
            # Ensure ymin is strictly positive for log scale
            ymin = max(1e-6, ymin)
            
            ymax = max_val * 1.05
            if target_value is not None:
                ymax = max(ymax, target_value * 1.05)
            if ymax <= ymin:
                ymax = ymin * 2.0
    return ymin, ymax

def interpolate_curve(dfs, x_col, y_col, num_points=150):
    """Interpolates curves to a uniform grid and computes mean/std."""
    all_x = []
    for df in dfs:
        all_x.extend(df[x_col].tolist())
    
    if not all_x:
        return None, None, None

    min_x = min(all_x)
    max_x = max(all_x)
    
    grid_x = np.linspace(min_x, max_x, num_points)
    
    interpolated_metrics = []
    for df in dfs:
        interp_val = np.interp(grid_x, df[x_col], df[y_col], right=df[y_col].iloc[-1])
        interpolated_metrics.append(interp_val)
        
    mean_curve = np.nanmean(interpolated_metrics, axis=0)
    std_curve = np.nanstd(interpolated_metrics, axis=0)
    std_curve = np.nan_to_num(std_curve, nan=0.0)
    
    return grid_x, mean_curve, std_curve

def plot_curve(ax, x, mean, std, style, x_scale=1.0):
    """Plots a mean curve with shaded std deviation."""
    if x is None:
        return
    scaled_x = x / x_scale
    ax.plot(scaled_x, mean, 
             color=style['color'], linestyle=style['linestyle'],
             label=style['label'], alpha=style['alpha'], linewidth=2.5)
    
    ax.fill_between(scaled_x, 
                     mean - std, 
                     mean + std, 
                     color=style['color'], alpha=0.10, edgecolor='none')

def setup_axes(ax, ymin, ymax, zoom, higher_is_better, target_value, xlabel, ylabel):
    """Applies styling and limits to an axis."""
    if zoom == 'log' and not higher_is_better:
        ax.set_yscale('log')
    else:
        ax.set_yscale('linear')
        
    ax.set_ylim(ymin, ymax)
    
    if target_value is not None:
        ax.axhline(y=target_value, color='#D0021B', linestyle=':', linewidth=1.5, label=f'Target ({target_value})')
        
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, edgecolor='#e5e5e5')
    ax.grid(True, which="major", color="#e8e8e8", linestyle="-", linewidth=0.8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    ax.set_xlabel(xlabel, color='#333333', fontweight='semibold')
    ax.set_ylabel(ylabel, color='#333333', fontweight='semibold')

def plot_workload(workload, algo_name, submissions, base_log_dir, base_save_dir, zoom, show_plot=False):
    """Orchestrates loading, interpolating, and plotting for a single workload."""
    target_metric, target_value = get_target_metric_and_value(base_log_dir, workload, submissions)
    if not target_metric:
        print(f"Could not find target metric for {workload}, skipping.")
        return

    csv_col_name = f"validation/{target_metric}"
    higher_is_better = any(x in target_metric.lower() for x in ['accuracy', 'auc', 'map', 'bleu', 'ssim', 'precision', 'score'])

    workload_curves, has_data, all_metric_values = load_workload_data(base_log_dir, workload, submissions, csv_col_name)

    if not has_data:
        return

    ymin, ymax = calculate_y_limits(all_metric_values, target_value, zoom, higher_is_better)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    title_suffix = " (Log Scale)" if (zoom == 'log' and not higher_is_better) else ""
    fig.suptitle(f"{algo_name} on {workload} (Metric: {target_metric}){title_suffix}", fontweight='bold', y=0.98)

    for sub, (dfs, style) in workload_curves.items():
        # Time-based Interpolation
        grid_times, mean_time, std_time = interpolate_curve(dfs, 'accumulated_submission_time', csv_col_name)
        plot_curve(ax1, grid_times, mean_time, std_time, style, x_scale=3600.0) # Convert to hours

        # Step-based Interpolation
        grid_steps, mean_step, std_step = interpolate_curve(dfs, 'global_step', csv_col_name)
        plot_curve(ax2, grid_steps, mean_step, std_step, style, x_scale=1000.0) # Convert to k-steps

    setup_axes(ax1, ymin, ymax, zoom, higher_is_better, target_value, 'Accumulated Time (hours)', f'Validation {target_metric.upper()}')
    setup_axes(ax2, ymin, ymax, zoom, higher_is_better, target_value, 'Global Steps (x10³)', f'Validation {target_metric.upper()}')

    plt.tight_layout()

    save_dir = base_save_dir / ALGO_CONFIGS[algo_name]['sub_dir']
    save_dir.mkdir(exist_ok=True, parents=True)
    png_path = save_dir / f'{workload}_curves.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved PNG to {png_path}")

def main():
    configure_styling()
    args = parse_arguments()

    if args.algo == 'all':
        algos_to_run = list(ALGO_CONFIGS.keys())
    else:
        algos_to_run = [args.algo]

    print(f"Running plotter for algorithms: {algos_to_run} with zoom={args.zoom}")

    base_log_dir = Path('~/submissions_algorithms/logs/self_tuning').expanduser()
    base_save_dir = Path('~/submissions_algorithms/logs/curve_plotting').expanduser()

    for algo_name in algos_to_run:
        print(f"\n=========================================")
        print(f"PLOTTING ALGORITHM: {algo_name}")
        print(f"=========================================")
        
        try:
            config = ALGO_CONFIGS[algo_name]
            submissions = config['submissions']
            
            workloads = find_workloads(base_log_dir, submissions)
            print(f"Found workloads for {algo_name}: {workloads}")
            
            for workload in workloads:
                print(f"\nProcessing workload: {workload}")
                plot_workload(workload, algo_name, submissions, base_log_dir, base_save_dir, args.zoom, show_plot=False)
                
        except Exception as e:
            print(f"ERROR: Failed to plot algorithm '{algo_name}': {e}")
            traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    main()
