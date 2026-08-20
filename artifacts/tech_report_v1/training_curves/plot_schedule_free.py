import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# Configure styling params
plt.rcParams.update({
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
})

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

import argparse

parser = argparse.ArgumentParser(description="Publication-grade plotting script for self-tuning benchmarks.")
parser.add_argument('--algo', type=str, choices=list(ALGO_CONFIGS.keys()) + ['all'], default='all',
                    help="Algorithm name to plot (choices: sfadamw, muon, ademamix, cautious_nadamw, lion, nadamw, all)")
parser.add_argument('--zoom', type=str, choices=['percentile', 'log'], default='log',
                    help="Type of zoom/scaling to use for the y-axis: 'percentile' (trims initial spikes with a linear scale) or 'log' (default, uses a logarithmic scale with the full data range).")
args = parser.parse_args()

# Select algorithms to run
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
        save_dir = base_save_dir / config['sub_dir']
        
        # Find all workloads for this algorithm
        workloads = set()
        for sub in submissions.keys():
            path = os.path.join(base_log_dir, sub, 'study_*', '*')
            dirs = glob.glob(path)
            for d in dirs:
                if os.path.isdir(d):
                    dirname = os.path.basename(d)
                    base_name = dirname.replace('_pytorch', '').replace('_jax', '')
                    workloads.add(base_name)
        
        print(f"Found workloads for {algo_name}: {workloads}")
        
        for workload in workloads:
            print(f"\nProcessing workload: {workload}")
            
            # Find target metric and value from the first available trial
            target_metric = None
            target_value = None
            
            for sub in submissions.items():
                pattern = os.path.join(base_log_dir, sub[0], 'study_*', f"{workload}*", 'trial_*', 'meta_data_0.json')
                files = glob.glob(pattern)
                if files:
                    try:
                        with open(files[0], 'r') as f:
                            data = json.load(f)
                            target_metric = data.get('workload.target_metric_name')
                            target_value = data.get('workload.validation_target_value')
                            print(f"Found target metric: {target_metric}, value: {target_value} from {files[0]}")
                            break
                    except Exception as e:
                        print(f"Error reading {files[0]}: {e}")
                        continue
                        
            if not target_metric:
                print(f"Could not find target metric for {workload}, skipping.")
                continue
                
            csv_col_name = f"validation/{target_metric}"
            
            # Check if metric is "higher is better" (Accuracy, BLEU, SSIM, AUC, MAP)
            higher_is_better = any(x in target_metric.lower() for x in ['accuracy', 'auc', 'map', 'bleu', 'ssim', 'precision', 'score'])
            
            # Prepare plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
            title_suffix = " (Log Scale)" if (args.zoom == 'log' and not higher_is_better) else ""
            fig.suptitle(f"Workload: {workload} (Metric: {target_metric}){title_suffix}", fontweight='bold', y=0.98)
            
            has_data = False
            all_metric_values = []
            
            # Step 1: Gather and inspect raw data curves for bounds checking
            workload_curves = {}
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
                    except Exception as e:
                        pass
                        
                if dfs:
                    workload_curves[sub] = (dfs, style)
                    has_data = True
        
            if not has_data:
                plt.close(fig)
                continue
                
            # Calculate y-limits depending on the zoom strategy
            if all_metric_values:
                sorted_vals = sorted(all_metric_values)
                n = len(sorted_vals)
                
                if args.zoom == 'percentile':
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
                else: # args.zoom == 'log'
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
            else:
                ymin, ymax = 0.0, 1.0
        
            # Second pass: interpolate and plot
            for sub, (dfs, style) in workload_curves.items():
                # --- Time-based Interpolation ---
                # Find global time range for this submission
                all_times = []
                for df in dfs:
                    all_times.extend(df['accumulated_submission_time'].tolist())
                
                if all_times:
                    min_time = min(all_times)
                    max_time = max(all_times)
                    
                    # Create a uniform grid of 150 points for smooth rendering
                    grid_times = np.linspace(min_time, max_time, 150)
                    
                    # Interpolate
                    interpolated_metrics = []
                    for df in dfs:
                        interp_val = np.interp(grid_times, df['accumulated_submission_time'], df[csv_col_name], right=df[csv_col_name].iloc[-1])
                        interpolated_metrics.append(interp_val)
                        
                    # Compute mean and standard deviation
                    mean_time_curve = np.nanmean(interpolated_metrics, axis=0)
                    std_time_curve = np.nanstd(interpolated_metrics, axis=0)
                    std_time_curve = np.nan_to_num(std_time_curve, nan=0.0)
                    
                    # Plot Time Curves (in hours)
                    time_hours = grid_times / 3600.0
                    ax1.plot(time_hours, mean_time_curve, 
                             color=style['color'], linestyle=style['linestyle'],
                             label=style['label'], alpha=style['alpha'], linewidth=2.5)
                    
                    ax1.fill_between(time_hours, 
                                     mean_time_curve - std_time_curve, 
                                     mean_time_curve + std_time_curve, 
                                     color=style['color'], alpha=0.10, edgecolor='none')
                                     
                # --- Step-based Interpolation ---
                all_steps = []
                for df in dfs:
                    all_steps.extend(df['global_step'].tolist())
                    
                if all_steps:
                    min_step = min(all_steps)
                    max_step = max(all_steps)
                    
                    grid_steps = np.linspace(min_step, max_step, 150)
                    
                    interpolated_steps = []
                    for df in dfs:
                        interp_val = np.interp(grid_steps, df['global_step'], df[csv_col_name], right=df[csv_col_name].iloc[-1])
                        interpolated_steps.append(interp_val)
                        
                    mean_step_curve = np.nanmean(interpolated_steps, axis=0)
                    std_step_curve = np.nanstd(interpolated_steps, axis=0)
                    std_step_curve = np.nan_to_num(std_step_curve, nan=0.0)
                    
                    steps_k = grid_steps / 1000.0
                    ax2.plot(steps_k, mean_step_curve, 
                             color=style['color'], linestyle=style['linestyle'],
                             label=style['label'], alpha=style['alpha'], linewidth=2.5)
                    
                    ax2.fill_between(steps_k, 
                                     mean_step_curve - std_step_curve, 
                                     mean_step_curve + std_step_curve, 
                                     color=style['color'], alpha=0.10, edgecolor='none')
        
            # Configure axes
            for ax in [ax1, ax2]:
                if args.zoom == 'log' and not higher_is_better:
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
                
            ax1.set_xlabel('Accumulated Time (hours)', color='#333333', fontweight='semibold')
            ax1.set_ylabel(f'Validation {target_metric.upper()}', color='#333333', fontweight='semibold')
            
            ax2.set_xlabel('Global Steps (x10³)', color='#333333', fontweight='semibold')
            ax2.set_ylabel(f'Validation {target_metric.upper()}', color='#333333', fontweight='semibold')
            
            plt.tight_layout()
            
            # Save plots
            save_dir.mkdir(exist_ok=True, parents=True)
            
            png_path = save_dir / f'{workload}_curves.png'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            
            plt.close(fig)
            print(f"Saved PNG to {png_path}")
            
    except Exception as e:
        print(f"ERROR: Failed to plot algorithm '{algo_name}': {e}")
        traceback.print_exc()

print("\nDone.")
