import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Define the target mapping of submission keys to display names
OPTIMIZERS = {
    'schedule_free_adamw_jax': 'JAX Schedule-Free v1',
    'schedule_free_adamw_jax_v2': 'JAX Schedule-Free v2',
    'schedule_free_adamw': 'PyTorch Schedule-Free v1',
    'schedule_free_adamw_v2': 'PyTorch Schedule-Free v2'
}

base_log_dir = Path('~/submissions_algorithms/logs/self_tuning').expanduser()

# Find all workloads present in any of the target folders
workloads = set()
for sub in OPTIMIZERS.keys():
    pattern = os.path.join(base_log_dir, sub, 'study_*', '*')
    dirs = glob.glob(pattern)
    for d in dirs:
        if os.path.isdir(d):
            dirname = os.path.basename(d)
            base_name = dirname.replace('_pytorch', '').replace('_jax', '')
            workloads.add(base_name)

workloads = sorted(list(workloads))
print(f"Found workloads: {workloads}")

# Data structure to hold results: results[opt_name][workload] = list of step_times
results = {opt: {wl: [] for wl in workloads} for opt in OPTIMIZERS.values()}

for sub_key, opt_display in OPTIMIZERS.items():
    print(f"\nProcessing {opt_display} ({sub_key})...")
    for wl in workloads:
        # Find all trials for this workload and optimizer
        pattern = os.path.join(base_log_dir, sub_key, 'study_*', f"{wl}*", 'trial_*', 'measurements.csv')
        files = glob.glob(pattern)
        
        trial_times = []
        for f in files:
            try:
                df = pd.read_csv(f)
                # Ensure we have the necessary columns and drop rows that don't have them
                if 'accumulated_submission_time' in df.columns and 'global_step' in df.columns:
                    df_valid = df.dropna(subset=['accumulated_submission_time', 'global_step'])
                    if len(df_valid) >= 2:
                        first_row = df_valid.iloc[0]
                        last_row = df_valid.iloc[-1]
                        
                        delta_t = last_row['accumulated_submission_time'] - first_row['accumulated_submission_time']
                        delta_s = last_row['global_step'] - first_row['global_step']
                        
                        if delta_s > 0:
                            avg_step_time_ms = (delta_t / delta_s) * 1000.0
                            trial_times.append(avg_step_time_ms)
            except Exception as e:
                print(f"Error processing trial file {f}: {e}")
                
        results[opt_display][wl] = trial_times

# Format table values as "mean ± std" or "-" if no data
formatted_table = {}
raw_table = {} # Store numeric mean for optional sorting/analysis

for opt_display in OPTIMIZERS.values():
    formatted_table[opt_display] = {}
    raw_table[opt_display] = {}
    for wl in workloads:
        times = results[opt_display][wl]
        if times:
            mean_val = np.mean(times)
            std_val = np.std(times)
            if len(times) > 1 and std_val > 0.01:
                formatted_table[opt_display][wl] = f"{mean_val:.1f} ± {std_val:.1f}"
            else:
                formatted_table[opt_display][wl] = f"{mean_val:.1f}"
            raw_table[opt_display][wl] = mean_val
        else:
            formatted_table[opt_display][wl] = "N/A"
            raw_table[opt_display][wl] = None

# Generate Markdown table manually
headers = ["Optimizer"] + workloads
md_lines = []
md_lines.append("| " + " | ".join(headers) + " |")
md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
for opt in OPTIMIZERS.values():
    row_vals = [formatted_table[opt][wl] for wl in workloads]
    md_lines.append("| " + opt + " | " + " | ".join(row_vals) + " |")
markdown_table = "\n".join(md_lines)

# Generate LaTeX table
latex_lines = []
latex_lines.append("\\begin{table*}[t]")
latex_lines.append("\\centering")
latex_lines.append("\\caption{Step Execution Time Comparison (milliseconds per step) across different workloads.}")
latex_lines.append("\\label{tab:step_time_comparison}")

# Column alignment: l followed by r for each workload column
col_align = "l" + "r" * len(workloads)
latex_lines.append(f"\\begin{{tabular}}{{{col_align}}}")
latex_lines.append("\\toprule")

# Header row (clean workload names formatted for LaTeX)
escaped_workloads = [wl.replace('_', '\\_') for wl in workloads]
latex_lines.append("Optimizer & " + " & ".join(escaped_workloads) + " \\\\")
latex_lines.append("\\midrule")

# Data rows
for opt in OPTIMIZERS.values():
    row_vals = []
    for wl in workloads:
        val = formatted_table[opt][wl]
        # Replace ± with \pm for LaTeX math mode
        if "±" in val:
            parts = val.split(" ± ")
            row_vals.append(f"${parts[0]} \\pm {parts[1]}$")
        elif val == "N/A":
            row_vals.append("---")
        else:
            row_vals.append(f"${val}$")
    latex_lines.append(f"{opt} & " + " & ".join(row_vals) + " \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table*}")

latex_table = "\n".join(latex_lines)

# Output results to files and console
output_dir = Path('~/submissions_algorithms/logs/step_size_analysis/sfadamw_step_size').expanduser()
output_dir.mkdir(exist_ok=True, parents=True)

with open(output_dir / 'step_time_comparison.md', 'w') as f:
    f.write("# Step Execution Time Comparison of Schedule Free Adamw in Markdown (ms/step)\n\n")
    f.write(markdown_table)
    f.write("\n\n## LaTeX Source Code\n\n```latex\n")
    f.write(latex_table)
    f.write("\n```\n")

print("\n=================== MARKDOWN TABLE ===================")
print(markdown_table)
print("\n==================== LAATEX TABLE ====================")
print(latex_table)
print(f"\nSaved tables to {output_dir / 'step_time_comparison.md'}")
