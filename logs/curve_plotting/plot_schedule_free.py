import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define submissions and their styles
submissions = {
    'schedule_free_adamw': {'color': 'skyblue', 'label': 'PyTorch v1', 'alpha': 0.8},
    'schedule_free_adamw_v2': {'color': 'darkblue', 'label': 'PyTorch v2', 'alpha': 0.8},
    'schedule_free_adamw_jax': {'color': 'wheat', 'label': 'JAX v1', 'alpha': 0.8},
    'schedule_free_adamw_jax_v2': {'color': 'darkorange', 'label': 'JAX v2', 'alpha': 0.8}
}

base_log_dir = Path('~/submissions_algorithms/logs/self_tuning').expanduser()
save_dir = Path('~/submissions_algorithms/logs/curve_plotting/sfadamw').expanduser()

# Find all workloads
workloads = set()
for sub in submissions.keys():
    path = os.path.join(base_log_dir, sub, 'study_*', '*')
    dirs = glob.glob(path)
    for d in dirs:
        if os.path.isdir(d):
            dirname = os.path.basename(d)
            base_name = dirname.replace('_pytorch', '').replace('_jax', '')
            workloads.add(base_name)

print(f"Found workloads: {workloads}")

for workload in workloads:
    print(f"\nProcessing workload: {workload}")
    
    # Find target metric and value from the first available trial
    target_metric = None
    target_value = None
    
    for sub in submissions.keys():
        pattern = os.path.join(base_log_dir, sub, 'study_*', f"{workload}*", 'trial_*', 'meta_data_0.json')
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
    
    # Prepare plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Workload: {workload} (Target: {target_metric})")
    
    has_data = False
    
    for sub, style in submissions.items():
        pattern = os.path.join(base_log_dir, sub, 'study_*', f"{workload}*", 'trial_*', 'eval_measurements.csv')
        files = glob.glob(pattern)
        
        if not files:
            print(f"  No data for {sub}")
            continue
            
        print(f"  Found {len(files)} trials for {sub}")
        
        all_dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if csv_col_name in df.columns:
                    all_dfs.append(df)
                else:
                    print(f"    Column {csv_col_name} not found in {f}")
            except Exception as e:
                print(f"    Error reading {f}: {e}")
                
        if not all_dfs:
            continue
            
        has_data = True

        # 1. Plot individual trial runs faintly to show raw trajectories
        for df in all_dfs:
            ax1.plot(df['accumulated_submission_time'], df[csv_col_name], 
                     color=style['color'], alpha=0.15, linewidth=1)
            ax2.plot(df['global_step'], df[csv_col_name], 
                     color=style['color'], alpha=0.15, linewidth=1)
        
        # 2. Align metrics and calculate Mean + Standard Deviation across trials
        combined_df = pd.concat(all_dfs)
        stats_df = combined_df.groupby('global_step')[csv_col_name].agg(['mean', 'std']).reset_index()
        
        # Average the time per step to align the time-series plot
        time_df = combined_df.groupby('global_step')['accumulated_submission_time'].mean().reset_index()
        stats_df = pd.merge(stats_df, time_df, on='global_step')
        
        # Fill missing std calculations (e.g., if step counts differ slightly between runs)
        stats_df['std'] = stats_df['std'].fillna(0)
        
        # 3. Plot the bold average line
        ax1.plot(stats_df['accumulated_submission_time'], stats_df['mean'], 
                 color=style['color'], label=style['label'], alpha=1.0, linewidth=2.5)
        
        ax2.plot(stats_df['global_step'], stats_df['mean'], 
                 color=style['color'], label=style['label'], alpha=1.0, linewidth=2.5)
        
        # 4. Fill the shaded area representing +/- 1 standard deviation
        ax1.fill_between(stats_df['accumulated_submission_time'], 
                         stats_df['mean'] - stats_df['std'], 
                         stats_df['mean'] + stats_df['std'], 
                         color=style['color'], alpha=0.10)
        
        ax2.fill_between(stats_df['global_step'], 
                         stats_df['mean'] - stats_df['std'], 
                         stats_df['mean'] + stats_df['std'], 
                         color=style['color'], alpha=0.10)
                 
    if not has_data:
        plt.close(fig)
        continue
        
    # Configure axes
    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        if target_value is not None:
            ax.axhline(y=target_value, color='r', linestyle='--', label=f'Target ({target_value})')
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
    ax1.set_xlabel('Accumulated Submission Time (s)')
    ax1.set_ylabel(csv_col_name)
    ax1.set_title(f'{csv_col_name} vs Time')
    
    ax2.set_xlabel('Global Step')
    ax2.set_ylabel(csv_col_name)
    ax2.set_title(f'{csv_col_name} vs Step')
    
    plt.tight_layout()
    
    # Save plot
    save_dir.mkdir(exist_ok=True, parents=True)
    out_path = save_dir / f'{workload}_curves.png'
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")

print("Done.")
