import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import OrderedDict

# Registry of algorithms mapping algorithm key -> display name & submission versions/languages
ALGO_CONFIGS = {
    'sfadamw': {
        'display_name': 'Schedule-Free AdamW',
        'submissions': OrderedDict([
            ('schedule_free_adamw_jax', 'JAX Schedule-Free v1'),
            ('schedule_free_adamw_jax_v2', 'JAX Schedule-Free v2'),
            ('schedule_free_adamw', 'PyTorch Schedule-Free v1'),
            ('schedule_free_adamw_v2', 'PyTorch Schedule-Free v2')
        ])
    },
    'muon': {
        'display_name': 'Muon',
        'submissions': OrderedDict([
            ('muon', 'JAX Muon v1'),
            ('muon_torch', 'PyTorch Muon v1'),
            ('muon_torch_jax_hps_lr_fix', 'PyTorch Muon v2 (JAX HPS)'),
            ('muon_torch_replicated_jax_hps', 'PyTorch Muon Replicated (JAX HPS)'),
            ('muon_torch_replicated_torch_hps', 'PyTorch Muon Replicated (Torch HPS)')
        ])
    },
    'nadamw': {
        'display_name': 'NAdamW',
        'submissions': OrderedDict([
            ('nadamw', 'JAX NAdamW v1'),
            ('nadamw_baselinev05', 'JAX NAdamW Baseline v0.5'),
            ('nadamw_resnet', 'JAX NAdamW ResNet')
        ])
    },
    'ademamix': {
        'display_name': 'AdemaMix',
        'submissions': OrderedDict([
            ('ademamix', 'PyTorch AdemaMix')
        ])
    },
    'cautious_nadamw': {
        'display_name': 'Cautious NAdamW',
        'submissions': OrderedDict([
            ('cautious_nadamw', 'JAX Cautious NAdamW')
        ])
    },
    'lion': {
        'display_name': 'Lion',
        'submissions': OrderedDict([
            ('lion', 'PyTorch Lion')
        ])
    },
    'diloco': {
        'display_name': 'DiLoCo',
        'submissions': OrderedDict([
            ('single_worker_diloco', 'PyTorch DiLoCo v1'),
            ('single_worker_dilocov2', 'PyTorch DiLoCo v2')
        ])
    }
}

def resolve_selected_algorithms(algo_args):
    """
    Parses and validates requested algorithm flags.
    Supports single strings, list of strings, or comma-separated lists.
    If 'all' is requested, returns all available algorithm keys.
    """
    if isinstance(algo_args, str):
        algo_args = [algo_args]
        
    resolved = []
    for item in algo_args:
        for part in item.split(','):
            part_clean = part.strip().lower()
            if not part_clean:
                continue
            if part_clean == 'all':
                return list(ALGO_CONFIGS.keys())
            elif part_clean in ALGO_CONFIGS:
                if part_clean not in resolved:
                    resolved.append(part_clean)
            else:
                valid_keys = ", ".join(list(ALGO_CONFIGS.keys()) + ['all'])
                raise ValueError(f"Unknown algorithm '{part_clean}'. Available choices: {valid_keys}")
                
    if not resolved:
        return list(ALGO_CONFIGS.keys())
    return resolved

def get_selected_submissions(selected_algos):
    """
    Gathers all submission keys and their display names across selected algorithms.
    Returns an OrderedDict mapping: sub_key -> (opt_display, algo_key).
    Ensures every version and language of the selected algorithms is included.
    """
    submissions_map = OrderedDict()
    for algo in selected_algos:
        if algo not in ALGO_CONFIGS:
            continue
        for sub_key, opt_display in ALGO_CONFIGS[algo]['submissions'].items():
            submissions_map[sub_key] = (opt_display, algo)
    return submissions_map

def find_workloads(base_log_dir, selected_submissions_map):
    """
    Finds distinct workload names present in any of the selected submission folders.
    """
    workloads = set()
    for sub_key in selected_submissions_map.keys():
        pattern = os.path.join(base_log_dir, sub_key, 'study_*', '*')
        dirs = glob.glob(pattern)
        for d in dirs:
            if os.path.isdir(d):
                dirname = os.path.basename(d)
                base_name = dirname.replace('_pytorch', '').replace('_jax', '')
                workloads.add(base_name)
    return sorted(list(workloads))

def collect_step_times(base_log_dir, selected_submissions_map, workloads):
    """
    Scans trial measurements.csv files and computes step execution times (ms/step).
    Returns `results` dict: results[opt_display][workload] = list of trial step times.
    And `raw_table` dict: raw_table[opt_display][workload] = mean step time across trials (or None).
    """
    opt_displays = [disp for disp, _ in selected_submissions_map.values()]
    results = {disp: {wl: [] for wl in workloads} for disp in opt_displays}
    raw_table = {disp: {wl: None for wl in workloads} for disp in opt_displays}
    
    for sub_key, (opt_display, _) in selected_submissions_map.items():
        for wl in workloads:
            pattern = os.path.join(base_log_dir, sub_key, 'study_*', f"{wl}*", 'trial_*', 'measurements.csv')
            files = glob.glob(pattern)
            
            trial_times = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                    if 'accumulated_submission_time' in df.columns and 'global_step' in df.columns:
                        df_valid = df.dropna(subset=['accumulated_submission_time', 'global_step'])
                        if len(df_valid) >= 2:
                            df_valid = df_valid.sort_values('global_step')
                            first_row = df_valid.iloc[0]
                            last_row = df_valid.iloc[-1]
                            
                            delta_t = last_row['accumulated_submission_time'] - first_row['accumulated_submission_time']
                            delta_s = last_row['global_step'] - first_row['global_step']
                            
                            if delta_s > 0:
                                avg_step_time_ms = (delta_t / delta_s) * 1000.0
                                trial_times.append(avg_step_time_ms)
                except Exception as e:
                    print(f"Warning: error processing trial file {f}: {e}")
                    
            results[opt_display][wl] = trial_times
            if trial_times:
                raw_table[opt_display][wl] = np.mean(trial_times)
                
    return results, raw_table

def normalize_step_times(results, base_log_dir, workloads, submissions_map, normalize_choice='sfadamw_v2'):
    """
    Normalizes step execution times (trial times) relative to a baseline algorithm.
    - If normalize_choice == 'sfadamw_v2': divides JAX by 'JAX Schedule-Free v2' and PyTorch by 'PyTorch Schedule-Free v2'.
    - If normalize_choice == 'sfadamw_v1': divides JAX by 'JAX Schedule-Free v1' and PyTorch by 'PyTorch Schedule-Free v1'.
    - If normalize_choice == 'nadamw': divides by 'JAX NAdamW v1'.
    Returns (normalized_results, raw_normalized_table, caption).
    """
    if normalize_choice == 'none' or not normalize_choice:
        return results, None, "Step Execution Time Comparison (milliseconds per step) across different workloads."
        
    # Map normalize_choice to baseline submission keys
    if normalize_choice == 'sfadamw_v2':
        jax_base_key = 'schedule_free_adamw_jax_v2'
        pt_base_key = 'schedule_free_adamw_v2'
        caption = "Step Execution Time Comparison (Normalized Ratios relative to Schedule-Free AdamW v2) across different workloads."
    elif normalize_choice == 'sfadamw_v1':
        jax_base_key = 'schedule_free_adamw_jax'
        pt_base_key = 'schedule_free_adamw'
        caption = "Step Execution Time Comparison (Normalized Ratios relative to Schedule-Free AdamW v1) across different workloads."
    elif normalize_choice == 'nadamw':
        jax_base_key = 'nadamw'
        pt_base_key = 'nadamw'
        caption = "Step Execution Time Comparison (Normalized Ratios relative to JAX NAdamW v1) across different workloads."
    else:
        raise ValueError(f"Unknown normalization choice '{normalize_choice}'")

    base_keys = {'JAX': jax_base_key, 'PyTorch': pt_base_key}
    base_means = {'JAX': {}, 'PyTorch': {}}
    
    reverse_map = {sub_k: disp for sub_k, (disp, _) in submissions_map.items()}
    
    for fw, b_key in base_keys.items():
        if b_key in reverse_map and reverse_map[b_key] in results:
            disp = reverse_map[b_key]
            for wl in workloads:
                times = results[disp].get(wl, [])
                base_means[fw][wl] = np.mean(times) if times else None
        else:
            for wl in workloads:
                pattern = os.path.join(base_log_dir, b_key, 'study_*', f"{wl}*", 'trial_*', 'measurements.csv')
                files = glob.glob(pattern)
                trial_times = []
                for f in files:
                    try:
                        df = pd.read_csv(f)
                        if 'accumulated_submission_time' in df.columns and 'global_step' in df.columns:
                            df_valid = df.dropna(subset=['accumulated_submission_time', 'global_step'])
                            if len(df_valid) >= 2:
                                df_valid = df_valid.sort_values('global_step')
                                first_row = df_valid.iloc[0]
                                last_row = df_valid.iloc[-1]
                                delta_t = last_row['accumulated_submission_time'] - first_row['accumulated_submission_time']
                                delta_s = last_row['global_step'] - first_row['global_step']
                                if delta_s > 0:
                                    trial_times.append((delta_t / delta_s) * 1000.0)
                    except Exception:
                        pass
                base_means[fw][wl] = np.mean(trial_times) if trial_times else None

    normalized_results = {}
    raw_normalized_table = {}
    for opt_display, wl_dict in results.items():
        normalized_results[opt_display] = {}
        raw_normalized_table[opt_display] = {}
        is_jax = opt_display.startswith('JAX')
        fw = 'JAX' if is_jax else 'PyTorch'
        
        for wl in workloads:
            times = wl_dict.get(wl, [])
            base_m = base_means[fw].get(wl)
            if times and base_m and base_m > 0:
                norm_times = [t / base_m for t in times]
                normalized_results[opt_display][wl] = norm_times
                raw_normalized_table[opt_display][wl] = np.mean(norm_times)
            else:
                normalized_results[opt_display][wl] = []
                raw_normalized_table[opt_display][wl] = None
                
    return normalized_results, raw_normalized_table, caption

def format_table_values(results, workloads, is_normalized=False):
    """
    Formats numeric step times as 'mean ± std' (if std > 0.01 and multiple trials)
    or 'mean' or 'N/A'. Ratios use 2 decimal places, raw ms/step uses 1 decimal place.
    Returns formatted_table dict: formatted_table[opt_display][workload] = string.
    """
    formatted_table = {}
    for opt_display, wl_dict in results.items():
        formatted_table[opt_display] = {}
        for wl in workloads:
            times = wl_dict.get(wl, [])
            if times:
                mean_val = np.mean(times)
                std_val = np.std(times)
                if is_normalized:
                    if len(times) > 1 and std_val > 0.01:
                        formatted_table[opt_display][wl] = f"{mean_val:.2f} ± {std_val:.2f}"
                    else:
                        formatted_table[opt_display][wl] = f"{mean_val:.2f}"
                else:
                    if len(times) > 1 and std_val > 0.01:
                        formatted_table[opt_display][wl] = f"{mean_val:.1f} ± {std_val:.1f}"
                    else:
                        formatted_table[opt_display][wl] = f"{mean_val:.1f}"
            else:
                formatted_table[opt_display][wl] = "N/A"
    return formatted_table

def generate_markdown_table(formatted_table, workloads, opt_displays):
    """
    Constructs a Markdown table representing the step execution time comparison.
    """
    headers = ["Optimizer"] + workloads
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for opt in opt_displays:
        if opt in formatted_table:
            row_vals = [formatted_table[opt][wl] for wl in workloads]
            md_lines.append("| " + opt + " | " + " | ".join(row_vals) + " |")
    return "\n".join(md_lines)

def generate_latex_table(formatted_table, workloads, opt_displays,
                         caption="Step Execution Time Comparison (milliseconds per step) across different workloads.",
                         label="tab:step_time_comparison"):
    """
    Constructs a publication-ready LaTeX table representing the step execution time comparison.
    """
    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{'l' + 'r' * len(workloads)}}}",
        "\\toprule"
    ]
    
    escaped_workloads = [wl.replace('_', '\\_') for wl in workloads]
    latex_lines.append("Optimizer & " + " & ".join(escaped_workloads) + " \\\\")
    latex_lines.append("\\midrule")
    
    for opt in opt_displays:
        if opt not in formatted_table:
            continue
        row_vals = []
        for wl in workloads:
            val = formatted_table[opt][wl]
            if "±" in val:
                parts = val.split(" ± ")
                row_vals.append(f"${parts[0]} \\pm {parts[1]}$")
            elif val == "N/A":
                row_vals.append("---")
            else:
                row_vals.append(f"${val}$")
        escaped_opt = opt.replace('_', '\\_')
        latex_lines.append(f"{escaped_opt} & " + " & ".join(row_vals) + " \\\\")
        
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    return "\n".join(latex_lines)

def generate_tables(algo_args='all', base_log_dir='~/submissions_algorithms/logs/self_tuning', save_dir=None, normalize='sfadamw_v2'):
    """
    High-level orchestrator function exposed for both CLI and Jupyter/Colab notebooks.
    Takes algorithm flags/names, loads step times across all versions/languages, normalizes them (if requested),
    and generates both Markdown and LaTeX tables.
    """
    base_log_path = Path(base_log_dir).expanduser()
    selected_algos = resolve_selected_algorithms(algo_args)
    submissions_map = get_selected_submissions(selected_algos)
    
    workloads = find_workloads(base_log_path, submissions_map)
    results, raw_table = collect_step_times(base_log_path, submissions_map, workloads)
    
    is_normalized = (normalize != 'none' and normalize is not False and normalize is not None)
    if is_normalized:
        active_results, active_raw, caption = normalize_step_times(
            results, base_log_path, workloads, submissions_map, normalize_choice=normalize
        )
    else:
        active_results, active_raw = results, raw_table
        caption = "Step Execution Time Comparison (milliseconds per step) across different workloads."
        
    formatted_table = format_table_values(active_results, workloads, is_normalized=is_normalized)
    
    opt_displays = [disp for disp, _ in submissions_map.values()]
    markdown_table = generate_markdown_table(formatted_table, workloads, opt_displays)
    latex_table = generate_latex_table(formatted_table, workloads, opt_displays, caption=caption)
    
    if save_dir:
        save_path = Path(save_dir).expanduser()
        save_path.mkdir(exist_ok=True, parents=True)
        md_file = save_path / 'step_time_comparison.md'
        tex_file = save_path / 'step_time_comparison.tex'
        
        title_text = f"# {caption}\n\n"
        with open(md_file, 'w') as f:
            f.write(title_text)
            f.write(markdown_table)
            f.write("\n\n## LaTeX Source Code\n\n```latex\n")
            f.write(latex_table)
            f.write("\n```\n")
        print(f"\nSaved tables to {md_file} and {tex_file}")
        
    return {
        'markdown_table': markdown_table,
        'latex_table': latex_table,
        'formatted_table': formatted_table,
        'raw_table': active_raw,
        'raw_ms_table': raw_table,
        'workloads': workloads,
        'opt_displays': opt_displays,
        'results': active_results,
        'raw_ms_results': results,
        'selected_algos': selected_algos,
        'normalization': normalize
    }

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate publication-grade step execution time comparison tables across algorithms and workloads."
    )
    parser.add_argument(
        '--algo',
        type=str,
        nargs='+',
        default=['all'],
        help="Algorithm(s) to include in the table. Options: 'all' or one/more of: " + ", ".join(ALGO_CONFIGS.keys())
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='~/submissions_algorithms/logs/self_tuning',
        help="Path to the self-tuning log directory."
    )
    parser.add_argument(
        '--normalize',
        type=str,
        choices=['sfadamw_v2', 'sfadamw_v1', 'nadamw', 'none'],
        default='sfadamw_v2',
        help="Normalize step execution times relative to a baseline ('sfadamw_v2' by default, or 'none' for raw ms/step)."
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help="Optional directory to save generated markdown and LaTeX table files."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(f"Generating step size table for algorithms: {args.algo} (normalization: {args.normalize}) ...")
    output = generate_tables(
        algo_args=args.algo,
        base_log_dir=args.log_dir,
        save_dir=args.save_dir,
        normalize=args.normalize
    )
    
    print("\n=================== MARKDOWN TABLE ===================")
    print(output['markdown_table'])
    print("\n==================== LATEX TABLE =====================")
    print(output['latex_table'])

if __name__ == "__main__":
    main()

