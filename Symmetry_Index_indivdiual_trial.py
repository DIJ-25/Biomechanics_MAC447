#!/usr/bin/env python3
"""plot_symmetry_index_figure_b5.py

Generate Individual trial Symmetry Index as 3 subplots.
Each subplot shows SI for one parameter (Flexion, Valgus, Internal Rotation) across all trials.
Bars colored: Blue (Flexion), Green (Valgus), Purple (Internal Rotation).
"""

from pathlib import Path
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- USER SETTINGS ----------
ROOT_FOLDER = Path.cwd()
DATA_ROOT = ROOT_FOLDER / 'Part_B_Data'
PARTICIPANT_FOLDERS = ['Participant_1', 'Participant_2']
TRIAL_INCLUDE_PATTERN = re.compile(r'walking', re.IGNORECASE)
TRIAL_EXCLUDE_PATTERN = re.compile(r'sitting', re.IGNORECASE)
OUTPUT_FOLDER = ROOT_FOLDER / 'results'
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Column names for knee angles
COLUMN_NAMES = {
    'Right': ['RKneeAngles_X', 'RKneeAngles_Y', 'RKneeAngles_Z'],
    'Left': ['LKneeAngles_X', 'LKneeAngles_Y', 'LKneeAngles_Z'],
}

PARAMETER_NAMES = [
    'Knee Flexion Angle',
    'Knee Valgus Angle',
    'Knee Internal Rotation Angle'
]

GAIT_CYCLE_POINTS = 101

# Colors for subplots
SUBPLOT_COLORS = ['#3377CC', '#00AA55', '#9933CC']  # Blue, Green, Purple

# -----------------------------------

def find_walking_csv_files():
    files = []
    for participant in PARTICIPANT_FOLDERS:
        participant_dir = DATA_ROOT / participant
        if not participant_dir.exists():
            print(f'Warning: participant folder not found: {participant_dir}')
            continue
        for csv_path in sorted(participant_dir.glob('*.csv')):
            name = csv_path.name
            if TRIAL_INCLUDE_PATTERN.search(name) and not TRIAL_EXCLUDE_PATTERN.search(name):
                files.append(csv_path)
    if not files:
        raise FileNotFoundError('No walking trial CSV files found. Check DATA_ROOT and patterns.')
    return files


def read_csv_file(csv_path):
    """Read a CSV with the OpenSim/Plug-in-Gait multi-row header format."""
    with open(csv_path, 'r', encoding='utf-8', errors='replace', newline='') as fh:
        reader = csv.reader(fh)
        header_rows = [next(reader) for _ in range(5)]

    group_row = header_rows[2]
    name_row = header_rows[3]

    # Forward-fill group names for X/Y/Z columns
    filled_groups = []
    current_group = ''
    for item in group_row:
        if item and item.strip():
            current_group = item.strip()
        filled_groups.append(current_group)

    flat_headers = []
    for group, name in zip(filled_groups, name_row):
        key = name.strip() if name else group
        if group and name and group != name:
            flat_headers.append(f'{group}_{name.strip()}')
        else:
            flat_headers.append(key)

    try:
        data = pd.read_csv(csv_path, skiprows=5, header=None)
    except Exception as exc:
        raise IOError(f'Could not read CSV file: {csv_path}') from exc

    if data.shape[1] != len(flat_headers):
        raise ValueError(
            f'Header length mismatch in {csv_path}: {len(flat_headers)} headers vs {data.shape[1]} columns.'
        )

    data.columns = flat_headers
    return data


def get_time_vector(df):
    candidates = [col for col in df.columns if 'time' in col.lower()]
    if not candidates:
        return np.arange(len(df), dtype=float)
    return df[candidates[0]].to_numpy(dtype=float)


def extract_signal(df, column_name):
    if column_name in df.columns:
        return df[column_name].to_numpy(dtype=float)

    normalized = {col.lower().replace(' ', '').replace(':', ''): col for col in df.columns}
    key = column_name.lower().replace(' ', '').replace(':', '')
    if key in normalized:
        return df[normalized[key]].to_numpy(dtype=float)

    for col in df.columns:
        simple = col.lower().replace(' ', '').replace(':', '')
        if key in simple or simple in key:
            return df[col].to_numpy(dtype=float)

    raise KeyError(f'Column not found: {column_name}')


def time_normalize_signal(time_vec, signal, n_points=GAIT_CYCLE_POINTS):
    time_vec = np.asarray(time_vec, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_vec) < 2:
        raise ValueError('Time vector must contain at least two points.')

    sorted_idx = np.argsort(time_vec)
    time_sorted = time_vec[sorted_idx]
    signal_sorted = signal[sorted_idx]
    unique_time, unique_idx = np.unique(time_sorted, return_index=True)
    if len(unique_time) < 2:
        raise ValueError('Time vector must contain at least two unique samples.')
    signal_unique = signal_sorted[unique_idx]

    valid = np.isfinite(signal_unique)
    if not np.any(valid):
        raise ValueError('Signal contains no finite values.')

    valid_time = unique_time[valid]
    valid_signal = signal_unique[valid]
    if valid_time.size < 2:
        return np.full(n_points, valid_signal[0], dtype=float)

    target_time = np.linspace(valid_time[0], valid_time[-1], n_points)
    norm_signal = np.interp(target_time, valid_time, valid_signal,
                            left=valid_signal[0], right=valid_signal[-1])
    return norm_signal


def plot_symmetry_index_by_trial(trial_si_data):
    """Plot SI for each trial across 3 subplots (Flexion, Valgus, Internal Rotation)."""
    trial_names = [path.stem for path, _ in trial_si_data]
    all_si_values = [si_value for _, si_tuple in trial_si_data for si_value in si_tuple]
    y_min = min(all_si_values)
    y_max = max(all_si_values)
    y_range = y_max - y_min
    margin = max(5.0, y_range * 0.1)
    y_limits = (y_min - margin, y_max + margin)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for param_idx in range(3):
        ax = axes[param_idx]
        color = SUBPLOT_COLORS[param_idx]
        
        # Extract SI values for this parameter across all trials
        si_values = [si_tuple[param_idx] for _, si_tuple in trial_si_data]
        
        # Create bar chart
        bars = ax.bar(range(len(trial_names)), si_values, color=color, alpha=0.75, edgecolor='black', linewidth=0.5)
        
        # Add horizontal dashed line at y=0
        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.7)
        
        # Set labels and title
        if param_idx == 0:
            ax.set_ylabel('SI (%)', fontsize=11)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.set_title(PARAMETER_NAMES[param_idx], fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(trial_names)))
        ax.set_xticklabels(trial_names, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        ax.set_ylim(y_limits)
        
        # Add value labels on bars
        for bar, value in zip(bars, si_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + np.sign(height) * 1,
                   f'{value:.1f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    #fig.suptitle('Symmetry Index by Trial (Figure B5 Style)', fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    filename = OUTPUT_FOLDER / 'Symmetry_Index_by_Trial_B5.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {filename}')


def main():
    csv_files = find_walking_csv_files()
    print(f'Found {len(csv_files)} walking trial files:')
    for path in csv_files:
        print(f'  {path}')

    trial_si_data = []  # List of (csv_path, (si_flexion, si_valgus, si_rotation))
    excluded_trials = []

    for csv_path in csv_files:
        df = read_csv_file(csv_path)
        time_vec = get_time_vector(df)

        # Extract and normalize all three parameters for both sides
        trial_norm = {'Right': [], 'Left': []}
        for side in ['Right', 'Left']:
            for column_name in COLUMN_NAMES[side]:
                signal = extract_signal(df, column_name)
                norm_signal = time_normalize_signal(time_vec, signal)
                trial_norm[side].append(norm_signal)

        # Check valgus outlier: if either side has Valgus angle > 20 degrees, exclude
        right_valgus = trial_norm['Right'][1]
        left_valgus = trial_norm['Left'][1]
        # if np.max(np.abs(right_valgus)) > 20 or np.max(np.abs(left_valgus)) > 20:
        #     print(f'Trial {csv_path.name} excluded due to Valgus outlier.')
        #     excluded_trials.append(csv_path.name)
        #     continue

        # Calculate peak values for each parameter
        si_values = []
        for param_idx in range(3):
            right_peak = np.max(np.abs(trial_norm['Right'][param_idx]))
            left_peak = np.max(np.abs(trial_norm['Left'][param_idx]))
            
            # Calculate SI for this trial and parameter
            si = ((right_peak - left_peak) / (0.5 * (right_peak + left_peak))) * 100
            si_values.append(si)
        
        trial_si_data.append((csv_path, tuple(si_values)))
        print(f'Trial {csv_path.name} accepted. SI: Flexion={si_values[0]:.1f}%, Valgus={si_values[1]:.1f}%, Rotation={si_values[2]:.1f}%')

    if not trial_si_data:
        raise RuntimeError('No valid trials remain after outlier rejection.')

    print(f'\nAccepted {len(trial_si_data)} trials. Generating Figure B5-style plot...')
    plot_symmetry_index_by_trial(trial_si_data)


if __name__ == '__main__':
    main()
