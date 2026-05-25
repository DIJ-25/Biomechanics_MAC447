#!/usr/bin/env python3
"""plot_knee_flexion_figure_b4.py

Generate Figure B4-style plot: Individual Knee Flexion trials with gait event markers.
Each trial plotted as a distinct solid line with IC (circle) and TO (cross) markers.
Legend displays trial filenames outside the plot box.
"""

from pathlib import Path
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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

GAIT_CYCLE_POINTS = 101

# Color palette for individual trials
TRIAL_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'
]

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


def find_local_maxima(signal):
    """Find indices of local maxima in a signal."""
    maxima = []
    signal = np.asarray(signal)
    for i in range(1, len(signal) - 1):
        if np.isfinite(signal[i]) and np.isfinite(signal[i-1]) and np.isfinite(signal[i+1]):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                maxima.append(i)
    return maxima


def extract_gait_events(df, time_vec):
    """Extract IC and TO frame indices from foot progress angle."""
    foot_angle = None
    for col_pattern in ['RFootProgressAngles_Z', 'RFootProgressAngles', 'LFootProgressAngles_Z', 'LFootProgressAngles']:
        try:
            foot_angle = extract_signal(df, col_pattern)
            if np.any(np.isfinite(foot_angle)):
                break
        except KeyError:
            continue
    
    if foot_angle is None or not np.any(np.isfinite(foot_angle)):
        return None, None
    
    valid_idx = np.where(np.isfinite(foot_angle))[0]
    if len(valid_idx) == 0:
        return None, None
    
    ic_frame_idx = valid_idx[np.argmin(foot_angle[valid_idx])]
    foot_angle_after_ic = foot_angle[ic_frame_idx:]
    local_maxima = find_local_maxima(foot_angle_after_ic)
    
    if local_maxima:
        to_frame_idx = ic_frame_idx + local_maxima[0]
    else:
        return ic_frame_idx, None
    
    return ic_frame_idx, to_frame_idx


def plot_individual_trials_with_events(csv_files, accepted_trials):
    """Plot individual Knee Flexion trials with gait event markers (Figure B4 style)."""
    gait_cycle = np.linspace(0, 100, GAIT_CYCLE_POINTS)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    trial_handles = []
    trial_labels = []
    
    for trial_idx, csv_path in enumerate(accepted_trials):
        color = TRIAL_COLORS[trial_idx % len(TRIAL_COLORS)]
        trial_name = csv_path.stem
        
        try:
            df = read_csv_file(csv_path)
            time_vec = get_time_vector(df)
            
            # Extract and normalize Right Knee Flexion (X component)
            flexion_signal = extract_signal(df, COLUMN_NAMES['Right'][0])
            norm_flexion = time_normalize_signal(time_vec, flexion_signal)
            
            # Plot trial line
            line = ax.plot(gait_cycle, norm_flexion, color=color, linewidth=2, label=trial_name)
            trial_handles.append(line[0])
            trial_labels.append(trial_name)
            
            # Add gait event markers
            ic_frame, to_frame = extract_gait_events(df, time_vec)
            
            if ic_frame is not None:
                total_frames = len(df)
                ic_percent = (ic_frame / total_frames) * 100
                ic_flexion = np.interp(ic_percent, gait_cycle, norm_flexion)
                ax.plot(ic_percent, ic_flexion, 'o', markersize=8, color=color,
                       markerfacecolor='white', markeredgewidth=1.5, zorder=10)
            
            if to_frame is not None:
                total_frames = len(df)
                to_percent = (to_frame / total_frames) * 100
                to_flexion = np.interp(to_percent, gait_cycle, norm_flexion)
                ax.plot(to_percent, to_flexion, 'x', markersize=10, color=color,
                       markeredgewidth=1.5, zorder=10)
        
        except Exception as exc:
            print(f'Warning: Could not process {csv_path.name}: {exc}')
    
    ax.set_xlabel('Gait Cycle (%)', fontsize=12)
    ax.set_ylabel('Flexion Angle (deg)', fontsize=12)
    #ax.set_title('Knee Flexion Angle - Individual Trials with Gait Events', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle=':', alpha=0.4)
    
    ic_marker = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                              markersize=8, markerfacecolor='white', markeredgewidth=1.5)
    to_marker = mlines.Line2D([], [], color='gray', marker='x', linestyle='None',
                              markersize=10, markeredgewidth=1.5)
    
    # Append custom markers
    trial_handles.extend([ic_marker, to_marker])
    trial_labels.extend(['Initial Contact (IC)', 'Toe-Off (TO)'])
    # ----------------------------------

    # Legend outside plot box on the right
    ax.legend(handles=trial_handles, labels=trial_labels, loc='center left', 
             bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=10)
    
    filename = OUTPUT_FOLDER / 'Knee_Flexion_Individual_Trials_B4.png'
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {filename}')


def main():
    csv_files = find_walking_csv_files()
    print(f'Found {len(csv_files)} walking trial files:')
    for path in csv_files:
        print(f'  {path}')

    accepted_trials = []
    excluded_trials = []

    for csv_path in csv_files:
        df = read_csv_file(csv_path)
        time_vec = get_time_vector(df)

        # Extract and normalize all three parameters
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

        accepted_trials.append(csv_path)
        print(f'Trial {csv_path.name} accepted.')

    if not accepted_trials:
        raise RuntimeError('No valid trials remain after outlier rejection.')

    print(f'\nAccepted {len(accepted_trials)} trials. Generating Figure B4-style plot...')
    plot_individual_trials_with_events(csv_files, accepted_trials)


if __name__ == '__main__':
    main()
