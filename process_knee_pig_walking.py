#!/usr/bin/env python3
"""process_knee_pig_walking.py

Load Plug-in Gait knee angle CSVs from healthy walking trials, time-normalise
each trial to 101 gait cycle points, compute mean and standard deviation,
plot mean +/- SD, extract peak values, and compute a symmetry index bar chart.

Change the folder settings and column names below to match CSV format.
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

# Column names for knee angles in each plane.
COLUMN_NAMES = {
    'Right': ['RKneeAngles_X', 'RKneeAngles_Y', 'RKneeAngles_Z'],
    'Left': ['LKneeAngles_X', 'LKneeAngles_Y', 'LKneeAngles_Z'],
}

PARAMETER_NAMES = [
    'Knee Flexion Angle',          # Sagittal plane / X
    'Knee Valgus Angle',           # Frontal plane / Y
    'Knee Internal Rotation Angle' # Transverse plane / Z
]
PLANE_LABELS = ['Sagittal', 'Frontal', 'Transverse']
GAIT_CYCLE_POINTS = 101
COLORS = {'Right': '#3377CC', 'Left': '#CC3333'}

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
    unit_row = header_rows[4]

    # Forward-fill group names so that repeated X/Y/Z columns inherit the correct group.
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
            f'Header length mismatch in {csv_path}: {len(flat_headers)} headers vs {data.shape[1]} columns '
            'in data rows.'
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

    # Fall back to substring matching for headers like Maciej:RKneeAngles_X.
    for col in df.columns:
        simple = col.lower().replace(' ', '').replace(':', '')
        if key in simple or simple in key:
            return df[col].to_numpy(dtype=float)

    raise KeyError(f'Column not found: {column_name}')


def time_normalize_signal(time_vec, signal, n_points=GAIT_CYCLE_POINTS):
    time_vec = np.asarray(time_vec, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_vec) < 2:
        raise ValueError('Time vector must contain at least two points for interpolation.')

    sorted_idx = np.argsort(time_vec)
    time_sorted = time_vec[sorted_idx]
    signal_sorted = signal[sorted_idx]
    unique_time, unique_idx = np.unique(time_sorted, return_index=True)
    if len(unique_time) < 2:
        raise ValueError('Time vector must contain at least two unique samples.')
    signal_unique = signal_sorted[unique_idx]

    valid = np.isfinite(signal_unique)
    if not np.any(valid):
        raise ValueError('Signal contains no finite values for interpolation.')

    valid_time = unique_time[valid]
    valid_signal = signal_unique[valid]
    if valid_time.size < 2:
        # Constant signal: fill the whole gait cycle with the single value.
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
    """Extract Initial Contact (IC) and Toe-Off (TO) frame indices from foot progress angle.
    
    Returns:
        (ic_frame_idx, to_frame_idx) as frame indices in the original data, or (None, None) if detection fails.
    """
    # Try to find foot progress angle Z (right preferred, fall back to left)
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
    
    # Find IC: absolute minimum in foot progress angle trajectory (heel strike)
    valid_idx = np.where(np.isfinite(foot_angle))[0]
    if len(valid_idx) == 0:
        return None, None
    
    ic_frame_idx = valid_idx[np.argmin(foot_angle[valid_idx])]
    
    # Find TO: first local maximum after IC (toe-off)
    foot_angle_after_ic = foot_angle[ic_frame_idx:]
    local_maxima = find_local_maxima(foot_angle_after_ic)
    
    if local_maxima:
        to_frame_idx = ic_frame_idx + local_maxima[0]
    else:
        return ic_frame_idx, None
    
    return ic_frame_idx, to_frame_idx


def plot_mean_sd(parameter_index, gait_cycle, trials_by_side, means_by_side, stds_by_side, accepted_trials=None):
    parameter_name = PARAMETER_NAMES[parameter_index]
    plane_label = PLANE_LABELS[parameter_index]
    fig, ax = plt.subplots(figsize=(10, 6))

    for side in ['Right', 'Left']:
        trials = trials_by_side[side][parameter_index]
        color = COLORS[side]
        for trial in trials:
            ax.plot(gait_cycle, trial, color=color, alpha=0.12, linewidth=1)

        mean_curve = means_by_side[side][parameter_index]
        std_curve = stds_by_side[side][parameter_index]
        ax.fill_between(
            gait_cycle,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.25,
            edgecolor='none',
            label=f'{side} ± SD',
        )
        ax.plot(gait_cycle, mean_curve, color=color, linewidth=2.5, label=f'{side} mean')

    # Add gait event markers for Knee Flexion plot with accepted trials
    if parameter_index == 0 and accepted_trials and len(accepted_trials) > 0:
        right_trials = trials_by_side['Right'][parameter_index]
        
        for trial_idx in range(min(3, len(accepted_trials))):  # Mark first 3 trials
            csv_path = accepted_trials[trial_idx]
            
            # Get the trial signal for this accepted trial
            if trial_idx < len(right_trials):
                trial_signal = right_trials[trial_idx]
                color = COLORS['Right']
                
                try:
                    df = read_csv_file(csv_path)
                    time_vec = get_time_vector(df)
                    ic_frame, to_frame = extract_gait_events(df, time_vec)
                    
                    if ic_frame is not None:
                        # Convert frame index to gait cycle percentage
                        total_frames = len(df)
                        ic_percent = (ic_frame / total_frames) * 100
                        ic_flexion = np.interp(ic_percent, gait_cycle, trial_signal)
                    
                    if to_frame is not None:
                        # Convert frame index to gait cycle percentage
                        total_frames = len(df)
                        to_percent = (to_frame / total_frames) * 100
                        to_flexion = np.interp(to_percent, gait_cycle, trial_signal)
                 
                except Exception as exc:
                    print(f'Note: Could not add gait event markers for {csv_path.name}: {exc}')

    #ax.set_title(f'{parameter_name} ({plane_label} plane)')
    ax.set_xlabel('Gait Cycle (%)')
    ax.set_ylabel('Angle (deg)')
    ax.set_xlim(0, 100)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.4)

    filename = OUTPUT_FOLDER / f'Knee_{parameter_name.replace(" ", "_")}_MeanSD.png'
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f'Saved {filename}')


def plot_symmetry_index(symmetry_index):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(PARAMETER_NAMES)), symmetry_index, color='#4C72B0', alpha=0.85)
    ax.set_xticks(range(len(PARAMETER_NAMES)))
    ax.set_xticklabels(PARAMETER_NAMES, rotation=15, ha='right')
    ax.set_ylabel('Symmetry Index (%)')
    #ax.set_title('Peak Symmetry Index for Knee Angles (Right vs Left)')
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', linestyle=':', alpha=0.4)

    for bar, value in zip(bars, symmetry_index):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + np.sign(value) * 0.5,
            f'{value:.1f}%',
            ha='center',
            va='bottom' if value >= 0 else 'top',
            fontsize=10,
        )

    filename = OUTPUT_FOLDER / 'Knee_SymmetryIndex.png'
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f'Saved {filename}')


def main():
    csv_files = find_walking_csv_files()
    print(f'Found {len(csv_files)} walking trial files:')
    for path in csv_files:
        print(f'  {path}')

    trials_by_side = {'Right': [[] for _ in PARAMETER_NAMES], 'Left': [[] for _ in PARAMETER_NAMES]}
    peaks_by_side = {'Right': [[] for _ in PARAMETER_NAMES], 'Left': [[] for _ in PARAMETER_NAMES]}
    excluded_trials = []
    accepted_trials = []  # Track which CSV files were accepted

    for csv_path in csv_files:
        df = read_csv_file(csv_path)
        time_vec = get_time_vector(df)

        # Extract and time-normalize all three parameters for this trial
        trial_norm = {'Right': [], 'Left': []}
        
        for side in ['Right', 'Left']:
            for column_name in COLUMN_NAMES[side]:
                signal = extract_signal(df, column_name)
                # Time-normalize the entire file to 101 points
                norm_signal = time_normalize_signal(time_vec, signal)
                trial_norm[side].append(norm_signal)

        # Check valgus outlier: if either side has Valgus angle > 20 degrees, exclude this trial
        right_valgus = trial_norm['Right'][1]
        left_valgus = trial_norm['Left'][1]
   

        # Add to the collective trials
        accepted_trials.append(csv_path)  # Track this accepted trial
        for side in ['Right', 'Left']:
            for i, norm_signal in enumerate(trial_norm[side]):
                trials_by_side[side][i].append(norm_signal)
                peaks_by_side[side][i].append(np.max(np.abs(norm_signal)))

    if not any(trials_by_side['Right']):
        raise RuntimeError('No valid trials remain after outlier rejection.')

    gait_cycle = np.linspace(0, 100, GAIT_CYCLE_POINTS)
    means_by_side = {'Right': [], 'Left': []}
    stds_by_side = {'Right': [], 'Left': []}

    for side in ['Right', 'Left']:
        for i in range(len(PARAMETER_NAMES)):
            side_trials = np.vstack(trials_by_side[side][i])
            means_by_side[side].append(np.mean(side_trials, axis=0))
            stds_by_side[side].append(np.std(side_trials, axis=0, ddof=0))
    
    for i in range(len(PARAMETER_NAMES)):
        plot_mean_sd(i, gait_cycle, trials_by_side, means_by_side, stds_by_side, accepted_trials)

    # peak_mean_right = np.array([np.mean(peaks_by_side['Right'][i]) for i in range(len(PARAMETER_NAMES))])
    # peak_mean_left = np.array([np.mean(peaks_by_side['Left'][i]) for i in range(len(PARAMETER_NAMES))])
    # symmetry_index = ((peak_mean_right - peak_mean_left) / (0.5 * (peak_mean_right + peak_mean_left))) * 100

    # print('\nPeak values (mean absolute peak across trials):')
    # for i, param in enumerate(PARAMETER_NAMES):
    #     print(f'  {param}: Right = {peak_mean_right[i]:.2f} deg, Left = {peak_mean_left[i]:.2f} deg, SI = {symmetry_index[i]:.2f}%')
    peak_mean_right = np.array([np.mean(peaks_by_side['Right'][i]) for i in range(len(PARAMETER_NAMES))])
    peak_mean_left = np.array([np.mean(peaks_by_side['Left'][i]) for i in range(len(PARAMETER_NAMES))])
    
    # Calculate Standard Deviation for the peaks
    peak_std_right = np.array([np.std(peaks_by_side['Right'][i], ddof=0) for i in range(len(PARAMETER_NAMES))])
    peak_std_left = np.array([np.std(peaks_by_side['Left'][i], ddof=0) for i in range(len(PARAMETER_NAMES))])
    
    symmetry_index = ((peak_mean_right - peak_mean_left) / (0.5 * (peak_mean_right + peak_mean_left))) * 100

    print('\nPeak values (Mean ± SD absolute peak across trials):')
    for i, param in enumerate(PARAMETER_NAMES):
        print(f'  {param}: Right = {peak_mean_right[i]:.2f} ± {peak_std_right[i]:.2f} deg, '
              f'Left = {peak_mean_left[i]:.2f} ± {peak_std_left[i]:.2f} deg, '
              f'SI = {symmetry_index[i]:.2f}%')
        
    plot_symmetry_index(symmetry_index)


if __name__ == '__main__':
    main()
