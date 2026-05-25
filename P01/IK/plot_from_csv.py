import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot time-series data from CSV produced from .mot")
    parser.add_argument("csv", type=Path, help="Input CSV file")
    parser.add_argument("--columns", nargs="+", default=["hip_flexion_r", "knee_angle_r", "ankle_angle_r"],
                        help="Column names to plot (default right hip/knee/ankle)")
    parser.add_argument("--time", default="time", help="Time column name (default 'time')")
    parser.add_argument("--outfile", type=Path, default=None, help="Save figure to file (e.g., out.png)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI for saved image")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.time not in df.columns:
        raise ValueError(f"Time column '{args.time}' not found in {args.csv}")

    t = df[args.time]

    plt.figure(figsize=(10, 5))
    for col in args.columns:
        if col not in df.columns:
            raise ValueError(f"Column not found: {col}")
        plt.plot(t, df[col], label=col, linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title(f"{args.csv.name} : {', '.join(args.columns)}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    if args.outfile:
        plt.savefig(args.outfile, dpi=args.dpi)
        print(f"Figure saved: {args.outfile}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
