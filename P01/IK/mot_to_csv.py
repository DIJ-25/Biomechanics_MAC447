import argparse
from pathlib import Path
import io

import pandas as pd


def read_mot(mot_path: Path) -> pd.DataFrame:
    with mot_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # find endheader index
    end_idx = next((i for i, line in enumerate(lines) if line.strip().lower() == "endheader"), None)
    if end_idx is None:
        raise ValueError("Could not find 'endheader' in .mot file")

    # header is the line after endheader
    header_line = lines[end_idx + 1].strip()
    colnames = header_line.split()

    # data lines are after header
    data_lines = lines[end_idx + 2 :]
    # join lines and read with pandas
    raw_data = "\n".join(" ".join(l.strip().split()) for l in data_lines if l.strip())
    df = pd.read_csv(io.StringIO(raw_data), delim_whitespace=True, names=colnames, header=None)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenSim .mot motion file to CSV")
    parser.add_argument("--mot", type=Path, help="Input .mot file")
    parser.add_argument("--csv", type=Path, nargs="?", help="Output .csv file (optional)")
    args = parser.parse_args()

    mot_path = args.mot
    if not mot_path.exists():
        raise FileNotFoundError(f"Input file not found: {mot_path}")

    df = read_mot(mot_path)

    out_path = args.csv if args.csv else mot_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
