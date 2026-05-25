import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# file path
path = Path("P01") / "IK" / "Walking01.mot"

# read .mot skipping header lines until `endheader`
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# find where data header starts (line after "endheader")
end_idx = next(i for i, L in enumerate(lines) if L.strip().lower() == "endheader")
header_line = lines[end_idx + 1].strip().split()
data_lines = lines[end_idx + 2 :]

# parse with pandas
colnames = header_line
df = pd.read_csv(
    pd.compat.StringIO(" ".join(" ".join(l.split()) for l in data_lines)),
    delim_whitespace=True,
    names=colnames,
    header=None,
)

# or simpler via pandas read_table with skiprows:
# df = pd.read_table(path, delim_whitespace=True, skiprows=end_idx+1)

time = df["time"]
hip_r   = df["hip_flexion_r"]
knee_r  = df["knee_angle_r"]
ankle_r = df["ankle_angle_r"]

plt.figure(figsize=(10, 5))
plt.plot(time, hip_r, label="hip flexion r", linewidth=1)
plt.plot(time, knee_r, label="knee angle r", linewidth=1)
plt.plot(time, ankle_r, label="ankle angle r", linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.title("P01 Walking01 IK joint angles (right leg)")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()