
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Set up the project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "detect_dataset.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load the original data for the plots
df = pd.read_csv(DATA_FILE)

# Remove the empty columns and rename the fault label
df = df.dropna(axis=1, how="all")
df = df.rename(columns={"Output (S)": "fault"})

# Plot 1: Three phase currents for the first 2000 rows
first_rows = df.iloc[:2000]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(first_rows.index, first_rows["Ia"], label="Ia")
ax.plot(first_rows.index, first_rows["Ib"], label="Ib")
ax.plot(first_rows.index, first_rows["Ic"], label="Ic")

# Shade the areas where a fault occurs
fault_rows = first_rows["fault"].to_numpy() == 1
ax.fill_between(
    first_rows.index,
    0,
    1,
    where=fault_rows,
    transform=ax.get_xaxis_transform(),
    alpha=0.2,
    color="red",
    label="Fault"
)

ax.set_title("Three-Phase Currents for the First 2000 Rows")
ax.set_xlabel("Row Number")
ax.set_ylabel("Current (A)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_1_currents.png", dpi=300)
plt.close()

# Plot 2: Ia against Ib, coloured by fault label
fig, ax = plt.subplots(figsize=(8, 6))

for label, name in [(0, "No Fault"), (1, "Fault")]:
    class_rows = df[df["fault"] == label]
    ax.scatter(
        class_rows["Ia"],
        class_rows["Ib"],
        s=12,
        alpha=0.6,
        label=name
    )

ax.set_title("Phase Current Ia vs Ib")
ax.set_xlabel("Ia (A)")
ax.set_ylabel("Ib (A)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_2_current_scatter.png", dpi=300)
plt.close()

# Plot 3: Compare Ia distributions for fault and no-fault rows
fig, ax = plt.subplots(figsize=(9, 5))

no_fault_current = df.loc[df["fault"] == 0, "Ia"]
fault_current = df.loc[df["fault"] == 1, "Ia"]

ax.hist(no_fault_current, bins=40, alpha=0.6, label="No Fault")
ax.hist(fault_current, bins=40, alpha=0.6, label="Fault")

ax.set_title("Distribution of Ia Current")
ax.set_xlabel("Ia (A)")
ax.set_ylabel("Number of Rows")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_3_current_histogram.png", dpi=300)
plt.close()

print("EDA plots created successfully.")
print("Saved in:", RESULTS_DIR)
print("Plot 1: eda_1_currents.png")
print("Plot 2: eda_2_current_scatter.png")
print("Plot 3: eda_3_current_histogram.png")
