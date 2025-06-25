# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 22:24:45 2025

@author: mjmde
"""

#######################################################################
#######################################################################
# NaSch Model
#######################################################################
#######################################################################


# -------------------------------------------------------------------------
# NaSch SIMULATION
# -------------------------------------------------------------------------

import random
import csv

# Parameters
N = 1000            # Number of cells on the circular track
time_steps = 1000000
v_max = 4           # Max velocity (cells per time step)
p_brake = 0.10      # Random slowdown probability
M_values = [50, 100, 170, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
random.seed(55)  


for M in M_values:
    output_path = f"D:/Master Thesis/Thesis Car Simulation Code/NaSch Data/Data_NaSch_{M}.csv"

    # Initialize positions: evenly spaced
    initial_positions = [(i * N) // M + 1 for i in range(M)]
    
    # Each car is a tuple: (car_id, position)
    cars = [(f"C{i+1}", pos) for i, pos in enumerate(initial_positions)]

    # Initialize velocities: start at v_max
    velocities = {car_id: v_max for car_id, _ in cars}

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time Step'] + [f"Car {i+1}" for i in range(M)])
        writer.writerow([0] + [pos for _, pos in cars])

        for t in range(1, time_steps + 1):
            # Sort cars by position for consistent update order
            cars = sorted(cars, key=lambda x: x[1])
            new_positions = []
            new_velocities = {}

            for i in range(M):
                car_id, position = cars[i]
                velocity = velocities[car_id]

                # Determine position of the car ahead (wrap around)
                front_pos = cars[(i + 1) % M][1]
                gap = (front_pos - position - 1) % N  # Gap modulo N

                # Rule 1: Acceleration
                if velocity < v_max:
                    velocity += 1

                # Rule 2: Slowing down due to other cars
                if velocity > gap:
                    velocity = gap

                # Rule 3: Random braking
                if velocity > 0 and random.random() < p_brake:
                    velocity -= 1

                # Rule 4: Movement
                new_position = (position + velocity - 1) % N + 1  # 1-based positions

                new_positions.append((car_id, new_position))
                new_velocities[car_id] = velocity


            # VALIDATION CHECKPOINTS
            prev_positions_dict = {car_id: pos for car_id, pos in cars}  # Before update
            step_differences = {}
            
            position_set = set()
            violations = []
            
            for car_id, new_pos in new_positions:
                old_pos = prev_positions_dict[car_id]
                
                # Step difference (accounting for circularity)
                step = (new_pos - old_pos) % N
                step_differences[car_id] = step
            
                # --- Check 1: No backward movement ---
                if step < 0 or step > v_max:
                    violations.append(f"[Step Error] Car {car_id} moved invalid steps: {step}")
            
                # --- Check 2: No duplicate positions ---
                if new_pos in position_set:
                    violations.append(f"[Collision] Position {new_pos} occupied by multiple cars!")
                position_set.add(new_pos)
            
            # --- Check 3: Circular ID order (logical) ---
            sorted_ids = [int(car_id[1:]) for car_id, _ in sorted(new_positions, key=lambda x: x[1])]
            for i in range(len(sorted_ids)):
                expected_next = (sorted_ids[i] + 1) if sorted_ids[i] < M else 1
                actual_next = sorted_ids[(i + 1) % M]
                # We tolerate circular wrap (e.g., 950 → 1)
                if actual_next != expected_next:
                    violations.append(f"[Order Error] ID {sorted_ids[i]} followed by {actual_next}, expected {expected_next}")
            
            # --- Raise error if any violations ---
            if violations:
                print(f"\nValidation failed at time step t={t}:")
                for v in violations:
                    print(" ", v)
                raise ValueError("Validation errors encountered. Simulation halted.")


            
            # Write to file
            ordered_output = [pos for _, pos in sorted(new_positions, key=lambda x: int(x[0][1:]))]
            writer.writerow([t] + ordered_output)

            # Update state
            cars = new_positions
            velocities = new_velocities

    print(f"Finished simulation for M={M}. Results in {output_path}")


# -------------------------------------------------------------------------
# CREATE VELOCITY FILES FROM POSITION FILES
# -------------------------------------------------------------------------

import csv
import winsound
import os

N = 1000  
M_values = [50, 100, 170, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]

base_path = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Data"

for M in M_values:
    position_file = os.path.join(base_path, f"Data_NaSch_{M}.csv")
    velocity_file = os.path.join(base_path, f"car_velocity_NaSch_{M}.csv")

    try:
        with open(position_file, 'r', newline='') as infile, \
             open(velocity_file, 'w', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            header = next(reader)
            writer.writerow(header)

            first_row = next(reader)
            velocity_row = first_row[:]
            for i in range(1, len(velocity_row)):
                velocity_row[i] = "0"
            writer.writerow(velocity_row)

            previous_row = first_row

            for row in reader:
                time_step_str = row[0]
                new_velocity_row = [time_step_str]

                for col_index in range(1, len(row)):
                    pos_prev = float(previous_row[col_index])
                    pos_curr = float(row[col_index])
                    diff = pos_curr - pos_prev
                    if diff < 0:
                        diff += N  # wrap-around
                    new_velocity_row.append(str(diff))

                writer.writerow(new_velocity_row)
                previous_row = row

        print(f"Velocity file created: {velocity_file}")

    except FileNotFoundError:
        print(f"File not found: {position_file}, skipping...")
    except Exception as e:
        print(f"Error with M={M}: {e}")




# -------------------------------------------------------------------------
# ANALYZE VELOCITY FILES AND PLOT RESULTS
# -------------------------------------------------------------------------


import csv
import matplotlib.pyplot as plt
import os
import time  # to track progress time

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Setup ===
M_values = [50, 100, 170, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
N = 1000
t1 = 100
T = 1000000 - t1
densities, global_speeds, flows = [], [], []

# === Paths ===
base_path = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Data"
graph_path = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Graphs"
file_pattern = os.path.join(base_path, "car_velocity_NaSch_{M}.csv")
output_csv = os.path.join(graph_path, "density_speed_flow_progress.csv")

# Ensure output folder exists
os.makedirs(graph_path, exist_ok=True)

# === Initialize CSV output file ===
with open(output_csv, "w", newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["density", "global_speed", "flow"])

# === Process velocity files ===
for M in M_values:
    start_time = time.time()
    input_file = file_pattern.format(M=M)
    rho = M / N
    sum_speeds = 0.0
    count_rows = 0

    try:
        with open(input_file, "r", newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            
            # Skip first t1 rows
            for _ in range(t1):
                next(reader, None)
            
            # Process T rows line by line
            for _ in range(T):
                try:
                    row = next(reader)
                except StopIteration:
                    print(f"File for M={M} ended early after {count_rows} rows.")
                    break

                # Slightly faster sum (avoids creating generator)
                row_sum = 0.0
                row_data = row[1:1+M]
                for col in row_data:
                    row_sum += float(col)

                sum_speeds += row_sum
                count_rows += 1

    except FileNotFoundError:
        print(f"Velocity file for M={M} not found.")
        continue

    if count_rows == 0:
        continue

    # Compute global speed and flow
    global_speed = sum_speeds / (count_rows * M)
    flow = rho * global_speed

    # Store results in memory (for plot)
    densities.append(rho)
    global_speeds.append(global_speed)
    flows.append(flow)

    # === Write progress to CSV ===
    with open(output_csv, "a", newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow([rho, global_speed, flow])

    elapsed = time.time() - start_time
    print(f"Processed M={M}: density={rho:.4f}, speed={global_speed:.4f}, flow={flow:.4f} in {elapsed:.1f} sec.")

# === Plot 1: Density vs Global Space-Mean Speed ===
plt.figure(figsize=(6, 4.2))
plt.plot(densities, global_speeds, color='black', linestyle='-')
plt.xlabel("Global density (vehicles/cell)")
plt.ylabel("Global Space-Mean Speed (cells/time step)")
plt.title("Speed-density relation")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "density_vs_speed.png"), dpi=300)
plt.close()

# === Plot 2: Density vs Flow ===
plt.figure(figsize=(6, 4.2))
plt.plot(densities, flows, color='black', linestyle='-')
plt.xlabel("Global density (vehicles/cell)")
plt.ylabel("Global flow (vehicles/time step)")
plt.title("Flow-density relation")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "density_vs_flow.png"), dpi=300)
plt.close()

print(f"\nPlots saved in: {graph_path}")
print(f"Progress CSV saved in: {output_csv}")





# -------------------------------------------------------------------------
# PLOT DENSITY vs SPEED and FLOW from Progress CSV with additional layout
# -------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import os

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Paths ===
graph_path = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Graphs"
progress_csv = os.path.join(graph_path, "density_speed_flow_progress.csv")

# === Load CSV ===
df = pd.read_csv(progress_csv)
df = df.sort_values("density")  # Ensure correct order

# === Identify density with SECOND largest flow ===
df_sorted_flow = df.sort_values("flow", ascending=False).reset_index()

# The first row → max flow
rho_c = df_sorted_flow.loc[1, "density"]
flow_c = df_sorted_flow.loc[1, "flow"]
speed_c = df_sorted_flow.loc[1, "global_speed"]




# === Plotting ===
fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# Panel a) — Flow vs Density
ax1 = axes[0]
ax1.plot(df["density"], df["flow"], color='black', linestyle='-', marker='o', markersize=4)
ax1.axvline(rho_c, color='black', linestyle='--', linewidth=1)
ax1.annotate(rf"$\rho_c = {rho_c:.2f}$", xy=(rho_c, flow_c),
             xytext=(rho_c + 0.01, flow_c -0.10),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=12)

ax1.set_ylabel("Flow (vehicles/time step)")
ax1.set_title("a) Flow-density relation")
ax1.grid(True, linestyle='--', linewidth=0.5)

# Panel b) — Speed vs Density 
ax2 = axes[1]
ax2.plot(df["density"], df["global_speed"], color='black', linestyle='-', marker='o', markersize=4)
ax2.axvline(rho_c, color='black', linestyle='--', linewidth=1)
ax2.annotate(rf"$\rho_c = {rho_c:.2f}$", xy=(rho_c, speed_c),
             xytext=(rho_c + 0.05, speed_c - 0.10),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=12)

ax2.set_xlabel("Global density (vehicles/cell)")
ax2.set_ylabel("Global Space-Mean Speed (cells/time step)")
ax2.set_title("b) Speed-density relation")
ax2.grid(True, linestyle='--', linewidth=0.5)

# === Final layout and save ===
plt.tight_layout()
output_plot_path = os.path.join(graph_path, "density_speed_flow_WITH_CRITICAL_combined_metric.png")
plt.savefig(output_plot_path, dpi=300)
plt.close()

print(f"\nPlot with critical density (combined metric) saved to: {output_plot_path}")






# -------------------------------------------------------------------------
# CREATE A LOOP TIME FILE WITH LOOP INDEX AND LOOP TIME COLUMN FOR EACH M
# FOR CAR 1 only
# -------------------------------------------------------------------------

import os
import csv

# === Parameters ===
N = 1000
M_values = [50, 100, 170, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
data_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Data"
output_csv = os.path.join(data_folder, "Loop_Times_Car1.csv")
temp_loop_dict = {}  # Keep loop times per M

# === First pass: process each file line-by-line ===
max_loops_overall = 0

for M in M_values:
    file_path = os.path.join(data_folder, f"Data_NaSch_{M}.csv")
    label = f"M={M}"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    loop_times = []
    total_distance = 0
    step_counter = 0

    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # read header
            car1_index = header.index("Car 1")

            prev_position = None
            for row in reader:
                try:
                    pos = int(row[car1_index])
                except:
                    continue  # skip bad rows

                if prev_position is None:
                    prev_position = pos
                    continue

                delta = (pos - prev_position) % N
                total_distance += delta
                step_counter += 1

                if total_distance >= N:
                    loop_times.append(step_counter)
                    total_distance = 0
                    step_counter = 0

                prev_position = pos

    except Exception as e:
        print(f"Error processing {label}: {e}")
        continue

    temp_loop_dict[label] = loop_times
    max_loops_overall = max(max_loops_overall, len(loop_times))
    print(f"Processed {label}: {len(loop_times)} loops")

# === Write row-by-row to CSV ===
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ["Loop Index"] + list(temp_loop_dict.keys())
    writer.writerow(header)

    for i in range(max_loops_overall):
        row = [i + 1]  # Loop Index
        for M in M_values:
            label = f"M={M}"
            loops = temp_loop_dict.get(label, [])
            row.append(loops[i] if i < len(loops) else "")
        writer.writerow(row)

print(f"\n Finished writing loop times to: {output_csv}")



# -------------------------------------------------------------------------
# CREATE A LOOP TIME FILE WITH LOOP INDEX AND LOOP TIME COLUMN FOR EACH M
# FOR CAR 25 cars
# -------------------------------------------------------------------------

import os
import csv
import zipfile

# === Parameters ===
N = 1000
M_values = sorted(list(range(50, 1000, 50)) + [170])

data_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Data"
zip_path    = os.path.join(data_folder, "Data_NaSch_M.zip")

# Prepare structures for 25 cars
car_ids   = list(range(1, 26))
loop_data = {
    car: { f"M={M}": [] for M in M_values }
    for car in car_ids
}
max_loops = { car: 0 for car in car_ids }

# === Open the zip once, then iterate each M inside it ===
with zipfile.ZipFile(zip_path, 'r') as zf:
    for M in M_values:
        entry_name = f"Data_NaSch_{M}.csv"
        if entry_name not in zf.namelist():
            print(f"[WARNING] {entry_name} not found in zip")
            continue

        # per‐car rolling state for this M
        prev_pos   = { c: None for c in car_ids }
        total_dist = { c: 0    for c in car_ids }
        step_cnt   = { c: 0    for c in car_ids }
        loops_here = { c: []   for c in car_ids }

        with zf.open(entry_name, 'r') as raw:
            reader = csv.reader(line.decode('utf-8') for line in raw)
            header = next(reader)
            # map car# to its column index
            col_idx = { c: header.index(f"Car {c}") for c in car_ids }

            for row in reader:
                for car in car_ids:
                    try:
                        pos = int(row[col_idx[car]])
                    except (ValueError, IndexError):
                        continue

                    if prev_pos[car] is None:
                        prev_pos[car] = pos
                        continue

                    delta = (pos - prev_pos[car]) % N
                    total_dist[car] += delta
                    step_cnt[car]   += 1

                    if total_dist[car] >= N:
                        loops_here[car].append(step_cnt[car])
                        total_dist[car] = 0
                        step_cnt[car]   = 0

                    prev_pos[car] = pos

        # save results for this M
        label = f"M={M}"
        for car in car_ids:
            loop_data[car][label] = loops_here[car]
            max_loops[car] = max(max_loops[car], len(loops_here[car]))

        print(f"Done M={M}: " +
              ", ".join(f"Car{c}={len(loops_here[c])}" for c in car_ids))

# === Write one CSV per car ===
for car in car_ids:
    out_csv = os.path.join(data_folder, f"Loop_Times_Car{car}.csv")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Loop Index"] + [f"M={M}" for M in M_values])

        for i in range(max_loops[car]):
            row = [i+1] + [
                loop_data[car][f"M={M}"][i]
                if i < len(loop_data[car][f"M={M}"]) else ""
                for M in M_values
            ]
            writer.writerow(row)

    print(f"Wrote: {out_csv}")




# -------------------------------------------------------------------------
# PLOT LOOP TIMES OF CAR 1 FOR ALL M IN ONE GRAPH
# -------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import os

# === Paths ===
data_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Data"
graph_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Graphs"
os.makedirs(graph_folder, exist_ok=True)

csv_file = os.path.join(data_folder, "Loop_Times_Car1.csv")
output_path = os.path.join(graph_folder, "loop_times_car1.png")

# === Load data ===
df = pd.read_csv(csv_file)
df.set_index("Loop Index", inplace=True)

# === Plot settings ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.2,
    "text.usetex": False
})

# === Plot ===
plt.figure(figsize=(8, 5))

# Find highest M
max_M = max([int(col.split('=')[1]) for col in df.columns])
target_col = f"M={max_M}"

# Find last valid Loop Index for highest M
last_valid_index = df[target_col].last_valid_index()
print(f"Restricting plot to Loop Index 1 to {last_valid_index} (based on M={max_M})")

# Plot all columns up to that index
for col in df.columns:
    plt.plot(df.index[df.index <= last_valid_index], df[col][df.index <= last_valid_index],
             label=col, linestyle='-')

plt.xlabel("Loop Index")
plt.ylabel("Loop Time [seconds]")
plt.title("Loop Times of Car 1 for Different Densities")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# Explicitly limit x-axis
plt.xlim(1, last_valid_index)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved to: {output_path}")


# -------------------------------------------------------------------------
# AVERAGE LOOP TIME FOR ALL 25 CARS PLOTTED OVER DENSITY, 
# AVERAGE STANDARD DEVIATION OF LOOP TIME FOR 25 CARS PLOTTED OVER DENSITY
# -------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Setup ===
N = 1000
data_folder  = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Data"
graph_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Graphs"
os.makedirs(graph_folder, exist_ok=True)

# === 1) Read & melt all 25 car files ===
car_ids = range(1, 26)
df_list = []

for car in car_ids:
    path = os.path.join(data_folder, f"Loop_Times_Car{car}.csv")
    df   = pd.read_csv(path)
    # wide → long
    long = df.melt(
        id_vars=["Loop Index"],
        var_name="M",
        value_name="Loop Time"
    ).dropna(subset=["Loop Time"])
    long["Loop Time"] = long["Loop Time"].astype(int)
    long["M"]         = long["M"].str.replace("M=", "").astype(int)
    long["Car"]       = car
    df_list.append(long)

big_df = pd.concat(df_list, ignore_index=True)

# === 2) Per-car stats at each M ===
car_stats = (
    big_df
    .groupby(["M","Car"])["Loop Time"]
    .agg(['mean','std'])
    .reset_index()
)

# === 3) Average across cars per M ===
m_grouped     = car_stats.groupby("M")
avg_mean      = m_grouped["mean"].mean()
avg_std       = m_grouped["std"].mean()
densities     = avg_mean.index / N

# === 4) Save CSV of mean & std vs M/density ===
out_csv = os.path.join(data_folder, "Mean_Loop_Times_per_M.csv")
summary = pd.DataFrame({
    "M":                avg_mean.index,
    "Density":          densities,
    "Mean Loop Time":   avg_mean.values,
    "Std Dev Loop Time":avg_std.values
})
summary.to_csv(out_csv, index=False)
print("Wrote summary CSV →", out_csv)

# === 5) Plot with dark-grey line + white circles ===
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":      12,
    "axes.titlesize":14,
    "axes.labelsize":12,
    "axes.edgecolor":"black",
    "axes.linewidth":1.0,
    "lines.linewidth":1.4,
    "text.usetex":   False,
})

# Marker & color settings
grey   = "#444444"
marker = 'o'

# Plot 1: Mean Loop Time vs Density
plt.figure(figsize=(6.5, 4.2))
plt.plot(
    densities,
    avg_mean.values,
    color=grey,
    marker=marker,
    markerfacecolor='white',
    markeredgecolor=grey,
    linestyle='-'
)
plt.xlabel("Density (vehicles per cell)")
plt.ylabel("Mean Loop Time [seconds]")
plt.title("Mean Loop Time vs. Density")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(graph_folder, "mean_loop_time_vs_density.png"),
    dpi=300
)
plt.close()

# Plot 2: StdDev of Loop Time vs Density
plt.figure(figsize=(6.5, 4.2))
plt.plot(
    densities,
    avg_std.values,
    color=grey,
    marker=marker,
    markerfacecolor='white',
    markeredgecolor=grey,
    linestyle='-'
)
plt.xlabel("Density (vehicles per cell)")
plt.ylabel("StdDev of Loop Time [seconds]")
plt.title("Standard Deviation of Loop Time vs. Density")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(graph_folder, "std_loop_time_vs_density.png"),
    dpi=300
)
plt.close()

print("Plots saved to:", graph_folder)



# -------------------------------------------------------------------------
# CREATE DELAY FILE AS CSV (CAR 1–25, DELAY = max(actual - mean, 0))
# -------------------------------------------------------------------------

import os
import re
import glob
import pandas as pd

# === Parameters ===
data_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch Data"
output_csv  = os.path.join(data_folder, "Delay_Times_per_loop.csv")

# === 1) Discover all your Loop_Times_CarX.csv files ===
pattern = os.path.join(data_folder, "Loop_Times_Car*.csv")
files = sorted(glob.glob(pattern))
if not files:
    raise FileNotFoundError(f"No loop‐time CSVs found in {data_folder}")

# === 2) Read & stack them ===
df_list = []
for path in files:
    # extract car number from filename
    m = re.search(r"Loop_Times_Car(\d+)\.csv$", os.path.basename(path))
    if not m:
        continue
    car = int(m.group(1))

    df = pd.read_csv(path)
    long = (
        df.melt(
            id_vars=["Loop Index"],
            var_name="M",
            value_name="Loop Time"
        )
        .dropna(subset=["Loop Time"])
    )
    long["Loop Time"] = long["Loop Time"].astype(int)
    long["M"]         = long["M"].str.replace("M=", "").astype(int)
    long["Car"]       = car

    df_list.append(long)

big_df = pd.concat(df_list, ignore_index=True)

# === 3) Compute mean per (Car, M) and then delay ===
big_df["Mean Loop Time"] = big_df.groupby(["Car","M"])["Loop Time"].transform("mean")
big_df["Delay"] = (
    (big_df["Loop Time"] - big_df["Mean Loop Time"])
    .clip(lower=0)
    .astype(int)
)

# === 4) Save just (Car, M, Loop Index, Delay) ===
out = big_df[["Car","M","Loop Index","Delay"]]
out.to_csv(output_csv, index=False)

print(f"Delay data written to: {output_csv}")




# -------------------------------------------------------------------------
# PLOT DELAY TIMES OF CAR 1 FOR ALL M IN ONE GRAPH
# -------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import os

# === File paths ===
base_folder = "D:/Master Thesis/Thesis Car Simulation Code"
data_file = os.path.join(base_folder, "NaSch Data", "Delay_Times_per_loop.csv")
graph_folder = os.path.join(base_folder, "NaSch Graphs")
os.makedirs(graph_folder, exist_ok=True)

# === Load CSV and filter for Car 1 ===
df = pd.read_csv(data_file)
df_car1 = df[df["Car"] == 1]

# === Plot Style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.2,
    "text.usetex": False
})

# === Plot ===
plt.figure(figsize=(8, 5))

for m_val, group in df_car1.groupby("M"):
    plt.plot(group["Loop Index"], group["Delay"], label=f"M={m_val}", linestyle='-')

plt.xlabel("Lap Number")
plt.ylabel("Delay (Time Steps)")
plt.title("Delay of Car 1 per Lap for Different M Values")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.tight_layout()

# === Save ===
output_path = os.path.join(graph_folder, "delay_times_car1.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved to: {output_path}")


# -------------------------------------------------------------------------
# PLOT SORTED DELAY TIMES OF CAR 1 FOR ALL M IN ONE GRAPH
# -------------------------------------------------------------------------


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === File paths ===
base_folder  = "D:/Master Thesis/Thesis Car Simulation Code"
data_file    = os.path.join(base_folder, "NaSch Data", "Delay_Times_per_loop.csv")
graph_folder = os.path.join(base_folder, "NaSch Graphs")
os.makedirs(graph_folder, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(data_file)

# === Plot Style ===
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     12,
    "axes.titlesize":14,
    "axes.labelsize":12,
    "xtick.labelsize":11,
    "ytick.labelsize":11,
    "legend.fontsize":8,
    "axes.edgecolor":"black",
    "axes.linewidth":1.0,
    "lines.linewidth":0.0,   # no connecting lines
    "text.usetex":   False
})

plt.figure(figsize=(8, 5))

# For each M, sort delays across ALL cars combined, then plot
for m_val, group in df.groupby("M"):
    delays_sorted = np.sort(group["Delay"].values)
    ranks         = np.arange(1, len(delays_sorted) + 1)
    plt.plot(
        ranks,
        delays_sorted,
        marker='.',        # small dot
        linestyle='none',  # no lines
        markersize=3,
        label=f"M={m_val}"
    )

plt.xlabel("Sorted Instance Rank")
plt.ylabel("Delay [seconds]")
plt.title("Sorted Delay for Different M Values")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=False)

plt.tight_layout()

# === Save ===
output_path = os.path.join(graph_folder, "sorted_delay_times_all_cars.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved to: {output_path}")





# -------------------------------------------------------------------------
# AVALANCHE DATA FOR DIFFERENT M: SIZE & DURATION OF DELAY AVALANCHES
# -------------------------------------------------------------------------


import os
import pandas as pd
import numpy as np

# === Paths (note the updated input filename) ===
base_dir          = "D:/Master Thesis/Thesis Car Simulation Code"
data_file         = os.path.join(base_dir, "NaSch Data", "Delay_Times_per_loop.csv")
output_durations  = os.path.join(base_dir, "NaSch Data", "Avalanche_Durations.csv")
output_sizes      = os.path.join(base_dir, "NaSch Data", "Avalanche_Sizes.csv")

# === Load the flat delay file ===
df = pd.read_csv(data_file)

# === Prepare accumulators ===
duration_records = []
size_records     = []

# === For each Car, M combination ===
for (car, M), group in df.groupby(["Car", "M"]):
    # sort by loop index and get the Delay values as a numpy array
    series = group.sort_values("Loop Index")["Delay"].values

    # find all indices where delay == 0 (these mark the boundaries)
    zeros = np.where(series == 0)[0]

    # walk each adjacent pair of zeros
    avalanche_idx = 1
    for start_zero, end_zero in zip(zeros, zeros[1:]):
        # only consider if there is at least one positive delay between
        if end_zero > start_zero + 1:
            # slice out just the positive‐delay region
            segment = series[start_zero+1 : end_zero]
            duration = len(segment)                  # how many time‐steps
            size     = np.trapz(segment, dx=1)       # approximate area under the delay curve

            duration_records.append({
                "Car": car,
                "M": M,
                "Avalanche Index": avalanche_idx,
                "Duration": duration
            })
            size_records.append({
                "Car": car,
                "M": M,
                "Avalanche Index": avalanche_idx,
                "Size": size
            })

            avalanche_idx += 1

# === Write out results exactly as before ===
pd.DataFrame(duration_records).to_csv(output_durations, index=False)
pd.DataFrame(size_records).to_csv(output_sizes,     index=False)

print(f"Saved avalanche durations to: {output_durations}")
print(f"Saved avalanche sizes to:     {output_sizes}")



# -------------------------------------------------------------------------
# AVALANCHE PLOTS: MEAN DURATION VS DENSITY AND MEAN SIZE VS DENSITY
# -------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import os

# === Paths ===
base_dir     = "D:/Master Thesis/Thesis Car Simulation Code"
data_folder  = os.path.join(base_dir, "NaSch Data")
graph_folder = os.path.join(base_dir, "NaSch Graphs")
os.makedirs(graph_folder, exist_ok=True)

duration_file = os.path.join(data_folder, "Avalanche_Durations.csv")
size_file     = os.path.join(data_folder, "Avalanche_Sizes.csv")

# === Load CSVs ===
df_durations = pd.read_csv(duration_file)
df_sizes     = pd.read_csv(size_file)

# === Compute Density (vehicles per cell) ===
df_durations["Density"] = df_durations["M"] / 1000
df_sizes["Density"]     = df_sizes["M"] / 1000

# === Summaries: mean over all cars & avalanches at each density ===
duration_summary = (
    df_durations
    .groupby("Density")["Duration"]
    .mean()
    .reset_index()
)
size_summary = (
    df_sizes
    .groupby("Density")["Size"]
    .mean()
    .reset_index()
)

# === Global plot style ===
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":      12,
    "axes.titlesize":14,
    "axes.labelsize":12,
    "axes.edgecolor":"black",
    "axes.linewidth":1.0,
    "lines.linewidth":1.4,
    "text.usetex":   False,
})

# === Marker & color settings ===
grey   = "#444444"
marker = 'o'

# === Plot 1: Mean Avalanche Duration vs Density ===
plt.figure(figsize=(6.5, 4.2))
plt.plot(
    duration_summary["Density"],
    duration_summary["Duration"],
    color=grey,
    marker=marker,
    markerfacecolor='white',
    markeredgecolor=grey,
    linestyle='-'
)
plt.xlabel("Density (vehicles per cell)")
plt.ylabel("Mean Avalanche Duration (time steps)")
plt.title("Mean Avalanche Duration vs. Density")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(graph_folder, "mean_avalanche_duration_vs_density.png"),
    dpi=300
)
plt.close()

# === Plot 2: Mean Avalanche Size vs Density ===
plt.figure(figsize=(6.5, 4.2))
plt.plot(
    size_summary["Density"],
    size_summary["Size"],
    color=grey,
    marker=marker,
    markerfacecolor='white',
    markeredgecolor=grey,
    linestyle='-'
)
plt.xlabel("Density (vehicles per cell)")
plt.ylabel("Mean Avalanche Size (area)")
plt.title("Mean Avalanche Size vs. Density")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(graph_folder, "mean_avalanche_size_vs_density.png"),
    dpi=300
)
plt.close()

print("Plots saved to:", graph_folder)




#####################################################################
# PANEL of 6 for my thesis (with log–log for mean & critical arrow only in (b))
#####################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Paths ===
base_dir      = "D:/Master Thesis/Thesis Car Simulation Code"
data_folder   = os.path.join(base_dir, "NaSch Data")
graph_folder  = os.path.join(base_dir, "NaSch Graphs")
progress_csv  = os.path.join(graph_folder, "density_speed_flow_progress.csv")
os.makedirs(graph_folder, exist_ok=True)

# === Compute critical density from progress CSV ===
df_prog     = pd.read_csv(progress_csv).sort_values("density")
flow_sorted = df_prog.sort_values("flow", ascending=False).reset_index(drop=True)
rho_c       = flow_sorted.loc[1, "density"]  # second‐largest flow

# === Global plot style ===
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":      12,
    "axes.titlesize":14,
    "axes.labelsize":12,
    "xtick.labelsize":11,
    "ytick.labelsize":11,
    "legend.fontsize":8,
    "axes.edgecolor":"black",
    "axes.linewidth":1.0,
    "lines.linewidth":1.2,
    "text.usetex":   False
})

grey   = "#444444"
marker = 'o'

# === Load data for each subplot ===

# a) Loop Times of Car 1
df_car1 = pd.read_csv(os.path.join(data_folder, "Loop_Times_Car1.csv"))
df_car1.set_index("Loop Index", inplace=True)
max_M      = max(int(c.split('=')[1]) for c in df_car1.columns)
last_valid = df_car1[f"M={max_M}"].last_valid_index()

# b,c) Mean & Std Dev Loop Time across cars
summary    = pd.read_csv(os.path.join(data_folder, "Mean_Loop_Times_per_M.csv"))
densities  = summary["Density"].values
mean_loop  = summary["Mean Loop Time"].values
std_loop   = summary["Std Dev Loop Time"].values

# compute y‐value at rho_c for the mean plot arrow
mean_at_rho_c = np.interp(rho_c, densities, mean_loop)

# d) Sorted Delay Times
df_delay = pd.read_csv(os.path.join(data_folder, "Delay_Times_per_loop.csv"))

# e,f) Avalanche summaries
df_dur      = pd.read_csv(os.path.join(data_folder, "Avalanche_Durations.csv"))
df_dur["Density"] = df_dur["M"] / 1000
dur_summary = df_dur.groupby("Density")["Duration"].mean().reset_index()

df_size      = pd.read_csv(os.path.join(data_folder, "Avalanche_Sizes.csv"))
df_size["Density"] = df_size["M"] / 1000
size_summary = df_size.groupby("Density")["Size"].mean().reset_index()

# === Create 2×3 panel ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) Loop Times of Car 1
ax = axes[0, 0]
for col in df_car1.columns:
    ax.plot(
        df_car1.index[df_car1.index <= last_valid],
        df_car1[col][df_car1.index <= last_valid],
        linestyle='-',
        label=col
    )
ax.set_xlim(1, last_valid)
ax.set_xlabel("Loop Index")
ax.set_ylabel("Loop Time [seconds]")
ax.set_title("a) Loop Times of Car 1 for Different Densities")
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# (b) Mean Loop Time vs. Density (log–log, with critical arrow)
ax = axes[0, 1]
ax.plot(
    densities, mean_loop,
    color=grey, marker=marker,
    markerfacecolor='white', markeredgecolor=grey,
    linestyle='-'
)
ax.set_xscale('log')
ax.set_yscale('log')
ax.axvline(rho_c, color='black', linestyle='--', linewidth=1)
ax.annotate(
     rf"$\rho_c = {rho_c:.2f}$",
    xy=(rho_c, mean_at_rho_c),
    xytext=(rho_c*1.2, mean_at_rho_c*0.85),   
    arrowprops=dict(arrowstyle='->', color='black', lw=1),
    fontsize=12
)

ax.set_xlabel("Density (vehicles per cell) (log)")
ax.set_ylabel("Mean Loop Time [seconds] (log)")
ax.set_title("b) Mean Loop Time vs. Density (log–log)")
ax.grid(True, linestyle='--', linewidth=0.5, which='both')

# (c) Std Dev Loop Time vs. Density (linear, no critical arrow)
ax = axes[0, 2]
ax.plot(
    densities, std_loop,
    color=grey, marker=marker,
    markerfacecolor='white', markeredgecolor=grey,
    linestyle='-'
)
ax.set_xlabel("Density (vehicles per cell)")
ax.set_ylabel("Std Dev Loop Time [seconds]")
ax.set_title("c) Std Dev of Loop Time vs. Density")
ax.grid(True, linestyle='--', linewidth=0.5)

# (d) Sorted Delay for Different M Values
ax = axes[1, 0]
for m_val, grp in df_delay.groupby("M"):
    sd = np.sort(grp["Delay"].values)
    rk = np.arange(1, len(sd)+1)
    ax.plot(
        rk, sd,
        marker='.', linestyle='none', markersize=3,
        label=f"M={m_val}"
    )
ax.set_xlabel("Sorted Instance Rank")
ax.set_ylabel("Delay [seconds]")
ax.set_title("d) Sorted Delay for Different M Values")
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# (e) Mean Avalanche Duration vs. Density
ax = axes[1, 1]
ax.plot(
    dur_summary["Density"], dur_summary["Duration"],
    color=grey, marker=marker,
    markerfacecolor='white', markeredgecolor=grey,
    linestyle='-'
)
ax.set_xlabel("Density (vehicles per cell)")
ax.set_ylabel("Mean Avalanche Duration (time steps)")
ax.set_title("e) Mean Avalanche Duration vs. Density")
ax.grid(True, linestyle='--', linewidth=0.5)

# (f) Mean Avalanche Size vs. Density
ax = axes[1, 2]
ax.plot(
    size_summary["Density"], size_summary["Size"],
    color=grey, marker=marker,
    markerfacecolor='white', markeredgecolor=grey,
    linestyle='-'
)
ax.set_xlabel("Density (vehicles per cell)")
ax.set_ylabel("Mean Avalanche Size (area)")
ax.set_title("f) Mean Avalanche Size vs. Density")
ax.grid(True, linestyle='--', linewidth=0.5)

# Finalize
plt.tight_layout()
output_path = os.path.join(graph_folder, "panel_2x3_loglog_b_only_with_critical.png")
plt.savefig(output_path, dpi=300)
plt.close(fig)

print(f"Panel saved to: {output_path}")





#######################################################################
#######################################################################
# NaSch + IDM Model
#######################################################################
#######################################################################


# -------------------------------------------------------------------------
# NaSch + IDM SIMULATION (PARALLELIZED)
# -------------------------------------------------------------------------

import random
import csv
import math
import numpy as np
import os
from multiprocessing import Pool, cpu_count

# Parameters
N = 1000
time_steps = 1000000
batch_size = 1000
M_values = [50, 75, 100, 125, 150, 170, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
random.seed(55)

# IDM parameters by driver type
idm_params = {
    "risk-averse":  {"s0": 0, "T": 1.2, "p_brake": 0.12},
    "normal":       {"s0": 0, "T": 1.0, "p_brake": 0.10},
    "risk-seeking": {"s0": 0, "T": 0.8, "p_brake": 0.08}
}

def assign_driver_type():
    return np.random.choice(["risk-averse", "normal", "risk-seeking"], p=[0.025, 0.95, 0.025])

def sample_vmax(driver_type):
    if driver_type == "risk-averse":
        return np.random.choice([3, 4], p=[0.5, 0.5])
    elif driver_type == "normal":
        return 4
    else:
        return np.random.choice([4, 5], p=[0.5, 0.5])

def get_desired_space(v, driver_type):
    p = idm_params[driver_type]
    return int(math.floor(p["s0"] + p["T"] * v))

def run_simulation(M):
    print(f"\n--- Working on M = {M} ---")
    output_path = f"D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Data/Data_NaSch_IDM_{M}.csv"

    # Initial setup
    initial_positions = [(i * N) // M + 1 for i in range(M)]
    cars = [(f"C{i+1}", pos) for i, pos in enumerate(initial_positions)]
    driver_types = {car_id: assign_driver_type() for car_id, _ in cars}
    desired_speeds = {car_id: sample_vmax(driver_types[car_id]) for car_id, _ in cars}
    velocities = {car_id: desired_speeds[car_id] for car_id, _ in cars}

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time Step'] + [f"Car {i+1}" for i in range(M)])
        writer.writerow([0] + [pos for _, pos in cars])

    for batch_start in range(1, time_steps + 1, batch_size):
        batch_end = min(batch_start + batch_size, time_steps + 1)
        rows_to_write = []

        for t in range(batch_start, batch_end):
            cars = sorted(cars, key=lambda x: x[1])
            new_positions = []
            new_velocities = {}

            for i in range(M):
                car_id, pos = cars[i]
                front_pos = cars[(i + 1) % M][1]
                velocity = velocities[car_id]
                driver_type = driver_types[car_id]
                desired_v = desired_speeds[car_id]
                p_brake = idm_params[driver_type]["p_brake"]
                desired_space = get_desired_space(velocity, driver_type)

                gap = (front_pos - pos - 1) % N

                # Rule 1: Acceleration
                if velocity < desired_v:
                    velocity += 1
                # Rule 2: Space constraint
                if velocity > (gap - desired_space):
                    velocity = max(gap - desired_space, 0)
                # Rule 3: Random braking
                if velocity > 0 and random.random() < p_brake:
                    velocity -= 1

                new_pos = (pos + velocity - 1) % N + 1
                new_positions.append((car_id, new_pos))
                new_velocities[car_id] = velocity

            # Validation
            prev_positions = {car_id: pos for car_id, pos in cars}
            position_set = set()
            violations = []

            for car_id, new_pos in new_positions:
                old_pos = prev_positions[car_id]
                step = (new_pos - old_pos) % N
                if step < 0 or step > 5:
                    violations.append(f"[Step Error] Car {car_id} moved invalid steps: {step}")
                if new_pos in position_set:
                    violations.append(f"[Collision] Position {new_pos} occupied by multiple cars!")
                position_set.add(new_pos)

            sorted_ids = [int(car_id[1:]) for car_id, _ in sorted(new_positions, key=lambda x: x[1])]
            for i in range(len(sorted_ids)):
                expected_next = (sorted_ids[i] + 1) if sorted_ids[i] < M else 1
                actual_next = sorted_ids[(i + 1) % M]
                if actual_next != expected_next:
                    violations.append(f"[Order Error] ID {sorted_ids[i]} followed by {actual_next}, expected {expected_next}")

            if violations:
                print(f"Validation failed at time step t={t}:")
                for v in violations:
                    print(" ", v)
                raise ValueError("Validation errors encountered. Simulation halted.")

            rows_to_write.append([t] + [pos for _, pos in sorted(new_positions, key=lambda x: int(x[0][1:]))])
            cars = new_positions
            velocities = new_velocities

        with open(output_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows_to_write)

    print(f"Finished simulation for M = {M}. Data saved to {output_path}")

if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        pool.map(run_simulation, M_values)



# -------------------------------------------------------------------------
# CREATE VELOCITY FILES FROM POSITION FILES
# -------------------------------------------------------------------------

import csv
import winsound
import os

N = 1000  
M_values = [50, 75, 100, 125, 150, 170, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]

base_path = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Data"

for M in M_values:
    position_file = os.path.join(base_path, f"Data_NaSch_IDM_{M}.csv")
    velocity_file = os.path.join(base_path, f"car_velocity_NaSch_IDM_{M}.csv")

    try:
        with open(position_file, 'r', newline='') as infile, \
             open(velocity_file, 'w', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            header = next(reader)
            writer.writerow(header)

            first_row = next(reader)
            velocity_row = first_row[:]
            for i in range(1, len(velocity_row)):
                velocity_row[i] = "0"
            writer.writerow(velocity_row)

            previous_row = first_row

            for row in reader:
                time_step_str = row[0]
                new_velocity_row = [time_step_str]

                for col_index in range(1, len(row)):
                    pos_prev = float(previous_row[col_index])
                    pos_curr = float(row[col_index])
                    diff = pos_curr - pos_prev
                    if diff < 0:
                        diff += N  # wrap-around
                    new_velocity_row.append(str(diff))

                writer.writerow(new_velocity_row)
                previous_row = row

        print(f"Velocity file created: {velocity_file}")

    except FileNotFoundError:
        print(f"File not found: {position_file}, skipping...")
    except Exception as e:
        print(f"Error with M={M}: {e}")




# -------------------------------------------------------------------------
# ANALYZE VELOCITY FILES AND PLOT RESULTS
# -------------------------------------------------------------------------


import csv
import matplotlib.pyplot as plt
import os
import time  # to track progress time

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Setup ===
M_values = [50, 75, 100, 125, 150, 170, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
N = 1000
t1 = 100
T = 1000000 - t1
densities, global_speeds, flows = [], [], []

# === Paths ===
base_path = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Data"
graph_path = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Graphs"
file_pattern = os.path.join(base_path, "car_velocity_NaSch_IDM_{M}.csv")
output_csv = os.path.join(graph_path, "density_speed_flow_progress.csv")

# Ensure output folder exists
os.makedirs(graph_path, exist_ok=True)

# === Initialize CSV output file ===
with open(output_csv, "w", newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["density", "global_speed", "flow"])

# === Process velocity files ===
for M in M_values:
    start_time = time.time()
    input_file = file_pattern.format(M=M)
    rho = M / N
    sum_speeds = 0.0
    count_rows = 0

    try:
        with open(input_file, "r", newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            
            # Skip first t1 rows
            for _ in range(t1):
                next(reader, None)
            
            # Process T rows line by line
            for _ in range(T):
                try:
                    row = next(reader)
                except StopIteration:
                    print(f"File for M={M} ended early after {count_rows} rows.")
                    break

                # Slightly faster sum (avoids creating generator)
                row_sum = 0.0
                row_data = row[1:1+M]
                for col in row_data:
                    row_sum += float(col)

                sum_speeds += row_sum
                count_rows += 1

    except FileNotFoundError:
        print(f"Velocity file for M={M} not found.")
        continue

    if count_rows == 0:
        continue

    # Compute global speed and flow
    global_speed = sum_speeds / (count_rows * M)
    flow = rho * global_speed

    # Store results in memory (for plot)
    densities.append(rho)
    global_speeds.append(global_speed)
    flows.append(flow)

    # === Write progress to CSV ===
    with open(output_csv, "a", newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow([rho, global_speed, flow])

    elapsed = time.time() - start_time
    print(f"Processed M={M}: density={rho:.4f}, speed={global_speed:.4f}, flow={flow:.4f} in {elapsed:.1f} sec.")

# === Plot 1: Density vs Global Space-Mean Speed ===
plt.figure(figsize=(6, 4.2))
plt.plot(densities, global_speeds, color='black', linestyle='-')
plt.xlabel("Global density (vehicles/cell)")
plt.ylabel("Global Space-Mean Speed (cells/time step)")
plt.title("Speed-density relation")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "density_vs_speed.png"), dpi=300)
plt.close()

# === Plot 2: Density vs Flow ===
plt.figure(figsize=(6, 4.2))
plt.plot(densities, flows, color='black', linestyle='-')
plt.xlabel("Global density (vehicles/cell)")
plt.ylabel("Global flow (vehicles/time step)")
plt.title("Flow-density relation")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(graph_path, "density_vs_flow.png"), dpi=300)
plt.close()

print(f"\nPlots saved in: {graph_path}")
print(f"Progress CSV saved in: {output_csv}")





# -------------------------------------------------------------------------
# PLOT DENSITY vs SPEED and FLOW from Progress CSV with additional layout
# -------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import os

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Paths ===
graph_path = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Graphs"
progress_csv = os.path.join(graph_path, "density_speed_flow_progress.csv")

# === Load CSV ===
df = pd.read_csv(progress_csv)
df = df[df["density"] != 0.025]
df = df.sort_values("density")  # Ensure correct order

# === Identify density with SECOND largest flow ===
df_sorted_flow = df.sort_values("flow", ascending=False).reset_index()

# The first row → max flow
rho_c = df_sorted_flow.loc[0, "density"]
flow_c = df_sorted_flow.loc[0, "flow"]
speed_c = df_sorted_flow.loc[0, "global_speed"]



# === Plotting ===
fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# Panel a) — Flow vs Density
ax1 = axes[0]
ax1.plot(df["density"], df["flow"], color='black', linestyle='-', marker='o', markersize=4)
ax1.axvline(rho_c, color='black', linestyle='--', linewidth=1)
ax1.annotate(rf"$\rho_c = {rho_c:.2f}$", xy=(rho_c, flow_c),
             xytext=(rho_c + 0.01, flow_c -0.10),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=12)

ax1.set_ylabel("Flow (vehicles/time step)")
ax1.set_title("a) Flow-density relation")
ax1.grid(True, linestyle='--', linewidth=0.5)

# Panel b) — Speed vs Density 
ax2 = axes[1]
ax2.plot(df["density"], df["global_speed"], color='black', linestyle='-', marker='o', markersize=4)
ax2.axvline(rho_c, color='black', linestyle='--', linewidth=1)
ax2.annotate(rf"$\rho_c = {rho_c:.2f}$", xy=(rho_c, speed_c),
             xytext=(rho_c + 0.05, speed_c - 0.10),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=12)

ax2.set_xlabel("Global density (vehicles/cell)")
ax2.set_ylabel("Global Space-Mean Speed (cells/time step)")
ax2.set_title("b) Speed-density relation")
ax2.grid(True, linestyle='--', linewidth=0.5)

# === Final layout and save ===
plt.tight_layout()
output_plot_path = os.path.join(graph_path, "density_speed_flow_WITH_CRITICAL_combined_metric.png")
plt.savefig(output_plot_path, dpi=300)
plt.close()

print(f"\nPlot with critical density (combined metric) saved to: {output_plot_path}")





# -------------------------------------------------------------------------
# CREATE A LOOP TIME FILE WITH LOOP INDEX AND LOOP TIME COLUMN FOR EACH M
# FOR CAR 1 only
# -------------------------------------------------------------------------

import os
import csv

# === Parameters ===
N = 1000
M_values = [50, 75, 100, 125, 150, 170, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
data_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Data"
output_csv = os.path.join(data_folder, "Loop_Times_Car1.csv")
temp_loop_dict = {}  # Keep loop times per M

# === First pass: process each file line-by-line ===
max_loops_overall = 0

for M in M_values:
    file_path = os.path.join(data_folder, f"Data_NaSch_IDM_{M}.csv")
    label = f"M={M}"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    loop_times = []
    total_distance = 0
    step_counter = 0

    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # read header
            car1_index = header.index("Car 1")

            prev_position = None
            for row in reader:
                try:
                    pos = int(row[car1_index])
                except:
                    continue  # skip bad rows

                if prev_position is None:
                    prev_position = pos
                    continue

                delta = (pos - prev_position) % N
                total_distance += delta
                step_counter += 1

                if total_distance >= N:
                    loop_times.append(step_counter)
                    total_distance = 0
                    step_counter = 0

                prev_position = pos

    except Exception as e:
        print(f"Error processing {label}: {e}")
        continue

    temp_loop_dict[label] = loop_times
    max_loops_overall = max(max_loops_overall, len(loop_times))
    print(f"Processed {label}: {len(loop_times)} loops")

# === Write row-by-row to CSV ===
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ["Loop Index"] + list(temp_loop_dict.keys())
    writer.writerow(header)

    for i in range(max_loops_overall):
        row = [i + 1]  # Loop Index
        for M in M_values:
            label = f"M={M}"
            loops = temp_loop_dict.get(label, [])
            row.append(loops[i] if i < len(loops) else "")
        writer.writerow(row)

print(f"\n Finished writing loop times to: {output_csv}")




# -------------------------------------------------------------------------
# CREATE A LOOP TIME FILE WITH LOOP INDEX AND LOOP TIME COLUMN FOR EACH M
# FOR CAR 25 cars
# -------------------------------------------------------------------------

import os
import csv
import zipfile

# === Parameters ===
N = 1000
M_values = [50, 75, 100, 125, 150, 170, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]

data_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Data"


# Prepare structures for 25 cars
car_ids   = list(range(2, 27))
loop_data = {
    car: { f"M={M}": [] for M in M_values }
    for car in car_ids
}
max_loops = { car: 0 for car in car_ids }

# === Open the zip once, then iterate each M inside it ===
for M in M_values:
    entry_path = os.path.join(data_folder, f"Data_NaSch_IDM_{M}.csv")
    if not os.path.exists(entry_path):
        print(f"[WARNING] {entry_path} not found")
        continue

    # per‐car rolling state for this M
    prev_pos   = { c: None for c in car_ids }
    total_dist = { c: 0    for c in car_ids }
    step_cnt   = { c: 0    for c in car_ids }
    loops_here = { c: []   for c in car_ids }

    with open(entry_path, 'r', newline='') as raw:
        reader = csv.reader(raw)

        header = next(reader)
        # map car# to its column index
        col_idx = { c: header.index(f"Car {c}") for c in car_ids }

        for row in reader:
            for car in car_ids:
                try:
                    pos = int(row[col_idx[car]])
                except (ValueError, IndexError):
                    continue

                if prev_pos[car] is None:
                    prev_pos[car] = pos
                    continue

                delta = (pos - prev_pos[car]) % N
                total_dist[car] += delta
                step_cnt[car]   += 1

                if total_dist[car] >= N:
                    loops_here[car].append(step_cnt[car])
                    total_dist[car] = 0
                    step_cnt[car]   = 0

                prev_pos[car] = pos

    # save results for this M
    label = f"M={M}"
    for car in car_ids:
        loop_data[car][label] = loops_here[car]
        max_loops[car] = max(max_loops[car], len(loops_here[car]))

    print(f"Done M={M}: " +
          ", ".join(f"Car{c}={len(loops_here[c])}" for c in car_ids))

# === Write one CSV per car ===
for car in car_ids:
    out_csv = os.path.join(data_folder, f"Loop_Times_Car{car}.csv")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Loop Index"] + [f"M={M}" for M in M_values])

        for i in range(max_loops[car]):
            row = [i+1] + [
                loop_data[car][f"M={M}"][i]
                if i < len(loop_data[car][f"M={M}"]) else ""
                for M in M_values
            ]
            writer.writerow(row)

    print(f"Wrote: {out_csv}")




# -------------------------------------------------------------------------
# PLOT LOOP TIMES OF CAR 1 FOR ALL M IN ONE GRAPH
# -------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import os

# === Paths ===
data_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Data"
graph_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Graphs"
os.makedirs(graph_folder, exist_ok=True)

csv_file = os.path.join(data_folder, "Loop_Times_Car1.csv")
output_path = os.path.join(graph_folder, "loop_times_car1.png")

# === Load data ===
df = pd.read_csv(csv_file)
df.set_index("Loop Index", inplace=True)

# === Plot settings ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.2,
    "text.usetex": False
})

# === Plot ===
plt.figure(figsize=(8, 5))

# Find highest M
max_M = max([int(col.split('=')[1]) for col in df.columns])
target_col = f"M={max_M}"

# Find last valid Loop Index for highest M
last_valid_index = df[target_col].last_valid_index()
print(f"Restricting plot to Loop Index 1 to {last_valid_index} (based on M={max_M})")

# Plot all columns up to that index
for col in df.columns:
    plt.plot(df.index[df.index <= last_valid_index], df[col][df.index <= last_valid_index],
             label=col, linestyle='-')

plt.xlabel("Loop Index")
plt.ylabel("Loop Time [seconds]")
plt.title("Loop Times of Car 1 for Different Densities")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# Explicitly limit x-axis
plt.xlim(1, last_valid_index)

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved to: {output_path}")




# -------------------------------------------------------------------------
# AVERAGE LOOP TIME FOR ALL 25 CARS PLOTTED OVER DENSITY, 
# AVERAGE STANDARD DEVIATION OF LOOP TIME FOR 25 CARS PLOTTED OVER DENSITY
# -------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Setup ===
N = 1000
data_folder  = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Data"
graph_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Graphs"
os.makedirs(graph_folder, exist_ok=True)

# === 1) Read & melt all 25 car files ===
car_ids = range(1, 26)
df_list = []

for car in car_ids:
    path = os.path.join(data_folder, f"Loop_Times_Car{car}.csv")
    df   = pd.read_csv(path)
    # wide → long
    long = df.melt(
        id_vars=["Loop Index"],
        var_name="M",
        value_name="Loop Time"
    ).dropna(subset=["Loop Time"])
    long["Loop Time"] = long["Loop Time"].astype(int)
    long["M"]         = long["M"].str.replace("M=", "").astype(int)
    long["Car"]       = car
    df_list.append(long)

big_df = pd.concat(df_list, ignore_index=True)

# === 2) Per-car stats at each M ===
car_stats = (
    big_df
    .groupby(["M","Car"])["Loop Time"]
    .agg(['mean','std'])
    .reset_index()
)

# === 3) Average across cars per M ===
m_grouped     = car_stats.groupby("M")
avg_mean      = m_grouped["mean"].mean()
avg_std       = m_grouped["std"].mean()
densities     = avg_mean.index / N

# === 4) Save CSV of mean & std vs M/density ===
out_csv = os.path.join(data_folder, "Mean_Loop_Times_per_M.csv")
summary = pd.DataFrame({
    "M":                avg_mean.index,
    "Density":          densities,
    "Mean Loop Time":   avg_mean.values,
    "Std Dev Loop Time":avg_std.values
})
summary.to_csv(out_csv, index=False)
print("Wrote summary CSV →", out_csv)

# === 5) Plot with dark-grey line + white circles ===
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":      12,
    "axes.titlesize":14,
    "axes.labelsize":12,
    "axes.edgecolor":"black",
    "axes.linewidth":1.0,
    "lines.linewidth":1.4,
    "text.usetex":   False,
})

# Marker & color settings
grey   = "#444444"
marker = 'o'

# Plot 1: Mean Loop Time vs Density
plt.figure(figsize=(6.5, 4.2))
plt.plot(
    densities,
    avg_mean.values,
    color=grey,
    marker=marker,
    markerfacecolor='white',
    markeredgecolor=grey,
    linestyle='-'
)
plt.xlabel("Density (vehicles per cell)")
plt.ylabel("Mean Loop Time [seconds]")
plt.title("Mean Loop Time vs. Density")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(graph_folder, "mean_loop_time_vs_density.png"),
    dpi=300
)
plt.close()

# Plot 2: StdDev of Loop Time vs Density
plt.figure(figsize=(6.5, 4.2))
plt.plot(
    densities,
    avg_std.values,
    color=grey,
    marker=marker,
    markerfacecolor='white',
    markeredgecolor=grey,
    linestyle='-'
)
plt.xlabel("Density (vehicles per cell)")
plt.ylabel("StdDev of Loop Time [seconds]")
plt.title("Standard Deviation of Loop Time vs. Density")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(graph_folder, "std_loop_time_vs_density.png"),
    dpi=300
)
plt.close()

print("Plots saved to:", graph_folder)



# -------------------------------------------------------------------------
# CREATE DELAY FILE AS CSV (CAR 1–25, DELAY = max(actual - mean, 0))
# -------------------------------------------------------------------------

import os
import re
import glob
import pandas as pd

# === Parameters ===
data_folder = "D:/Master Thesis/Thesis Car Simulation Code/NaSch + IDM Data"
output_csv  = os.path.join(data_folder, "Delay_Times_per_loop.csv")

# === 1) Discover all your Loop_Times_CarX.csv files ===
pattern = os.path.join(data_folder, "Loop_Times_Car*.csv")
files = sorted(glob.glob(pattern))
if not files:
    raise FileNotFoundError(f"No loop‐time CSVs found in {data_folder}")

# === 2) Read & stack them ===
df_list = []
for path in files:
    # extract car number from filename
    m = re.search(r"Loop_Times_Car(\d+)\.csv$", os.path.basename(path))
    if not m:
        continue
    car = int(m.group(1))

    df = pd.read_csv(path)
    long = (
        df.melt(
            id_vars=["Loop Index"],
            var_name="M",
            value_name="Loop Time"
        )
        .dropna(subset=["Loop Time"])
    )
    long["Loop Time"] = long["Loop Time"].astype(int)
    long["M"]         = long["M"].str.replace("M=", "").astype(int)
    long["Car"]       = car

    df_list.append(long)

big_df = pd.concat(df_list, ignore_index=True)

# === 3) Compute mean per (Car, M) and then delay ===
big_df["Mean Loop Time"] = big_df.groupby(["Car","M"])["Loop Time"].transform("mean")
big_df["Delay"] = (
    (big_df["Loop Time"] - big_df["Mean Loop Time"])
    .clip(lower=0)
    .astype(int)
)

# === 4) Save just (Car, M, Loop Index, Delay) ===
out = big_df[["Car","M","Loop Index","Delay"]]
out.to_csv(output_csv, index=False)

print(f"Delay data written to: {output_csv}")




# -------------------------------------------------------------------------
# PLOT DELAY TIMES OF CAR 1 FOR ALL M IN ONE GRAPH
# -------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import os

# === File paths ===
base_folder = "D:/Master Thesis/Thesis Car Simulation Code"
data_file = os.path.join(base_folder, "NaSch + IDM Data", "Delay_Times_per_loop.csv")
graph_folder = os.path.join(base_folder, "NaSch + IDM Graphs")
os.makedirs(graph_folder, exist_ok=True)

# === Load CSV and filter for Car 1 ===
df = pd.read_csv(data_file)
df_car1 = df[df["Car"] == 1]

# === Plot Style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.2,
    "text.usetex": False
})

# === Plot ===
plt.figure(figsize=(8, 5))

for m_val, group in df_car1.groupby("M"):
    plt.plot(group["Loop Index"], group["Delay"], label=f"M={m_val}", linestyle='-')

plt.xlabel("Lap Number")
plt.ylabel("Delay (Time Steps)")
plt.title("Delay of Car 1 per Lap for Different M Values")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

plt.tight_layout()

# === Save ===
output_path = os.path.join(graph_folder, "delay_times_car1.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved to: {output_path}")



# -------------------------------------------------------------------------
# PLOT SORTED DELAY TIMES OF CAR 1 FOR ALL M IN ONE GRAPH
# -------------------------------------------------------------------------


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === File paths ===
base_folder = "D:/Master Thesis/Thesis Car Simulation Code"
data_file = os.path.join(base_folder, "NaSch + IDM Data", "Delay_Times_per_loop.csv")
graph_folder = os.path.join(base_folder, "NaSch + IDM Graphs")
os.makedirs(graph_folder, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(data_file)

# === Plot Style ===
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     12,
    "axes.titlesize":14,
    "axes.labelsize":12,
    "xtick.labelsize":11,
    "ytick.labelsize":11,
    "legend.fontsize":8,
    "axes.edgecolor":"black",
    "axes.linewidth":1.0,
    "lines.linewidth":0.0,   # no connecting lines
    "text.usetex":   False
})

plt.figure(figsize=(8, 5))

# For each M, sort delays across ALL cars combined, then plot
for m_val, group in df.groupby("M"):
    delays_sorted = np.sort(group["Delay"].values)
    ranks         = np.arange(1, len(delays_sorted) + 1)
    plt.plot(
        ranks,
        delays_sorted,
        marker='.',        # small dot
        linestyle='none',  # no lines
        markersize=3,
        label=f"M={m_val}"
    )

plt.xlabel("Sorted Instance Rank")
plt.ylabel("Delay [seconds]")
plt.title("Sorted Delay for Different M Values")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=False)

plt.tight_layout()

# === Save ===
output_path = os.path.join(graph_folder, "sorted_delay_times_all_cars.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved to: {output_path}")




# -------------------------------------------------------------------------
# AVALANCHE DATA FOR DIFFERENT M: SIZE & DURATION OF DELAY AVALANCHES
# -------------------------------------------------------------------------


import os
import pandas as pd
import numpy as np

# === Paths (note the updated input filename) ===
base_dir          = "D:/Master Thesis/Thesis Car Simulation Code"
data_file         = os.path.join(base_dir, "NaSch + IDM Data", "Delay_Times_per_loop.csv")
output_durations  = os.path.join(base_dir, "NaSch + IDM Data", "Avalanche_Durations.csv")
output_sizes      = os.path.join(base_dir, "NaSch + IDM Data", "Avalanche_Sizes.csv")

# === Load the flat delay file ===
df = pd.read_csv(data_file)

# === Prepare accumulators ===
duration_records = []
size_records     = []

# === For each Car, M combination ===
for (car, M), group in df.groupby(["Car", "M"]):
    # sort by loop index and get the Delay values as a numpy array
    series = group.sort_values("Loop Index")["Delay"].values

    # find all indices where delay == 0 (these mark the boundaries)
    zeros = np.where(series == 0)[0]

    # walk each adjacent pair of zeros
    avalanche_idx = 1
    for start_zero, end_zero in zip(zeros, zeros[1:]):
        # only consider if there is at least one positive delay between
        if end_zero > start_zero + 1:
            # slice out just the positive‐delay region
            segment = series[start_zero+1 : end_zero]
            duration = len(segment)                  # how many time‐steps
            size     = np.trapz(segment, dx=1)       # approximate area under the delay curve

            duration_records.append({
                "Car": car,
                "M": M,
                "Avalanche Index": avalanche_idx,
                "Duration": duration
            })
            size_records.append({
                "Car": car,
                "M": M,
                "Avalanche Index": avalanche_idx,
                "Size": size
            })

            avalanche_idx += 1

# === Write out results exactly as before ===
pd.DataFrame(duration_records).to_csv(output_durations, index=False)
pd.DataFrame(size_records).to_csv(output_sizes,     index=False)

print(f"Saved avalanche durations to: {output_durations}")
print(f"Saved avalanche sizes to:     {output_sizes}")




# -------------------------------------------------------------------------
# AVALANCHE PLOTS: MEAN DURATION VS DENSITY AND MEAN SIZE VS DENSITY
# -------------------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import os

# === Paths ===
base_dir     = "D:/Master Thesis/Thesis Car Simulation Code"
data_folder  = os.path.join(base_dir, "NaSch + IDM Data")
graph_folder = os.path.join(base_dir, "NaSch + IDM Graphs")
os.makedirs(graph_folder, exist_ok=True)

duration_file = os.path.join(data_folder, "Avalanche_Durations.csv")
size_file     = os.path.join(data_folder, "Avalanche_Sizes.csv")

# === Load CSVs ===
df_durations = pd.read_csv(duration_file)
df_sizes     = pd.read_csv(size_file)

# === Compute Density (vehicles per cell) ===
df_durations["Density"] = df_durations["M"] / 1000
df_sizes["Density"]     = df_sizes["M"] / 1000

# === Summaries: mean over all cars & avalanches at each density ===
duration_summary = (
    df_durations
    .groupby("Density")["Duration"]
    .mean()
    .reset_index()
)
size_summary = (
    df_sizes
    .groupby("Density")["Size"]
    .mean()
    .reset_index()
)

# === Global plot style ===
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":      12,
    "axes.titlesize":14,
    "axes.labelsize":12,
    "axes.edgecolor":"black",
    "axes.linewidth":1.0,
    "lines.linewidth":1.4,
    "text.usetex":   False,
})

# === Marker & color settings ===
grey   = "#444444"
marker = 'o'

# === Plot 1: Mean Avalanche Duration vs Density ===
plt.figure(figsize=(6.5, 4.2))
plt.plot(
    duration_summary["Density"],
    duration_summary["Duration"],
    color=grey,
    marker=marker,
    markerfacecolor='white',
    markeredgecolor=grey,
    linestyle='-'
)
plt.xlabel("Density (vehicles per cell)")
plt.ylabel("Mean Avalanche Duration (time steps)")
plt.title("Mean Avalanche Duration vs. Density")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(graph_folder, "mean_avalanche_duration_vs_density.png"),
    dpi=300
)
plt.close()

# === Plot 2: Mean Avalanche Size vs Density ===
plt.figure(figsize=(6.5, 4.2))
plt.plot(
    size_summary["Density"],
    size_summary["Size"],
    color=grey,
    marker=marker,
    markerfacecolor='white',
    markeredgecolor=grey,
    linestyle='-'
)
plt.xlabel("Density (vehicles per cell)")
plt.ylabel("Mean Avalanche Size (area)")
plt.title("Mean Avalanche Size vs. Density")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(
    os.path.join(graph_folder, "mean_avalanche_size_vs_density.png"),
    dpi=300
)
plt.close()

print("Plots saved to:", graph_folder)




#####################################################################
# PANEL of 6 for my thesis (with log–log for mean & critical arrow only in (b))
#####################################################################

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Paths ===
base_dir      = "D:/Master Thesis/Thesis Car Simulation Code"
data_folder   = os.path.join(base_dir, "NaSch + IDM Data")
graph_folder  = os.path.join(base_dir, "NaSch + IDM Graphs")
progress_csv  = os.path.join(graph_folder, "density_speed_flow_progress.csv")
os.makedirs(graph_folder, exist_ok=True)

# === Compute critical density from progress CSV ===
df_prog     = pd.read_csv(progress_csv).sort_values("density")
flow_sorted = df_prog.sort_values("flow", ascending=False).reset_index(drop=True)
rho_c       = flow_sorted.loc[0, "density"]  

# === Global plot style ===
plt.rcParams.update({
    "font.family":   "serif",
    "font.size":      12,
    "axes.titlesize":14,
    "axes.labelsize":12,
    "xtick.labelsize":11,
    "ytick.labelsize":11,
    "legend.fontsize":8,
    "axes.edgecolor":"black",
    "axes.linewidth":1.0,
    "lines.linewidth":1.2,
    "text.usetex":   False
})

grey   = "#444444"
marker = 'o'

# === Load data for each subplot ===

# a) Loop Times of Car 1
df_car1 = pd.read_csv(os.path.join(data_folder, "Loop_Times_Car1.csv"))
df_car1.set_index("Loop Index", inplace=True)
max_M      = max(int(c.split('=')[1]) for c in df_car1.columns)
last_valid = df_car1[f"M={max_M}"].last_valid_index()

# b,c) Mean & Std Dev Loop Time across cars
summary    = pd.read_csv(os.path.join(data_folder, "Mean_Loop_Times_per_M.csv"))
densities  = summary["Density"].values
mean_loop  = summary["Mean Loop Time"].values
std_loop   = summary["Std Dev Loop Time"].values

# compute y‐value at rho_c for the mean plot arrow
mean_at_rho_c = np.interp(rho_c, densities, mean_loop)

# d) Sorted Delay Times
df_delay = pd.read_csv(os.path.join(data_folder, "Delay_Times_per_loop.csv"))

# e,f) Avalanche summaries
df_dur      = pd.read_csv(os.path.join(data_folder, "Avalanche_Durations.csv"))
df_dur["Density"] = df_dur["M"] / 1000
dur_summary = df_dur.groupby("Density")["Duration"].mean().reset_index()

df_size      = pd.read_csv(os.path.join(data_folder, "Avalanche_Sizes.csv"))
df_size["Density"] = df_size["M"] / 1000
size_summary = df_size.groupby("Density")["Size"].mean().reset_index()

# === Create 2×3 panel ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (a) Loop Times of Car 1
ax = axes[0, 0]
for col in df_car1.columns:
    ax.plot(
        df_car1.index[df_car1.index <= last_valid],
        df_car1[col][df_car1.index <= last_valid],
        linestyle='-',
        label=col
    )
ax.set_xlim(1, last_valid)
ax.set_xlabel("Loop Index")
ax.set_ylabel("Loop Time [seconds]")
ax.set_title("a) Loop Times of Car 1 for Different Densities")
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# (b) Mean Loop Time vs. Density (log–log, with critical arrow)
ax = axes[0, 1]
ax.plot(
    densities, mean_loop,
    color=grey, marker=marker,
    markerfacecolor='white', markeredgecolor=grey,
    linestyle='-'
)
ax.set_xscale('log')
ax.set_yscale('log')
ax.axvline(rho_c, color='black', linestyle='--', linewidth=1)
ax.annotate(
     rf"$\rho_c = {rho_c:.2f}$",
    xy=(rho_c, mean_at_rho_c),
    xytext=(rho_c*1.2, mean_at_rho_c*0.85),   
    arrowprops=dict(arrowstyle='->', color='black', lw=1),
    fontsize=12
)

ax.set_xlabel("Density (vehicles per cell) (log)")
ax.set_ylabel("Mean Loop Time [seconds] (log)")
ax.set_title("b) Mean Loop Time vs. Density (log–log)")
ax.grid(True, linestyle='--', linewidth=0.5, which='both')

# (c) Std Dev Loop Time vs. Density (linear, no critical arrow)
ax = axes[0, 2]
ax.plot(
    densities, std_loop,
    color=grey, marker=marker,
    markerfacecolor='white', markeredgecolor=grey,
    linestyle='-'
)
ax.set_xlabel("Density (vehicles per cell)")
ax.set_ylabel("Std Dev Loop Time [seconds]")
ax.set_title("c) Std Dev of Loop Time vs. Density")
ax.grid(True, linestyle='--', linewidth=0.5)

# (d) Sorted Delay for Different M Values
ax = axes[1, 0]
for m_val, grp in df_delay.groupby("M"):
    sd = np.sort(grp["Delay"].values)
    rk = np.arange(1, len(sd)+1)
    ax.plot(
        rk, sd,
        marker='.', linestyle='none', markersize=3,
        label=f"M={m_val}"
    )
ax.set_xlabel("Sorted Instance Rank")
ax.set_ylabel("Delay [seconds]")
ax.set_title("d) Sorted Delay for Different M Values")
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

# (e) Mean Avalanche Duration vs. Density
ax = axes[1, 1]
ax.plot(
    dur_summary["Density"], dur_summary["Duration"],
    color=grey, marker=marker,
    markerfacecolor='white', markeredgecolor=grey,
    linestyle='-'
)
ax.set_xlabel("Density (vehicles per cell)")
ax.set_ylabel("Mean Avalanche Duration (time steps)")
ax.set_title("e) Mean Avalanche Duration vs. Density")
ax.grid(True, linestyle='--', linewidth=0.5)

# (f) Mean Avalanche Size vs. Density
ax = axes[1, 2]
ax.plot(
    size_summary["Density"], size_summary["Size"],
    color=grey, marker=marker,
    markerfacecolor='white', markeredgecolor=grey,
    linestyle='-'
)
ax.set_xlabel("Density (vehicles per cell)")
ax.set_ylabel("Mean Avalanche Size (area)")
ax.set_title("f) Mean Avalanche Size vs. Density")
ax.grid(True, linestyle='--', linewidth=0.5)

# Finalize
plt.tight_layout()
output_path = os.path.join(graph_folder, "panel_2x3_loglog_b_only_with_critical.png")
plt.savefig(output_path, dpi=300)
plt.close(fig)

print(f"Panel saved to: {output_path}")







