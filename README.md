
# Traffic Simulation Models – NaSch and NaSch + IDM

This repository contains the Python code used for the traffic flow simulations in the MSc thesis _"Timeliness Criticality in Supply Chains"_ by Merel de Leur (2025). The simulations explore the dynamics of decentralized systems through two models:

- **NaSch Model**: A classical cellular automaton traffic model
- **NaSch + IDM Model**: A hybrid version incorporating heterogeneous driver behavior based on the Intelligent Driver Model (IDM)

---

## Features

- Simulate traffic with varying densities and driver types
- Generate position and velocity data
- Analyze loop times, delays, and avalanche effects
- Create publication-quality plots for thesis figures

---

## Structure

```
Thesis Car Simulations.py     # Main script: runs both NaSch and NaSch+IDM models
NaSch Data/                   # Output CSVs from NaSch simulation
NaSch Graphs/                 # Plots from NaSch analysis
NaSch + IDM Data/             # Output CSVs from NaSch+IDM simulation
NaSch + IDM Graphs/           # Plots from NaSch+IDM analysis
```

---

## Requirements

Install required Python libraries with:

```bash
pip install numpy matplotlib pandas
```

> Tested with Python 3.10+

---

## Running the Code

Make sure to adjust file paths in the script to match your local environment.

To run the full simulation and analysis pipeline:

```bash
python traffic_simulation.py
```

This will:
- Simulate traffic across a range of densities
- Validate simulation correctness at each step
- Save data and plots to the appropriate folders

---

## Key Outputs

- **Flow-density plots**: `density_vs_flow.png`
- **Speed-density plots**: `density_vs_speed.png`
- **Loop time statistics**: `mean_loop_time_vs_density.png`
- **Delay and avalanche metrics**: `delay_times_car1.png`, `mean_avalanche_duration_vs_density.png`
- **Thesis-ready panel figure**: `panel_2x3_loglog_b_only_with_critical.png`

---

## Model Notes

### NaSch Model
- Discrete time, discrete space
- Simple rules for acceleration, braking, and random slowing
- Idealized traffic dynamics for understanding congestion emergence

### NaSch + IDM Model
- Heterogeneous agents with differing risk profiles
- Driver-specific max speeds and time headways
- Models more realistic driver behavior while preserving cellular structure

---

## Description of the Code

The code runs a complete simulation and post-processing pipeline:

### 1. **Position Simulation**
- Each car is initialized with a position (`evenly spaced`) and a velocity (`v_max` or heterogeneous if IDM).
- In each time step (up to 1 million), cars update positions based on:

  #### For NaSch:
  - **Acceleration:** If `v < v_max`, increment speed by 1.
  - **Slowing Down:** Adjust to avoid collisions (gap between cars).
  - **Random Braking:** With probability `p`, reduce speed by 1.
  - **Movement:** Move forward by `v` cells, wrap around circularly.

  #### For NaSch+IDM:
  - Each car has:
    - A driver type (`risk-averse`, `normal`, `risk-seeking`)
    - A personal `desired speed` and `T` (safe time gap)
    - A personal `braking probability`
  - Each step:
    - Attempt to accelerate toward desired speed
    - Compute gap to car ahead and calculate `desired space`
    - If gap too small, reduce velocity accordingly
    - Apply random braking
    - Move car based on final velocity

  - After position updates, **validation checks**:
    - No backward moves
    - No collisions (unique positions)
    - Consistent ID ordering

- Car positions at each time step are saved to CSV (`Data_NaSch_*.csv` or `Data_NaSch_IDM_*.csv`)

---

### 2. **Velocity Extraction**
- From position files, the per-step velocity for each car is computed by the difference in position.
- Handles circular track wrap-around (`modulo N`).
- Results are saved in `car_velocity_*.csv`.

---

### 3. **Fundamental Diagram Analysis**
- From velocity data:
  - Compute **global mean speed** (space-mean) and **global flow** 
- Plot:
  - Speed vs. Density
  - Flow vs. Density
  - Mark critical density with max flow

---

### 4. **Loop Time Extraction**
- For car 1 (and 25 others), calculate how long it takes to complete one full loop (i.e., travel `N` cells cumulatively).
- Loop time data is saved per car across `M` values.
- Plots generated:
  - Loop time vs. loop index
  - Mean & Std deviation of loop time vs. density

---

### 5. **Delay and Avalanche Analysis**
- Define **delay** as: `actual_loop_time - mean_loop_time` (only if positive).
- Analyze:
  - Delay vs. loop index
  - Sorted delays (for all M)
  - **Avalanche events:** contiguous positive delays between zero delays
    - Extract **duration** (length) and **size** (area under curve)
    - Plot vs. density

---

### 6. **Panel Plots for Thesis**
- 2x3 panels combining:
  - Loop times
  - Delay plots
  - Avalanche statistics
  - All include scientific formatting and annotation of critical density

---

All results are saved in folders, and graphs are reproducible.

---

## Citation

If you use this code, please cite:

> de Leur, M. J. M. (2025). _Timeliness Criticality in Supply Chains_. MSc Finance Thesis. Vrije Universiteit Amsterdam.

Also cite the models used:

- Nagel, K., & Schreckenberg, M. (1992). A cellular automaton model for freeway traffic. *Journal de Physique I*, 2(12), 2221–2229. https://doi.org/10.1051/jp1:1992277  
- Tian, J., Jiang, R., Li, G., Treiber, M., Jia, B., & Zhu, C. (2016). Improved 2D intelligent driver model... *Transportation Research Part F*, 41, 55–65. https://doi.org/10.1016/j.trf.2016.06.005

---

## Contact

Merel de Leur  
mjm.deleur@gmail.com
