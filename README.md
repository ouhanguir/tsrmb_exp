# Two-Stage Robust Matching Experiments

This repository contains the numerical experiments for the paper "Matching Drivers to Riders: A Two-stage Robust Approach". The code implements various algorithms for robust matching problems using real taxi trajectory data from Shenzhen, China.


## Files Structure

### 1. `two_stage_experiments.py` - Main Experiment Driver
The primary file that orchestrates the two-stage robust matching experiments.

**Key Functions:**
- `construct_df_hr()`: Loads and cleans taxi data for specified days and hours
- `graph_construction()`: Builds bipartite graphs for matching drivers to riders
- `bottleneck_matching_graph_bs_single()`: Finds bottleneck matching using binary search
- `tsrmb_single_scenario_graph()`: Two-stage robust matching for a scenario
- `tsrmb_two_scenarios_graph_1try()`: Two-stage robust matching for two scenarios
- `tsrmb_greedy()`: Greedy algorithm for two-stage robust matching

**Main Experiment Loop:**
- Runs experiments for specific hours 
- Performs iterations per hour for statistical significance
- Compares multiple algorithms: optimal, greedy, and robust approaches
- Saves results to a csv file

### 2. `utils.py` - Utility Functions
Contains shared utility functions used by the experiment files.

**Key Functions:**
- `gps_dist()`: Computes distance between GPS coordinates using Haversine formula
- `compute_pickup_dropoff()`: Extracts pickup/dropoff events from taxi trajectory data
- `graph_construction()`: Builds bipartite graphs for matching
- `graph_construction_k()`: Multi-stage graph construction for k future stages
- `construct_second_stage_riders_graph()`: Creates graphs for second stage riders
- `bottleneck_matching_graph_bs()`: Bottleneck matching with binary search
- `tsrmb_evaluate()`: Evaluates two-stage robust matching solutions
- `tsrmm_evaluate()`: Evaluates two-stage robust matching with mean cost

### 3. `multi_stage_experiments.py` - Multi-Stage Experiments
Similar to `two_stage_experiments.py` but designed for experiments involving more than two future stages.

**Features:**
- Extends the two-stage framework to multiple stages
- Handles more complex uncertainty scenarios
- Implements multi-stage robust optimization algorithms

## Data Processing

### Input Data
- **Source**: Taxi trajectory data from Shenzhen (2009). The data is stored in [Baidu Netdesk](https://pan.baidu.com/s/1FlNO1CfXiyi15WLuXn_ucA) (code: w2uu) 
- **Format**: CSV files with columns: taxi_id, date_time, longitude, latitude, speed, direction, occupied, other
- **Location**: `./taxi_data/sz_taxi_data/TRK200909*.txt`

### Data Cleaning Steps
1. **Type Conversion**: Convert data types and handle malformed entries
2. **Geographic Filtering**: Focus on Shenzhen city center (114.075°E ± 0.075°, 22.54°N ± 0.03°)
3. **Occupancy Filtering**: Remove invalid occupancy values
4. **Pickup/Dropoff Extraction**: Identify state transitions in taxi occupancy
5. **Time Window Filtering**: Extract data for specific hours and time intervals

## Algorithms Implemented

### 1. Two-Stage Robust Matching (TSRM)
- **Objective**: Minimize worst-case total cost
- **First Stage**: Average cost of current matches
- **Second Stage**: Bottleneck cost of future matches

### 2. Two-Stage Robust Bottleneck Matching (TSRBB)
- **Objective**: Minimize worst-case bottleneck cost
- **Approach**: Uses binary search on edge weights

### 3. Greedy Algorithms
- **Simple Greedy**: Maximum weight matching ignoring future uncertainty
- **Robust Greedy**: Greedy approach with robust evaluation

### 4. Integer Programming (IP)
- **Solver**: Gurobi optimization solver
- **Formulation**: Mixed-integer programming for exact solutions


## Dependencies

- **Optimization**: `gurobipy` for integer programming
- **Data Processing**: `pandas`, `numpy`
- **Graph Algorithms**: `networkx`
- **Geospatial**: `geopandas`, `shapely`
