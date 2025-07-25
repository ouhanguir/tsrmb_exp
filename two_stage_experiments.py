#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
two_stage_experiments.py

Main experiment driver for two-stage robust matching experiments on taxi data.
- Loads and cleans raw taxi data
- Constructs graphs for matching
- Runs matching algorithms (robust, greedy, etc.)
- Evaluates and visualizes results

This file was adapted from read_data.py for clarity and maintainability.
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import utils as util
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pylab as plt
plt.style.use('ggplot')
import os
import glob
import datashader as ds
import datashader.transfer_functions as tf
import networkx as nx
from networkx.algorithms import bipartite
from functools import partial
from datashader.utils import export_image
from datashader.colors import colormap_select, Greys9, Hot, viridis, inferno

# Gurobi API options (for optimization)
options = {
    "WLSACCESSID": "c3937ebe-bb0c-4d11-9019-12773b62428c",
    "WLSSECRET": "0cb6a97a-99ba-497f-b239-15fe7ce960aa",
    "LICENSEID": 759319,
}

time1 = "19:00-19:01"
time2 = "19:01-19:02"

# ----------------------
# Data Preparation
# ----------------------
def construct_df_hr(days=['03', '10'], nb_future_stages=2, day_d=['17'], hr=21):
    """
    Loads and cleans taxi data for specified days and hour.
    Returns first stage, future stage, realized future stage dataframes, and available drivers.
    """
    column_names = ['taxi_id', 'date_time', 'longitude', 'latitude', 'speed', 'direction', 'occupied', 'other']
    future_stage_scenarios = [[] for _ in range(1, nb_future_stages)]
    stages = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    for day in days:
        filename = f'./work/shenzhen/all/TRK200909{day}.txt'
        df = pd.read_csv(filename, names=column_names, sep=',', dtype={'latitude': np.float64})
        df["taxi_id"] = df["taxi_id"].astype(str)
        df["occupied"] = pd.to_numeric(df["occupied"])
        df["latitude"] = pd.to_numeric(df["latitude"])
        # Remove rows with malformed longitude
        df_long_b = df.longitude.apply(lambda x: len(repr(x)))
        df = df.drop(df_long_b[df_long_b < 4].index.values)
        df_long = df.longitude.apply(lambda x: repr(x)[3])
        df = df.drop(df_long[df_long != '.'].index.values)
        df["longitude"] = pd.to_numeric(df["longitude"])
        df['date_time'] = pd.to_datetime(df.date_time)
        # Remove rows with invalid occupancy
        df_occup = df[(df.occupied != 0) & (df.occupied != 1)]
        df = df.drop(df_occup.index.values)
        df = df.drop_duplicates().sort_values(by=['date_time'])
        # Filter to city region
        maindt = df[(abs(df.longitude - 114.075) <= 0.075) & (abs(df.latitude - 22.54) <= 0.03)]
        # Build future stage scenarios
        for k in range(1, nb_future_stages):
            df_new = maindt.loc[(maindt.date_time >= f'2009-09-{day} {hr}:{stages[k-1]}:00') &
                                (maindt.date_time <= f'2009-09-{day} {hr}:{stages[k]}:00')]
            df_pickup_dropoff = util.compute_pickup_dropoff(df_new)
            df_pickup_dropoff = df_pickup_dropoff[df_pickup_dropoff['occupied'] == 1]
            future_stage_scenarios[k-1].append(df_pickup_dropoff)
        print(f"Future Stage Scenarios from Day {day} added.")
    # Realized future stages for a specific day
    realized_future_stages = []
    for day in day_d:
        filename = f'./work/shenzhen/all/TRK200909{day}.txt'
        df = pd.read_csv(filename, names=column_names, sep=',', dtype={'latitude': np.float64})
        df["taxi_id"] = df["taxi_id"].astype(str)
        df["occupied"] = pd.to_numeric(df["occupied"])
        df["latitude"] = pd.to_numeric(df["latitude"])
        df_long_b = df.longitude.apply(lambda x: len(repr(x)))
        df = df.drop(df_long_b[df_long_b < 4].index.values)
        df_long = df.longitude.apply(lambda x: repr(x)[3])
        df = df.drop(df_long[df_long != '.'].index.values)
        df["longitude"] = pd.to_numeric(df["longitude"])
        df['date_time'] = pd.to_datetime(df.date_time)
        df_occup = df[(df.occupied != 0) & (df.occupied != 1)]
        df = df.drop(df_occup.index.values)
        df = df.drop_duplicates().sort_values(by=['date_time'])
        maindt = df[(abs(df.longitude - 114.075) <= 0.075) & (abs(df.latitude - 22.54) <= 0.03)]
        available_drivers_geo = pd.DataFrame()
        df_new = maindt.loc[(maindt.date_time >= f'2009-09-{day} {hr}:00:00') &
                            (maindt.date_time <= f'2009-09-{day} {hr}:01:00')]
        df_new_drivers = maindt.loc[(maindt.date_time >= f'2009-09-{day} {hr-1}:54:00') &
                                    (maindt.date_time <= f'2009-09-{day} {hr}:04:00')]
        df_pickup_dropoff = util.compute_pickup_dropoff(df_new)
        available_drivers = df_new_drivers[df_new_drivers['occupied'] == 0]
        available_drivers_ids = available_drivers.taxi_id.unique()
        for t_id in available_drivers_ids:
            t_df = available_drivers[available_drivers['taxi_id'] == t_id]
            available_drivers_geo = pd.concat([available_drivers_geo, t_df.head(1)], ignore_index=True)
        first_stage = df_pickup_dropoff[df_pickup_dropoff['occupied'] == 1]
        for k in range(1, nb_future_stages):
            df_new_future_stage = maindt.loc[(maindt.date_time >= f'2009-09-{day} {hr}:{stages[k-1]}:00') &
                                             (maindt.date_time <= f'2009-09-{day} {hr}:{stages[k]}:00')]
            df_pickup_dropoff_future = util.compute_pickup_dropoff(df_new_future_stage)
            df_pickup_dropoff_future = df_pickup_dropoff_future[df_pickup_dropoff_future['occupied'] == 1]
            realized_future_stages.append(df_pickup_dropoff_future)
        print("first and realized future stages computed")
        return first_stage, future_stage_scenarios, realized_future_stages, available_drivers_geo


def construct_df(days=['03', '10'], nb_future_stages=2, day_d=['17'], hr='21'):
    """
    Alternative data loading function with different parameter handling.
    Similar to construct_df_hr but with string hour parameter and different data processing.
    """
    column_names = ['taxi_id', 'date_time', 'longitude', 'latitude', 'speed', 'direction', 'occupied', 'other']
    future_stage_scenarios = [[] for k in range(1, nb_future_stages)]
    stages = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    
    # Load and process data for scenario building days
    for day in days:
        filename = f'./work/shenzhen/all/TRK200909{day}.txt'
        df = pd.read_csv(filename, names=column_names, sep=',', dtype={'latitude': np.float64})
        df["taxi_id"] = df["taxi_id"].apply(lambda x: str(x))
        df["occupied"] = pd.to_numeric(df["occupied"])
        df["latitude"] = pd.to_numeric(df["latitude"])
        
        # Remove rows with malformed longitude
        df_long_b = df.longitude.apply(lambda x: len(repr(x)))
        df_long_b = df_long_b[df_long_b < 4].index.values
        df = df.drop(df_long_b)
        df_long = df.longitude.apply(lambda x: repr(x)[3])
        df_long = df_long[df_long != '.']
        to_delete = df_long.index.values
        df = df.drop(to_delete)
        df["longitude"] = pd.to_numeric(df["longitude"])
        df['date_time'] = pd.to_datetime(df.date_time)
        
        # Remove rows with invalid occupancy
        df_occup = df[df.occupied != 0]
        df_occup = df_occup[df_occup.occupied != 1]
        to_delete = df_occup.index.values
        df = df.drop(to_delete)
        df = df.drop_duplicates()
        df = df.sort_values(by=['date_time'])
        
        # Filter to city region
        maindt = df.copy()
        maindt = maindt[abs(maindt.longitude - 114.075) <= 0.075]
        maindt = maindt[abs(maindt.latitude - 22.54) <= 0.03]
        
        # Build future stage scenarios
        for k in range(1, nb_future_stages):
            df_new = maindt.loc[(maindt.date_time >= f'2009-09-{day} 21:{stages[k-1]}:00') & 
                                (maindt.date_time <= f'2009-09-{day} 21:{stages[k]}:00')]
            df_pickup_dropoff = util.compute_pickup_dropoff(df_new)
            df_pickup_dropoff = df_pickup_dropoff[df_pickup_dropoff['occupied'] == 1]
            future_stage_scenarios[k-1].append(df_pickup_dropoff)
        
        print(f"Future Stage Scenarios from Day {day} added.")
    
    # Load and process data for realized future stages
    realized_future_stages = []
    for day in day_d:
        filename = f'./work/shenzhen/all/TRK200909{day}.txt'
        df = pd.read_csv(filename, names=column_names, sep=',', dtype={'latitude': np.float64})
        print("Dataframe read")
       
        df["taxi_id"] = df["taxi_id"].apply(lambda x: str(x))
        df["occupied"] = pd.to_numeric(df["occupied"])
        df["latitude"] = pd.to_numeric(df["latitude"])
        
        # Remove rows with malformed longitude
        df_long_b = df.longitude.apply(lambda x: len(repr(x)))
        df_long_b = df_long_b[df_long_b < 4].index.values
        df = df.drop(df_long_b)
        df_long = df.longitude.apply(lambda x: repr(x)[3])
        df_long = df_long[df_long != '.']
        to_delete = df_long.index.values
        df = df.drop(to_delete)
        df["longitude"] = pd.to_numeric(df["longitude"])
        df['date_time'] = pd.to_datetime(df.date_time)
        
        # Remove rows with invalid occupancy
        df_occup = df[df.occupied != 0]
        df_occup = df_occup[df_occup.occupied != 1]
        to_delete = df_occup.index.values
        df = df.drop(to_delete)
        df = df.drop_duplicates()
        df = df.sort_values(by=['date_time'])
        print("Dataframe typed and sorted")
        
        # Filter to city region
        maindt = df.copy()
        maindt = maindt[abs(maindt.longitude - 114.075) <= 0.075]
        maindt = maindt[abs(maindt.latitude - 22.54) <= 0.03]
        
        # Extract available drivers and first stage data
        available_drivers_geo = pd.DataFrame()
        df_new = maindt.loc[(maindt.date_time >= f'2009-09-{day} 21:00:00') & 
                            (maindt.date_time <= f'2009-09-{day} 21:01:00')]
        df_new_drivers = maindt.loc[(maindt.date_time >= f'2009-09-{day} 20:54:00') & 
                                    (maindt.date_time <= f'2009-09-{day} 21:04:00')]
        df_pickup_dropoff = util.compute_pickup_dropoff(df_new)
        available_drivers = df_new_drivers[df_new_drivers['occupied'] == 0]
        available_drivers_ids = available_drivers.taxi_id.unique()
        
        # Get one record per available driver
        for t_id in available_drivers_ids:
            t_df = available_drivers[available_drivers['taxi_id'] == t_id]
            available_drivers_geo = pd.concat([available_drivers_geo, t_df.head(1)], ignore_index=True)
        
        first_stage = df_pickup_dropoff[df_pickup_dropoff['occupied'] == 1]
        
        # Build realized future stages
        for k in range(1, nb_future_stages):  
            df_new_future_stage = maindt.loc[(maindt.date_time >= f'2009-09-{day} 21:{stages[k-1]}:00') & 
                                             (maindt.date_time <= f'2009-09-{day} 21:{stages[k]}:00')]
            df_pickup_dropoff_future = util.compute_pickup_dropoff(df_new_future_stage)
            df_pickup_dropoff_future = df_pickup_dropoff_future[df_pickup_dropoff_future['occupied'] == 1]
            realized_future_stages.append(df_pickup_dropoff_future)
        
        print("first and realized future stages computed")
        return first_stage, future_stage_scenarios, realized_future_stages, available_drivers_geo


def gps_dist(a, b, c, d):
    """
    Compute the distance (in meters) between two GPS locations.
    
    Args:
        a: longitude of first point
        b: latitude of first point  
        c: longitude of second point
        d: latitude of second point
    
    Returns:
        Distance in meters using Haversine formula
    """
    r = 0.0174533  # 1 degree in radians
    return 2 * 6371000 * np.arcsin(np.sqrt(  # https://en.wikipedia.org/wiki/Haversine_formula
        np.sin(r*(d - b)/2.0)**2 + np.cos(r*b) * np.cos(r*d) * np.sin(r*(c - a)/2.0)**2))


def graph_construction(available_drivers, first_stage, second_stage_scenarios):
    """
    Constructs a bipartite graph for matching drivers to riders.
    
    Args:
        available_drivers: DataFrame of available drivers
        first_stage: DataFrame of first stage riders
        second_stage_scenarios: List of DataFrames for second stage scenarios
    
    Returns:
        G: NetworkX graph with weighted edges
        drivers: List of driver IDs
        first_stage_riders: List of first stage rider IDs
        second_stage_sc: List of second stage scenario rider IDs
        opt2_range: List of edge weights for second stage
    """
    G = nx.Graph()
    opt2_range = []
    drivers = available_drivers.taxi_id.unique()
    G.add_nodes_from(drivers, bipartite=0)
    first_stage_riders = ['R' + str(i) for i in range(len(first_stage))]
    G.add_nodes_from(first_stage_riders, bipartite=1)
    
    # Create second stage rider nodes
    second_stage_riders = []
    second_stage_sc = []
    for i in range(len(second_stage_scenarios)):
        s = []
        for j in range(len(second_stage_scenarios[i])):
            s.append('S' + str(i) + '_' + str(j))
        second_stage_sc.append(s)
        G.add_nodes_from(s, bipartite=1)
    
    # Add weighted edges between drivers and riders
    for d in drivers:
        d_df = available_drivers[available_drivers['taxi_id'] == d].iloc[0]
        d_long = d_df.longitude
        d_lat = d_df.latitude
        
        # Add edges to first stage riders
        for r1 in first_stage_riders:
            r1_id = int(r1[1:])
            r1_df = first_stage.iloc[r1_id]
            w = -int(gps_dist(d_long, d_lat, r1_df.longitude, r1_df.latitude))
            G.add_weighted_edges_from([(d, r1, w)])
        
        # Add edges to second stage riders
        for i in range(len(second_stage_sc)):
            for r2 in second_stage_sc[i]:
                r2_id = int(r2[3:])
                r2_df = second_stage_scenarios[i].iloc[r2_id]
                w = -int(gps_dist(d_long, d_lat, r2_df.longitude, r2_df.latitude))
                opt2_range.append(-w)
                G.add_weighted_edges_from([(d, r2, w)])
    
    return G, drivers, first_stage_riders, second_stage_sc, opt2_range


# ----------------------
# Missing Functions from Original read_data.py
# ----------------------

def bottleneck_matching_graph_bs(graph, a, b, drivers):
    """
    Find bottleneck matching using binary search on edge weights.
    
    Args:
        graph: NetworkX graph
        a: Number of drivers
        b: Number of riders
        drivers: List of driver IDs
    
    Returns:
        M_min: Optimal matching
        w_min: Minimum bottleneck weight
    """
    G = graph.copy()
    w_arr = []
    w_dic = {}
    M_dic = {}
    M_min = []
    w_min = -float('inf')
    
    # Collect all edge weights
    for e in G.edges():
        w = G[e[0]][e[1]]['weight']
        w_dic.update({e: w})
    
    # Sort unique weights in descending order
    w_arr = list(w_dic.values())
    w_arr_bs = w_arr
    w_arr_bs = set(w_arr_bs)
    w_arr_bs = list(w_arr_bs)
    w_arr_bs.sort(reverse=True)
    
    # Binary search setup
    low = 0
    high = len(w_arr_bs) - 1
    mid = (high + low) // 2
    
    # Focus on top half of weights
    w_arr_bs = w_arr_bs[low:mid]
    high = len(w_arr_bs) - 1
    mid = (high + low) // 2
    
    # Binary search for optimal bottleneck weight
    while w_arr_bs:
        try:
            w = w_arr_bs[mid]
            G_w = G.copy()
            to_delete = [e for e, w_e in w_dic.items() if w_e < w]
            G_w.remove_edges_from(to_delete)
            
            # Try to find perfect matching
            try:
                M_dic = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G_w, top_nodes=drivers)
                if len(M_dic)/2 == np.min([a, b]):
                    M_min = M_dic
                    w_min = w
                    high = mid - 1
                else:
                    low = mid + 1
            except:
                print("Problem")
            
            w_arr_bs = w_arr_bs[low:high+1]
            low = 0
            high = len(w_arr_bs) - 1
            mid = (high + low) // 2
        except IndexError:
            print("indexError")
            break
    
    return M_min, w_min


def bottleneck_matching_graph_bs_single(graph, a, b, drivers, threshold_opt1=None, w_2=None, wtot_min=None):
    """
    Find bottleneck matching using binary search on edge weights (single scenario version).
    
    Args:
        graph: NetworkX graph
        a: Number of drivers
        b: Number of riders
        drivers: List of driver IDs
        threshold_opt1: Optional threshold for first stage weights
        w_2: Optional second stage weight
        wtot_min: Optional minimum total weight
    
    Returns:
        M_min: Optimal matching
        w_min: Minimum bottleneck weight
    """
    G = graph.copy()
    w_arr = []
    w_dic = {}
    M_min = []
    M_dic = {}
    w_min = -float('inf')
    
    # Collect all edge weights
    for e in G.edges():
        w = G[e[0]][e[1]]['weight']
        w_dic.update({e: w})
    
    # Sort unique weights in descending order
    w_arr = list(w_dic.values())
    w_arr_bs = w_arr
    w_arr_bs = set(w_arr_bs)
    w_arr_bs = list(w_arr_bs)
    w_arr_bs.sort(reverse=True)
    
    # Binary search setup
    low = 0
    high = len(w_arr_bs) - 1
    mid = (high + low) // 4
    
    # Focus on top quarter of weights
    w_arr_bs = w_arr_bs[low:mid]
    high = len(w_arr_bs) - 1
    mid = (high + low) // 2
    
    # Binary search for optimal bottleneck weight
    while w_arr_bs:
        try:
            w = w_arr_bs[mid]
            G_w = G.copy()
            to_delete = [e for e, w_e in w_dic.items() if w_e < w]
            G_w.remove_edges_from(to_delete)
            
            # Try to find perfect matching
            M_dic = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G_w, top_nodes=drivers)
            if len(M_dic)/2 == np.min([a, b]):
                M_min = M_dic
                w_min = w
                high = mid - 1
            else:
                if w + w_2 <= wtot_min and w_2 is not None and wtot_min is not None:
                    break
                low = mid + 1
            
            w_arr_bs = w_arr_bs[low:high+1]
            low = 0
            high = len(w_arr_bs) - 1
            mid = (high + low) // 2
        except IndexError:
            print("indexError")
            break
    
    return M_min, w_min


def tsrmb_single_scenario_graph(graph, r1, b,  second_stage_riders, drivers, threshold_opt1=None, threshold_opt2=None, threshold_cost=None):
    """
    Two-stage robust matching with bottleneck constraint for a single scenario.
    
    Args:
        graph: NetworkX graph
        b: Number of first and second stage riders
        r1: Number of first stage riders
        second_stage_riders: List of second stage rider IDs
        drivers: List of driver IDs
        threshold_opt1: Optional threshold for first stage weights
        threshold_opt2: Optional threshold for second stage weights
        threshold_cost: Optional minimum total cost
    
    Returns:
        D_1: Optimal first stage decision
        first_stage_min: First stage cost
        second_stage_min: Second stage cost
    """
    G = graph.copy()
    w_arr = []
    first_stage_min = -float('inf')
    second_stage_min = -float('inf')
    total_cost_min = -float('inf')
    
    if threshold_cost is not None:
        total_cost_min = threshold_cost
    
    w_dic = {}
    second_stage_edges = []
    deletable = []
    
    # Filter edges based on second stage riders and thresholds
    for e in G.edges():
        if e[0] in second_stage_riders or e[1] in second_stage_riders:
            w = G[e[0]][e[1]]['weight']
            if threshold_opt2 is not None and w < threshold_opt2:
                deletable.append(e)
            else:
                w_dic.update({e: w})
                second_stage_edges.append(e)
    
    G.remove_edges_from(deletable)
    w_arr = list(w_dic.values())
    w_arr = set(w_arr)
    w_arr = list(w_arr)
    w_arr.sort(reverse=True)
    
    # Iterate through weights to find optimal solution
    while w_arr:
        try:
            w = w_arr.pop()
            G_w = G.copy()
            to_delete = [e for e, w_e in w_dic.items() if w_e < w]
            G_w.remove_edges_from(to_delete)
            
            # Set second stage edge weights to 0 for first stage optimization
            for e in second_stage_edges:
                if e in G_w.edges():
                    G_w[e[0]][e[1]]['weight'] = 0
            
            try:
                w_1 = 0
                M = nx.max_weight_matching(G_w, maxcardinality=True, weight='weight')
                for e in M:
                    w_e = G_w[e[0]][e[1]]['weight']
                    w_1 = w_1 + w_e
                w_1 = w_1 * 1.0 / r1
                
                if len(M) == b and w_1 + w >= total_cost_min:
                    first_stage_min = w_1
                    second_stage_min = w
                    total_cost_min = w_1 + w
                    D1 = [m[0] for m in M if m[1] in first_stage_riders] + [m[1] for m in M if m[0] in first_stage_riders]
            except TypeError:
                print("typerror")
                continue
        except IndexError:
            print("index error in single scenario")
            break
    
    return D1, first_stage_min, second_stage_min


def tsrmb_two_scenarios_graph_1try(graph, drivers, first_stage_riders, scenario1, scenario2, graph_second_stage, opt_2=None):
    """
    Two-stage robust matching for two scenarios with one attempt.
    
    Args:
        graph: NetworkX graph
        drivers: List of driver IDs
        first_stage_riders: List of first stage rider IDs
        scenario1: First scenario rider IDs
        scenario2: Second scenario rider IDs
        graph_second_stage: Second stage graph
        opt_2: guess for the optimal threshold of second stage
    
    Returns:
        D_1: First stage driver assignments
        first_cost: First stage cost
        second_cost: Second stage cost
    """
    G2 = graph_second_stage.copy()
    w2_dic = {}
    w2_G2_dic = {}
    M2 = {}
    second_stage_edges = []
    G2_to_delete = []
    G = graph.copy()
    
    # Collect edges for both scenarios
    for e in G.edges():
        if e[0] in scenario1 or e[1] in scenario1 or e[0] in scenario2 or e[1] in scenario2:
            w = G[e[0]][e[1]]['weight']
            w2_dic.update({e: w})
            second_stage_edges.append(e)
    
    # Filter second stage graph edges based on threshold
    for e in G2.edges():
        w2_G2 = G2[e[0]][e[1]]['weight']
        if w2_G2 >= 2*opt_2 and opt_2 is not None:
            w2_G2_dic.update({e: w2_G2})
        else:
            G2_to_delete.append(e)
    
    # Create filtered second stage graph
    G2_w2 = G2.copy()
    G2_w2.remove_edges_from(G2_to_delete)
    
    # Find matching in second stage graph
    M2 = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G2_w2, top_nodes=scenario1)
    try:
        S2_match = [r2 for r1, r2 in M2.items() if r1 in scenario1]
    except IndexError:
        print("IndexError in S2_match")
        S2_match = []
    
    # Remove matched riders from graph
    G.remove_nodes_from(S2_match)
    S2_unmatch = [r for r in scenario2 if r not in S2_match]
    second_stage_riders = []
    second_stage_riders = second_stage_riders + scenario1
    second_stage_riders = second_stage_riders + S2_unmatch
    
    # Solve the two-stage robust matching problem with a single scenario S1 U S2_unmatch
    # print(G)
    # D_1, first_cost, second_cost = tsrmb_single_scenario_graph(G, drivers, len(first_stage_riders), len(first_stage_riders)+len(second_stage_riders), [second_stage_riders])
    D_1, first_cost, second_cost, total_cost = ip_tsrmb(drivers, first_stage_riders, [second_stage_riders], G, True, s1_length=None)
    
    return D_1, first_cost, second_cost


def tsrmb_two_secnarios_alg(graph, drivers, first_stage_riders, scenario1, scenario2, graph_second_stage, w_range):
    """
    Two-stage robust matching algorithm for two scenarios.
    
    Args:
        graph: NetworkX graph
        drivers: List of driver IDs
        first_stage_riders: List of first stage rider IDs
        scenario1: First scenario rider IDs
        scenario2: Second scenario rider IDs
        graph_second_stage: Second stage graph
        w_range: Range of second stage OPT2 weights to try
    
    Returns:
        D_1_min: Optimal first stage driver assignments
        cost_min: Minimum total cost
        alg2: Second stage cost
    """
    cost_min = float('inf')
    alg2 = float('inf')
    D_1_min = []
    
    for w in w_range:
        # For every guess of OPT2, we run the algorithm
        print("w = " + str(w))
        D_1 = tsrmb_two_scenarios_graph_1try(graph, drivers, first_stage_riders, scenario1, scenario2, graph_second_stage, opt_2=-w)[0]
        evalu = tsrmb_evaluate(graph, D_1, drivers, first_stage_riders, [scenario1, scenario2])
        if evalu[0] < cost_min:
            D_1_min = D_1
            cost_min = evalu[0]
            alg2 = evalu[-1]
    
    return D_1_min, cost_min, alg2


def tsrmb_greedy(graph, drivers, first_stage_riders, scenarios):
    """
    Greedy algorithm for two-stage robust matching.
    
    Args:
        graph: NetworkX graph
        drivers: List of driver IDs
        first_stage_riders: List of first stage rider IDs
        scenarios: List of scenario rider IDs
    
    Returns:
        cost: Total cost
        first_st_cost: First stage cost
        worst_case_cost: Worst case second stage cost
        D_1: First stage driver assignments
    """
    G = graph.copy()
    
    # Remove all scenario riders from graph
    for scenario in scenarios:
        G.remove_nodes_from(scenario)
    
    # Find maximum weight matching
    M = nx.max_weight_matching(G, maxcardinality=True, weight='weight')
    
    # Extract first stage driver assignments
    D_1 = [m[0] for m in M if m[1] in first_stage_riders] + [m[1] for m in M if m[0] in first_stage_riders]
    
    # Evaluate the solution
    cost, first_st_cost, worst_case_cost = tsrmb_evaluate(graph, D_1, drivers, first_stage_riders, scenarios)
    
    return cost, first_st_cost, worst_case_cost, D_1


def tsrmb_evaluate(graph, first_stage_decision, drivers, first_stage_riders, scenarios):
    """
    Evaluate two-stage robust matching solution.
    
    Args:
        graph: NetworkX graph
        first_stage_decision: List of drivers assigned in first stage
        drivers: List of all driver IDs
        first_stage_riders: List of first stage rider IDs
        scenarios: List of scenario rider IDs
    
    Returns:
        total_cost: Total cost (first stage + worst case second stage)
        first_stage_cost: First stage cost
        worst_case_cost: Worst case second stage cost
    """
    worst_case_cost = 0
    second_stage_drivers = [d for d in drivers if d not in first_stage_decision]
    
    # Evaluate worst case second stage cost across all scenarios
    for i in range(len(scenarios)):
        G2 = graph.copy()
        G2.remove_nodes_from(first_stage_decision)
        G2.remove_nodes_from(first_stage_riders)
        for j in range(len(scenarios)):
            if i != j:
                G2.remove_nodes_from(scenarios[j])
        try:
            M, w2 = bottleneck_matching_graph_bs(G2, len(drivers), len(scenarios[i]), second_stage_drivers)
            if -w2 > worst_case_cost: 
                worst_case_cost = -w2
        except TypeError:
            continue
    
    # Calculate first stage cost
    w1 = 0
    G2 = graph.copy()
    G2.remove_nodes_from(second_stage_drivers)
    for j in range(len(scenarios)):
        G2.remove_nodes_from(scenarios[j])
    
    M = nx.max_weight_matching(G2, maxcardinality=True, weight='weight')
    for e in M:
        w_e = G2[e[0]][e[1]]['weight']
        w1 = w1 + w_e
    w1 = w1 * 1.0 / len(first_stage_decision)
    total_cost = -w1 + worst_case_cost
    
    return total_cost, -w1, worst_case_cost


def tsrmm_evaluate(graph, first_stage_decision, drivers, first_stage_riders, scenarios):
    """
    Evaluate two-stage robust matching with mean cost.
    
    Args:
        graph: NetworkX graph
        first_stage_decision: List of drivers assigned in first stage
        drivers: List of all driver IDs
        first_stage_riders: List of first stage rider IDs
        scenarios: List of scenario rider IDs
    
    Returns:
        total_cost: Total cost (first stage + worst case second stage)
        first_stage_cost: First stage cost
        worst_case_cost: Worst case second stage cost
    """
    worst_case_cost = 0
    second_stage_drivers = [d for d in drivers if d not in first_stage_decision]
    
    # Evaluate worst case second stage cost across all scenarios
    for i in range(len(scenarios)):
        G2 = graph.copy()
        G2.remove_nodes_from(first_stage_decision)
        G2.remove_nodes_from(first_stage_riders)
        for j in range(len(scenarios)):
            if i != j:
                G2.remove_nodes_from(scenarios[j])
        try:
            w2 = 0
            M = nx.max_weight_matching(G2, maxcardinality=True, weight='weight')
            for e in M:
                w_e = G2[e[0]][e[1]]['weight']
                w2 = w2 + w_e
            if -w2 > worst_case_cost: 
                worst_case_cost = -w2
        except TypeError:
            continue
    
    # Calculate first stage cost
    w1 = 0
    G2 = graph.copy()
    G2.remove_nodes_from(second_stage_drivers)
    for j in range(len(scenarios)):
        G2.remove_nodes_from(scenarios[j])
    
    M = nx.max_weight_matching(G2, maxcardinality=True, weight='weight')
    for e in M:
        w_e = G2[e[0]][e[1]]['weight']
        w1 = w1 + w_e
    
    total_cost = -w1 + worst_case_cost
    return total_cost, -w1, worst_case_cost


def ip_tsrmb(available_drivers, first_stage, second_stage_scenarios, G, flag_subroutine=False, s1_length=None):
    """
    Integer programming formulation for two-stage robust matching.
    
    Args:
        available_drivers: DataFrame of available drivers
        first_stage: DataFrame of first stage riders
        second_stage_scenarios: List of DataFrames for second stage scenarios
        G: NetworkX graph
        flag_subroutine: Flag for subroutine mode
        s1_length: Length parameter for scenario 1
    
    Returns:
        D1: List of drivers assigned in first stage
        first_stage_obj: First stage objective value
        second_stage_obj: Second stage objective value
        total_obj: Total objective value
    """
    with gp.Env(params=options) as env, gp.Model(env=env) as m:
        m.setParam("OutputFlag", 0)
        drivers = range(len(available_drivers))
        
        # Decision variables
        z = m.addVar(vtype=GRB.CONTINUOUS, name="z")
        x = [[m.addVar(vtype=GRB.BINARY, name='x_'+str(i) +',' + str(d)) for d in drivers] for i in range(len(first_stage))]
        y = [[[m.addVar(vtype=GRB.BINARY, name='y_'+str(i) +'_' + str(j) + ',' + str(d)) for d in drivers] for j in range(len(second_stage_scenarios[i]))] for i in range(len(second_stage_scenarios))]
        
        # Constraints for second stage bottleneck
        if flag_subroutine:
            for i in range(len(second_stage_scenarios)):
                for j in range(len(second_stage_scenarios[i])):
                    for d in drivers:
                        try:
                            dist_i_d = -G.get_edge_data('S'+str(i) + '_' + str(second_stage_scenarios[i][j][3:]), available_drivers[d])['weight']
                        except TypeError:
                            dist_i_d = -G.get_edge_data('S'+str(i+1) + '_' + str(second_stage_scenarios[i][j][3:]), available_drivers[d])['weight']
                        m.addConstr(z >= dist_i_d*y[i][j][d], "c_z_y_"+str(i) +'_' + str(j) + ',' + str(d))
        else:
            for i in range(len(second_stage_scenarios)):
                for j in range(len(second_stage_scenarios[i])):
                    for d in drivers:
                        dist_i_d = -G.get_edge_data('S'+str(i) + '_' + str(j), available_drivers[d])['weight']
                        m.addConstr(z >= dist_i_d*y[i][j][d], "c_z_y_"+str(i) +'_' + str(j) + ',' + str(d))
        
        # Every first stage rider needs to be matched once
        for i in range(len(first_stage)):
            cstr_i = 0
            for d in drivers:
                cstr_i += x[i][d]
            m.addConstr(cstr_i == 1, "match_x_"+str(i))
        
        # Every second stage rider needs to be matched once
        for i in range(len(second_stage_scenarios)):
            for j in range(len(second_stage_scenarios[i])):
                cstr_j = 0
                for d in drivers:
                    cstr_j += y[i][j][d]
                m.addConstr(cstr_j == 1, "match_y_"+str(i) +'_' + str(j))
        
        # Each driver can be matched at most once across all stages
        for s in range(len(second_stage_scenarios)):
            for d in drivers:
                cstr = 0
                for i in range(len(first_stage)):
                    cstr += x[i][d]
                for j in range(len(second_stage_scenarios[s])):
                    cstr += y[s][j][d]
                m.addConstr(cstr <= 1, "d_s_"+ str(s))
        
        # Objective function
        obj_function = z
        
        # Add first stage cost
        tmp = 0
        for i in range(len(first_stage)):
            for d in drivers:
                dist_i_d = -G.get_edge_data('R' + str(i), available_drivers[d])['weight']
                tmp += dist_i_d*x[i][d]
        
        obj_function += (1.0/len(first_stage))*tmp
        m.setObjective(obj_function, GRB.MINIMIZE)
        m.optimize()
        
        if m.Status == GRB.OPTIMAL:
            print("Optimal objective value:", m.ObjVal)
            total_obj = m.ObjVal
            second_stage_obj = m.getVarByName("z").X
            first_stage_obj = total_obj - second_stage_obj
            D1 = []
            for i in range(len(first_stage)):
                for d in drivers:
                    if m.getVarByName('x_'+str(i) +',' + str(d)).X > 0:
                        D1.append(available_drivers[d])
            
            return D1, first_stage_obj, second_stage_obj, total_obj
        else:
            print("No optimal solution found")
            return None, None, None, None


# ----------------------
# Main Experiment Loop
# ----------------------

nb_future_stages = 2
hr_arr = range(9, 22)
sim_df = pd.DataFrame(columns=['hr', 'ALG2', 'Greedy2', 'total_alg', 'total_greedy'])
n_iter = 15

# Run experiments for each hour
for h in range(9, 22):
    # Load data for current hour
    first_stage, future_stage_scenarios, realized_future_stages, available_drivers_geo = construct_df_hr(
        days=['03', '10'], nb_future_stages=2, day_d=['17'], hr=h)
    
    # Run multiple iterations for statistical significance
    for k in range(n_iter):
        # Sample available drivers
        available_drivers_geo_sample = available_drivers_geo.sample(150*(nb_future_stages+1))
        
        # Construct graphs for optimization and evaluation
        G_opt, drivers_opt, first_stage_riders_opt, second_stage_sc_opt, opt_opt2_range = graph_construction(
            available_drivers_geo_sample, first_stage, [realized_future_stages[0]])
        G, drivers, first_stage_riders, second_stage_sc, opt2_range = graph_construction(
            available_drivers_geo_sample, first_stage, future_stage_scenarios[0])
        G2 = util.construct_second_stage_riders_graph(future_stage_scenarios[0])
        
        # Define weight range for algorithm
        # w_range = list(set([round(x) for x in opt2_range]))

        #For quick testing, we can focus on lower percentile values, as the optimal value OPT2 is usually in the lower percentile
        w_range = [round(np.percentile(opt2_range, 5)), round(np.percentile(opt2_range, 2.5)), 
                  round(np.percentile(opt2_range, 2)), round(np.percentile(opt2_range, 10)), 
                  round(np.percentile(opt2_range, 7.5)), 200, 300, 400]
        
        # Run algorithms
        opt_s_star = ip_tsrmb(drivers_opt, first_stage, second_stage_sc_opt, G_opt)
        alg = tsrmb_two_secnarios_alg(G, drivers, first_stage_riders, second_stage_sc[0], second_stage_sc[1], G2, w_range)
        opt = ip_tsrmb(drivers, first_stage_riders, second_stage_sc, G)
        greedy = tsrmb_greedy(G, drivers, first_stage_riders, second_stage_sc)
        
        # Evaluate algorithms on realized scenario
        alg_realized = tsrmb_evaluate(G_opt, alg[0], drivers, first_stage_riders, second_stage_sc_opt)
        greedy_realized = tsrmb_evaluate(G_opt, greedy[-1], drivers, first_stage_riders, second_stage_sc_opt)
        opt_realized = tsrmb_evaluate(G_opt, opt[0], drivers, first_stage_riders, second_stage_sc_opt)
        
        # Compute total matching costs
        total_matching_alg = util.tsrmm_evaluate(G, alg[0], drivers, first_stage_riders, second_stage_sc)
        total_matching_greedy = util.tsrmm_evaluate(G, greedy[-1], drivers, first_stage_riders, second_stage_sc)
        alg2 = alg[-1]
        greedy2 = greedy[2]
        
        # Store results
        new_row = pd.DataFrame([[h, alg2, greedy2, total_matching_alg, total_matching_greedy]], columns=sim_df.columns)
        sim_df = pd.concat([sim_df, new_row], ignore_index=True)

# Save results to CSV
sim_df.to_csv('sim_df_table_2.csv')   



