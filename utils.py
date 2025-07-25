# -*- coding: utf-8 -*-
"""
utils_minimal.py

Minimal utility functions for two-stage experiments.
Contains only the functions actually used in two_stage_experiments.py:
- compute_pickup_dropoff: Extract pickup and dropoff events from taxi trajectory data
- construct_second_stage_riders_graph: Construct graph for second stage riders
- tsrmm_evaluate: Evaluate two-stage robust matching with mean cost
"""

import pandas as pd
import numpy as np
import networkx as nx


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


def compute_pickup_dropoff(df_input):
    """
    Extract pickup and dropoff events from taxi trajectory data.
    
    Args:
        df_input: DataFrame with taxi trajectory data including 'taxi_id', 'date_time', 'occupied' columns
    
    Returns:
        DataFrame with pickup and dropoff events only
    """
    column_names = ['taxi_id', 'date_time', 'longitude', 'latitude', 'speed', 'direction', 'occupied', 'other']
    id_array = df_input['taxi_id'].unique()
    df_pickup_dropoff = pd.DataFrame({}, columns=column_names)
    
    for t_id in id_array:
        df_taxi = df_input.loc[df_input['taxi_id'] == t_id]
        df_taxi = df_taxi.sort_values(by=['date_time'])
        to_look_for = 0
        to_delete = []
        
        # Find state transitions (pickup/dropoff events)
        for i in df_taxi.index:
            if df_taxi['occupied'][i] == to_look_for:
                to_look_for = 1 - to_look_for
            else:
                to_delete.append(i)
        
        # Remove non-transition points
        df_taxi = df_taxi.drop(index=to_delete)
        df_pickup_dropoff = pd.concat([df_pickup_dropoff, df_taxi], ignore_index=True)
    
    return df_pickup_dropoff


def construct_second_stage_riders_graph(second_stage_scenarios):
    """
    Construct a graph for second stage riders.
    
    Args:
        second_stage_scenarios: List of DataFrames for second stage scenarios
    
    Returns:
        G: NetworkX graph with weighted edges between second stage riders
    """
    G = nx.Graph()
    second_stage_riders_sc = []
    scenario = []
    for i in range(len(second_stage_scenarios)):
        scenario = []
        for j in range(len(second_stage_scenarios[i])):
            scenario.append('S'+str(i) + '_' + str(j))
        second_stage_riders_sc.append(scenario)
        G.add_nodes_from(scenario, bipartite = i)
    for r1 in second_stage_riders_sc[0]:
        r1_id  = int(r1[3:])
        r1_df = second_stage_scenarios[0].iloc[r1_id]
        for r2 in second_stage_riders_sc[1]:
            r2_id  = int(r2[3:])
            r2_df = second_stage_scenarios[1].iloc[r2_id]
            w = -int(gps_dist(r1_df.longitude, r1_df.latitude, r2_df.longitude, r2_df.latitude))
            G.add_weighted_edges_from([(r1,r2,w)])
            
    return G


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