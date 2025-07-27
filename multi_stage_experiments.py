#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_stage_experiments.py

Experiments for multi-stage (more than two) robust matching on taxi data.
- Loads and processes taxi data for multiple stages
- Constructs graphs and runs multi-stage matching algorithms
- Evaluates and visualizes results
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
import bokeh, bokeh.plotting, bokeh.models
from bokeh.io import output_notebook, show
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
# Data Preparation for Multi-Stage Experiments
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


def construct_df(days= ['03', '10'], nb_future_stages = 2, day_d = ['17'], hr = '21'):

    #days = ['03', '10']
    column_names = ['taxi_id', 'date_time', 'longitude', 'latitude', 'speed', 'direction', 'occupied','other']
    #gdf_days = gpd.GeoDataFrame()
    #second_stage_scenarios = []
    future_stage_scenarios = [[] for k in range(1,nb_future_stages)]
    stages = ['01','02', '03', '04', '05', '06', '07', '08', '09','10']
    # k = 10
    for day in days:
        #filename = './TRK200909'+ day +'/work/shenzhen/all/TRK200909' + day + '.txt'
        filename = './work/shenzhen/all/TRK200909' + day + '.txt'
        #df_master = pd.concat(pd.read_csv(f, names=column_names) for f in taxi_files)  #glue all data into the dataframe
        df = pd.read_csv(filename, names=column_names, sep =',', dtype = {'latitude':np.float64})
        #df['date_time'] = pd.to_datetime(df.date_time) # Correct the type in date_time column
        df["taxi_id"] = df["taxi_id"].apply(lambda x: str(x))
        taxi_id_values = df['taxi_id'].unique()
        df["occupied"] = pd.to_numeric(df["occupied"])
        df["latitude"] = pd.to_numeric(df["latitude"])
        df_long_b = df.longitude.apply(lambda x:len(repr(x)))
        df_long_b = df_long_b[df_long_b <4].index.values
        df = df.drop(df_long_b)
        df_long = df.longitude.apply(lambda x:repr(x)[3])
        df_long = df_long[df_long != '.']
        to_delete = df_long.index.values
        df = df.drop(to_delete)
        df["longitude"] = pd.to_numeric(df["longitude"])
        df['date_time'] = pd.to_datetime(df.date_time)
        df_occup = df[df.occupied != 0]
        df_occup = df_occup[df_occup.occupied != 1]
        to_delete = df_occup.index.values
        df = df.drop(to_delete)
        df = df.drop_duplicates()
        df = df.sort_values(by=['date_time'])
        
        maindt = df.copy()
        
        maindt = maindt[abs(maindt.longitude -114.075) <= 0.075]
        maindt = maindt[abs(maindt.latitude - 22.54) <= 0.03]
  
        for k in range(1,nb_future_stages):
            df_new = maindt.loc[(maindt.date_time >='2009-09-'+ day +' 21:' + stages[k-1] +':00') & (maindt.date_time <='2009-09-' + day +' 21:' + stages[k] +':00')]
            df_pickup_dropoff = util.compute_pickup_dropoff(df_new)
            df_pickup_dropoff = df_pickup_dropoff[df_pickup_dropoff['occupied']==1]
            future_stage_scenarios[k-1].append(df_pickup_dropoff)
        
        print("Future Stage Scenarios from Day " + str(day) + " added.")
    
    realized_future_stages = []

    for day in day_d:
        filename = './work/shenzhen/all/TRK200909' + day + '.txt'
        df = pd.read_csv(filename, names=column_names, sep =',', dtype = {'latitude':np.float64})
        print("Dataframe read")
       
        df["taxi_id"] = df["taxi_id"].apply(lambda x: str(x))
        taxi_id_values = df['taxi_id'].unique()
        df["occupied"] = pd.to_numeric(df["occupied"])
        df["latitude"] = pd.to_numeric(df["latitude"])
        df_long_b = df.longitude.apply(lambda x:len(repr(x)))
        df_long_b = df_long_b[df_long_b <4].index.values
        df = df.drop(df_long_b)
        df_long = df.longitude.apply(lambda x:repr(x)[3])
        df_long = df_long[df_long != '.']
        to_delete = df_long.index.values
        df = df.drop(to_delete)
        df["longitude"] = pd.to_numeric(df["longitude"])
        df['date_time'] = pd.to_datetime(df.date_time)
        df_occup = df[df.occupied != 0]
        df_occup = df_occup[df_occup.occupied != 1]
        to_delete = df_occup.index.values
        df = df.drop(to_delete)
        df = df.drop_duplicates()
        df = df.sort_values(by=['date_time'])
        print("Dataframe typed and sorted")
        
        maindt = df.copy()

        
        maindt = maindt[abs(maindt.longitude -114.075) <= 0.075]
        maindt = maindt[abs(maindt.latitude - 22.54) <= 0.03]
        
        available_drivers_geo = pd.DataFrame()
        df_new = maindt.loc[(maindt.date_time >='2009-09-'+ day +' 21:00:00') & (maindt.date_time <='2009-09-' + day + ' 21:01:00')]
        # df_new_second_stage = maindt.loc[(maindt.date_time >='2009-09-'+ day +' 21:01:00') & (maindt.date_time <='2009-09-' + day + ' 21:02:00')]
        df_new_drivers = maindt.loc[(maindt.date_time >='2009-09-'+ day +' 20:54:00') & (maindt.date_time <='2009-09-' + day + ' 21:04:00')]
        df_pickup_dropoff = util.compute_pickup_dropoff(df_new)

        available_drivers = df_new_drivers[df_new_drivers['occupied']==0]
        available_drivers_ids = available_drivers.taxi_id.unique()
        for t_id in available_drivers_ids:
            t_df = available_drivers[available_drivers['taxi_id']== t_id]
            available_drivers_geo = pd.concat([available_drivers_geo, t_df.head(1)], ignore_index=True)
            # available_drivers_geo = available_drivers_geo.append(t_df.head(1))
        first_stage = df_pickup_dropoff[df_pickup_dropoff['occupied']==1]
        for k in range(1,nb_future_stages):  
            df_new_future_stage = maindt.loc[(maindt.date_time >='2009-09-'+ day +' 21:' + stages[k-1] +':00') & (maindt.date_time <='2009-09-' + day +' 21:' + stages[k] +':00')]
            df_pickup_dropoff_future = util.compute_pickup_dropoff(df_new_future_stage)
            df_pickup_dropoff_future = df_pickup_dropoff_future[df_pickup_dropoff_future['occupied']==1]
            realized_future_stages.append(df_pickup_dropoff_future)
        # realized_second_stage = df_pickup_dropoff_second_stage[df_pickup_dropoff_second_stage['occupied']==1]
        print("first and realized future stages computed")
        
        return first_stage, future_stage_scenarios, realized_future_stages, available_drivers_geo


def gps_dist(a, b, c, d):    
    '''Compute the distance (in meters) between two gps locations. Input is assumed to be a = longitude, b = latitude, etc.'''
    r = 0.0174533  # 1 degree in radians
    return 2 * 6371000 * np.arcsin( np.sqrt( # https://en.wikipedia.org/wiki/Haversine_formula
        np.sin(r*(d - b)/2.0)**2 + np.cos(r*b) * np.cos(r*d) * np.sin(r*(c - a)/2.0)**2))


def graph_construction(available_drivers, first_stage, second_stage_scenarios):
    #constructs the metric graph
    #input is three pandas dataframes
    G = nx.Graph()
    opt2_range = []
    drivers = available_drivers.taxi_id.unique()
    G.add_nodes_from(drivers, bipartite = 0)
    first_stage_riders = ['R' + str(i) for i in range(len(first_stage))]
    G.add_nodes_from(first_stage_riders, bipartite = 1) 
        # TODO : optimize this for loop into one call
        # TODO : check bipartiteness
    second_stage_riders = []
    second_stage_sc = []
    #print('range(len(second_stage_scenarios)) ' + str(range(len(second_stage_scenarios))))
    for i in range(len(second_stage_scenarios)):
        #print(range(len(second_stage_scenarios[i])))
        s = []
        for j in range(len(second_stage_scenarios[i])):
            s.append('S'+str(i) + '_' + str(j))
        second_stage_sc.append(s)
        G.add_nodes_from(s, bipartite = 1)
    
    # print("nodes_created")
    #now adding the edges
    for d in drivers:
        d_df = available_drivers[available_drivers['taxi_id']==d].iloc[0]
        d_long = d_df.longitude
        d_lat = d_df.latitude
        for r1 in first_stage_riders:
            r1_id = int(r1[1:])
            r1_df = first_stage.iloc[r1_id]
            w = -int(gps_dist(d_long, d_lat, r1_df.longitude, r1_df.latitude))
            G.add_weighted_edges_from([(d,r1,w)])
        for i in range(len(second_stage_sc)):
            for r2 in second_stage_sc[i]:
                # TODO : correct this when S > 9
                r2_id  = int(r2[3:])
                r2_df = second_stage_scenarios[i].iloc[r2_id]
                w = -int(gps_dist(d_long, d_lat, r2_df.longitude, r2_df.latitude))
                opt2_range.append(-w)
                G.add_weighted_edges_from([(d,r2,w)])
    
    return G, drivers, first_stage_riders, second_stage_sc, opt2_range


def bottleneck_matching_graph_bs_single(graph, a, b, drivers, threshold_opt1 = None,  w_2 = None, wtot_min = None):
    G = graph.copy()
    w_arr = []
    w_dic = {}
    M_min = []
    M_dic = {}
    w_min = -float('inf')
    for e in G.edges():
        w = G[e[0]][e[1]]['weight']
        #if(w <= threshold_opt1 and  threshold_opt1!= None):
        w_dic.update({e : w})
        #sort w_arr
    w_arr = list(w_dic.values())
    w_arr_bs = w_arr
    w_arr_bs = set(w_arr_bs)
    w_arr_bs = list(w_arr_bs)
    w_arr_bs.sort(reverse=True)
    low = 0
    high = len(w_arr_bs) - 1
    mid = (high + low) // 4
    
    #no need to care about half of the weights
    w_arr_bs = w_arr_bs[low:mid]
    high = len(w_arr_bs) - 1
    mid = (high + low) // 2
    #TODO : Binary Search
    #TODO : take unique values in w_arr
    while w_arr_bs:
        
        try:
            w = w_arr_bs[mid]
            #print("length of array " + str(len(w_arr_bs)) + " , first stage bottleneck = " + str(w))
#            print("w = " + str(w))
#            print("w_min = " + str(w_min))
#            print("low = " + str(low) + " ,high = " + str(high) + " , mid = " + str(mid))
            G_w = G.copy()
            to_delete = [e for e, w_e in w_dic.items() if w_e < w]
            #print("deleting_edges")
            G_w.remove_edges_from(to_delete)
            #print("computing matching")
            #M = nx.max_weight_matching(G_w, maxcardinality=True, weight='weight')
            M_dic = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G_w, top_nodes=drivers)
            if(len(M_dic)/2 == np.min([a,b])):
                M_min = M_dic
                w_min = w
                high = mid - 1
            else:
                if(w + w_2 <= wtot_min and w_2 != None and wtot_min != None):
                    break
                #print("no perfect matching")
                low = mid + 1
            w_arr_bs = w_arr_bs[low:high+1]
            low = 0
            high = len(w_arr_bs) - 1
            mid = (high + low) // 2
        except IndexError:
            print("indexError")
#            print(w_arr_bs)
#            print(len(w_arr_bs))
#            print(mid)
#            print(high)
#            print(low)
            break
    
    return M_min, w_min



def tsrbb_single_scenario_graph(graph, a, b, second_stage_riders, drivers, threshold_opt1 = None, threshold_opt2 = None, threshold_cost = None):
    #compute the graph
    G = graph.copy()
    #G_c = G.copy()
    w_arr = []
    M_min = []
    first_stage_min = -float('inf')
    second_stage_min = -float('inf')
    total_cost_min = -float('inf')
    if(threshold_cost!= None):
        total_cost_min = threshold_cost
    w_dic = {}
    second_stage_edges = []
    deletable = []
    for e in G.edges():
        if e[0] in second_stage_riders or e[1] in second_stage_riders:
            w = G[e[0]][e[1]]['weight']
            if(threshold_opt2 != None and w < threshold_opt2):
                deletable.append(e)
            else:
                w_dic.update({e : w})
                second_stage_edges.append(e)
    
    G.remove_edges_from(deletable)
    w_arr = list(w_dic.values())
    w_arr_bs = w_arr
    w_arr_bs = set(w_arr_bs)
    w_arr_bs = list(w_arr_bs)
    w_arr_bs.sort(reverse=True)
    
    # low = 0
    # high = len(w_arr_bs) - 1
    # mid = (high + low) // 2
    #print("highest values in array  = " + str(w_arr_bs[high]))
    
     #no need to care about half of the weights
#    w_arr_bs = w_arr_bs[low:mid]
#    high = len(w_arr_bs) - 1
#    mid = (high + low) // 2
    
    while w_arr_bs:
        try:
            #print('size of w_arr_bs from single ' + str(len(w_arr_bs)))
            w = w_arr_bs.pop()
            #print('second stage bottleneck = ' + str(w) + " ,highest values in array  = " + str(w_arr_bs[high]))
            G_w = G.copy()
            #print("deleting stuff")
            to_delete = [e for e, w_e in w_dic.items() if w_e < w]
            G_w.remove_edges_from(to_delete)
            #print("stuff deleted")
            for e in second_stage_edges:
                if(e in G_w.edges()):
                    G_w[e[0]][e[1]]['weight'] = 0
            try:
                 #print("trying matching")
                 M, w_1 = bottleneck_matching_graph_bs_single(G_w, a, b, drivers, threshold_opt1 = threshold_opt1, w_2 = w , wtot_min = total_cost_min )
                 if(len(M)/2 == b and w_1 + w >= total_cost_min):
                    # high = mid - 1
                    # M_min = M
                    # first_stage_min = w_1
                    second_stage_min = w
                    total_cost_min = w_1 + w
                    #print("total cost changed, new cost = " + str(total_cost_min))
                 # else:
                 #    #print("going up")
                 #    low = mid + 1
                 # w_arr_bs = w_arr_bs[low:high+1]
                 # low = 0
                 # high = len(w_arr_bs) - 1
                 # mid = (high + low) // 2
                 #print("low = " + str(low) + ", high = " + str(high) + " , mid = " + str(mid))
            except TypeError:
                print("typerror")
                continue
        except IndexError:
            print("index error in single scenario")
            break
        
    return M_min, first_stage_min, second_stage_min



def tsrmb_single_scenario_graph(graph, a, b, r1, second_stage_riders, drivers, threshold_opt1 = None, threshold_opt2 = None, threshold_cost = None):
    #compute the graph
    G = graph.copy()
    #G_c = G.copy()
    w_arr = []
    M_min = []
    first_stage_min = -float('inf')
    second_stage_min = -float('inf')
    total_cost_min = -float('inf')
    if(threshold_cost!= None):
        total_cost_min = threshold_cost
    w_dic = {}
    second_stage_edges = []
    deletable = []
    for e in G.edges():
        if e[0] in second_stage_riders or e[1] in second_stage_riders:
            w = G[e[0]][e[1]]['weight']
            if(threshold_opt2 != None and w < threshold_opt2):
                deletable.append(e)
            else:
                w_dic.update({e : w})
                second_stage_edges.append(e)
    
    G.remove_edges_from(deletable)
    w_arr = list(w_dic.values())
    w_arr_bs = w_arr
    w_arr_bs = set(w_arr_bs)
    w_arr_bs = list(w_arr_bs)
    w_arr_bs.sort(reverse=True)
    
    # low = 0
    # high = len(w_arr_bs) - 1
    # mid = (high + low) // 2
#    print("highest values in array  = " + str(w_arr_bs[high]))
    
     #no need to care about half of the weights
#    w_arr_bs = w_arr_bs[low:mid]
#    high = len(w_arr_bs) - 1
#    mid = (high + low) // 2
    
    while w_arr_bs:
        try:
            #print('size of w_arr_bs from single ' + str(len(w_arr_bs)))
            w = w_arr_bs.pop()
            #print('second stage bottleneck = ' + str(w) + " ,highest values in array  = " + str(w_arr_bs[high]))
            G_w = G.copy()
            #print("deleting stuff")
            to_delete = [e for e, w_e in w_dic.items() if w_e < w]
            G_w.remove_edges_from(to_delete)
            #print("stuff deleted")
            for e in second_stage_edges:
                if(e in G_w.edges()):
                    G_w[e[0]][e[1]]['weight'] = 0
            try:
                 #print("trying matching")
                 w_1 = 0
                 M = nx.max_weight_matching(G_w, maxcardinality=True, weight='weight')
                 for e in M:
                     w_e = G_w[e[0]][e[1]]['weight']
                     w_1 = w_1 + w_e
                 w_1 = w_1*1.0/r1
                 #print("w_1 = " + str(w_1) + ", len(M) = " + str(len(M)) + ", b = " + str(b))
                 #M, w_1 = bottleneck_matching_graph_bs_single(G_w, a, b, drivers, threshold_opt1 = threshold_opt1, w_2 = w , wtot_min = total_cost_min )
                 if(len(M) == b and w_1 + w >= total_cost_min):
                    # high = mid - 1
                    # M_min = M
                    first_stage_min = w_1
                    second_stage_min = w
                    total_cost_min = w_1 + w
                    #print("total cost changed, new cost = " + str(total_cost_min))
                 # else:
                 #    #print("going up")
                 #    low = mid + 1
                 # w_arr_bs = w_arr_bs[low:high+1]
                 # low = 0
                 # high = len(w_arr_bs) - 1
                 # mid = (high + low) // 2
                 #print("low = " + str(low) + ", high = " + str(high) + " , mid = " + str(mid))
            except TypeError:
                print("typerror")
                continue
        except IndexError:
            print("index error in single scenario")
            break
        
    return M_min, first_stage_min, second_stage_min



def ip_tsrmb(available_drivers, first_stage, second_stage_scenarios, G, flag_subroutine = False, s1_length = None):
    # first_stage_riders = ['R' + str(i) for i in range(len(first_stage))]
    with gp.Env(params=options) as env, gp.Model(env=env) as m:
    # m = gp.Model("mip1")
        m.setParam("OutputFlag", 0)
        drivers = range(len(available_drivers))
        
        z = m.addVar(vtype=GRB.CONTINUOUS, name="z")
        # x = [[]]
        # for i in range(len(first_stage)):
        #     for d in drivers:
        #         x[i][d] = m.addVar(vtype=GRB.BINARY, name='x_'+str(i) +',' + str(d))
                
        x = [[m.addVar(vtype=GRB.BINARY, name='x_'+str(i) +',' + str(d)) for d in drivers] for i in range(len(first_stage))]
        y = [[[m.addVar(vtype=GRB.BINARY, name='y_'+str(i) +'_' + str(j) + ',' + str(d)) for d in drivers] for j in range(len(second_stage_scenarios[i]))] for i in range(len(second_stage_scenarios))]
        
        if flag_subroutine:
            for i in range(len(second_stage_scenarios)):
                for j in range(len(second_stage_scenarios[i])):
                    for d in drivers:
                        # if G.get_edge_data('S'+str(i) + '_' + str(j),available_drivers[d]) is None:
                        #     print(j)
                        #     print(second_stage_scenarios[i][j])
                        #     print('S'+str(i) + '_' + str(second_stage_scenarios[i][j][:3]))
                        #     print('S'+str(i+1) + '_' + str(second_stage_scenarios[i][j][:3]))
                        try:
                            dist_i_d = -G.get_edge_data('S'+str(i) + '_' + str(second_stage_scenarios[i][j][3:]),available_drivers[d])['weight']
                        except TypeError:
                            dist_i_d = -G.get_edge_data('S'+str(i+1) + '_' + str(second_stage_scenarios[i][j][3:]),available_drivers[d])['weight']
                        # if G.get_edge_data('S'+str(i) + '_' + str(j),available_drivers[d]) is None:
                        #     print('S'+str(i) + '_' + str(j))
                        #     print(available_drivers[d])
                       
                        # print(j)
                        # print(second_stage_scenarios[i][j])
                        # print('S'+str(i) + '_' + str(second_stage_scenarios[i][j][3:]))
                        # print('S'+str(i+1) + '_' + str(second_stage_scenarios[i][j][3:]))
                        # dist_i_d = -G.get_edge_data('S'+str(i) + '_' + str(j),available_drivers[d])['weight']
                        m.addConstr(z >= dist_i_d*y[i][j][d], "c_z_y_"+str(i) +'_' + str(j) + ',' + str(d)) 
        
        else:
            for i in range(len(second_stage_scenarios)):
                for j in range(len(second_stage_scenarios[i])):
                    for d in drivers:
                        dist_i_d = -G.get_edge_data('S'+str(i) + '_' + str(j),available_drivers[d])['weight']
                        m.addConstr(z >= dist_i_d*y[i][j][d], "c_z_y_"+str(i) +'_' + str(j) + ',' + str(d)) 
            
        #every first stage rider needs to be matched once
        for i in range(len(first_stage)):
            cstr_i = 0
            for d in drivers:
                cstr_i += x[i][d]
            m.addConstr(cstr_i == 1, "match_x_"+str(i))
        
        
        #every second stage rider needs to be matched once
        for i in range(len(second_stage_scenarios)):
            for j in range(len(second_stage_scenarios[i])):
                cstr_j = 0
                for d in drivers:
                    cstr_j += y[i][j][d]
                m.addConstr(cstr_j == 1, "match_y_"+str(i) +'_' + str(j))
        
        
        
        for s in range(len(second_stage_scenarios)):
            for d in drivers:
                cstr = 0
                for i in range(len(first_stage)):
                    cstr += x[i][d]
                for j in range(len(second_stage_scenarios[s])):
                    cstr += y[s][j][d]
                m.addConstr(cstr <= 1, "d_s_"+ str(s))
                    
            #print(range(len(second_stage_scenarios[i])))
        
        obj_function = z 
        
        tmp = 0
        for i in range(len(first_stage)):
            for d in drivers:
                dist_i_d = -G.get_edge_data('R' + str(i),available_drivers[d])['weight']
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
            
        
def bottleneck_matching_graph_bs(graph, a, b, drivers):
    G = graph.copy()
    w_arr = []
    w_dic = {}
    M_dic = {}
    M_min = []
    w_min = -float('inf')
    for e in G.edges():
        w = G[e[0]][e[1]]['weight']
        w_dic.update({e : w})
        #sort w_arr
    w_arr = list(w_dic.values())
    w_arr_bs = w_arr
    w_arr_bs = set(w_arr_bs)
    w_arr_bs = list(w_arr_bs)
    w_arr_bs.sort(reverse=True)
    low = 0
    high = len(w_arr_bs) - 1
    mid = (high + low) // 2
    
    #no need to care about half of the weights
    w_arr_bs = w_arr_bs[low:mid]
    high = len(w_arr_bs) - 1
    mid = (high + low) // 2
    #TODO : Binary Search
    #TODO : take unique values in w_arr
    while w_arr_bs:
        #print("length of weight array " + str(len(w_arr_bs)))
        try:
            w = w_arr_bs[mid]
            #print("w = " + str(w))
            #print("w_min = " + str(w_min))
            #print("low = " + str(low) + " ,high = " + str(high) + " , mid = " + str(mid))
            G_w = G.copy()
            to_delete = [e for e, w_e in w_dic.items() if w_e < w]
            #print("deleting_edges")
            G_w.remove_edges_from(to_delete)
            #print("computing matching")
            #M = nx.max_weight_matching(G_w, maxcardinality=True, weight='weight')
            try:
                #print("trying")
                M_dic = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G_w, top_nodes=drivers)
                #print(len(M_dic))
                if(len(M_dic)/2 == np.min([a,b])):
                    M_min = M_dic
                    w_min = w
                    high = mid - 1
                else:
                    low = mid + 1
            except:
                print("Problem")
                #print(M_dic)
            w_arr_bs = w_arr_bs[low:high+1]
            low = 0
            high = len(w_arr_bs) - 1
            mid = (high + low) // 2
        except IndexError:
            print("indexError")
            break
    
    return M_min, w_min


def tsrmb_evaluate(graph, first_stage_decision, drivers,first_stage_riders,scenarios):
    worst_case_cost = 0
    #w2 = 0
    second_stage_drivers = [d for d in drivers if d not in first_stage_decision]
    for i in range(len(scenarios)):
        G2 = graph.copy()
        G2.remove_nodes_from(first_stage_decision)
        G2.remove_nodes_from(first_stage_riders)
        for j in range(len(scenarios)):
            if i!=j:
                G2.remove_nodes_from(scenarios[j])
        try:
            #print(G2.edges())
            M, w2 = bottleneck_matching_graph_bs(G2, len(drivers), len(scenarios[i]), second_stage_drivers)
            if -w2 > worst_case_cost: 
                worst_case_cost = -w2
        except TypeError:
            continue
            
    #print("sceond stage cost " + str(worst_case_cost))
    w1 = 0
    G2 = graph.copy()
    #print("excluding second stage")
    #second_stage_drivers = [d for d in drivers if d not in first_stage_decision]
    G2.remove_nodes_from(second_stage_drivers)
    #print("excluded")
    for j in range(len(scenarios)):
        G2.remove_nodes_from(scenarios[j])
    #print("computing first stage cost")
    M = nx.max_weight_matching(G2, maxcardinality=True, weight='weight')
    for e in M:
        w_e = G2[e[0]][e[1]]['weight']
        w1 = w1 + w_e
    w1 = w1*1.0/len(first_stage_decision)
    total_cost = -w1 + worst_case_cost
    return total_cost, -w1, worst_case_cost



def tsrmm_evaluate(graph, first_stage_decision, drivers,first_stage_riders,scenarios):
    worst_case_cost = 0
    #w2 = 0
    second_stage_drivers = [d for d in drivers if d not in first_stage_decision]
    for i in range(len(scenarios)):
        G2 = graph.copy()
        G2.remove_nodes_from(first_stage_decision)
        G2.remove_nodes_from(first_stage_riders)
        for j in range(len(scenarios)):
            if i!=j:
                G2.remove_nodes_from(scenarios[j])
        try:
            #print(G2.edges())
            w2 = 0
            M = nx.max_weight_matching(G2, maxcardinality=True, weight='weight')
            #M, w2 = bottleneck_matching_graph_bs(G2, len(drivers), len(scenarios[i]), second_stage_drivers)
            for e in M:
                w_e = G2[e[0]][e[1]]['weight']
                w2 = w2 + w_e
            if -w2 > worst_case_cost: 
                worst_case_cost = -w2
        except TypeError:
            continue
            
    w1 = 0
    G2 = graph.copy()

    G2.remove_nodes_from(second_stage_drivers)
    for j in range(len(scenarios)):
        G2.remove_nodes_from(scenarios[j])
    #print("computing first stage cost")
    M = nx.max_weight_matching(G2, maxcardinality=True, weight='weight')
    for e in M:
        w_e = G2[e[0]][e[1]]['weight']
        w1 = w1 + w_e
    total_cost = -w1 + worst_case_cost
    return total_cost, -w1, worst_case_cost



def tsrmb_two_scenarios_graph_1try(graph, drivers,first_stage_riders, scenario1, scenario2, graph_second_stage, opt_2 = None):
    G2 = graph_second_stage.copy()
    w2_dic = {}
    w2_G2_dic = {}
    M2 = {}
    second_stage_edges = []
    G2_to_delete = []
    G = graph.copy()
    for e in G.edges():
        if e[0] in scenario1 or e[1] in scenario1 or e[0] in scenario2 or e[1] in scenario2:
            w = G[e[0]][e[1]]['weight']
            w2_dic.update({e : w})
            second_stage_edges.append(e)
    for e in G2.edges():
        w2_G2 = G2[e[0]][e[1]]['weight']
        if(w2_G2 >= 2*opt_2 and opt_2!= None):
            w2_G2_dic.update({e : w2_G2})
        else:
            G2_to_delete.append(e)
#    
#    G2.remove_nodes_from(G2_to_delete)

    #w2 = opt_2
    G2_w2 = G2.copy()
    #to_delete = [e for e, w_e in w2_G2_dic.items() if w_e < 2*w2]
    G2_w2.remove_edges_from(G2_to_delete)
    #M2 = nx.max_weight_matching(G2_w2, maxcardinality=True, weight=None)
    # print("####### G2 Done")
    M2 = nx.algorithms.bipartite.matching.hopcroft_karp_matching(G2_w2, top_nodes=scenario1)
    try:
        #S2_match = [m[0] for m in M2 if m[1] in scenario1] + [m[1] for m in M2 if m[0] in scenario1]
        S2_match = [r2 for r1,r2 in M2.items() if r1 in scenario1]
    except IndexError:
        print("IndexError in S2_match")
        S2_match = []
        #print("S2 match is " + str(S2_match))
    #print("length of S2_match = " + str(len(S2_match)))
    #print("removing S2_match")
    G.remove_nodes_from(S2_match)
    S2_unmatch = [r for r in scenario2 if r not in S2_match]
    second_stage_riders = []
    second_stage_riders = second_stage_riders + scenario1
    second_stage_riders = second_stage_riders + S2_unmatch
    
    D_1, first_cost, second_cost, total_cost = ip_tsrmb(drivers,first_stage_riders,[second_stage_riders], G, True, s1_length = None)

    return D_1, first_cost, second_cost


def tsrmb_two_secnarios_alg(graph, drivers,first_stage_riders, scenario1, scenario2, graph_second_stage, w_range):
    cost_min = float('inf')
    alg2 = float('inf')
    D_1_min = []
    for w in w_range:
        print("w = " + str(w))
        D_1 = tsrmb_two_scenarios_graph_1try(graph, drivers,first_stage_riders, scenario1, scenario2, graph_second_stage, opt_2 = -w)[0]
        evalu = tsrmb_evaluate(graph, D_1, drivers,first_stage_riders,[scenario1, scenario2])
        if(evalu[0] < cost_min):
            D_1_min = D_1
            cost_min = evalu[0]
            alg2 = evalu[-1]
    return D_1_min, cost_min, alg2


def tsrmb_greedy(graph, drivers, first_stage_riders, scenarios):
    G = graph.copy()
    for scenario in scenarios:
        G.remove_nodes_from(scenario)
    M = nx.max_weight_matching(G, maxcardinality=True, weight='weight')

    D_1 = [m[0] for m in M if m[1] in first_stage_riders] + [m[1] for m in M if m[0] in first_stage_riders]
    #D_1 = [d for d,r in M.items() if r in first_stage_riders]
    #print(D_1)
    cost, first_st_cost, worst_case_cost = tsrmb_evaluate(graph, D_1, drivers,first_stage_riders,scenarios)
    #print('first stage cost = ' +str(first_st_cost) + ", second stage cost = " + str(worst_case_cost))
    return cost, first_st_cost, worst_case_cost, D_1
 

nb_future_sc_arr = [2,5,9]
#array contain hours of the day
h_arr = [15,18,21]
n_iter = 5 

for h in h_arr:
    for nb_future_sc in nb_future_sc_arr:
        print('h = ', h, ' ,  nb_future_sc = ', nb_future_sc)
        for i in range(n_iter):
            first_stage, future_stage_scenarios, realized_future_stages, available_drivers_geo = construct_df_hr(days= ['03', '10'], nb_future_stages = nb_future_sc + 1, day_d = ['17'], hr = h)
            print("iteration " + str(i))
            realized_greedy_cost = []
            realized_optimal_cost = []
            realized_alg_cost = []
            s1s2_greedy_cost = []
            s1s2_optimal_cost = []
            s1s2_alg_cost = []
            available_drivers_geo_sample = available_drivers_geo.sample(150*(nb_future_sc+1))
            available_drivers_geo_cp = available_drivers_geo_sample.copy()
            D_1 = []
            for k in range(nb_future_sc):
                # For each future stage k:
                # 1. Compute greedy and optimal costs for realized scenarios
                # 2. Update available drivers by removing those matched in previous stages
                # 3. Construct graphs for both realized and predicted future scenarios
                # 4. Run greedy matching and optimization algorithms to get costs
                # 5. Store costs for comparison between realized vs predicted scenarios
                drivers_to_drop = available_drivers_geo_sample[available_drivers_geo_sample['taxi_id'].isin(D_1)]
                available_drivers_geo_sample = available_drivers_geo_sample.drop(drivers_to_drop.index, axis=0)
                if k == 0:
                    fs = first_stage
                else:
                    fs = realized_future_stages[k-1]
                
                #Constructs graphs for both realized and predicted future scenarios
                G_opt, drivers_opt, first_stage_riders_opt, second_stage_sc_opt, opt_opt2_range = graph_construction(available_drivers_geo_sample, fs, [realized_future_stages[k]])
                G, drivers, first_stage_riders, second_stage_sc, opt2_range = graph_construction(available_drivers_geo_sample, fs, future_stage_scenarios[k])
                

                print("G constructed for " + str(k+1) +  "scenario, drivers = " + str(len(drivers_opt)) + " , first_stage = " + str(len(first_stage_riders_opt)) + " ,second stage = " + str(len(second_stage_sc_opt[0])))
                G2 = util.construct_second_stage_riders_graph(future_stage_scenarios[k])
                cost_g, first_stage_threshold, threshold, D_1 = tsrmb_greedy(G_opt, drivers_opt, first_stage_riders_opt, second_stage_sc_opt)
                realized_greedy_cost.append(cost_g)
                print("greedy computed for k = ", str(k))
            
            
                b = len(fs) + len(realized_future_stages[k])
                a = len(available_drivers_geo_cp)
                th_cost = - cost_g - 5
                size_first_stage = len(fs)
                cost_o = util.tsrmb_single_scenario_graph(G_opt, a, b, size_first_stage, second_stage_sc_opt[0], drivers_opt, threshold_opt1 = -first_stage_threshold, threshold_opt2 = -threshold, threshold_cost = th_cost)
                realized_optimal_cost.append(-cost_o[2])
                print("Optimal cost computed for k = ", str(k))
                
                
                cost_g2, first_stage_threshold2, threshold2, D_1_tmp = tsrmb_greedy(G, drivers, first_stage_riders, second_stage_sc)
                cost_g3 = util.tsrmm_greedy(G_opt, drivers_opt, first_stage_riders_opt, second_stage_sc_opt)
                cost_g4 = util.tsrmm_greedy(G_opt, drivers_opt, first_stage_riders_opt, second_stage_sc)
                
                s1s2_greedy_cost.append(cost_g2)
                
                
                w_range = [100, 200, 300, 400, 500, 600]
                costs_realized = list()
                costs_model = list()
                costs_matching = list()
                costs_matching_model = list()
                costs_model_all = list()
                for w in w_range:
                    print(str(w))
                    D_1 = tsrmb_two_scenarios_graph_1try(G, drivers,first_stage_riders, second_stage_sc[0], second_stage_sc[1], G2, opt_2 = -w)[0]
                    if(len(D_1) == 0):
                        break
                    evalu = tsrmb_evaluate(G_opt, D_1, drivers,first_stage_riders,second_stage_sc_opt)
                    evalu1 = tsrmb_evaluate(G, D_1, drivers,first_stage_riders,second_stage_sc)
                    evalu2 = util.tsrmm_evaluate(G_opt, D_1, drivers,first_stage_riders,second_stage_sc_opt)
                    evalu3 = util.tsrmm_evaluate(G, D_1, drivers,first_stage_riders,second_stage_sc)
                    costs_realized.append(evalu[0])
                    costs_model.append(evalu1[0])
                    costs_model_all.append(evalu1[0])
                    costs_matching.append(evalu2[0])
                    costs_matching_model.append(evalu3[0])
                    D_1 = tsrmb_two_scenarios_graph_1try(G, drivers,first_stage_riders, second_stage_sc[1], second_stage_sc[0], G2, opt_2 = -w)[0]
                    if(len(D_1) == 0):
                        break
                    evalu = tsrmb_evaluate(G_opt, D_1, drivers,first_stage_riders,second_stage_sc_opt)
                    evalu1 = tsrmb_evaluate(G, D_1, drivers,first_stage_riders,second_stage_sc)
                    evalu2 = util.tsrmm_evaluate(G_opt, D_1, drivers,first_stage_riders,second_stage_sc_opt)
                    costs_realized.append(evalu[0])
                    costs_model.append(evalu1[0])
                    costs_model_all.append(evalu1[0])
                    costs_matching.append(evalu2[0])
                    costs_matching_model.append(evalu3[0])
                realized_alg_cost.append(costs_realized[-2:])
                s1s2_optimal_cost.append(costs_model_all[-2:])
                print("all costs computed for k = ", str(k))
                
            # print("greedy S = (" + str(sum(realized_greedy_cost)) + "), ALG = " + str(sum(realized_alg_cost)))
            # print("greedy S_1, S_2 = " + str(sum(s1s2_greedy_cost)) + "), ALG = " + str(sum(s1s2_optimal_cost)))
            # print("greedy matching cost = " + str(cost_g3) + ", ALG matching cost = " + str(costs_matching[-2:]))
