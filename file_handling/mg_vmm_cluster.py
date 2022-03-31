#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mg_vmm_cluster.py: Contains functions which clusters data from the Multi-Grid
                   detector obtained with the VMM read-out system.
"""

import os
import numpy as np
import pandas as pd
import pcapng as pg

def cluster_vmm_data(events_df, time_window=2e-6):
    """ Clusters data obtained with the VMM readout and stores the obtained
        clusters in a dataframe.

    Args:
        events_df (pd.DataFrame): DataFrame containing the events

    Returns:
        clusters_df (pd.DataFrame): DataFrame containing the clusters

    """        
    # Declare parameters
    est_size = round(len(events_df)) + 1 # Add one extra element for dummy cluster
    column_labels = events_df.columns.values.tolist()
    events_np = events_df.to_numpy()
    LO_to_s = 1 / (88.0525 * 1e6)
    
    # Pre-allocated space for clusters
    clusters_dict = {'wch': (-1) * np.ones([est_size], dtype=int),
                     'gch_max': (-1) * np.ones([est_size], dtype=int),
                     'gch_cog': np.zeros([est_size], dtype=float),
                     'wadc': np.zeros([est_size], dtype=int),
                     'gadc': np.zeros([est_size], dtype=int),
                     'wm': np.zeros([est_size], dtype=int),
                     'gm': np.zeros([est_size], dtype=int),
                     'time': (-1) * np.ones([est_size], dtype=float),
                     'tof': (-1) * np.ones([est_size], dtype=float),
                     'fen': (-1) * np.ones([est_size], dtype=int),
                     'ring': (-1) * np.ones([est_size], dtype=int),
                     'gchs_adjacent': np.zeros([est_size], dtype=int),
                     'same_fen': np.zeros([est_size], dtype=int),
                     'same_ring': np.zeros([est_size], dtype=int)}
    
    # Get indices of data columns for events
    idx_PulseTimeHI = column_labels.index('PulseTimeHI')
    idx_PulseTimeLO = column_labels.index('PulseTimeLO')
    idx_PrevPulseTimeHI = column_labels.index('PrevPulseTimeHI')
    idx_PrevPulseTimeLO = column_labels.index('PrevPulseTimeLO')
    idx_time_hi = column_labels.index('time_hi')
    idx_time_lo = column_labels.index('time_lo')
    idx_tdc = column_labels.index('tdc')
    idx_vmm = column_labels.index('vmm')
    idx_channel = column_labels.index('channel')
    idx_adc = column_labels.index('adc')
    idx_fen = column_labels.index('fen')
    idx_ring = column_labels.index('ring')
    
    # Iterate through data
    idx_cluster = 0
    max_wadc, max_gadc, grids_in_cluster = 0, 0, 0
    same_fen = True
    same_ring = True
    cluster_fen = -1
    cluster_ring = -1
    gch_min, gch_max = np.inf, -np.inf
    start_time = -np.inf
    for event in events_np:
        # Extract event parameters
        time = event[idx_time_hi] + event[idx_time_lo] * LO_to_s
        adc = event[idx_adc]
        vmm = event[idx_vmm]
        channel = event[idx_channel]
        fen = event[idx_fen]
        ring = event[idx_ring]
        hybrid = vmm // 2
        if fen != cluster_fen: same_fen = False
        if ring//2 != cluster_ring//2: same_ring = False
        is_wire = hybrid == 1
        # Check if event is in same cluster
        if 0 <= (time - start_time) <= time_window:
            # Continue on current cluster
            if is_wire:
                # Wires
                clusters_dict['wadc'][idx_cluster] += adc
                clusters_dict['wm'][idx_cluster] += 1
                if adc > max_wadc:
                    max_wadc = adc
                    wch = ((vmm % 2) * 64 + channel - 32) ^ 1
                    clusters_dict['wch'][idx_cluster] = wch
            else: 
                # Grids
                clusters_dict['gadc'][idx_cluster] += adc
                clusters_dict['gm'][idx_cluster] += 1
                gch = channel
                adc_gch = adc * gch
                clusters_dict['gch_cog'][idx_cluster] += adc * gch
                if adc > max_gadc:
                    max_gadc = adc
                    clusters_dict['gch_max'][idx_cluster] = gch
                if channel > gch_max: gch_max = channel
                if channel < gch_min: gch_min = channel
                grids_in_cluster += 1
        else:
            # End current cluster
            if ((gch_max - gch_min) == (grids_in_cluster - 1)):
                clusters_dict['gchs_adjacent'][idx_cluster] = 1
            if same_fen:
                clusters_dict['same_fen'][idx_cluster] = 1
            if same_ring:
                clusters_dict['same_ring'][idx_cluster] = 1
            grid_charge_total = clusters_dict['gadc'][idx_cluster]
            if grid_charge_total > 0:
                clusters_dict['gch_cog'][idx_cluster] *= (1/grid_charge_total)
            # Start new cluster
            idx_cluster += 1
            grids_in_cluster = 0
            gch_min, gch_max = np.inf, -np.inf
            max_wadc, max_gadc = 0, 0
            start_time = time
            cluster_fen, same_fen = event[idx_fen], True
            cluster_ring, same_ring = event[idx_ring], True
            # Insert wadc, gadc, gch and wch
            if is_wire:
                # Wires
                clusters_dict['wadc'][idx_cluster] += adc
                clusters_dict['wm'][idx_cluster] += 1
                max_wadc = adc
                wch = ((vmm % 2) * 64 + channel - 32) ^ 1
                clusters_dict['wch'][idx_cluster] = wch
            else:
                # Grids
                clusters_dict['gadc'][idx_cluster] += adc
                clusters_dict['gm'][idx_cluster] += 1
                max_gadc = adc
                gch = channel
                clusters_dict['gch_max'][idx_cluster] = gch
                if channel > gch_max: gch_max = channel
                if channel < gch_min: gch_min = channel
                grids_in_cluster += 1
            # Insert fen and ring
            clusters_dict['fen'][idx_cluster] = cluster_fen
            clusters_dict['ring'][idx_cluster] = cluster_ring
            # Insert tof
            PrevPulseTime = event[idx_PrevPulseTimeHI] + event[idx_PrevPulseTimeLO] * LO_to_s
            PulseTime = event[idx_PulseTimeHI] + event[idx_PulseTimeLO] * LO_to_s
            tdc = event[idx_tdc]
            if time < PulseTime:
                tof = time - PrevPulseTime
            else:
                tof = time - PulseTime
            clusters_dict['tof'][idx_cluster] = tof
            clusters_dict['time'][idx_cluster] = time
            
    
    # Remove empty elements
    for key in clusters_dict:
        # Start at index 1 to throw away first dummy
        # End one cluster from end to throw away potentially incomplete clusters
        clusters_dict[key] = clusters_dict[key][1:idx_cluster]
    
    # Convert to dataframe
    clusters_df = pd.DataFrame(clusters_dict)
    
    return clusters_df
