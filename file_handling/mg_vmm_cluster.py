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

def cluster_vmm_data(events_df):
    """ Clusters data obtained with the VMM readout and stores the obtained
        clusters in a dataframe.

    Args:
        events_df (pd.DataFrame): DataFrame containing the events

    Returns:
        clusters_df (pd.DataFrame): DataFrame containing the clusters

    """        
    # Declare parameters
    est_size = round(len(events_df)/2) + 1 # Add one extra element for dummy cluster
    column_labels = events_df.columns.values.tolist()
    events_np = events_df.to_numpy()
    time_window = 2e-6 # in seconds
    LO_to_s = 1 / (88.0525 * 1e6)
    
    # Pre-allocated space for clusters
    clusters_dict = {'wch': (-1) * np.ones([est_size], dtype=int),
                     'gch_max': (-1) * np.ones([est_size], dtype=int),
                     'gch_cog': (-1) * np.ones([est_size], dtype=int),
                     'wadc': np.zeros([est_size], dtype=int),
                     'gadc_sum': np.zeros([est_size], dtype=int),
                     'wm': np.zeros([est_size], dtype=int),
                     'gm': np.zeros([est_size], dtype=int),
                     'time': (-1) * np.ones([est_size], dtype=int),
                     'tof': (-1) * np.ones([est_size], dtype=int),
                     'gchs_adjacent': np.zeros([est_size], dtype=int)}
    
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
    
    # Iterate through data
    idx_cluster = 0
    max_wadc, max_gadc, cluster_size = 0, 0, 1
    gch_min, gch_max = np.inf, -np.inf
    start_time = -np.inf
    for event in events_np:
        # Extract event parameters
        time = event[idx_PulseTimeHI] + event[idx_PulseTimeLO] * LO_to_s
        adc = event[idx_adc]
        vmm = event[idx_vmm]
        channel = event[idx_channel]
        is_wire = vmm < 4
        # Check if event is in same cluster
        if (time - start_time) < time_window:
            # Continue on current cluster
            cluster_size += 1
            if is_wire:
                # Wires
                clusters_dict['wadc'][idx_cluster] += adc
                clusters_dict['wm'][idx_cluster] += 1
                if adc > max_wadc:
                    max_wadc = adc
                    wch = (vmm // 2) * (-32) + vmm * 64 + channel
                    clusters_dict['wch'][idx_cluster] = wch
            else: 
                # Grids
                clusters_dict['gadc_sum'][idx_cluster] += adc
                clusters_dict['gm'][idx_cluster] += 1
                gch = (vmm - 4) * 51 + channel
                clusters_dict['gch_cog'][idx_cluster] += adc * gch
                if adc > max_gadc:
                    max_gadc = adc
                    clusters_dict['gch_max'][idx_cluster] = gch
                if channel > gch_max: gch_max = channel
                if channel < gch_min: gch_min = channel
        else:
            # End current cluster
            if (gch_max - gch_min) == (cluster_size - 1):
                clusters_dict['gchs_adjacent'][idx_cluster] = 1
            grid_charge_total = clusters_dict['gadc_sum'][idx_cluster]
            if grid_charge_total > 0:
                clusters_dict['gch_cog'][idx_cluster] *= (1/grid_charge_total)
            # Start new cluster
            idx_cluster += 1
            cluster_size = 1
            gch_min, gch_max = np.inf, -np.inf
            start_time = time
            # Insert wadc, gadc, gch and wch
            if is_wire:
                # Wires
                clusters_dict['wadc'][idx_cluster] += adc
                clusters_dict['wm'][idx_cluster] += 1
                max_wadc = adc
                wch = (vmm // 2) * (-32) + vmm * 64 + channel
                clusters_dict['wch'][idx_cluster] = wch
            else:
                # Grids
                clusters_dict['gadc_sum'][idx_cluster] += adc
                clusters_dict['gm'][idx_cluster] += 1
                max_gadc = adc
                gch = (vmm - 4) * 51 + channel
                clusters_dict['gch_max'][idx_cluster] = gch
                if channel > gch_max: gch_max = channel
                if channel < gch_min: gch_min = channel
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
        clusters_dict[key] = clusters_dict[key][1:idx_cluster+1] # Start at index 1 to throw away first dummy cluster
    
    # Convert to dataframe
    clusters_df = pd.DataFrame(clusters_dict)
    
    return clusters_df
