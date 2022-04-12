#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manage.py: Contains functions to manage mesytec Multi-Grid data
"""

import numpy as np
import pandas as pd

import file_handling.mg_mesytec_ref_read_and_cluster as mg_read
from IPython.display import clear_output

# ==============================================================================
#                                  FILTER DATA
# ==============================================================================

def filter_data(df, parameters):
    """
    Filters clusters based on preferences set on GUI.

    Args:
        df (DataFrame): Clustered events
        parameters (dict): Dictionary containing information on which
                           parameters to filter on, and within which range.

    Returns:
        df_red (DataFrame): DataFrame containing the reduced data according to
                            the specifications in "parameters".
    """

    df_red = df
    for parameter, (min_val, max_val, filter_on) in parameters.items():
        if filter_on:
            if parameter == 'layer':
                df_red = df_red[((df_red.wch % 20) >= min_val) &
                                ((df_red.wch % 20) <= max_val)]
            elif parameter == 'row':
                df_red = df_red[(((df_red.bus * 4) + df_red.wch//20) >= min_val) &
                                (((df_red.bus * 4) + df_red.wch//20) <= max_val)]
            else:
                df_red = df_red[(df_red[parameter] >= min_val) &
                                (df_red[parameter] <= max_val)]
    return df_red

# ==============================================================================
#                                MERGE FILES
# ==============================================================================

def merge_files(dfs):
    """
    Function to merge DataFrames containing Multi-Grid data. When subsequent
    files are appended, the timestampe is updateded on each file such that the
    first file has the time=0 reference.

    Args:
        dfs (np.array): Array of DataFrames containing the Multi-Grid data

    Returns:
        df_full (pd.DataFrame): DataFrame containing appended Multi-Grid data

    """

    # Declare first element and get max time
    df_full = dfs[0]
    start = df_full.shape[0]
    max_time = df_full.tail(1)['time'].array[0]
    # Append all files so that subsequent files get an increased timestamp
    for df in dfs[1:]:
        df_full = df_full.append(df, ignore_index=True)
        df_full['time'].loc[start:] = df_full['time'].loc[start:] + max_time
        max_time = df_full.tail(1)['time'].array[0]
        start = df_full.shape[0]
    return df_full

# ==============================================================================
#                             EXTRACT AND SAVE DATA
# ==============================================================================

def extract_and_save(run, raw_path, processed_folder):
    """
    Function to extract, cluster and save data.

    Args:
        run (str): File run, as specified in the 'Paths'-declaration below
        raw_path (str): Path to the raw data in the '.zip'-file
        processed_folder (str): Location of clusters and events

    Yields:
        Clusters and events are extracted from the raw data and saved in the
        'processed'-folder.

    """
    clusters_path = processed_folder + run + '_clu.h5'
    events_path = processed_folder + run + '_ev.h5'
    extract_data(raw_path, clusters_path, events_path)


# ==============================================================================
#                            LOAD CLUSTERS AND EVENTS
# ==============================================================================

def load_clusters_and_events(run, processed_folder):
    """
    Function to load data from a specific run.

    Args:
        run (str): File run, as specified in the 'Paths'-declaration below
        processed_folder (str): Location of clusters and events

    Returns:
        Clusters (DataFrame)
        Events (DataFrame)

    """
    clusters_path = processed_folder + run + '_clu.h5'
    events_path = processed_folder + run + '_ev.h5'
    return load_data(clusters_path), load_data(events_path)


# ==============================================================================
#                             HELPER FUNCTIONS
# ==============================================================================

def extract_data(zipped_path, clusters_save_path, events_save_path):
    """
    Function to extract, cluster and save data.

    Args:
        zipped_path (str): Location of raw data
        clusters_save_path (str): Destination for clusters
        events_save_path (str): Destination for events

    Yields:
        Clusters and events are extracted from the raw data and saved at the
        specified locations

    """
    unzipped_path = mg_read.unzip_data(zipped_path)
    data = mg_read.import_data(unzipped_path)
    # Extract clusters and save to disc
    clusters = mg_read.extract_clusters(data)
    save_data(clusters, clusters_save_path)
    clusters = None
    # Extract events and save to disc
    events = mg_read.extract_events(data)
    save_data(events, events_save_path)
    events = None
    # Clear data
    data = None

def save_data(df, path):
    """
    Saves clusters or events to specified HDF5-path.

    Args:
        path (str): Path to HDF5 containing the saved DataFrame
        df (DataFrame): Data

    Yields:
        Data is saved to path
    """
    # Export to HDF5
    df.to_hdf(path, 'df', complevel=9)


def load_data(path):
    """
    Loads clustered data from specified HDF5-path.

    Args:
        path (str): Path to HDF5 containing the saved data

    Returns:
        df (DataFrame): Data
    """
    df = pd.read_hdf(path, 'df')
    return df

def reorder_channels_clusters(data, num_w_in_row=16,bus1=3, rows_w1=2, num_gr1=0, bus2=2, rows_w2=4, num_gr2=37, name_new_bus=9):
    """
    Reorders and mearges clusters from two busses.
    
    Args:
        data = Dataframe data
        bus1 (int): First bus: Will be placed first
        num_w_in_row (int): Number of wires in each row
        rows_w1: How many rows are connected to bus 1
        num_gr1: How manny grids are connected to bus 1
        bus2: Second bus
        rows_w2: Rows of wires in bus 1
        num_gr2: Number of grids in bus 2
        name_new_bus: number of new bus

    Returns:
        df (DataFrame): Data
    """
    print('Reordering cluster channels')
    pd.options.mode.chained_assignment = None  # default='warn'
    max_w=20        # MAximum amount of wires in one row
    diff=max_w-num_w_in_row
    start=diff//2        # Start of data from each row of wires
    end=max_w-diff/2-1  # End of data of each row of wires
    list_emty_wch=[0,1,18,19,20,21,38,39,40,41,58,59,60,61,78,79]
    data_bus1_unfiltered=data[(data.bus==bus1)]
    data_bus2_unfiltered=data[(data.bus==bus2)]
    data_bus1=data_bus1_unfiltered[~(data_bus1_unfiltered.wch.isin(list_emty_wch))]
    data_bus2=data_bus2_unfiltered[~(data_bus2_unfiltered.wch.isin(list_emty_wch))]
    if rows_w1>0:
        for wi in range(rows_w1):
            for i in range(num_w_in_row):
                indx_old=wi*max_w+start+i
                indx_new=wi*num_w_in_row+i
                data_bus1.loc[(data_bus1.wch == indx_old), 'wch'] = indx_new
    else:
        pass
    if rows_w2>0:
        for wi in range(rows_w2):
            for i in range(num_w_in_row):
                indx_old=79-(wi*max_w+start+i)
                indx_new=95-(wi*num_w_in_row+i)
                data_bus2.loc[(data_bus2.wch == indx_old), 'wch'] = indx_new
    else:
        pass
    
    max_gr=40
    diff_g=max_gr-num_gr2+num_gr1
    start= int(diff_g)
    if num_gr1>0:  
        for gi in range(num_gr1):
            indx_old=117-gi
            indx_new=132-gi
            data_bus1.loc[(data_bus1.gch == indx_old), 'gch'] = indx_new
    else:
        pass
    if num_gr2>0:  
        for gi in range(num_gr2):
            indx_old=117-gi
            indx_new=132-gi
            data_bus2.loc[(data_bus2.gch == indx_old), 'gch'] = indx_new
            
    else:
        pass

    data_ordered= pd.concat([data_bus1,data_bus2])
    data_ordered['bus'] = data_ordered['bus'].replace([bus1,bus2],name_new_bus)
    data = None
    return data_ordered
    
    
def reorder_channels_events(data, num_w_in_row=16, bus1=3, rows_w1=2, num_gr1=0, bus2=2, rows_w2=4, num_gr2=37, name_new_bus=9):
    """
    Reorders and mearges clusters from two busses.
    
    Args:
        data = Dataframe data
        bus1 (int): First bus: Will be placed first
        num_w_in_row (int): Number of wires in each row
        rows_w1: How many rows are connected to bus 1
        num_gr1: How manny grids are connected to bus 1
        bus2: Second bus
        rows_w2: Rows of wires in bus 1
        num_gr2: Number of grids in bus 2
        name_new_bus: number of new bus

    Returns:
        df (DataFrame): Data
    """
    pd.options.mode.chained_assignment = None  # default='warn'
    print('Reordering cluster channels')
    list_emty_wch=[0,1,18,19,20,21,38,39,40,41,58,59,60,61,78,79,80]
    data_bus1_unfiltered=data[(data.bus==bus1)]
    data_bus2_unfiltered=data[(data.bus==bus2)]
    data_bus1=data_bus1_unfiltered[~(data_bus1_unfiltered.ch.isin(list_emty_wch))]
    data_bus2=data_bus2_unfiltered[~(data_bus2_unfiltered.ch.isin(list_emty_wch))]
 
    start= 1
    if num_gr1>0:  
        for gi in range(num_gr1):
            indx_old=117-gi
            indx_new=132-gi
            data_bus1['ch'] = data_bus1['ch'].replace([indx_old],indx_new)
    if num_gr2>0:  
        for gi in range(num_gr2):
            indx_old=117-gi
            indx_new=132-gi
            data_bus2['ch'] = data_bus2['ch'].replace([indx_old],indx_new)
    max_w=20        # MAximum amount of wires in one row
    diff=max_w-num_w_in_row
    start=diff//2    
    if rows_w1>0:
        for wi in range(rows_w1):
            for i in range(num_w_in_row):
                indx_old=wi*max_w+start+i
                indx_new=wi*num_w_in_row+i
                data_bus1['ch'] = data_bus1['ch'].replace([indx_old],indx_new)
    if rows_w2>0:
        for wi in range(rows_w2):
            for i in range(num_w_in_row):
                indx_old=79-(wi*max_w+start+i)
                indx_new=95-(wi*num_w_in_row+i)
                data_bus2['ch'] = data_bus2['ch'].replace([indx_old],indx_new)  
    data_ordered= pd.concat([data_bus1,data_bus2])
    data_ordered['bus'] = data_ordered['bus'].replace([bus1,bus2],name_new_bus)
    data = None
    return data_ordered
    

def merge_events_by_time(data, time_s):
    """
    Merge datapoints close to echother.
    
    Args:
        data = Dataframe data belonging to the same bus (named)
        time_diff = time difference between 2 clusters to be considered to be the same


    Returns:
        df (DataFrame): Data
    """
    tdc_to_s= 62.5e-9
    pd.options.mode.chained_assignment = None  # default='warn'
    data_filt=data
    time_diff=int(time_s/tdc_to_s)
    print(time_diff)
    #data_filt=data_filt[data_filt.gm>0]
    for ind in range(len(data_filt)):
        if ind>0:
            diff_old=abs(data_filt.iloc[ind].time - data_filt.iloc[ind-1].time)
            if (diff_old<time_diff):
                data_filt.iloc[ind].wm += data_filt.iloc[ind-1].wm
                data_filt.iloc[ind].gm += data_filt.iloc[ind-1].gm
                if data_filt.iloc[ind].wadc >= data_filt.iloc[ind-1].wadc:
                    pass
                else:
                    data_filt.iloc[ind].wch=data_filt.iloc[ind-1].wch
                if data_filt.iloc[ind].gadc >= data_filt.iloc[ind-1].gadc:
                    pass
                else:
                    data_filt.iloc[ind].gch=data_filt.iloc[ind-1].gch
                data_filt.iloc[ind].wadc += data_filt.iloc[ind-1].wadc
                data_filt.iloc[ind].gadc += data_filt.iloc[ind-1].gadc
        if ind < len(data_filt) -1:
            diff_new=abs(data_filt.iloc[ind].time - data_filt.iloc[ind+1].time)
            if (diff_new<time_diff):
                data_filt.iloc[ind].wm += data_filt.iloc[ind+1].wm
                data_filt.iloc[ind].gm += data_filt.iloc[ind+1].gm
                if data_filt.iloc[ind].wadc >= data_filt.iloc[ind+1].wadc:
                    pass
                else:
                    data_filt.iloc[ind].wch=data_filt.iloc[ind+1].wch
                if data_filt.iloc[ind].gadc >= data_filt.iloc[ind+1].gadc:
                    pass
                else:
                    data_filt.iloc[ind].gch=data_filt.iloc[ind+1].gch
                data_filt.iloc[ind].wadc += data_filt.iloc[ind+1].wadc
                data_filt.iloc[ind].gadc += data_filt.iloc[ind+1].gadc
        if ind % (len(data_filt)//10) == 1:
            percentage_finished = int(round((ind/len(data_filt))*100))
            # Decide how much of the data should be read
            if percentage_finished>100:
                stop=True
            clear_output(wait=True)
            #print('Percentage: %d' % percentage_finished)
            print('Mearging clusters')
            print((percentage_finished//10)* '*' +(40-percentage_finished//10)*' ' + str(percentage_finished) + ' %' )   
    clear_output(wait=True)
    print(40*' '+'Finished mearging clusters')   

    data = None
    
    return data_filt