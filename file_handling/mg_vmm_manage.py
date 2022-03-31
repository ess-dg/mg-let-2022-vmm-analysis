#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mg_vmm_filter.py: Contains functions which filters Multi-Grid data 
                  obtained with the VMM read-out electronics.
"""

import os
import numpy as np
import pandas as pd
import pcapng as pg
import file_handling.mg_vmm_read as mg_read
import file_handling.mg_vmm_cluster as mg_cluster

def filter_data(df, parameters):
    """
    Filters clusters based on preferences.

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
            df_red = df_red[(df_red[parameter] >= min_val) &
                            (df_red[parameter] <= max_val)]
    return df_red

def channel_to_xyz(wch, gch, ring):
    x = (wch // 16) * 0.025
    y = (wch % 16) * 10
    

def import_many_files(folder_path, file_name, number_files):
    # Import first data file
    first_path = folder_path + file_name + '_00000.pcapng' 
    print(first_path)
    df = mg_read.read_vmm_data(first_path)
    # Import rest of data
    for i in range(number_files-1):
        number = str(i+1)
        number_digits = len(number)
        path_i = folder_path + file_name + '_' + '0'*(5-number_digits) + number + '.pcapng'
        print(path_i)
        df_i = mg_read.read_vmm_data(path_i)
        df.append(df_i, ignore_index=True)
    return df
