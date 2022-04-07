#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mg_read.py: Contains functions which unzips, imports and clusters data from the
            Multi-Grid detector obtained with the Mesytec read-out system:
            https://www.mesytec.com/products/datasheets/VMMR.pdf
"""

import os
import struct
import shutil
import zipfile
import re
import numpy as np
import pandas as pd
from IPython.display import clear_output


# =============================================================================
#                     DICTIONARY FOR BINARY TRANSLATION
# =============================================================================

# MASKS
TYPE_MASK      =   0xC0000000     # 1100 0000 0000 0000 0000 0000 0000 0000
DATA_MASK      =   0xF0000000     # 1111 0000 0000 0000 0000 0000 0000 0000

CHANNEL_MASK   =   0x00FFF000     # 0000 0000 1111 1111 1111 0000 0000 0000
BUS_MASK       =   0x0F000000     # 0000 1111 0000 0000 0000 0000 0000 0000
ADC_MASK       =   0x00000FFF     # 0000 0000 0000 0000 0000 1111 1111 1111
TIMESTAMP_MASK =   0x3FFFFFFF     # 0011 1111 1111 1111 1111 1111 1111 1111
EXTS_MASK      =   0x0000FFFF     # 0000 0000 0000 0000 1111 1111 1111 1111
TRIGGER_MASK   =   0xCF000000     # 1100 1111 0000 0000 0000 0000 0000 0000

# DICTONARY
HEADER         =   0x40000000     # 0100 0000 0000 0000 0000 0000 0000 0000
DATA           =   0x00000000     # 0000 0000 0000 0000 0000 0000 0000 0000
EOE            =   0xC0000000     # 1100 0000 0000 0000 0000 0000 0000 0000

DATA_BUS_START =   0x30000000     # 0011 0000 0000 0000 0000 0000 0000 0000
DATA_EVENT     =   0x10000000     # 0001 0000 0000 0000 0000 0000 0000 0000
DATA_EXTS      =   0x20000000     # 0010 0000 0000 0000 0000 0000 0000 0000

TRIGGER        =   0x41000000     # 0100 0001 0000 0000 0000 0000 0000 0000

# BIT-SHIFTS
CHANNEL_SHIFT   =   12
BUS_SHIFT       =   24
EXTS_SHIFT      =   30


# ==============================================================================
#                                  UNZIP DATA
# ==============================================================================

def unzip_data(zip_source):
    """ Unzips mestyec .zip-files and extracts data-files:

            1. Extracts data in temporary folder
            2. Selects the relevant file and moves it to a temporary location
            3. Removes the temporary fodler where the rest of zipped data is

    Args:
        zip_source (str): Path to '.zip'-file that contains the data

    Returns:
        '.mesytec'-file path (str): Path to the extracted '.mesytec'-file

    """
    dirname = os.path.dirname(__file__)
    zip_temp_folder = os.path.join(dirname, '../zip_temp_folder/')
    mkdir_p(zip_temp_folder)
    file_temp_folder = os.path.join(dirname, '../')
    destination = ''
    with zipfile.ZipFile(zip_source, "r") as zip_ref:
        zip_ref.extractall(zip_temp_folder)
        temp_list = os.listdir(zip_temp_folder)
        source_file = None
        print(temp_list)
        for temp_file in temp_list:
            if temp_file[-8:] == '.mvmelst':
                source_file = temp_file
        source = zip_temp_folder + source_file
        destination = file_temp_folder + source_file
        shutil.move(source, destination)
        shutil.rmtree(zip_temp_folder, ignore_errors=True)
    return destination


# ==============================================================================
#                                IMPORT DATA
# ==============================================================================

def import_data(file_path, maximum_file_size_in_mb=np.inf):
    """ Imports mestyec data in three steps:

            1. Reads file as binary and saves data in 'content'
            2. Finds the end of the configuration text, i.e. '}\n}\n' followed
               by 0 to n spaces, then saves everything after this to
               'reduced_content'.
            3. Groups data into 'uint'-words of 4 bytes (32 bits) length

    Args:
        file_path (str): Path to '.mesytec'-file that contains the data
        maximum_file_size_in_mb (float): Maximum allowed file size to import

    Returns:
        data (tuple): A tuple where each element is a 32 bit mesytec word

    """

    # Get maximum file size in [bytes]
    ONE_MB_IN_BYTES = (1 << 20)
    maximum_file_size_in_bytes = maximum_file_size_in_mb * ONE_MB_IN_BYTES
    # Assign piece size in [bytes]
    piece_size = 1000 * ONE_MB_IN_BYTES
    # Import data
    with open(file_path, mode='rb') as bin_file:
        # Get first piece of data
        content = bin_file.read(piece_size)
        # Skip configuration text
        match = re.search(b'}\n}\n[ ]*', content)
        start = match.end()
        content = content[start:]
        # Split first piece of data into groups of 4 bytes
        data = struct.unpack('I' * (len(content)//4), content)
        # Repeat for the rest of data
        more_data = True
        imported_data = piece_size
        while more_data and imported_data <= maximum_file_size_in_bytes:
            imported_data += piece_size
            piece = bin_file.read(piece_size)
            if not piece:  # Reached end of file
                more_data = False
            else:
                data += struct.unpack('I' * (len(piece)//4), piece)
    # Remove original file
    os.remove(file_path)
    return data


# ==============================================================================
#                                 EXTRACT CLUSTERS
# ==============================================================================

def extract_clusters(data):
    """ Clusters the imported data and stores it in a DataFrame containing
        coicident events (i.e. clusters).

        Does this in the following fashion for coincident events:
            1. Reads one word at a time
            2. Checks what type of word it is (HEADER, DATA_BUS_START,
               DATA_EVENT, DATA_EXTS or EOE).
            3. When a HEADER is encountered, 'is_open' is set to 'True',
               signifying that a new cluster has been started. Data is then
               gathered into a single coincident event until EoE is encountered.
            4. When EOE is encountered, the cluster is formed with information
               about bus, channel, charge, time and multiplicity. A flag is set
               to 1 if more than one bus was encountered. The cluster is placed
               in the created dictionary.
            5. After the iteration through data is complete, the dictionary
               containing the coincident events is convereted to a DataFrame.

    Explanation of cluster parameters:
        'bus': Grid column which recorded the cluster
        'time': At what time the event occured
        'tof': At what time-of-flight the event occured
        'grid_channels': Which grid channels are activated
        'grid_adc': Charge collected by each grid
        'wch': Wire channel with most recorded charge
        'gch': Grid channel with most recorded charge
        'wadc': Total charge collected by all wires in coincidence
        'gadc': Total charge collected by all grids in coincidence
        'wm': Total number of wires in coincidence
        'gm': Total number of grids in coincidence
        'flag': Is 1 if more than one bus was encountered under the same HEADER,
                else -1

    Args:
        data (tuple): Tuple containing data, one word per element.

    Returns:
        clusters (DataFrame): DataFrame containing one neutron
                              event per row. Each neutron event has
                              information about: "bus", "time",
                              "tof", "grid_channels", "grid_adc", "wch", "gch", "wadc", "gadc",
                              "wm", "gm" and "flag".

    """
    size = len(data)
    # Initiate dictionary to store clusters
    ce_dict = {'bus': (-1) * np.ones([size], dtype=int),
               'time': (-1) * np.ones([size], dtype=int),
               'tof': (-1) * np.ones([size], dtype=int),
               'gridch': (-1)*np.ones((size,37),dtype=int),
               'gridadc': np.zeros((size,37),dtype=int),
               'wch': (-1) * np.ones([size], dtype=int),
               'gch': (-1) * np.ones([size], dtype=int),
               'gch_max': (-1) * np.ones([size], dtype=int),
               'wadc': np.zeros([size], dtype=int),
               'gadc': np.zeros([size], dtype=int),
               'wm': np.zeros([size], dtype=int),
               'gm': np.zeros([size], dtype=int),
               'flag_unconnected': (-1) * np.ones([size], dtype=int),
               'max_dist': np.zeros([size], dtype=int)}
    np.full([size], "", dtype=np.object)
                
    # Declare temporary boolean variables, related to words
    is_open, is_trigger, is_data, is_exts = False, False, False, False
    # Declare temporary variables, related to events
    previous_bus, bus = -1, -1
    max_adc_w, max_adc_g = 0, 0
    here1=False
    here2=False
    here3=False
    stop=False
    # Declare variables that track time and index for events and clusters
    time, trigger_time, ce_index = 0, 0, -1
    # Iterate through data
    for i, word in enumerate(data):
        # Five possibilities: Header, DataBusStart, DataEvent, DataExTs or EoE.
        if (word & TYPE_MASK) == HEADER:
            #print('HEADER')
            str_grch=' '
            str_gradc= ' '
            is_open = True
            is_trigger = (word & TRIGGER_MASK) == TRIGGER
            max_adc_w, max_adc_g = 0, 0
            ce_index += 1
        elif ((word & DATA_MASK) == DATA_BUS_START) & is_open:
            #print('DATA')
            # Extract Bus
            bus = (word & BUS_MASK) >> BUS_SHIFT
            is_data = True
            # Initiate temporary cluster variables and increase cluster index
            previous_bus = bus
        elif ((word & DATA_MASK) == DATA_EVENT) & is_open:
            #print('EVENT')
            # Extract Channel and ADC
            channel = ((word & CHANNEL_MASK) >> CHANNEL_SHIFT)
            adc = (word & ADC_MASK)
            bus = (word & BUS_MASK) >> BUS_SHIFT
            # Wires in bus 0 have channels between 0->63 and bus 1 have chanels betwwen 0 -> 31
            bus_bool= (bus==2 or bus==3)
            channel_bool=(0 <= channel <= 79)
            if ( bus_bool and channel_bool ):
                ch_new=reorder_w_channles(channel)
                if ch_new != -1:
                    if ((bus == 3) and (0 <= ch_new <= 32)):
                        # Save to new channels so all wires are beside eachother in the same bus
                        #print('Bus %d' %bus)
                        #print('Wch %d' %(ch_new ^1))
                        ce_dict['bus'][ce_index] = 9
                        ce_dict['wadc'][ce_index] += adc
                        ce_dict['wm'][ce_index] += 1
                        if adc > max_adc_w: max_adc_w, ce_dict['wch'][ce_index] = adc, ch_new ^ 1
                    elif (bus==2):
                        #print('Bus %d' %bus)
                        # Save to new channels so all wires are beside eachother in the same bus
                        ce_dict['bus'][ce_index] = 9
                        ce_dict['wadc'][ce_index] += adc
                        ce_dict['wm'][ce_index] += 1
                        #print('Wch %d' %((32+ch_new) ^1))
                        if adc > max_adc_w: max_adc_w, ce_dict['wch'][ce_index] = adc, (32+ch_new) ^ 1
                    else:
                        pass
            # Grids in bus 2 have channels 0 to 119
            elif (bus==2) and (80 <= channel <= 119):
                # Give new channelnumber
                ch_new = reorder_gr_channles(channel)
                if (ch_new != -1):
                    # Rename bus to bus 9
                    #print('Bus %d' %bus)
                    #print('Gch %d' %(ch_new ^1))
                    ce_dict['bus'][ce_index] = 9
                    ce_dict['gridch'][ce_index][ce_dict['gm'][ce_index]]=ch_new
                    ce_dict['gridadc'][ce_index][ce_dict['gm'][ce_index]]=adc
                    ce_dict['gm'][ce_index] += 1
                    ce_dict['gadc'][ce_index] += adc
                    # Use grid with largest collected charge as hit position
                    if adc > max_adc_g: max_adc_g, ce_dict['gch_max'][ce_index] = adc, ch_new
        elif ((word & DATA_MASK) == DATA_EXTS) & is_open:
            #print('EXT')
            extended_time_stamp = (word & EXTS_MASK) << EXTS_SHIFT
            is_exts = True
        elif ((word & TYPE_MASK) == EOE) & is_open:
            # Extract time_timestamp and add extended timestamp, if ExTs is used
            time_stamp = (word & TIMESTAMP_MASK)
            time = (extended_time_stamp | time_stamp) if is_exts else time_stamp
            #print('TIME: %d' %time)
            # Update Triggertime, if this was a trigger event
            if is_trigger: trigger_time = time
            # Save cluster data
            if is_data:
                ce_dict['time'][ce_index] = time
                ce_dict['tof'][ce_index] = time - trigger_time
                # See if grids are beside eachother and if there is a jump, how long is it:
                # Only do this bor bus 9 (only one we folow)
                bus = ce_dict['bus'][ce_index]
                if bus==9:
                    idx=ce_dict['gm'][ce_index]
                    if idx > 1:
                        channels=list(ce_dict['gridch'][ce_index][:])
                        if -1 in channels:
                            channels.remove(-1)
                            diff=max(channels)-min(channels)
                            ce_dict['max_dist'][ce_index] = (diff-idx)+1
                            if abs(diff-idx+1)>2:
                                ce_dict['flag_unconnected'][ce_index] = 1  
                        if idx > 2:
                            max_gch=ce_dict['gch'][ce_index]
                            dist_ch= [abs(x-max_gch) for x in channels]
                            dist=max(dist_ch)
                            if dist_ch.count(dist)==1:
                                ce_dict['gch'][ce_index]=channels[dist_ch.index(max(dist_ch))]
                            else:
                                ce_dict['gch'][ce_index]=ce_dict['gch_max'][ce_index]
                        else:
                            ce_dict['gch'][ce_index]=ce_dict['gch_max'][ce_index]
                    else:
                        ce_dict['gch'][ce_index]=ce_dict['gch_max'][ce_index]
                # Reset temporary variables, related to data in events
                previous_bus, bus = -1, -1
                max_adc_w, max_adc_g = 0, 0
                # Reset temporary boolean variables, related to word-headers
                is_open, is_trigger, is_data = False, False, False
                
        # Print progress of clustering process
        if stop:
            break
        if i % (len(data)//10) == 1:
            percentage_finished = int(round((i/len(data))*100))
            # Decide how much of the data should be read
            if percentage_finished>100:
                stop=True
            clear_output(wait=True)
            #print('Percentage: %d' % percentage_finished)
            print('Loading clusters ...')
            print((percentage_finished//10)* '*' +(40-percentage_finished//10)*' ' +  str(percentage_finished) + ' %' )
    
    clear_output(wait=True) 
    print(40*' '+'Finished!')
    # Remove empty elements in clusters and save in DataFrame for easier analysis
    data = None
    # Take out the 2D arrays for the gridchannels and the adc for each grid
    ce_dict['gridch'] = ce_dict['gridch'][0:ce_index]
    ce_dict['gridadc'] = ce_dict['gridadc'][0:ce_index]
    grid_channels=ce_dict.pop('gridch')
    grid_adc=ce_dict.pop('gridadc')
    # Save main data in datframe
    for key in ce_dict:
        ce_dict[key] = ce_dict[key][0:ce_index]
    ce_df = pd.DataFrame(ce_dict)
    # Return dataframe as well as the "D arrays for grid channels and grid adc
    return ce_df , grid_channels, grid_adc


# =============================================================================
#                               EXTRACT EVENTS
# =============================================================================

def extract_events(data):
    """ Extracts the events from the imported data and stores it in a DataFrame

        Does this in the following fashion:
            1. Reads one word at a time
            2. Checks what type of word it is (HEADER, DATA_BUS_START,
               DATA_EVENT, DATA_EXTS or EOE).
            3. Stores vent data into a dictionary
            4. After the iteration through the data is complete, the dictionary
               containing the events is convereted to a DataFrame.

    Explanation of event parameters:
        'bus': Grid column which recorded the cluster
        'ch': Channel
        'adc': Total charge collected by all wires in coincidence
        'time': At what time the event occured

    Args:
        data (tuple): Tuple containing data, one word per element.

    Returns:
        clusters (DataFrame): DataFrame containing one
                              event per row. Each event has
                              information about: "bus", "ch", "adc" and "time"
                              
    In this case this functions combines the data from two buses into a third bus

    """
    size = len(data)
    # Initiate dictionary to store events
    e_dict = {'bus': (-1) * np.ones([size], dtype=int),
              'ch': (-1) * np.ones([size], dtype=int),
              'adc': np.zeros([size], dtype=int),
              'time': (-1) * np.ones([size], dtype=int)}
    # Declare temporary boolean variables
    is_open, is_data = False, False
    # Declare variable that track index for events
    e_index = 0
    e_count = 0
    # Iterate through data
    for i, word in enumerate(data):
        # Five possibilities: Header, DataBusStart, DataEvent, DataExTs or EoE.
        if (word & TYPE_MASK) == HEADER:
            is_open = True
        elif ((word & DATA_MASK) == DATA_BUS_START) & is_open:
            is_data = True
        elif ((word & DATA_MASK) == DATA_EVENT) & is_open:
            # Extract Channel and ADC
            channel = ((word & CHANNEL_MASK) >> CHANNEL_SHIFT)
            adc = (word & ADC_MASK)
            bus = (word & BUS_MASK) >> BUS_SHIFT
            # Wires in bus 0 have channels between 0->63 and 0 -> 31 in bus 1
            bus_bool= (bus==2 or bus==3)
            channel_bool=(0 <= channel <= 79)
            if ( bus_bool and channel_bool ):
                ch_new=reorder_w_channles(channel)
                if (ch_new != -1):
                    e_dict['bus'][e_index] = 9
                    # Save event data and increase event index and event count, save to a third new bus, bus 9
                    if (bus == 3) and (0 <= ch_new <= 31):
                        e_dict['ch'][e_index] = ch_new ^ 1
                    elif (bus==2) :
                        # Save to new channels so all wires are beside eachother in the same bus
                        e_dict['ch'][e_index] = (32+ch_new) ^ 1
                    e_dict['adc'][e_index] = adc
                    e_index += 1
                    e_count += 1
            # Grids have channels between 80 and 119 (bus 3 has no grids atached to it)
            elif (bus==2) and (80 <= channel <= 119):
                ch_new=reorder_gr_channles(channel)
                if ch_new != -1:  
                    # Save event data and increase event index and event count
                    e_dict['bus'][e_index] = 9
                    e_dict['ch'][e_index] = ch_new 
                    e_dict['adc'][e_index] = adc
                    e_index += 1
                    e_count += 1
        elif ((word & DATA_MASK) == DATA_EXTS) & is_open:
            extended_time_stamp = (word & EXTS_MASK) << EXTS_SHIFT
            is_exts = True
        elif ((word & TYPE_MASK) == EOE) & is_open:
            # Extract time_timestamp and add extended timestamp, if ExTs is used
            time_stamp = (word & TIMESTAMP_MASK)
            time = (extended_time_stamp | time_stamp) if is_exts else time_stamp
            if is_data:
                e_dict['time'][e_index-e_count:e_index+1] = time
            # Reset temporary boolean variables, related to word-headers
            is_open, is_trigger, is_data = False, False, False
            e_count = 0

        # Print progress of clustering process
        if i % (len(data)//10) == 1:
            percentage_finished = int(round((i/len(data))*100))
            # Decide how much of the data should be read
            if percentage_finished>100:
                stop=True
            clear_output(wait=True)
            #print('Percentage: %d' % percentage_finished)
            print('Loading events ...')
            print((percentage_finished//10)* '*' +(40-percentage_finished//10)*' ' + str(percentage_finished) + ' %' )   
    clear_output(wait=True)
    print(40*' '+'Finished!')    
    # Remove empty elements in events and save in DataFrame for easier analysis
    for key in e_dict:
        e_dict[key] = e_dict[key][0:e_index+1]
    e_df = pd.DataFrame(e_dict)
    return e_df


# ==============================================================================
#                               HELPER FUNCTIONS
# ==============================================================================

def mkdir_p(my_path):
    """
    Creates a directory, equivalent to using mkdir -p on the command line.

    Args:
        my_path (str): Path to where the new folder should be created.

    Yields:
        A new folder at the requested path.
    """
    from errno import EEXIST
    from os import makedirs,path
    try:
        makedirs(my_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(my_path):
            pass
        else: raise

def reorder_w_channles(ch):
    """
    Returns the channel we want

    Args:
        ch (int): gives a channel some of which will be empty (should not be called but...)

    Yields:
        ch_new (int): channel we want
    """
    list_emty_wch=[0,1,18,19,20,21,38,39,40,41,58,59,60,61,78,79]
    if ch in list_emty_wch:
        return -1
    else:
        row=ch // 20
        level= ch % 20 -2  # -2 since we skipp 2 of the first pinns (and 2 of the last)
        ch_new = row*16+level
        return ch_new
def reorder_gr_channles(ch):
    """
    Returns the channel we want

    Args:
        ch (int): gives a channel some of which will be empty (should not be called but...)

    Yields:
        ch_new (int): channel we want
    """
    list_emty_gr=[80,118,119]
    if ch in list_emty_gr:
        return -1
    else:
        ch_new=ch+15
 
        return ch_new