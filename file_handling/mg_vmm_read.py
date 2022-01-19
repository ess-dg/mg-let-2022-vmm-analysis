#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mg_vmm_read.py: Contains functions which imports data from the Multi-Grid
                detector obtained with the VMM read-out system.
"""

import os
import numpy as np
import pandas as pd
import pcapng as pg

def read_vmm_data(file_path):
    """ Parses data obtained with the VMM readout and stores the obtained
        events in a dataframe.

    Args:
        file_path (str): Path to '.pcapng'-file that contains the data

    Returns:
        events_df (pd.DataFrame): DataFrame containing the events

    """
    # Define parameters
    packet_type = pg.blocks.EnhancedPacket
    ess_cookie = b'ESS'
    event_size = 20 # In bytes
    # Iterate through data
    with open(file_path, mode='rb') as bin_file:
        scanner = pg.FileScanner(bin_file)
        packets = [block for block in scanner if type(block) is packet_type]
        est_size = sum([packet.packet_len//event_size for packet in packets])
        # Pre-allocated space for events
        events_dict = {'instrument_id': (-1) * np.ones([est_size], dtype=int),
                       'data_length': (-1) * np.ones([est_size], dtype=int),
                       'OutputQ': (-1) * np.ones([est_size], dtype=int),
                       'TimeSrc': (-1) * np.ones([est_size], dtype=int),
                       'PulseTimeHI': (-1) * np.ones([est_size], dtype=int),
                       'PulseTimeLO': (-1) * np.ones([est_size], dtype=int),
                       'PrevPulseTimeHI': (-1) * np.ones([est_size], dtype=int),
                       'PrevPulseTimeLO': (-1) * np.ones([est_size], dtype=int),
                       'SeqNo': (-1) * np.ones([est_size], dtype=int),
                       'ring': (-1) * np.ones([est_size], dtype=int),
                       'fen': (-1) * np.ones([est_size], dtype=int),
                       'length': (-1) * np.ones([est_size], dtype=int),
                       'time_hi': (-1) * np.ones([est_size], dtype=int),
                       'time_lo': (-1) * np.ones([est_size], dtype=int),
                       'bc': (-1) * np.ones([est_size], dtype=int),
                       'ot': (-1) * np.ones([est_size], dtype=int),
                       'adc': (-1) * np.ones([est_size], dtype=int),
                       'om': (-1) * np.ones([est_size], dtype=int),
                       'geo': (-1) * np.ones([est_size], dtype=int),
                       'tdc': (-1) * np.ones([est_size], dtype=int),
                       'vmm': (-1) * np.ones([est_size], dtype=int),
                       'channel': (-1) * np.ones([est_size], dtype=int),
                       'packet_number': (-1) * np.ones([est_size], dtype=int),
                       'readout_number': (-1) * np.ones([est_size], dtype=int)}
        # Fill space with events
        counter = 0
        for i, packet in enumerate(packets):
            data = packet.packet_data
            idx_0 = data.find(ess_cookie) + len(ess_cookie)
            start, end = idx_0+25, len(data)
            readouts = [data[i:i+event_size] for i in range(start, end, event_size)]
            # Extract event data
            for j, readout in enumerate(readouts):
                events_dict['ring'][counter] = readout[0]
                events_dict['fen'][counter] = readout[1]
                events_dict['length'][counter] = int.from_bytes(readout[2:4], byteorder='little')
                events_dict['time_hi'][counter] = int.from_bytes(readout[4:8], byteorder='little')
                events_dict['time_lo'][counter] = int.from_bytes(readout[8:12], byteorder='little')
                events_dict['bc'][counter] = int.from_bytes(readout[12:14], byteorder='little')
                events_dict['ot'][counter] = int.from_bytes(readout[14:16], byteorder='little') >> 15
                events_dict['adc'][counter] = int.from_bytes(readout[14:16], byteorder='little') & 0x3FF
                events_dict['om'][counter] = readout[16] >> 7
                events_dict['geo'][counter] = readout[16] & 0x3F
                events_dict['tdc'][counter] = readout[17]
                events_dict['vmm'][counter] = readout[18]
                events_dict['channel'][counter] = readout[19]
                events_dict['packet_number'][counter] = i + 1
                events_dict['readout_number'][counter] = j + 1
                counter += 1
            # Extract header data
            idxs = np.arange(counter-len(readouts), counter, 1)
            events_dict['instrument_id'][idxs] = data[idx_0]
            events_dict['data_length'][idxs] = int.from_bytes(data[idx_0+1:idx_0+3], byteorder='little')
            events_dict['OutputQ'][idxs] = data[idx_0+3]
            events_dict['TimeSrc'][idxs] = data[idx_0+4]
            events_dict['PulseTimeHI'][idxs] = int.from_bytes(data[idx_0+5:idx_0+9], byteorder='little')
            events_dict['PulseTimeLO'][idxs] = int.from_bytes(data[idx_0+9:idx_0+13], byteorder='little')
            events_dict['PrevPulseTimeHI'][idxs] = int.from_bytes(data[idx_0+13:idx_0+17], byteorder='little')
            events_dict['PrevPulseTimeLO'][idxs] = int.from_bytes(data[idx_0+17:idx_0+21], byteorder='little')
            events_dict['SeqNo'][idxs] = int.from_bytes(data[idx_0+21:idx_0+25], byteorder='little')
    # Remove empty elements
    for key in events_dict:
        events_dict[key] = events_dict[key][0:counter]
    # Convert to dataframe
    events_df = pd.DataFrame(events_dict)
    return events_df
