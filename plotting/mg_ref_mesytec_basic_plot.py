#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mg_basic_plot.py: Contains the basic functions to plot Multi-Grid data.
"""
import os
import matplotlib.font_manager
matplotlib.font_manager._rebuild()
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

import plotting.helper_functions as plotting_hf
import file_handling.mg_mesytec_manage_ref as mg_manage
import matplotlib.font_manager







# ==============================================================================
#                                   PHS (1D)
# ==============================================================================

def phs_1d_plot(clusters, clusters_uf, number_bins, bus, duration):
    """
    Plot the 1D PHS with and without filters.

    Args:
        clusters (DataFrame): Clustered events, filtered
        clusters_uf (DataFrame): Clustered events, unfiltered
        number_bins (int): Number of bins to use in the histogram
        bus (int): Bus to plot
        duration (float): Measurement duration in seconds

    Yields:
        Figure containing a 1D PHS plot
    """
    # Clusters filtered
    plt.hist(clusters.wadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 8000], label='Wires (filtered)', color='blue',
             weights=(1/duration)*np.ones(len(clusters.wadc)))
    plt.hist(clusters.gadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 8000], label='Grids (filtered)', color='red',
             weights=(1/duration)*np.ones(len(clusters.gadc)))
    # Clusters unfiltered
    hist_w, bins_w, __ = plt.hist(clusters_uf.wadc, bins=number_bins,
                                  histtype='step', zorder=5, range=[0, 8000],
                                  label='Wires', color='cyan',
                                  weights=(1/duration)*np.ones(len(clusters_uf.wadc)))
    hist_g, bins_g, __ = plt.hist(clusters_uf.gadc, bins=number_bins,
                                  histtype='step', zorder=5, range=[0, 8000],
                                  label='Grids', color='magenta',
                                  weights=(1/duration)*np.ones(len(clusters_uf.gadc)))
    plt.title('Bus: %d' % bus)
    plt.xlabel('Charge (ADC channels)')
    plt.ylabel('Counts/s')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    #plt.ylim(1e-5, 1)
    plt.legend()
    # Save histograms
    bins_w_c = 0.5 * (bins_w[1:] + bins_w[:-1])
    bins_g_c = 0.5 * (bins_g[1:] + bins_g[:-1])
    dirname = os.path.dirname(__file__)
    path_wires = os.path.join(dirname,
                              '../output/seq_phs_unfiltered_wires_bus_%d.txt' % bus)
    path_grids = os.path.join(dirname,
                              '../output/seq_phs_unfiltered_grids_bus_%d.txt' % bus)
    np.savetxt(path_wires,
               np.transpose(np.array([bins_w_c, hist_w])),
               delimiter=",",header='bins, hist (counts/s)')
    np.savetxt(path_grids,
               np.transpose(np.array([bins_g_c, hist_g])),
               delimiter=",",header='bins, hist (counts/s)')


# ==============================================================================
#                                   PHS (2D)
# ==============================================================================

def phs_2d_plot(events, bus, vmin, vmax):
    """
    Histograms the ADC-values from each channel individually and summarises it
    in a 2D histogram plot, where the color scale indicates number of counts.
    Each bus is presented in an individual plot.

    Args:
        events (DataFrame): DataFrame containing individual events
        bus (int): Bus to plot
        vmin (float): Minimum value on color scale
        vmax (float): Maximum value on color scale

    Yields:
        Plot containing 2D PHS heatmap plot
    """
    plt.xlabel('Channel')
    plt.ylabel('Charge (ADC channels)')
    plt.title('Bus: %d' % bus)
    bins = [133,4095]
    if events.shape[0] > 1:
        plt.hist2d(events.ch, events.adc, bins=bins,
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   range=[[-0.5, 132.5], [0, 4400]],
                   cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Counts')


# ==============================================================================
#                           PHS (Wires vs Grids)
# ==============================================================================


def clusters_phs_plot(clusters, bus, duration, vmin, vmax):
    """
    Histograms ADC charge from wires vs grids, one for each bus, showing the
    relationship between charge collected by wires and charge collected by
    grids. In the ideal case, there should be linear relationship between these
    two quantities.

    Args:
        clusters (DataFrame): Clustered events
        bus (int): Bus to plot
        duration (float): Measurement duration in seconds
        vmin (float): Minimum value on color scale
        vmax (float): Maximum value on color scale

    Yields:
        Plot containing the PHS 2D scatter plot

    """

    plt.xlabel('Charge wires (ADC channels)')
    plt.ylabel('Charge grids (ADC channels)')
    plt.title('Bus: %d' % bus)
    bins = [200, 200]
    ADC_range = [[0, 10000], [0, 10000]]
    plt.hist2d(clusters.wadc, clusters.gadc, bins=bins,
               norm=LogNorm(vmin=vmin, vmax=vmax),
               range=ADC_range,
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wadc)))
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('Counts/s')


# ==============================================================================
#                          COINCIDENCE HISTOGRAM (2D)
# ==============================================================================

def clusters_2d_plot(clusters, title, vmin, vmax, duration):
    """
    Plots a 2D histograms of clusters: wires vs grids.

    Args:
        clusters (DataFrame): Clustered events, filtered
        vmin (float): Minimum value on color scale
        vmax (float): Maximum value on color scale
        duration (float): Measurement duration in seconds

    Yields:
        Plot containing the 2D coincidences

    """

    try:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch_max, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    except:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
        
    plt.xlabel('Wire (Channel number)')
    plt.ylabel('Grid (Channel number)')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    

# ==============================================================================
#                          COINCIDENCE SURFACE (3D)
# ==============================================================================

def clusters_3d_plot(clusters, title, vmin, vmax, duration):
    """
    Plots a 2D histograms of clusters: wires vs grids.

    Args:
        clusters (DataFrame): Clustered events, filtered
        vmin (float): Minimum value on color scale
        vmax (float): Maximum value on color scale
        duration (float): Measurement duration in seconds

    Yields:
        Plot containing the 2D coincidences

    """

    hf=plt.figure()
    try:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch_max, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    except:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    ha = hf.add_subplot(111, projection='3d')   
    x=[i+0.5 for i in xedges]
    y=[i+0.5 for i in yedges]
    X,Y=np.meshgrid(x[:-1],yedges[:-1])
    print(len(X))
    print(len(Y))
    print(len(hist[0]))
    ha.plot_surface(X,Y,np.transpose(hist),cmap=cm.jet,linewidth=0,antialiased=False)
    plt.xlabel('Wire (Channel number)')
    plt.ylabel('Grid (Channel number)')
    plt.title(title)

# ==============================================================================
#                               MULTIPLICITY
# ==============================================================================

def multiplicity_plot(clusters, bus, duration, vmin=None, vmax=None):
    """
    Plots a 2D histograms of wire-grid event multiplicity in the clusters

    Args:
        clusters (DataFrame): Clustered events, filtered
        bus (int): Bus to plot
        duration (float): Measurement duration in seconds
        vmin (float): Minimum value on color scale
        vmax (float): Maximum value on color scale

    Yields:
        Plot containing the 2D multiplicity distribution

    """
    # Declare parameters
    m_range = [0, 10, 0, 10]
    # Plot data
    hist, xbins, ybins, im = plt.hist2d(clusters.wm, clusters.gm,
                                        bins=[m_range[1]-m_range[0]+1,
                                              m_range[3]-m_range[2]+1],
                                        range=[[m_range[0], m_range[1]+1],
                                               [m_range[2], m_range[3]+1]],
                                        norm=LogNorm(vmin=vmin, vmax=vmax),
                                        cmap='jet',
                                        weights=(1/duration)*np.ones(len(clusters.wm)))
    # Iterate through all squares and write percentages
    tot = clusters.shape[0] * (1/duration)
    font_size = 6.5
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            if hist[j, i] > 0:
                text = plt.text(xbins[j]+0.5, ybins[i]+0.5,
                                '%.0f%%' % (100*(hist[j, i]/tot)),
                                color="w", ha="center", va="center",
                                fontweight="bold", fontsize=font_size)
                text.set_path_effects([path_effects.Stroke(linewidth=1,
                                                           foreground='black'),
                                       path_effects.Normal()])
    # Set ticks on axis
    ticks_x = np.arange(m_range[0], m_range[1]+1, 1)
    locs_x = np.arange(m_range[0] + 0.5, m_range[1]+1.5, 1)
    ticks_y = np.arange(m_range[2], m_range[3]+1, 1)
    locs_y = np.arange(m_range[2] + 0.5, m_range[3]+1.5, 1)
    plt.xticks(locs_x, ticks_x)
    plt.yticks(locs_y, ticks_y)
    plt.xlabel("Wire multiplicity")
    plt.ylabel("Grid multiplicity")
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    plt.title('Bus: %d' % bus)
    #plt.tight_layout()


# ==============================================================================
#                                       RATE
# ==============================================================================


def rate_plot(clusters, number_bins, bus):
    """
    Histograms the rate as a function of time.

    Args:
        clusters (DataFrame): Clustered events
        number_bins (int): The number of bins to histogram the data into
        bus (int): Bus to plot

    Yields:
        Plot containing the rate as a function of time

    """

    # Prepare figure
    plt.title('Bus: %d' % bus)
    plt.xlabel('Time (hours)')
    plt.ylabel('Rate (events/s)')
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    # Plot
    time = (clusters.time * 62.5e-9)/(60 ** 2)
    hist, bin_edges = np.histogram(time, bins=number_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    delta_t = (60 ** 2) * (bin_centers[1] - bin_centers[0])
    plt.errorbar(bin_centers, (hist/delta_t), np.sqrt(hist)/delta_t,
                 marker='.', linestyle='', zorder=5, color='black')


# ==============================================================================
#                              UNIFORMITY - GRIDS
# ==============================================================================

def grid_histogram(clusters, bus, duration):
    """
    Histograms the counts in each grid.

    Args:
        clusters (DataFrame): Clustered events
        bus(int): The bus of the data
        duration (float): Measurement duration in seconds

    Yields:
        Plot containing the grid histogram
    """

    # Prepare figure
    plt.title('Bus: %d' % bus)
    plt.xlabel('Grid channel')
    plt.ylabel('Counts/s')
    #plt.yscale('log')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    # Histogram data
    try:
        plt.hist(clusters.gch_max, bins=37, zorder=4, range=[95.5, 132.5],
             weights=(1/duration)*np.ones(len(clusters.gch)),
             histtype='step', color='black')
    except:
        plt.hist(clusters.gch, bins=37, zorder=4, range=[95.5, 132.5],
             weights=(1/duration)*np.ones(len(clusters.gch)),
             histtype='step', color='black')
        


# ==============================================================================
#                              UNIFORMITY - WIRES
# ==============================================================================

def wire_histogram(clusters, bus, duration):
    """
    Histograms the counts in each wire.

    Args:
        clusters (DataFrame): Clustered events
        bus(int): The bus of the data
        duration (float): Measurement duration in seconds

    Yields:
        Plot containing the wire histogram
    """

    # Prepare figure
    plt.title('Bus: %d' % bus)
    plt.xlabel('Wire channel')
    plt.ylabel('Counts/s')
    #plt.yscale('log')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    # Histogram data
    plt.hist(clusters.wch, bins=96, zorder=4, range=[-0.5, 95.5],
             weights=(1/duration)*np.ones(len(clusters.wch)),
             histtype='step', color='black')


# ==============================================================================
#                      PLOT ALL BASIC PLOTS FOR ONE BUS
# ==============================================================================

def mg_plot_basic_bus(run, bus, clusters_unfiltered, events, df_filter, area,save=False,
                      plot_title=''):
    """
    Function to plot all basic plots for SEQUOIA detector, for a single bus,
    such as PHS, Coincidences and rate.

    Ordering of plotting is:

    PHS 2D             - NOT FILTERED
    PHS 1D             - FILTERED AND NOT FILTERED
    MULTIPLICITY       - FILTERED
    PHS CORRELATION    - FILTERED
    COINCIDENCES 2D    - FILTERED
    RATE               - FILTERED
    UNIFORMITY (WIRES) - FILTERED
    UNIFORMITY (GRIDS) - FILTERED

    Note that all plots are filtered except the PHS 2D.

    Args:
        run (str): File run
        bus (int): Bus to plot
        clusters_unfiltered (DataFrame): Unfiltered clusteres
        events (DataFrame): Individual events
        df_filter (dict): Dictionary specifying the filter which will be used
                          on the clustered data
        area (float): Area in m^2 of the active detector surface
        plot_title (str): Title of PLOT

    Yields:
        Plots the basic analysis

    """

    plotting_hf.set_thick_labels(15)
    try:
        os.mkdir('../output/%s_%d' % (run,bus))
    except:
        pass
    output_path = '../output/%s_%d/%s_summary_bus_%d' % (run,bus,run, bus)

    # Filter clusters
    clusters = mg_manage.filter_data(clusters_unfiltered, df_filter)

    # Declare parameters
    duration_unf = ((clusters_unfiltered.time.values[-1]
                    - clusters_unfiltered.time.values[0]) * 62.5e-9)
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9

    # Filter data from only one bus
    events_bus = events[events.bus == bus]
    clusters_bus = clusters[clusters.bus == bus]
    clusters_uf_bus = clusters_unfiltered[clusters_unfiltered.bus == bus]

    fig = plt.figure()


    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    plt.suptitle(plot_title, fontsize=15, fontweight='bold', y=1.00005)
    # PHS - 2D
    plt.subplot(4, 2, 1)
    vmin = 1
    vmax = events.shape[0] // 1000 + 100
    if events_bus.shape[0] > 0:
        phs_2d_plot(events_bus, bus, vmin, vmax)
    plt.title('PHS vs Channel')

    # PHS - 1D
    plt.subplot(4, 2, 2)
    bins_phs_1d = 300
    phs_1d_plot(clusters_bus, clusters_uf_bus, bins_phs_1d, bus, duration)
    plt.yscale('log')
    plt.title('PHS')

    # Coincidences - 2D
    plt.subplot(4, 2, 5)
    if clusters.shape[0] != 0:
        vmin = (1 * 1/duration)
        vmax = (clusters.shape[0] // 450 + 5) * 1/duration
    else:
        duration = 1
        vmin = 1
        vmax = 1

    number_events = clusters_bus.shape[0]
    number_events_error = np.sqrt(clusters_bus.shape[0])
    events_per_s = number_events/duration
    events_per_s_m2 = events_per_s/area
    events_per_s_m2_error = number_events_error/(duration*area)
    title = ('Coincidences\n(%d events, %.3f±%.3f events/s/m$^2$)' % (number_events,
                                                                      events_per_s_m2,
                                                                      events_per_s_m2_error))
    if number_events > 1:
        clusters_2d_plot(clusters_bus, title, vmin, vmax, duration)

    # Rate
    plt.subplot(4, 2, 6)
    number_bins = 40
    rate_plot(clusters_bus, number_bins, bus)
    plt.title('Rate vs time')

    # Multiplicity
    plt.subplot(4, 2, 3)
    if clusters_bus.shape[0] > 1:
        multiplicity_plot(clusters_bus, bus, duration)
    plt.title('Event multiplicity')


    # Coincidences - PHS
    plt.subplot(4, 2, 4)
    if clusters.shape[0] != 0:
        vmin = 1/duration
        vmax = (clusters.shape[0] // 450 + 1000) / duration
    else:
        duration = 1
        vmin = 1
        vmax = 1
    if clusters_bus.shape[0] > 1:
        clusters_phs_plot(clusters_bus, bus, duration, vmin, vmax)
    plt.title('Charge coincidences')

    # Uniformity - grids
    plt.subplot(4, 2, 8)
    grid_histogram(clusters_bus, bus, duration)
    plt.title('Uniformity - grids')

    # Uniformity - wires
    plt.subplot(4, 2, 7)
    wire_histogram(clusters_bus, bus, duration)
    plt.title('Uniformity - wires')

    # Save data
    fig.set_figwidth(10)
    fig.set_figheight(16)
    plt.tight_layout()
    if save:
        fig.savefig(output_path+'.png', bbox_inches='tight')
        # open file for writing the filter
        f = open("../output/%s_%d/filter.txt" % (run,bus),"w+")
        # write file
        f.write( str(df_filter) )
        # close file
        f.close()
        mg_save_plot_basic_bus(run, bus, clusters_unfiltered, events, df_filter, area,save=True,
                      plot_title='')
        
# ==============================================================================
#                      Save all basic plots to new foalder
# ==============================================================================

def mg_save_plot_basic_bus(run, bus, clusters_unfiltered, events, df_filter, area,save=False,
                      plot_title=''):
    """
    Function to plot all basic plots for SEQUOIA detector, for a single bus,
    such as PHS, Coincidences and rate.

    Ordering of plotting is:

    PHS 2D             - NOT FILTERED
    PHS 1D             - FILTERED AND NOT FILTERED
    MULTIPLICITY       - FILTERED
    PHS CORRELATION    - FILTERED
    COINCIDENCES 2D    - FILTERED
    RATE               - FILTERED
    UNIFORMITY (WIRES) - FILTERED
    UNIFORMITY (GRIDS) - FILTERED

    Note that all plots are filtered except the PHS 2D.

    Args:
        run (str): File run
        bus (int): Bus to plot
        clusters_unfiltered (DataFrame): Unfiltered clusteres
        events (DataFrame): Individual events
        df_filter (dict): Dictionary specifying the filter which will be used
                          on the clustered data
        area (float): Area in m^2 of the active detector surface
        plot_title (str): Title of PLOT

    Yields:
        Plots the basic analysis

    """

    plotting_hf.set_thick_labels(15)
    output_path = '../output/%s_%d/%s_summary_bus_%d' % (run,bus,run, bus)

    # Filter clusters
    clusters = mg_manage.filter_data(clusters_unfiltered, df_filter)

    # Declare parameters
    duration_unf = ((clusters_unfiltered.time.values[-1]
                    - clusters_unfiltered.time.values[0]) * 62.5e-9)
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9

    # Filter data from only one bus
    events_bus = events[events.bus == bus]
    clusters_bus = clusters[clusters.bus == bus]
    clusters_uf_bus = clusters_unfiltered[clusters_unfiltered.bus == bus]

    fig = plt.figure()
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    # PHS - 2D
    vmin = 1
    vmax = events.shape[0] // 1000 + 100
    if events_bus.shape[0] > 0:
        phs_2d_plot(events_bus, bus, vmin, vmax)
    plt.title('PHS vs Channel')
    fig.savefig(output_path+'_PHS_vs_chan.png', bbox_inches='tight')

    # PHS log - 1D
    fig = plt.figure()
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    bins_phs_1d = 300
    phs_1d_plot(clusters_bus, clusters_uf_bus, bins_phs_1d, bus, duration)
    plt.yscale('log')
    plt.title('PHS')
    fig.savefig(output_path+'_PHS_log.png', bbox_inches='tight')
    
    # PHS lin - 1D
    fig = plt.figure()
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    bins_phs_1d = 300
    phs_1d_plot(clusters_bus, clusters_uf_bus, bins_phs_1d, bus, duration)
    plt.title('PHS')
    fig.savefig(output_path+'_PHS_lin.png', bbox_inches='tight')

    # Coincidences - 2D
    fig = plt.figure()
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    if clusters.shape[0] != 0:
        vmin = (1 * 1/duration)
        vmax = (clusters.shape[0] // 450 + 5) * 1/duration
    else:
        duration = 1
        vmin = 1
        vmax = 1

    number_events = clusters_bus.shape[0]
    number_events_error = np.sqrt(clusters_bus.shape[0])
    events_per_s = number_events/duration
    events_per_s_m2 = events_per_s/area
    events_per_s_m2_error = number_events_error/(duration*area)
    title = ('Coincidences\n(%d events, %.3f±%.3f events/s/m$^2$)' % (number_events,
                                                                      events_per_s_m2,
                                                                      events_per_s_m2_error))
    if number_events > 1:
        clusters_2d_plot(clusters_bus, title, vmin, vmax, duration)
    fig.savefig(output_path+'_coince_num.png', bbox_inches='tight')

    # Rate
    fig = plt.figure()
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    number_bins = 40
    rate_plot(clusters_bus, number_bins, bus)
    plt.title('Rate vs time')
    plt.yscale('log')
    fig.savefig(output_path+'_Rate_vs_time_log.png', bbox_inches='tight')
    
    # Rate Lin
    fig = plt.figure()
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    number_bins = 40
    rate_plot(clusters_bus, number_bins, bus)
    plt.title('Rate vs time')
    fig.savefig(output_path+'_Rate_vs_time_lin..png', bbox_inches='tight')
    
    # Multiplicity
    fig = plt.figure()
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    if clusters_bus.shape[0] > 1:
        multiplicity_plot(clusters_bus, bus, duration)
    plt.title('Event multiplicity')
    fig.savefig(output_path+'_Clu_mul.png', bbox_inches='tight')

    # Coincidences - PHS
    fig = plt.figure()
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    if clusters.shape[0] != 0:
        vmin = 1/duration
        vmax = (clusters.shape[0] // 450 + 1000) / duration
    else:
        duration = 1
        vmin = 1
        vmax = 1
    if clusters_bus.shape[0] > 1:
        clusters_phs_plot(clusters_bus, bus, duration, vmin, vmax)
    plt.title('Charge coincidences')
    fig.savefig(output_path+'_coince_charge.png', bbox_inches='tight')

    # Uniformity - grids
    fig = plt.figure()
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    grid_histogram(clusters_bus, bus, duration)
    plt.title('Uniformity - grids')
    fig.savefig(output_path+'_Uni_gr.png', bbox_inches='tight')

    # Uniformity - wires
    fig = plt.figure()
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Computer Modern']
    wire_histogram(clusters_bus, bus, duration)
    plt.title('Uniformity - wires')
    fig.savefig(output_path+'_Uni_wi.png', bbox_inches='tight')

# ==============================================================================
#          PLOT Charge distrobution for clusters with more than one grid 
# ==============================================================================

def mg_plot_grid_distrobution(clusters_unfiltered,gm=5,num_to_plot=1,
                      plot_title=''):
    """
    Function to plot the charge distrobution for the detection with the most grids connected to it.
    This to be able to see if the "brag peak" is pressent in the case of more grids being activated. Returns number of events with this multiplicety for the data provided.

    """
    fig = plt.figure()   
    plotting_hf.set_thick_labels(15)
    
    indx_most_grids=(clusters_unfiltered.gm).idxmax()
    indx_most_charge=(clusters_unfiltered.gadc).idxmax()
    
    clusters = clusters_unfiltered[clusters_unfiltered.gm==gm]
    #print(clusters)
    index=clusters.index.tolist()
    num_itms=len(index)
    if num_itms> num_to_plot:
        index=index[0:num_to_plot]
    if num_itms<=0: 
        print('No events with this multiplicety: ', gm)
        return 0
        
    print('Number of events with this multiplicety: ', num_itms)
    for ind in index:
        channel_nr=[clusters.gch1[ind], clusters.gch2[ind], clusters.gch3[ind],clusters.gch4[ind],clusters.gch5[ind],clusters.gch[ind]]
        charge=[clusters.gadc1[ind], clusters.gadc2[ind], clusters.gadc3[ind],clusters.gadc4[ind],clusters.gadc5[ind],clusters.gadc[ind]]
        plt.scatter(channel_nr,charge)
        print('The number of "skipped" grids are: ',clusters.max_dist[ind])
    plt.show()
    return num_itms  


# ==============================================================================
#          PLOT Pulse height spectra for 3 buses
# ==============================================================================

def mg_plot_pulses(title, bus0, bus1, bus2, clusters0, events0, clusters1, events1, clusters2, event2):
    """
    This function plottes the pulse height spectra from 3 busses. Obs: all filters and such have to be aplied beforehand
    
    Args:
        title: title of plot and filename
        bus0 (int): Firts bus
        bus1 (int): Second bus
        bus2 (int): Third bus
        clusters0: dataframe with all clusters belonging to bus0
        event0: dataframe with all events belonging to bus0
        clusters1: dataframe with all clusters belonging to bus1
        event1: dataframe with all events belonging to bus1
        clusters2: dataframe with all clusters belonging to bus2
        event2: dataframe with all events belonging to bus2

    Yields:
        Plots a superposition of the PHS from 3 busses

    """
    fig = plt.figure()
    number_bins=100
    if clusters0.size>100: 
        duration0 = (clusters0.time.values[-1] - clusters0.time.values[0]) * 62.5e-9
        plt.hist(clusters0.wadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 5000], label='Wires: bus %d' % bus0, color='forestgreen',
             weights=(1/duration0)*np.ones(len(clusters0.wadc)))
        plt.hist(clusters0.gadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 5000], label='Grids: bus %d' % bus0, color='lime',
             weights=(1/duration0)*np.ones(len(clusters0.gadc)))
    
    if clusters1.size>100:
        duration1 = (clusters1.time.values[-1] - clusters1.time.values[0]) * 62.5e-9
        plt.hist(clusters1.wadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 5000], label='Wires: bus %d' % bus1, color='darkorange',
             weights=(1/duration1)*np.ones(len(clusters1.wadc)))
        plt.hist(clusters1.gadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 5000], label='Grids: bus %d' % bus1, color='brown',
             weights=(1/duration1)*np.ones(len(clusters1.gadc)))
        
    if clusters2.size>100:
        duration2 = (clusters2.time.values[-1] - clusters2.time.values[0]) * 62.5e-9
        plt.hist(clusters2.wadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 5000], label='Wires: bus %d' % bus2, color='blue',
             weights=(1/duration2)*np.ones(len(clusters2.wadc)))
        plt.hist(clusters2.gadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 5000], label='Grids: bus %d' % bus2, color='magenta',
             weights=(1/duration2)*np.ones(len(clusters2.gadc)))
   
    plt.xlabel('Charge (ADC channels)')
    plt.ylabel('Counts/s')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    #plt.ylim(1e-5, 1)
    plt.title(title)
    plt.legend()
    #plt.yscale('log')
    plt.show()
    output_path = '../output/%s_PHS.png' % title
    plt.savefig(output_path, bbox_inches='tight')
    
# ==============================================================================
#          PLOT Charge distrobution for clusters with more than one grid 
# ==============================================================================

def plot_charge_distr(clusters_unfiltered,grid_channels, grid_adc):
    """
    Function to plot the charge distrobution in a histogram for the detection with the most grids connected to it.
    This to be able to see if the "brag peak" is pressent in the case of more grids being activated. Returns number of events with this multiplicety for the data provided.

    """
    fig = plt.figure()   
    plotting_hf.set_thick_labels(15)

    
    clusters = clusters_unfiltered[clusters_unfiltered.gm>2]
    #print(clusters)
    index=clusters.index.tolist()
    num_itms=len(index)
    charges_tot=np.zeros(40,dtype=int)
    channels_all=[]
    channels_all_adc=[]
    if num_itms<=0: 
        print('No clusters with more than 2 grids in coincidence')
        return 0
    print('Number of clusters with more than 2 grids in coincidence: ', num_itms)
    for ind in index:
        gm= clusters.gm[ind]
        gr_max= clusters.gch_max[ind]
        channel_nr=list(grid_channels[ind][:gm])
        if -1 in channel_nr:
            print(channel_nr)
            channel_nr.remove(-1)
        
        dist_ch= [(x-gr_max) for x in channel_nr]
        abs_dist_ch= [abs(x-gr_max) for x in channel_nr]
        if abs_dist_ch.count(max(abs_dist_ch))==1:
            if dist_ch[abs_dist_ch.index(max(abs_dist_ch))] > 0:
                dist_ch= [(-x) for x in dist_ch] 
        charge=list(grid_adc[ind][:gm])
        channels_all += dist_ch
        channels_all_adc += charge
        if len([*filter(lambda x: x < -40, dist_ch)]) > 0:
            print('Channel list: ', channel_nr)
            print('Max channel: ',gr_max)
            print('Distance list: ', dist_ch)
        for ch_ind in range(len(dist_ch)):
            charges_tot[dist_ch[ch_ind]+20] += charge[ch_ind]
    
    plt.hist(channels_all,bins=11,range=[-5.5,5.5], weights=channels_all_adc)  
    #plt.plot(charges_tot)
    #plt.yscale('log')
    plt.show()
    
    return charges_tot

# ==============================================================================
#          PLOT PHS for a list of wires from a dataset with clusters
# ==============================================================================


def mg_plot_wires(title, clusters, wires, save):
    """
    This function plots the pulse height spectra from a list of wires
    
    Args:
        title: title of plot and filename
        clusters (df): dataframe of clusters
        wires (list): List of wires to look at
        save (bolean): True if the image is to be saved, false otherwise

    Yields:
        Plots a superposition of the PHS from the wires in the list

    """
    fig = plt.figure()
    number_bins=50
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
    colors=['blue', 'green', 'orange', 'firebrick', 'magenta']
    int=0
    for w in wires:
        clusters_w=clusters[clusters['wch']==w]
        plt.hist(clusters_w.wadc, bins=number_bins, histtype='step', color =colors[int],
                 zorder=5, range=[0, 5000], label='Wire:  %d' % w,
                 weights=(1/duration)*np.ones(len(clusters_w.wadc)))
        int +=1
    plt.xlabel('Charge (ADC channels)')
    plt.ylabel('Counts/s')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    #plt.ylim(1e-5, 1)
    plt.title(title)
    plt.legend()
    plt.yscale('log')
    plt.show()
    fig.set_figwidth(7)
    fig.set_figheight(5)
    plt.tight_layout()
    if save: 
        output_path = '../output/%s_PHS.png' % title
        plt.savefig(output_path, bbox_inches='tight')

# ==============================================================================
#          PLOT PHS for a list of wires from a dataset with clusters
# ==============================================================================


def mg_plot_wires_sep(title, clusters, wires, save):
    """
    This function plots the pulse height spectra from a list of wires
    
    Args:
        title: title of plot and filename
        clusters (df): dataframe of clusters
        wires (list): List of wires to look at
        save (bolean): True if the image is to be saved, false otherwise

    Yields:
        Plots a superposition of the PHS from the wires in the list

    """
    number_bins=50
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
    colors=['blue', 'green', 'orange', 'firebrick', 'magenta']
    int=0;
    for w in wires:
        fig = plt.figure()
        clusters_w=clusters[clusters['wch']==w]
        plt.hist(clusters_w.wadc, bins=number_bins, histtype='step',
                 zorder=5, range=[0, 5000], label='Wire:  %d' % w, color =colors[int],
                 weights=(1/duration)*np.ones(len(clusters_w.wadc)))
        plt.xlabel('Charge (ADC channels)')
        plt.ylabel('Counts/s')
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title(title)
        plt.legend()
        plt.yscale('log')
        plt.show()
        int += 1
    mg_plot_wires(title, clusters, wires, save)

# ===================================================================================================================================
#        Plot the charge distrobution with: linefitting + standard deviation in x and y
# ===================================================================================================================================


def mg_charge_dist(title, clusters_use, bus, save):    
    """
    This function plots the pulse height spectra from a list of wires
    
    Args:
        title: title of plot and filename
        clusters (df): dataframe of clusters
        save (bolean): True if the image is to be saved, false otherwise

    Yields:
        Plots a charge distrobution with a linefitting inclusing the standard deviation in x and y

    """
    from matplotlib.colors import LogNorm
    from scipy.stats import linregress
    fig = plt.figure()
    duration = (clusters_use.time.values[-1] - clusters_use.time.values[0]) * 62.5e-9
    vmin = 1/duration
    vmax = (clusters_use.shape[0] // 450 + 1000) / duration
    plt.xlabel('Charge wires (ADC channels)')
    plt.ylabel('Charge grids (ADC channels)')
    bins = [4095, 4095]
    ADC_range = [[0, 4095], [0, 4095]]
    plt.hist2d(clusters_use.wadc, clusters_use.gadc, bins=bins,norm=LogNorm(vmin=vmin, vmax=vmax),range=ADC_range, cmap='jet',weights=(1/duration)*np.ones(len(clusters_use.wadc)))
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    result = linregress(clusters_use.wadc,clusters_use.gadc)
    stand_x=np.sqrt(sum([i**2 for i in (clusters_use.wadc-((clusters_use.gadc-result.intercept)/result.slope))])/len(clusters_use.wadc))
    stand_y=np.sqrt(sum([i**2 for i in (clusters_use.gadc-(clusters_use.wadc*result.slope+result.intercept))])/len(clusters_use.gadc))
    plt.plot(clusters_use.wadc,result.intercept+result.slope*clusters_use.wadc,color='black')
    plt.plot(clusters_use.wadc,result.intercept+stand_y+result.slope*clusters_use.wadc, color='green',label='S gradc:  %.3f' % stand_y)
    plt.plot(clusters_use.wadc,result.intercept-stand_y+result.slope*clusters_use.wadc, color='green',label='S gradc:  %.3f' % stand_y)
    plt.plot(clusters_use.wadc+stand_x,result.intercept+result.slope*(clusters_use.wadc), linestyle='dashed',color='red',label='S wadc:  %.3f' % stand_x)
    plt.plot(clusters_use.wadc-stand_x,result.intercept+result.slope*(clusters_use.wadc), linestyle='dashed',color='red',label='S wadc:  %.3f' % stand_x)    
    plt.legend()
    plt.title('')
    plt.text(200,3500,'Standard deviation in wadc: %.3f\n Standard deviation in gradc: %.3f ' %(stand_x, stand_y))
    plt.text(500,300,'Regression line: gradc = %.3f + %.3f * wadc ' %(result.intercept,result.slope))
    print(stand_x,stand_y)
    plt.show()
    if save: 
        output_path = '../output/%s_PHS.png' % title
        plt.savefig(output_path, bbox_inches='tight')
        

# ===================================================================================================================================
#        Plot the time difference distrobution
# ===================================================================================================================================


def mg_time_diff(clusters):    
    """
    This function plots the time difference between clusters in a histogram
    
    Args:
        clusters = datafram with clusters
    Yields:
        Plots a histogram of the time distrobution

    """
    fig = plt.figure()
    tdc_to_s= 62.5e-9
    delta_time = [x*tdc_to_s for x in np.diff(clusters.time) ]
    # Histogram data
    log_bins = np.logspace(-10, 3, 1000)
    plt.hist(delta_time, bins=log_bins, color='black', histtype='step')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel("Delta time (s)")
    plt.xscale('log')
    plt.xlim(1e-10, 1e3)
    plt.yscale('log')
    print((delta_time.count(0)))
    
# ===================================================================================================================================
#        Plot colormap of a grid
# ===================================================================================================================================


def mg_colormap_grid(clusters,gr_nm,num_rows=6):    
    """
     This function plots a colormap integrated over a grid
    
    Args:
        clusters (df) = dataframe with clusters
        gr_nm (int) = gridnumber 
    Yields:
        Plots a colormap of the number of hits in a grid

    """
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
    try:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch_max, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    except:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    
    try:
        clusters_gr=clusters[clusters.gch_max==gr_nm]
    except:
        clusters_gr=clusters[clusters.gch==gr_nm]    
    fig = plt.figure()
    fig.set_size_inches(9, 9,forward=True)
    array=np.zeros((16,num_rows), dtype=float)
    for i in range(num_rows):
        for j in range(16):
            array[j][i]=hist[(5-i)*16+j][gr_nm-96]
            
    plt.pcolormesh(array,cmap='jet',vmin=0, vmax=np.max(hist))
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    plt.title(gr_nm)
    plt.xlabel('Wire row')
    plt.ylabel('Wire number')
    plt.tight_layout()
    plt.show
    
# ===================================================================================================================================
#        Plot colormap of a set of grids
# ===================================================================================================================================


def mg_colormap_grids(clusters,gr_list,hist,num_rows=6):    
    """
     This function plots a colormap integrated over grids
    
    Args:
        clusters (df) = dataframe with clusters
        gr_list (int) = list of gridnumbers
    Yields:
        Plots a colormap of the number of hits in a grid

    """
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
    try:
        clusters_gr=clusters[clusters.gch_max.isin(gr_list)]
    except:
        clusters_gr=clusters[clusters.gch.isin(gr_list)]    
    #fig = plt.figure()
    #fig.set_size_inches(9, 9,forward=True)
    array=np.zeros((16,num_rows), dtype=float)
    for i in range(num_rows):
        for j in range(16):
            array[j][i]=((clusters_gr[clusters_gr.wch==((5-i)*16+j)]).size)/clusters.size*100
    plt.pcolormesh(array,cmap='jet',vmin=0, vmax=np.max(array))
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('% of total counts')
    #plt.title('First grid: %d , Last grid: %d' %(gr_list[0],gr_list[-1]))
    plt.xlabel('Wire row')
    plt.ylabel('Wire number')
    plt.show
    
# ===================================================================================================================================
#        Plot colormap of set of layers seen as from the front of the detector
# ===================================================================================================================================


def mg_colormap_layers(clusters,layer_list,hist,num_rows=6):    
    """
    This function plots a colormap integrated over layers 
    
    Args:
        clusters (df) = dataframe with clusters
        layer_list (int) = first and last layer to be integrated (first: 0, last: 15)
    Yields:
        Plots a colormap of the layers

    """
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
    #fig.set_size_inches(5, 20,forward=True)
    array=np.zeros((37,num_rows), dtype=float)
    for i in range(num_rows):
        for j in range(37):
            for int in range(layer_list[0],layer_list[-1]+1):
                array[j][i] += hist[16*(5-i)+int][j]
                              
    plt.pcolormesh(array,cmap='jet',vmin=0, vmax=np.max(array))
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    plt.title('First wire: %d , Last wire: %d' %(layer_list[0],layer_list[-1]))
    plt.xlabel('Wire row')
    plt.ylabel('Grid number')
    plt.show
    
# ===================================================================================================================================
#        Plot colormap of one row of wires , see the detector from the right
# ===================================================================================================================================


def mg_colormap_wirerows(clusters,row_num):    
    """
    This function plots a colormap integrated over layers 
    
    Args:
        clusters (df) = dataframe with clusters
        row_num (int) = which row of wires is studied [1,6]
    Yields:
        Plots a colormap of the layers

    """
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
 
    
    try:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch_max, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    except:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
        
    fig = plt.figure()
    fig.set_size_inches(5, 20,forward=True)
    array=np.zeros((37,16), dtype=float)
    for i in range(16):
        for j in range(37):
            array[j][i] = hist[row_num*16+i][j]
                              
    plt.pcolormesh(array,cmap='jet',vmin=0, vmax=np.max(hist))
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    plt.title('Row number: %d' %row_num)
    plt.xlabel('Wire number')
    plt.ylabel('Grid number')
    plt.tight_layout()
    plt.show

# ===================================================================================================================================
#        Plot colormap of all row of wires (integrated), see the detector from the right
# ===================================================================================================================================


def mg_colormap_wirerows_int(clusters,num_row,hist):    
    """
    This function plots a colormap integrated over layers 
    
    Args:
        clusters (df) = dataframe with clusters
        num_row (int) = number of rows to integrate over
    Yields:
        Plots a colormap of the layers

    """
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
        
    #fig.set_size_inches(5, 20,forward=True)
    array=np.zeros((37,16), dtype=float)
    for i in range(16):
        for j in range(37):
            for row in range(num_row) :
                array[j][i] += hist[(5-row)*16+i][j]
                              
    plt.pcolormesh(array,cmap='jet',vmin=0, vmax=np.max(array))
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    plt.xlabel('Wire number')
    plt.ylabel('Grid number')
    plt.show
# ===================================================================================================================================
#        Finds the largest peak and plots and saves 3 intersections as well as a an integrated image of all directions
# ===================================================================================================================================


def mg_intersect(clusters,title):    
    """
    This function plots a colormap integrated over layers 
    
    Args:
        clusters (df) = dataframe with clusters
        title (string)= string for title 
    Yields:
        Plots a colormap of the layers
"""
    from numpy import unravel_index
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
    try:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch_max, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    except:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    #index=unravel_index(hist.argmax(), hist.shape)

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    i=0
    x=np.zeros(37*16*6, dtype=float)
    y=np.zeros(37*16*6, dtype=float)
    z=np.zeros(37*16*6, dtype=float)
    t=np.zeros(37*16*6, dtype=float)
    for gr in range(37):
        for w in range(16):
            for row in range(6):
                x[i] = 5-row
                y[i] = w
                z[i] = gr
                t[i]=hist[row*16+w][gr]
                i += 1
    figure = ax.scatter(x, y, z, c=t, cmap='jet')

    ax.set_xlabel('Wire row')
    ax.set_ylabel('Wire number')
    ax.set_zlabel('Grid number')
    cbar = fig.colorbar(figure)
    cbar.set_label('Counts/s')
    plt.tight_layout()
    plt.show()

    
# ===================================================================================================================================
#        Plot the ADC PHS for one a list of wires
# ===================================================================================================================================


def mg_row_of_wires(clusters,list_wires,gr_num):    
    """
    This function plots a PHS over a list of wires
    
    Args:
        clusters (df) = dataframe with clusters
        wire_list (list) = list of wires 
        gr_num (int) = what grid to look at
    Yields:
        Plots a comperaive PHS for the wires entered in wire_list
"""


    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
    number_bins=300
    try:
        clusters_gr=clusters[clusters.gch_max==gr_num]
    except:
        clusters_gr=clusters[clusters.gch==gr_num]
    
    color=iter(cm.rainbow(np.linspace(0, 1, 16)))   
    for wire in list_wires:
        c = next(color)
        clusters_w=clusters_gr[clusters_gr.wch==wire]
        plt.hist(clusters_w.wadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 8000], label='Wire %d' %wire,
             weights=(1/duration)*np.ones(len(clusters_w.wadc)),color=c)
        
    plt.yscale('log')
    plt.legend()
    plt.show
        
    
    
    
    
# ===================================================================================================================================
#        Plot Front, side, topp view as well as rate over time and chargecoincidence
# ===================================================================================================================================


def mg_plott_sum(run, clusters,area,save=False,
                      plot_title=''):    
    """
    This function plots a PHS over a list of wires
    
    Args:
        clusters (df) = dataframe with clusters
        wire_list (list) = list of wires 
        gr_num (int) = what grid to look at
    Yields:
        Plots a comperaive PHS for the wires entered in wire_list
"""
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
    try:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch_max, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    except:
        hist,xedges,yedges,image=plt.hist2d(clusters.wch, clusters.gch, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    bus=9
    plotting_hf.set_thick_labels(15)
    try:
        os.mkdir('../output/%s_%d' % (run,bus))
    except:
        pass
    output_path = '../output/%s_%d/%s_summary_bus_%d' % (run,bus,run, bus)

    
    # Filter data from only one bus
    fig=plt.figure(figsize=(12, 6))
    plt.subplot(1,3,1)
    mg_colormap_layers(clusters,[0,15],hist,num_rows=4)
    plt.title('a)')
    plt.subplot(1,3,2)
    mg_colormap_wirerows_int(clusters,4,hist)
    plt.title('b)')
    plt.subplot(3,3,3)
    mg_colormap_grids(clusters,range(97,132),hist,4)
    plt.title('c)')
    plt.subplot(3,3,6)
    rate_plot(clusters, 30, 9)
    plt.title('d)')
    if clusters.shape[0] != 0:
        vmin = 1/duration
        vmax = (clusters.shape[0] // 450 + 1000) / duration
    else:
        duration = 1
        vmin = 1
        vmax = 1
    plt.subplot(3,3,9)
    clusters_phs_plot(clusters, 9, duration, vmin, vmax)
    plt.title('e)')
    
    
    number_events = clusters.shape[0]
    number_events_error = np.sqrt(clusters.shape[0])
    events_per_s = number_events/duration
    events_per_s_m2 = events_per_s/area
    events_per_s_m2_error = number_events_error/(duration*area)
    title = ('Coincidences\n(%d events, %.3f±%.3f events/s/m$^2$)' % (number_events,
                                                                      events_per_s_m2,
                                                                      events_per_s_m2_error))
    

    # Save data
    fig.set_figwidth(14)
    fig.set_figheight(16)
    plt.tight_layout()
    if save:
        fig.savefig(output_path+'_sides.png', bbox_inches='tight')
        # open file for writing the filter
        f = open("../output/%s_%d/filter.txt" % (run,bus),"w+")
        # write file
        f.write( str(df_filter) )
        # close file
        f.close()
        mg_save_plot_basic_bus(run, bus, clusters_unfiltered, events, df_filter, area,save=True,
                      plot_title='')
