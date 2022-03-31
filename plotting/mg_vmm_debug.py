import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects

def plot_channel_histograms(df, file_name):
    """ Function to plot channel histograms for each VMM in the data.
    
    Args:
        df (pd.DataFrame): DataFrame containing data
        file_name (str): File name
    
    Yields:
        Plot showing the channel histograms for all VMMs in the data.
    
    """
    # Declare parameters
    number_fens = 2
    number_vmms_per_fen = 4
    # Plot data 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                                 sharey=True, sharex=True)
    plt.suptitle(file_name)
    axes = [[ax1, ax3], [ax2, ax4]]
    colors = ['blue', 'red', 'green', 'orange']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for fen_id in range(number_fens):
        for vmm_id in range(number_vmms_per_fen):
            ax = axes[fen_id][vmm_id // 2]
            df_vmm = df[(df.fen == fen_id) & (df.vmm == vmm_id)]
            #hist, bins, __ = ax.hist(df_vmm.channel, color=colors[vmm_id], histtype='step',
            #                         range=[-0.5, 63.5], bins=64, label='VMM %d' % (vmm_id%2),
            #                         linestyle=linestyles[vmm_id])
            hist, bins = np.histogram(df_vmm.channel, range=[-0.5, 63.5], bins=64)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, hist, align='center', alpha=0.5, label='VMM %d' % (vmm_id%2))
            ax.legend(title='Hybrid %d' % (vmm_id // 2), ncol=2, loc=4)
    ax3.set_xlabel('Channel')
    ax4.set_xlabel('Channel')
    ax1.set_ylabel('Counts')
    ax3.set_ylabel('Counts')
    ax1.set_title('FEN 0')
    ax2.set_title('FEN 1')
    plt.tight_layout()
    
def plot_channel_histograms_large(df, file_name, print_text=False, lower_threshold=-np.inf, higher_threshold=np.inf):
    """
    
    """
    # Declare parameters
    number_rings = 2
    number_fens = 1
    number_vmms_per_fen = 4
    # Plot data 
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4,
                                                                     sharey=True, sharex=True)
    plt.suptitle(file_name)
    axes = [[ax1, ax5], [ax2, ax6], [ax3, ax7], [ax4, ax8]]
    colors = ['red', 'blue']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for ring_id in range(number_rings):
        for vmm_id in range(number_vmms_per_fen):
            ax = axes[2*ring_id + vmm_id // 2][vmm_id % 2]
            ax.set_title('Ring ID: %d\nHybrid: %d, VMM: %d' % (ring_id, (vmm_id // 2), (vmm_id % 2)))
            df_vmm = df[(df.ring // 2 == ring_id) & (df.vmm == vmm_id)]
            hist, bins = np.histogram(df_vmm.channel, range=[-0.5, 63.5], bins=64)
            if print_text:
                lower_idxs = np.where(hist < lower_threshold)
                higher_idxs = np.where(hist > higher_threshold)
                if (len(lower_idxs[0]) != 64) or (len(higher_idxs[0]) != 64):
                    print('-------------------------------------------------------------------')
                    print('Ring ID: %d, Hybrid: %d, VMM: %d' % (ring_id, (vmm_id // 2), (vmm_id % 2)))
                    print('-------------------------------------------------------------------')
                    print('Number silent channels: %d' % len(lower_idxs[0]))
                    print('Number noisy channels: %d' % len(higher_idxs[0]))
                    if(len(lower_idxs[0]) != 64):
                        print('Quiet channels', lower_idxs[0])
                    if(len(higher_idxs[0]) != 64):
                        print('Noisy channels', higher_idxs[0])
                    print('###################################################################')
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(bin_centers, hist, align='center', alpha=0.5, label='VMM %d' % (vmm_id%2), color=colors[vmm_id % 2])
            #ax.legend(title='Hybrid %d' % (vmm_id // 2), ncol=2, loc=4)
    ax5.set_xlabel('Channel')
    ax6.set_xlabel('Channel')
    ax7.set_xlabel('Channel')
    ax8.set_xlabel('Channel')
    ax1.set_ylabel('Counts')
    ax5.set_ylabel('Counts')
    fig.set_size_inches(13, 7)
    plt.tight_layout()
    

def plot_adc_histograms(df, file_name):
    """ Function to plot ADC histograms for each VMM in the data.
    
    Args:
        df (pd.DataFrame): DataFrame containing data
        file_name (str): File name
    
    Yields:
        Plot showing the adc histograms for all VMMs in the data.
    
    """
    # Declare parameters
    number_fens = 2
    number_vmms_per_fen = 4
    # Plot data 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                                 sharey=True, sharex=True)
    plt.suptitle(file_name)
    axes = [[ax1, ax3], [ax2, ax4]]
    colors = ['blue', 'red', 'green', 'orange']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for fen_id in range(number_fens):
        for vmm_id in range(number_vmms_per_fen):
            ax = axes[fen_id][vmm_id // 2]
            df_vmm = df[(df.fen == fen_id) & (df.vmm == vmm_id)]
            ax.hist(df_vmm.adc, color=colors[vmm_id], histtype='step',
                    range=[-0.5, 4094.5], bins=4095, label='VMM %d' % vmm_id,
                    linestyle=linestyles[vmm_id])
            ax.legend(title='Hybrid %d' % (vmm_id // 2), ncol=2)
    ax3.set_xlabel('ADC')
    ax4.set_xlabel('ADC')
    ax1.set_ylabel('Counts')
    ax3.set_ylabel('Counts')
    ax1.set_title('FEN 0')
    ax2.set_title('FEN 1')
    plt.tight_layout()

def plot_adc_histograms_large(df, file_name, number_bins=1024):
    """
    
    """
    # Declare parameters
    number_rings = 2
    number_fens = 1
    number_vmms_per_fen = 4
    # Plot data 
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4,
                                                                     sharey=True, sharex=True)
    plt.suptitle(file_name)
    axes = [[ax1, ax5], [ax2, ax6], [ax3, ax7], [ax4, ax8]]
    colors = ['red', 'blue']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    for ring_id in range(number_rings):
        for vmm_id in range(number_vmms_per_fen):
            ax = axes[2*ring_id + vmm_id // 2][vmm_id % 2]
            ax.set_title('Ring ID: %d\nHybrid: %d, VMM: %d' % (ring_id, (vmm_id // 2), (vmm_id % 2)))
            df_vmm = df[(df.ring // 2 == ring_id) & (df.vmm == vmm_id)]
            ax.hist(df_vmm.adc, range=[-0.5, 1022.5], bins=number_bins, alpha=0.5, label='VMM %d' % (vmm_id%2), color=colors[vmm_id % 2])
            #ax.legend(title='Hybrid %d' % (vmm_id // 2), ncol=2, loc=4)
    ax5.set_xlabel('ADC')
    ax6.set_xlabel('ADC')
    ax7.set_xlabel('ADC')
    ax8.set_xlabel('ADC')
    ax1.set_ylabel('Counts')
    ax5.set_ylabel('Counts')
    fig.set_size_inches(13, 7)
    plt.tight_layout()
    
    
def plot_channel_vs_adc_2d_histograms(df, file_name):
    """ Function to plot channel vs ADC 2D histograms for each VMM in the data.
    
    Args:
        df (pd.DataFrame): DataFrame containing data
        file_name (str): File name
    
    Yields:
        Plot showing the channel vs adc histograms for all VMMs in the data.
    
    """
    # Declare parameters
    number_fens = 2
    number_vmms_per_fen = 4
    # Plot data 
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2,
                                                                         sharey=True, sharex=True)
    plt.suptitle(file_name)
    axes = [[ax1, ax3, ax5, ax7], [ax2, ax4, ax6, ax8]]
    colors = ['blue', 'red']
    linestyles = ['solid', 'dashed']
    for fen_id in range(number_fens):
        for vmm_id in range(number_vmms_per_fen):
            ax = axes[fen_id][vmm_id]
            df_vmm = df[(df.fen == fen_id) & (df.vmm == vmm_id)]
            ax.set_title('Hybrid %d, VMM %d' % ((vmm_id // 2), vmm_id))
            if df_vmm.shape[0] > 0:
                h = ax.hist2d(df_vmm.channel, df_vmm.adc, bins=[64, 4095],
                              norm=LogNorm(),
                              range=[[-0.5, 63.5], [-0.5, 4094.5]],
                              cmap='jet')
                fig.colorbar(h[3], ax=ax)
    ax7.set_xlabel('Channel')
    ax8.set_xlabel('Channel')
    ax1.set_ylabel('ADC')
    ax3.set_ylabel('ADC')
    ax5.set_ylabel('ADC')
    ax7.set_ylabel('ADC')
    plt.tight_layout()
    #ax1.set_title('FEN 0')
    #ax2.set_title('FEN 1')

def plot_channel_vs_adc_2d_histograms_large(df, file_name, number_bins=1024):
    """
    
    """
    # Declare parameters
    number_rings = 2
    number_fens = 1
    number_vmms_per_fen = 4
    # Plot data 
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(nrows=2, ncols=4,
                                                                     sharey=True, sharex=True)
    plt.suptitle(file_name)
    axes = [[ax1, ax5], [ax2, ax6], [ax3, ax7], [ax4, ax8]]
    colors = ['red', 'blue']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    max_v = df.shape[0]/500
    for ring_id in range(number_rings):
        for vmm_id in range(number_vmms_per_fen):
            ax = axes[2*ring_id + vmm_id // 2][vmm_id % 2]
            ax.set_title('Ring ID: %d\nHybrid: %d, VMM: %d' % (ring_id, (vmm_id // 2), (vmm_id % 2)))
            df_vmm = df[(df.ring // 2 == ring_id) & (df.vmm == vmm_id)]
            if df_vmm.shape[0] > 0:
                h = ax.hist2d(df_vmm.channel, df_vmm.adc, bins=[64, 1024],
                              norm=LogNorm(vmin=1, vmax=max_v),
                              range=[[-0.5, 63.5], [-0.5, 1022.5]],
                              cmap='jet')
                fig.colorbar(h[3], ax=ax)
                #ax.set_xlabel('Channel')
            #else:
            #    print(df_vmm.shape[0])
            #    h = ax.hist2d([-1000], [-1000], bins=[64, 1024],
            #                  norm=LogNorm(vmin=1, vmax=max_v),
            #                  range=[[-0.5, 63.5], [-0.5, 1022.5]],
            #                  cmap='jet')
            #    fig.colorbar(h[3], ax=ax)
                #ax.set_xlabel('Channel')
    ax5.set_xlabel('Channel')
    ax6.set_xlabel('Channel')
    ax7.set_xlabel('Channel')
    ax8.set_xlabel('Channel')   
    ax1.set_ylabel('ADC')
    ax5.set_ylabel('ADC')
    fig.set_size_inches(13, 7)
    plt.tight_layout()
    
def plot_time_stamp(df, time_resolution, title):
    # Plot event time stamps
    fig = plt.figure()
    time_stamps = df['time_hi'].to_numpy() + df['time_lo'].to_numpy() * time_resolution
    pulse_times = df['PulseTimeHI'].to_numpy() + df['PulseTimeLO'].to_numpy() * time_resolution
    prev_pulse_times = df['PrevPulseTimeHI'].to_numpy() + df['PrevPulseTimeLO'].to_numpy() * time_resolution
    plt.plot(time_stamps, color='black', label='time')
    plt.plot(pulse_times, color='red', label='PulseTime')
    plt.plot(prev_pulse_times, color='blue', label='PrevPulseTime')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('Readout number')
    plt.ylabel('Time (s)')
    plt.legend(title='Data', loc=1)
    plt.title('%s.pcapng' % title)
    plt.tight_layout()
    plt.show()
    fig.savefig('../output/%s_timestamp_comparison.png' % title)
    
    
def plot_coincidences(df, title):
    # Extract data from the two columns
    df_1 = df[df['ring'] // 2 == 0]
    df_2 = df[df['ring'] // 2 == 1]
    # Declare parameters
    max_v = df.shape[0]/500
    area = 0.025 * 0.025 * 51 * 6
    if len(df['time'].to_numpy() > 2):
        duration = df['time'].to_numpy()[-1] - df['time'].to_numpy()[0]
        print('Duration', duration, 'seconds')
        rate_per_s_1 = df_1.shape[0] / duration
        rate_per_m_s_1 = (df_1.shape[0] / duration) / area
        rate_per_s_2 = df_2.shape[0] / duration
        rate_per_m_s_2 = (df_2.shape[0] / duration) / area
    else:
        duration = -1
        rate_per_s_1 = -1
        rate_per_m_s_1 = -1
        rate_per_s_2 = -1
        rate_per_m_s_2 = -1
    # Plot data in one column each
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, sharex=True)
    plt.suptitle('File: %s\nDuration: %d minutes' % (title, (duration/60)))
    if df_1.shape[0] > 0:
        h = ax1.hist2d(df_1.wch, df_1.gch_max, bins=[96, 51],
                      norm=LogNorm(vmin=1, vmax=max_v),
                      range=[[-0.5, 95.5], [-0.5, 50.5]],
                      cmap='jet')
        cbar = fig.colorbar(h[3], ax=ax1)
        cbar.set_label('Counts')
    if df_2.shape[0] > 0:
        h = ax2.hist2d(df_2.wch, df_2.gch_max, bins=[96, 51],
                      norm=LogNorm(vmin=1, vmax=max_v),
                      range=[[-0.5, 95.5], [-0.5, 50.5]],
                      cmap='jet')
        cbar = fig.colorbar(h[3], ax=ax2)
        cbar.set_label('Counts')
    ax1.set_xlabel('Wire channel')
    ax1.set_ylabel('Grid channel')
    ax1.set_title('Column 1\nClusters: %d\nRate: %.2f Hz/m$^2$' % (df_1.shape[0], rate_per_m_s_1))
    ax2.set_xlabel('Wire channel')
    ax2.set_ylabel('Grid channel')
    ax2.set_title('Column 2\nClusters: %d\nRate: %.2f Hz/m$^2$' % (df_2.shape[0], rate_per_m_s_2))
    fig.set_size_inches(13, 6)
    plt.tight_layout()
    
def plot_multiplicity(df, title):
    def plot_multiplicity_column(df_column, max_v, ax):
        # Histogram data
        hist, xbins, ybins, im = ax.hist2d(df_column.wm, df_column.gm, bins=[11, 11],
                                           #vmin=1, vmax=max_v,
                                           range=[[-0.5, 10.5], [-0.5, 10.5]],
                                           cmap='jet')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Counts')
        # Iterate through all squares and write percentages
        tot = df_column.shape[0]
        font_size = 10
        for i in range(len(ybins)-1):
            for j in range(len(xbins)-1):
                if hist[j, i] > 0:
                    text = ax.text(xbins[j]+0.5, ybins[i]+0.5,
                                    '%.0f%%' % (100*(hist[j, i]/tot)),
                                    color="w", ha="center", va="center",
                                    fontweight="bold", fontsize=font_size)
                    text.set_path_effects([path_effects.Stroke(linewidth=1,
                                                               foreground='black'),
                                           path_effects.Normal()])                    
        ticks_x = np.arange(0, 11, 1)
        locs_x = np.arange(0.5, 10.5, 1)
        ticks_y = np.arange(0, 11, 1)
        locs_y = np.arange(0.5, 10.5, 1)
        ax.set_xticks(ticks_x)
        ax.set_yticks(ticks_y)
        ax.set_xlabel("Wire multiplicity")
        ax.set_ylabel("Grid multiplicity")
        
    # Extract data from the two columns
    df_1 = df[df['ring'] // 2 == 0]
    df_2 = df[df['ring'] // 2 == 1]
    # Declare parameters
    max_v = df.shape[0]/2
    # Plot data in one column each
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, sharex=True)
    plt.suptitle(title)
    ax1.set_title('Column 1')
    if df_1.shape[0] > 0:
        plot_multiplicity_column(df_1, max_v, ax1)
    ax2.set_title('Column 2')
    if df_2.shape[0] > 0:
        plot_multiplicity_column(df_2, max_v, ax2)
    fig.set_size_inches(13, 6)
    plt.tight_layout()
    

def plot_phs_wires_vs_grids(df, title, number_bins=200):
    def plot_phs_wires_vs_grids_column(df_column, max_v, ax):
        # Histogram data
        hist, xbins, ybins, im = ax.hist2d(df_column.wadc, df_column.gadc, bins=[number_bins, number_bins],
                                           norm=LogNorm(vmin=1, vmax=max_v),
                                           range=[[-0.5, 1023.5], [-0.5, 1023.5]],
                                           cmap='jet')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Counts')
        ax.set_xlabel("Wire ADC")
        ax.set_ylabel("Grid ADC")
        
    # Extract data from the two columns
    df_1 = df[df['ring'] // 2 == 0]
    df_2 = df[df['ring'] // 2 == 1]
    # Declare parameters
    max_v = df.shape[0]/500
    # Plot data in one column each
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    plt.suptitle(title)
    ax1.set_title('Column 1')
    if df_1.shape[0] > 0:
        plot_phs_wires_vs_grids_column(df_1, max_v, ax1)
    ax2.set_title('Column 2')
    if df_2.shape[0] > 0:
        plot_phs_wires_vs_grids_column(df_2, max_v, ax2)
    fig.set_size_inches(13, 6)
    plt.tight_layout()
    
    

def plot_phs_wires_and_grids(df, title, number_bins=100):
    def plot_phs_wires_and_grids_column(df_column, ax, number_bins):
        # Histogram data
        ax.hist(df_column.wadc, bins=number_bins, range=[-0.5, 2047.5], label='Wires', color='blue', alpha=1, histtype='step')
        ax.hist(df_column.gadc, bins=number_bins, range=[-0.5, 2047.5], label='Grids', color='red', alpha=1, histtype='step')
        ax.set_yscale('log')
        ax.set_xlabel("ADC")
        ax.set_ylabel("Counts")
        ax.legend()
        
    # Extract data from the two columns
    df_1 = df[df['ring'] // 2 == 0]
    df_2 = df[df['ring'] // 2 == 1]
    # Plot data in one column each
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    plt.suptitle(title)
    ax1.set_title('Column 1')
    plot_phs_wires_and_grids_column(df_1, ax1, number_bins)
    ax2.set_title('Column 2')
    plot_phs_wires_and_grids_column(df_2, ax2, number_bins)
    fig.set_size_inches(13, 6)
    plt.tight_layout()
    
def plot_delta_time(df, title, number_bins=1000):
    def plot_delta_time_column(df_column, ax, number_bins):
        # Extraxt delta time
        delta_time = np.diff(df_column['time'])
        # Histogram data
        log_bins = np.logspace(-6, 2, number_bins)
        ax.hist(delta_time, bins=log_bins, color='black', histtype='step')
        ax.grid(True, which='major', linestyle='--', zorder=0)
        ax.grid(True, which='minor', linestyle='--', zorder=0)
        ax.set_xlabel("Delta time (s)")
        ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_xlim(1e-6, 1e2)
        
    # Extract data from the two columns
    df_1 = df[df['ring'] // 2 == 0]
    df_2 = df[df['ring'] // 2 == 1]
    # Plot data in one column each
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    plt.suptitle(title)
    ax1.set_ylabel("Counts")
    ax1.set_title('Column 1')
    plot_delta_time_column(df_1, ax1, number_bins)
    ax2.set_title('Column 2')
    plot_delta_time_column(df_2, ax2, number_bins)
    fig.set_size_inches(13, 6)
    plt.tight_layout()
    

def plot_time(df, title):
    def plot_time_column(df_column, ax):
        # Histogram data
        ax.plot(df_column['time'], color='black')
        ax.grid(True, which='major', linestyle='--', zorder=0)
        ax.grid(True, which='minor', linestyle='--', zorder=0)
        ax.set_xlabel('Index')
        
    # Extract data from the two columns
    df_1 = df[df['ring'] // 2 == 0]
    df_2 = df[df['ring'] // 2 == 1]
    # Plot data in one column each
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True)
    plt.suptitle(title)
    ax1.set_ylabel("Counts")
    ax1.set_title('Column 1')
    ax1.set_ylabel('Time (s)')
    plot_time_column(df_1, ax1)
    ax2.set_title('Column 2')
    plot_time_column(df_2, ax2)
    fig.set_size_inches(13, 6)
    plt.tight_layout()  

def plot_3D_hist(df, title):
    wchs = df['wch'].to_numpy()
    gchs = df['gch_max'].to_numpy()
    rings = df['ring'].to_numpy()
    # Calculate 3D histogram
    H, edges = np.histogramdd(np.array([wchs, gchs, rings]),
                              bins=(96, 51, 2),
                              range=((0, 96), (0, 51), (0, 2))
                              )
    # Insert results into an array
    hist = [[], [], [], []]
    loc = 0
    labels = []
    for wch in range(0, 96):
        for gch_max in range(0, 51):
            for fen in range(0, 2):
                x_pos = (wch // 16) * 22.5 + fen * (5 * 22.5 + 4 + 22.5)
                y_pos = (50 - gch_max) * 22.5
                z_pos = (wch % 16) * 10
                hist[0].append(x_pos)
                hist[1].append(y_pos)
                hist[2].append(z_pos)
                hist[3].append(H[wch, gch_max, fen])
                loc += 1
                labels.append('Wire channel: ' + str(wch) + '<br>'
                              + 'Grid channel: ' + str(gch_max) + '<br>'
                              + 'fen: ' + str(fen) + '<br>'
                              + 'Counts: ' + str(H[wch, gch_max, fen])
                              )
    # Produce 3D histogram plot
    MG_3D_trace = go.Scatter3d(x=hist[0],
                               y=hist[1],
                               z=hist[2],
                               mode='markers',
                               marker=dict(size=5,
                                           color=hist[3],
                                           colorscale='Jet',
                                           opacity=1,
                                           colorbar=dict(thickness=20,
                                                         title='Counts'
                                                         ),
                                           ),
                               text=labels,
                               name='Multi-Grid',
                               scene='scene1'
                               )
    # Introduce figure and put everything together
    fig = py.subplots.make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]])
    # Insert histogram
    fig.append_trace(MG_3D_trace, 1, 1)
    fig['layout']['scene1']['xaxis'].update(title='x (mm)', range=[-625, 625])
    fig['layout']['scene1']['yaxis'].update(title='y (mm)', range=[-50, 1200])
    fig['layout']['scene1']['zaxis'].update(title='z (mm)', range=[-625, 625])
    fig['layout'].update(title='Coincidences (3D)<br>Data set: ' + str(title) + '.pcapng')
    fig.layout.showlegend = False
    # Plot
    py.offline.init_notebook_mode()
    #py.offline.iplot(fig)
    py.offline.plot(fig,
                    filename='../output/coincident_events_histogram.html',
                    auto_open=True)