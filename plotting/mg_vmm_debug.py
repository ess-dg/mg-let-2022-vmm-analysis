import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
            ax.hist(df_vmm.channel, color=colors[vmm_id], histtype='step',
                    range=[-0.5, 63.5], bins=64, label='VMM %d' % vmm_id,
                    linestyle=linestyles[vmm_id])
            ax.legend(title='Hybrid %d' % (vmm_id // 2), ncol=2)
    ax3.set_xlabel('Channel')
    ax4.set_xlabel('Channel')
    ax1.set_ylabel('Counts')
    ax3.set_ylabel('Counts')
    ax1.set_title('FEN 0')
    ax2.set_title('FEN 1')
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
    
    