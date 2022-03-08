import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import plotly as py
import plotly.graph_objs as go
import matplotlib.patheffects as path_effects

# Declare visualization functions
def plot_vmm_and_channel(df, title):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    df_fen_0 = df[df.fen == 0]
    plt.hist2d(df_fen_0.wch, df_fen_0.gch_max, bins=[96, 51],
               range=[[-0.5, 95.5], [-0.5, 50.5]],
               norm=LogNorm(),
               cmap='jet')
    plt.xlabel('Wire (channel)')
    plt.ylabel('Grid (channel)')
    plt.title('fen 0')
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    plt.gca().invert_yaxis()
    plt.subplot(1, 2, 2)
    df_fen_1 = df[df.fen == 1]
    plt.hist2d(df_fen_1.wch, df_fen_1.gch_max, bins=[96, 51],
               norm=LogNorm(),
               range=[[-0.5, 95.5], [-0.5, 50.5]],
               cmap='jet')
    plt.xlabel('Wire (channel)')
    plt.ylabel('Grid (channel)')
    plt.title('fen 1')
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    plt.tight_layout()
    plt.gca().invert_yaxis()
    fig.set_figheight(3.5)
    fig.set_figwidth(10)
    fig.savefig('../output/%s_clusters_wch_vs_gch.png' % title)
    
def plot_tof(df, title):
    fig = plt.figure()
    plt.hist(df.tof, color='black', bins=100)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('tof (s)')
    plt.ylabel('Counts')
    plt.title('%s.pcapng\nTime-of-Flight' % title)
    fig.savefig('../output/%s_tof.png' % title)
    
def plot_multiplicity(df, title):
    font_size = 9
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    df_fen_0 = df[df.fen == 0]
    hist, xbins, ybins, im = plt.hist2d(df_fen_0.wm, df_fen_0.gm, bins=[5, 5],
                                        range=[[-0.5, 4.5], [-0.5, 4.5]],
                                      # norm=LogNorm(),
                                        cmap='jet')
    tot = df_fen_0.shape[0]
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            if hist[j, i] > 0:
                text = plt.text(xbins[j]+0.5, ybins[i]+0.5,
                                '%.1f%%' % (100*(hist[j, i]/tot)),
                                color="w", ha="center", va="center",
                                fontweight="bold", fontsize=font_size)
                text.set_path_effects([path_effects.Stroke(linewidth=1,
                                                           foreground='black'),
                                       path_effects.Normal()])
    plt.xlabel('Wires')
    plt.ylabel('Grids')
    plt.title('fen 0')
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    plt.subplot(1, 2, 2)
    df_fen_1 = df[df.fen == 1]
    hist, xbins, ybins, im = plt.hist2d(df_fen_1.wm, df_fen_1.gm, bins=[5, 5],
                                      # norm=LogNorm(),
                                       range=[[-0.5, 4.5], [-0.5, 4.5]],
                                       cmap='jet')
    tot = df_fen_1.shape[0]
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            if hist[j, i] > 0:
                text = plt.text(xbins[j]+0.5, ybins[i]+0.5,
                                '%.1f%%' % (100*(hist[j, i]/tot)),
                                color="w", ha="center", va="center",
                                fontweight="bold", fontsize=font_size)
                text.set_path_effects([path_effects.Stroke(linewidth=1,
                                                           foreground='black'),
                                       path_effects.Normal()])
    plt.xlabel('Wires')
    plt.ylabel('Grids')
    plt.title('fen 1')
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    fig.set_figheight(4)
    fig.set_figwidth(9)
    plt.tight_layout()
    fig.savefig('../output/%s_clusters_multiplicity.png' % title)

def plot_3D_hist(df, title):
    # Calculate 3D histogram
    H, edges = np.histogramdd(df[['wch', 'gch_max', 'fen']].values,
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
    
# Declare visualization plot function
def plot_vmm_data(df, title):
    columns = list(df)
    xy_plot_ids = np.array([4, 5, 6, 7, 8, 9, 13, 17, 18]) - 1
    for i, column in enumerate(columns):
        fig = plt.figure()
        df_temp = df[column].to_numpy()
        if i in xy_plot_ids:
            plt.plot(df_temp, color='black')
            plt.xlabel('Readout number')
            plt.ylabel(column)
        elif column == 'ring':
            plt.hist(df_temp, color='black', histtype='step', range=[-0.5, 23.5], bins=24)
            plt.xlabel(column)
            plt.ylabel('Counts')
        elif column == 'channel':
            plt.hist(df_temp, color='black', histtype='step', range=[-0.5, 63.5], bins=64)
            plt.xlabel(column)
            plt.ylabel('Counts')
        else:
            plt.hist(df_temp, color='black', histtype='step')
            plt.xlabel(column)
            plt.ylabel('Counts')
        plt.title(column)
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title('%s.pcapng\n%s' % (title, column))
        plt.tight_layout()
        fig.savefig('../output/%s_%s.png' % (title, column))
    
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
    fig.set_figheight(6)
    fig.set_figwidth(12)
    plt.legend(title='Data', bbox_to_anchor=(0.68, 1.13), ncol=3)
    plt.title('%s.pcapng\nTimestamp comparison\n\n\n' % title)
    plt.tight_layout()
    plt.show()
    fig.savefig('../output/%s_timestamp_comparison.png' % title)
    # Plot delta t
    fig = plt.figure()
    time_stamps = df['time_hi'].to_numpy() + df['time_lo'].to_numpy() * time_resolution
    delta_time_stamps = np.diff(time_stamps)
    plt.hist(delta_time_stamps, color='black', label='$\Delta$time', range=[0, 0.000005], bins=100)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.ylabel('Counts')
    plt.xlabel('Delta time (s)')
    plt.title('%s.pcapng\nDelta timestamp' % title)
    plt.tight_layout()
    plt.show()
    fig.savefig('../output/%s_delta_timestamp.png' % title)

    
def plot_fen_and_vmm(df, title):
    fig = plt.figure()
    plt.hist2d(df.fen, df.vmm, bins=[2, 3],
           range=[[-0.5, 1.5], [-0.5, 2.5]],
           cmap='jet'
              )
    plt.xlabel('fen')
    plt.xticks([0, 1])
    plt.yticks([0, 1, 2])
    plt.ylabel('vmm')
    plt.title('%s.pcapng\nfen vs vmm' % title)
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    fig.savefig('../output/%s_fen_vs_vmm.png' % title)
    
def plot_vmm_and_channel_events(df, title):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    df_fen_0 = df[df.fen == 0]
    plt.hist2d(df_fen_0.vmm, df_fen_0.channel, bins=[3, 64],
               range=[[-0.5, 2.5], [-0.5, 63.5]],
               norm=LogNorm(),
               cmap='jet')
    plt.xlabel('vmm')
    plt.xticks([0, 1, 2])
    plt.ylabel('channel')
    plt.title('fen 0')
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    plt.subplot(1, 2, 2)
    df_fen_1 = df[df.fen == 1]
    plt.hist2d(df_fen_1.vmm, df_fen_1.channel, bins=[3, 64],
               norm=LogNorm(),
               range=[[-0.5, 2.5], [-0.5, 63.5]],
               cmap='jet')
    plt.xlabel('vmm')
    plt.xticks([0, 1, 2])
    plt.ylabel('channel')
    plt.title('fen 1')
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    plt.tight_layout()
    fig.set_figheight(3.5)
    fig.set_figwidth(10)
    fig.savefig('../output/%s_vmm_vs_channel.png' % title)