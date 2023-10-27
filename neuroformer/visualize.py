from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
import numpy as np
import pandas as pd
import os

# from utils import *
# from analysis import *
# from SpikeVidUtils import *

def set_plot_params():
    ## fonts
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Ubuntu'
    # plt.rcParams['font.monospace'] = 'Ubuntu mono'
    # plt.rc('font',**{'family':'sans-serif','sans-serif':['Open Sans'],'size':10})
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Helvetica Neue') 
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'normal'
    
    # # font sizes
    # plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 6
    # plt.rcParams['xtick.labelsize'] = 10
    # plt.rcParams['ytick.labelsize'] = 10
    # plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['figure.titlesize'] = 8

    ## colors
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams["figure.facecolor"] = '202020'
    plt.rcParams['axes.facecolor']= '202020'
    plt.rcParams['savefig.facecolor']= '202020'

def set_plot_white():
    # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
    # plt.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})

    # Set the font used for MathJax - more on this later
    plt.rc('mathtext',**{'default':'regular'})
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams["figure.facecolor"] = 'white'
    plt.rcParams['axes.facecolor']= 'white'
    plt.rcParams['savefig.facecolor']= 'white'

def set_plot_black():
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams["figure.facecolor"] = '202020'
    plt.rcParams['axes.facecolor']= '202020'
    plt.rcParams['savefig.facecolor']= '202020'

def set_research_params():
    fs = 5
    LW = 0.001
    TW = 0.3
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    plt.rcParams['font.family'] = 'sans'
    plt.rcParams['xtick.labelsize']= fs
    plt.rcParams['ytick.labelsize']= fs
    plt.rcParams['axes.labelsize']= fs
    plt.rcParams['axes.titlesize']= 6
    plt.rcParams['legend.fontsize']= fs
    plt.rcParams['lines.linewidth']= 0.5
    plt.rcParams['xtick.major.size'] = 1.5
    plt.rcParams['ytick.major.size'] = 1.5
    # make ticks thinner
    plt.rcParams['xtick.major.width'] = TW
    plt.rcParams['ytick.major.width'] = TW
    
    # change grid si to make it lighter
    plt.rcParams['grid.linewidth'] = LW
    plt.rcParams['grid.alpha'] = 0.7

    plt.rcParams['axes.linewidth'] = 0.2


def nature_style():
    # Custom style settings
    nature_style = {
        "font.family": "Arial",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.titlesize": 12,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "lines.linewidth": 2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.figsize": [6, 4],
        "figure.dpi": 100,
        "figure.autolayout": True,
        "savefig.dpi": 300,
        "axes.prop_cycle": plt.cycler(color=[
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]),
    }

    # Apply the custom style
    plt.style.use(nature_style)



def plot_losses(trainer): 
    plt.figure(figsize=(20,5))
    
    # plotting train losses
    plt.subplot(1,2,1)
    plt.title('%s training losses' % str(trainer)[1:8])
    for i, losses in enumerate(trainer.train_losses):
            plt.plot(losses, label=i)
    plt.legend(title="epoch")
    
    # plotting testing losses
    plt.subplot(1,2,2)
    plt.title('%s testing losses' % str(trainer)[1:8])
    for i, losses in enumerate(trainer.test_losses):
            plt.plot(losses, label=i)
    plt.legend(title="epoch")

    plt.show()

def plot_losses_wattr(trainer, model_attr): 
    plt.figure(figsize=(20,5))
    
    # plotting train losses
    plt.subplot(1,2,1)
    plt.title('%s training losses' % model_attr)
    for i, losses in enumerate(trainer.train_losses):
            plt.plot(losses, label=i)
    plt.legend(title="epoch")
    
    # plotting testing losses
    plt.subplot(1,2,2)
    plt.title('%s testing losses' % model_attr)
    for i, losses in enumerate(trainer.test_losses):
            plt.plot(losses, label=i)
    plt.legend(title="epoch")

    plt.show()

# def horizontal_plot()


def tidy_axis(ax, top=False, right=False, left=False, bottom=False):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    ax.xaxis.set_tick_params(top='off', direction='out', width=0.5)
    ax.yaxis.set_tick_params(right='off', left='off', direction='out', width=0.5)

def plot_neurons(ax, df, neurons, color_map):
    # print(df.head())
    for idx, id_ in enumerate(neurons):
        df_id = df[df['ID'] == id_]
        if len(df_id) == 0:
            break
        color = color_map[id_]
        ax.scatter(df_id['Time'], df_id['ID'], color=color, marker="|", s=150, label='Simulated')
        
        # ax.set_ylim(0, len(neurons))
        xlim = int(max(df['Interval']))
        ax.set_xlim(0, xlim)
        ax.set_xticks(np.linspace(0, xlim, num=3))
        
        ax.tick_params(axis='y', labelsize=15) 
        ax.tick_params(axis='x', labelsize=15)  


def plot_raster_trial(df1, df2, trials, neurons):
    neurons_idx = {neuron: idx for idx, neuron in enumerate(neurons)}

    color_labels = neurons
    rgb_values = sns.color_palette("bright", len(neurons))
    color_map = dict(zip(color_labels, rgb_values))

    fig, ax = plt.subplots(nrows=len(trials), ncols=2, figsize=(12,10), squeeze=False)
    for n, trial in enumerate(trials):
        df1_trial_n = df1[df1['Trial'] == trial]
        df2_trial_n = df2[df2['Trial'] == trial]
        ax[n][0].set_ylabel(f'Trial {trial}')
        plot_neurons(ax[n][0], df1_trial_n, neurons, color_map)
        plot_neurons(ax[n][1], df2_trial_n, neurons, color_map)

        # ax[0][n].get_shared_x_axes().join(ax[0][0], ax[0][n])
        # ax[1][n].get_shared_x_axes().join(ax[0][0], ax[1][n])

        # make y axes catergorical
        ax[n][0].set_yticks(np.arange(0, len(neurons), 1))
        # ax[n][0].set_yticklabels(neurons)
        ax[n][1].set_yticks(np.arange(0, len(neurons), 1))
        # ax[n][1].set_yticklabels(neurons)

    # plt.setp(ax, yticks=np.arange(0, len(neurons)), yticklabels=neurons)
    ax[0][0].set_title('True')
    ax[0][1].set_title('Predicted')
    fig.supxlabel('Time (S)')
    fig.supylabel('Neuron ID')
    plt.tight_layout()
 

def get_id_intervals(df, n_id, intervals):
    id_intervals = np.zeros(len(intervals))
    interval_counts = df[df['ID'] == n_id].groupby(df['Interval']).size()
    id_intervals[interval_counts.index.astype(int).tolist()] = interval_counts.index.astype(int).tolist()
    return id_intervals.tolist()


def get_id_intervals(df, n_id, intervals):
    id_intervals = np.zeros(len(intervals))
    interval_counts = df[df['ID'] == n_id].groupby(df['Interval']).size()
    id_intervals[interval_counts.index.astype(int).tolist()] = interval_counts.index.astype(int).tolist()
    return id_intervals.tolist()

def plot_var(ax, df, variable, values, color_map=None, color=None, m_s=150, l_w=0.5):
    for value in values:
        color = color_map[value] if color_map is not None else color
        data = df[df[variable] == value]
        data[variable] = data[variable].astype('str')
        ax.scatter(data['Time'], data[variable], color=color,    # c=data[variable].map(color_map),
                   marker="|", s=m_s, linewidth=l_w)

        # ax.xaxis.set_tick_params(top='off', direction='out', width=1)
        ax.yaxis.set_tick_params(right='off', left='off', direction='out', width=1)
        
        ax.set_ylim(0, len(values))
        xlim = int(max(df['Interval'])) if len(df['Interval']) > 0 else 64
        ax.set_xlim(0, xlim)
        ax.set_xticks(np.linspace(0, xlim, num=3))

        ax.tick_params(axis='y', labelsize=5) 
        ax.tick_params(axis='x', labelsize=5)       

        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        ax.xaxis.set_tick_params(top='off', direction='out', width=l_w)
        # ax.yaxis.set_tick_params(right='off', direction='out', width=1)


ms_firing = 550    # 50
line_width = 2   # 0.75
lw_scatter = 2   # 0.3
def plot_firing_comparison(df_1, df_2, id, trials, intervals, figure_name=None):
    '''
    get trial averaged spikes (PSTH)
    '''
    id_ = id
    true = df_1[(df_1['Trial'].isin(trials)) & (df_1['ID'] == id_)].reset_index(drop=True)
    pred = df_2[(df_2['Trial'].isin(trials)) & (df_2['ID'] == id_)].reset_index(drop=True)
    rates_1_id = get_rates(true, [id_], intervals)[id_]
    rates_2_id = get_rates(pred, [id_], intervals)[id_]

    left, width = 0.15, 0.85
    bottom, height = 0.1, 0.1
    spacing = 0.005

    height_hist = 0.10
    rect_scatter_1 = [left, bottom*4, width, height]
    rect_scatter_2 = [left, bottom*3, width, height]
    rect_hist1 = [left, bottom*2, width, height_hist]
    # rect_hist2 = [left, bottom*1, width, height_hist]
    # rect_histy = [left + width + spacing, bottom, 0.2, height]

    if figure_name is None:
        fig = plt.figure(figsize=(10, 10))
    else:
        fig = figure_name


    # ax_rast_1 = fig.add_subaxes(rect_scatter_1)
    # ax_rast_2 = fig.add_axes(rect_scatter_2, sharex=ax_rast_1)
    # ax_hist_1 = fig.add_axes(rect_hist1, sharex=ax_rast_1)
    # ax_hist_2 = fig.add_axes(rect_hist2, sharex=ax_rast_1)

    tidy_axis(fig)
    no_top_right_ticks(fig)
    fig.set_yticks([])
    fig.set_yticklabels([])
    fig.axis('off')

    ax_rast_1 = fig.inset_axes(rect_scatter_1)
    ax_rast_2 = fig.inset_axes(rect_scatter_2, sharex=ax_rast_1)
    ax_hist_1 = fig.inset_axes(rect_hist1, sharex=ax_rast_1)

    ax_rast_2.axis('off')
    ax_rast_1.axis('off')

    axes_list = [ax_rast_1, ax_rast_2, ax_hist_1]
    # colors = sns.color_palette("gist_ncar_r", 2)
    colors = ['black', 'red']

    def plot_raster_scatter(ax, data, color, label):
        ax.scatter(data['Interval'], data['ID'], c=color, s=ms_firing, linewidth=lw_scatter, marker='|', label=label)
        ax.set_xlabel(label)

    # ax.scatter(true['Interval'], true['ID'].astype('str'), color='#069AF3', marker='|')
    plot_raster_scatter(ax_rast_2, pred, colors[0], 'Simulated')
    plot_raster_scatter(ax_rast_1, true, colors[1], 'True')

    # sns.distplot(true['Interval'], hist=False)
    # sns.distplot(pred['Interval'], hist=False)
    # sns.kdeplot(pred['Interval'], ax=ax_hist_1, bw_adjust=.25, color=colors[0], lw=line_width, alpha=0.7, warn_singular='False')    #plot(np.array(intervals), rates_1_id, color=colors[0],  lw=3)
    # sns.kdeplot(true['Interval'], ax=ax_hist_1, bw_adjust=.25, color=colors[1], lw=line_width, alpha=0.7, warn_singular='False')   #plot(np.array(intervals), rates_2_id, color=colors[1], lw=3)
    
    ax_hist_1.set_ylabel('')
    ax_hist_1.set_yticks([])
    sns.despine(top=True, left=True)
    # tidy_axis(ax_hist_1, bottom=True)
    # tidy_axis(ax_hist_2, bottom=True)
    ax_hist_1.set_xlabel([])
    # ax_hist_1.spines['bottom'].set_visible(False)
    # ax_rast_1.spines['bottom'].set_visible(False)
    # ax_rast_2.spines['bottom'].set_visible(False)
    # ax_hist_1.spines['top'].set_visible(False)
    # ax_hist_2.spines['top'].set_visible(False)

    # xlabels = np.arange(0, max(intervals) + 1, 60)
    # xticks, xlabels = xlabels, xlabels
    max_intervals = math.ceil(df_1['Interval'].max())
        # max_intervals = max(intervals)
    xticks, xlabels = [0,max_intervals // 2, max_intervals], [0,max_intervals // 2, max_intervals]
    xlabels = ['', '', '']

    yticks, ylabels = np.arange(len(trials)), list(map(str, trials))
    for ax in axes_list:
        tidy_axis(ax, bottom=True)
        no_top_right_ticks(ax)
        ax.set_xlim(0, max(intervals))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks([])
        ax.set_yticklabels([])


    # ax_hist_1.set_xlabel('Time (s)', fontsize=20)
    ax_hist_1.set_xlabel('', fontsize=20)
    # legend = fig.legend(bbox_to_anchor=(0.25, 0.01), ncol=3, frameon=True, fontsize=17.5)  # bbox_to_anchor=(0.75, 0.55)
    ax_rast_1.set_title("{}".format(id_), fontsize=20)

def plot_firing_comparison_sweeps(df_1, df_2, id, trials, intervals, figure_name=None):
    '''
    get trial averaged spikes (PSTH)
    '''

    if figure_name is None:
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        fig = plt.subplot()
    else:
        fig = figure_name

    default_size = (156.93750000000003, 53.706045112781794)
    current_size = fig.bbox.width, fig.bbox.height
    scale = current_size[0] / default_size[0], current_size[1] / default_size[1]
    
    line_width = 6 * scale[0]
    LW_scatter = 2.5 * scale[0]
    ms_firing = 7.75  * scale[1]

    scale = (1, 1)
    left, width = 0.05, 0.95
    bottom, height = 0.05 * scale[1] * 2, 0.1 * scale[1] * 2
    height_hist = 0.3 * scale[1]
    rect_hist1 = [left, bottom*2, width, height_hist]
    # rect_hist2 = [left, bottom*1, width, height_hist]
    # rect_histy = [left + width + spacing, bottom, 0.2, height]

    tidy_axis(fig)
    no_top_right_ticks(fig)
    fig.set_yticks([])
    fig.set_yticklabels([])
    fig.axis('off')

    ax_dict_true = dict()
    ax_dict_pred = dict()
    for n, trial in enumerate(trials):
        ax_dict_true[trial] = fig.inset_axes([left, bottom * (5 + (n)), width, height_hist])
        ax_dict_pred[trial] = fig.inset_axes([left, bottom * (5 + (len(trials) + n)), width, height_hist], sharex=ax_dict_true[trial])

        ax_dict_true[trial].axis('off')
        ax_dict_pred[trial].axis('off')
        
    ax_hist_1 = fig.inset_axes(rect_hist1, sharex=ax_dict_true[trials[0]])
    axes_list = [list(ax_dict_true.values()), list(ax_dict_pred.values()), [ax_hist_1]]

    # colors = sns.color_palette("gist_ncar_r", 2)
    colors = ['black', 'red']

    def plot_raster_scatter(ax, data, color, label):
        ax.scatter(data['Interval'], data['ID'], c=color, s=ms_firing, marker='|', linewidth=LW_scatter, label=label)
        ax.set_xlabel(label)

    # ax.scatter(true['Interval'], true['ID'].astype('str'), color='#069AF3', marker='|')
    for n, trial in enumerate(trials):
        id_ = id
        true = df_1[(df_1['Trial'] == trial) & (df_1['ID'] == id_)].reset_index(drop=True)
        pred = df_2[(df_2['Trial'] == trial) & (df_2['ID'] == id_)].reset_index(drop=True)
        if id_ == 345:
            print(true, pred)
        plot_raster_scatter(ax_dict_pred[trial], pred, colors[0], 'Simulated')
        plot_raster_scatter(ax_dict_true[trial], true, colors[1], 'True')

        sns.kdeplot(pred['Interval'], ax=ax_hist_1, bw_adjust=.25, color=colors[0], lw=line_width, alpha=0.7, warn_singular='False')    #plot(np.array(intervals), rates_1_id, color=colors[0],  lw=3)
        sns.kdeplot(true['Interval'], ax=ax_hist_1, bw_adjust=.25, color=colors[1], lw=line_width, alpha=0.7, warn_singular='False')   #plot(np.array(intervals), rates_2_id, color=colors[1], lw=3)
    
    max_intervals = df_1['Interval'].max()
    yticks, ylabels = np.arange(len(trials)), list(map(str, trials))
    xticks, xlabels = [0,max_intervals // 2, max_intervals], [0,max_intervals // 2, max_intervals]
    xlabels = ['', '', '']
    for ax in axes_list:
        ax = ax[0]
        tidy_axis(ax, bottom=True)
        no_top_right_ticks(ax)
        ax.set_xlim(0, max(intervals))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_yticks([])
        ax.set_yticklabels([])


    # sns.distplot(true['Interval'], hist=False)
    # sns.distplot(pred['Interval'], hist=False)
    
    # ax_hist_1.set_ylabel('')
    # ax_hist_1.set_yticks([])
    # sns.despine(top=True, left=True)
    # tidy_axis(ax_hist_1, bottom=True)
    # tidy_axis(ax_hist_2, bottom=True)
    # ax_hist_1.set_xlabel('')
    # ax_hist_1.set_xlabel('Time (s)', fontsize=20)
    # legend = fig.legend(bbox_to_anchor=(0.25, 0.01), ncol=3, frameon=True, fontsize=17.5)  # bbox_to_anchor=(0.75, 0.55)
    # list(ax_dict_pred.values())[-1].set_title("{}".format(id_), fontsize=5)

    # set_fontsize()
    

def get_psth(df, n_id, trials):
    df = df[df['ID'] == n_id]
    df = df[df['Trial'] == trial]
    df = df.groupby('Interval_dt').size().reset_index()
    df.columns = ['Interval_dt', 'Count']
    return df

def set_categorical_ticks(ax, yticks=None, ylabels=None, xticks=None, xlabels=None, fs=None):
    fs = fs if fs is not None else 10
    if yticks is not None:
        ax.set_ylim(0, len(ylabels))
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
    if xticks is not None:
        ax.set_xlim(0, max(xlabels))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
    ax.get_yaxis().tick_left()

def no_top_right_ticks(ax):
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_tick_params(top='off', direction='out')
    ax.yaxis.set_tick_params(top='off', right='off', left='on', direction='out')
    ax.tick_params(labelright='off', labeltop='off')
    ax.tick_params(axis='both', direction='out')
    ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
    ax.get_yaxis().tick_left()

def plot_neurons_trials_psth(df_1, df_2, neurons, trials, intervals, fig_title, figuresize=None): 

    fs = 6
    LW = 0.001
    # plt.rcParams['fig.supylabel']= fs
    line_width = 0.2

    set_fontsize()

    colors = ['black', 'red']
    
    df_1 = df_1.reset_index(drop=True)
    df_2 = df_2.reset_index(drop=True)
    dt = 5
    intervals_dt = [dt * n for n in range(int((intervals[-1]) // dt) + 1)]
    df_1['Interval_dt'] = pd.cut(df_1['Interval'], intervals_dt, include_lowest=True)
    df_2['Interval_dt'] = pd.cut(df_2['Interval'], intervals_dt, include_lowest=True)

    # neuron_list = list(map(str, sorted(top_corr[:6].index.tolist())))
    neurons = list(map(str, [i for i in neurons]))

    trials = df_1['Trial'].unique()
    # neuron_list = sorted(top_corr[:10].index.tolist())
    scale = 1
    nrows, ncols = 4, len(neurons)
    fig_size = figuresize if figuresize is not None else (2 * scale * len(neurons),10 * scale)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fig_size)
    variable = 'Trial'

    # color_labels = trials
    # rgb_values = sns.color_palette("gist_ncar_r", len(trials))
    # color_map = dict(zip(color_labels, rgb_values))

    max_freq = 0
    for n, neuron in enumerate(neurons):
        df_1['ID'] = df_1['ID'].astype('str')
        df_2['ID'] = df_2['ID'].astype('str') 
        df_1_id = df_1[df_1['ID'] == neuron]
        df_2_id = df_2[df_2['ID'] == neuron] 

        max_intervals = 32
        # max_intervals = max(intervals)
        yticks, ylabels = np.arange(len(trials)), list(map(str, trials))
        xticks, xlabels = [0,max_intervals // 2, max_intervals], [0,max_intervals // 2, max_intervals]

        m_s = 3
        l_w = 0.25
        plot_var(ax[0][n], df_1_id, variable, trials, color=colors[0], m_s=m_s, l_w=l_w)
        plot_var(ax[1][n], df_2_id, variable, trials, color=colors[1], m_s=m_s, l_w=l_w)

        set_categorical_ticks(ax[0][n], yticks, ylabels, xticks, xlabels)
        set_categorical_ticks(ax[1][n], yticks, ylabels, xticks, xlabels)

        # ax[0][n].hist2d(df_1_id['Interval'], df_1_id['Trial'], bins=intervals)
        # ax[1][n].hist2d(df_2_id['Interval'], df_2_id['Trial'], bins=intervals)

        ax[0][n].set_yticks([])
        ax[1][n].set_yticks([])
        x_margins = 10000
        ax[0][n].margins(x=x_margins)
        ax[1][n].margins(x=x_margins)

        # sns.kdeplot(df_1['Interval'], ax=ax[0][n], bw_adjust=.25, color=colors[0], lw=line_width, alpha=0.75)    #plot(np.array(intervals), rates_1_id, color=colors[0],  lw=3)
        # sns.kdeplot(df_2['Interval'], ax=ax[1][n], bw_adjust=.25, color=colors[1], lw=line_width, alpha=0.75)   #plot(np.array(intervals), rates_2_id, color=colors[1], lw=3)

        if n >= 1:
            no_top_right_ticks(ax[0][n])
            no_top_right_ticks(ax[1][n])

        df_1['ID'] = df_1['ID'].astype('int')
        df_2['ID'] = df_2['ID'].astype('int')  

        neuron_int = int(neuron)
        df_1_id = df_1[df_1['ID'] == neuron_int]
        df_2_id = df_2[df_2['ID'] == neuron_int]
        # rates_1 = get_rates(df_1, [neuron_int], intervals_dt)[neuroplotn_int]
        # rates_2 = get_rates(df_2, [neuron_int], intervals_dt)[neuron_int]

        freq_id_1 = df_1_id['Interval'].value_counts().reindex(intervals, fill_value=0)
        freq_id_2 = df_2_id['Interval'].value_counts().reindex(intervals, fill_value=0)
        bins = np.arange(len(intervals) // 2)
        # bins = len(intervals)
        # ax[2][n].bar(intervals_dt, freq_id_1)

        # ax[2][n].hist([freq_id_1, freq_id_2], bins=bins, histtype='step', edgecolor=['blue', 'red'], 
        #                lw=2, alpha=0.3, facecolor=['blue', 'red'], label=['True', 'Sim'])
        # c_2, c_1 = rgb_values[2], rgb_values[-1]

        ax[2][n].hist(df_1_id['Interval'], bins=bins, edgecolor=None, alpha=1, facecolor=colors[0], label='True')
        ax[3][n].hist(df_2_id['Interval'], bins=bins, edgecolor=None, alpha=1, facecolor=colors[1], label='Predicted') # histtype='step'
        ax[2][n].xaxis.set_tick_params(top='off', direction='out', width=l_w)
        ax[3][n].xaxis.set_tick_params(top='off', direction='out', width=l_w)
        # xticks, xlabels = [0, max(intervals) // 2, max(intervals)], [0, max(intervals) // 2, max(intervals)]
        set_categorical_ticks(ax[2][n], None, None, xticks, xlabels)
        ax[0][n].spines['left'].set_visible(False)
        ax[1][n].spines['left'].set_visible(False)
        ax[2][n].spines['right'].set_visible(False)
        ax[2][n].spines['top'].set_visible(False)

        set_categorical_ticks(ax[3][n], None, None, xticks, xlabels)
        ax[3][n].spines['right'].set_visible(False)
        ax[3][n].spines['top'].set_visible(False)

        if n >= 0:
            no_top_right_ticks(ax[2][n])
            ax[3][n].get_shared_y_axes().join(ax[2][n], ax[2][n-1])
            no_top_right_ticks(ax[3][n])
        max_lim = (max(ax[2][n].get_ylim()[1], ax[3][n].get_ylim()[1]))
        ax[0][n].set_xticklabels([])
        ax[1][n].set_xticklabels([])
        # ax[2][n].set_xticklabels([])
        # ax[3][n].set_xticklabels([])
        ax[2][n].set_ylim(0, max_lim)
        ax[3][n].set_ylim(0, max_lim)
        ax[2][n].get_shared_y_axes().join(ax[3][n], ax[3][n-1])

        
        # max_freq = max(freq_id_1.max(), freq_id_2.max(), max_freq)
        # yticks, ylabels = np.linspace(0, max(freq_id_1.max(), freq_id_2.max()), 3), [i for i in range(max(freq_id_1.max(), freq_id_2.max()))]
        # set_categorical_ticks(ax[2][n], yticks, ylabels, xticks, xlabels)

    plt.setp(ax[0])

    # ax[0][0].set_ylim(0, 32)
    # ax[0][0].set_ylabel('True')
    # ax[1][0].set_ylabel('Sim')
    # ax[2][0].set_ylabel('PSTH, True')
    # ax[3][0].set_ylabel('PSTH, Simulation')
    # ax[2][-1].legend()


    # ax[0][0].legend(bbox_to_anchor=(0,0,1,1))
    # fig.supxlabel('Time (S)', fontsize=15, y=0.07)
    # fig.supylabel('Trials')
    fig.suptitle(fig_title, fontsize=5, y=0.925)

    set_fontsize()
    # fig.gca().set_aspect('equal', adjustable='box')
    # plt.autoscale()
    # plt.tight_layout()
    


def get_boxplot_data(df_1, df_2, intervals, trials):
    data_boxplot_true = []
    data_boxplot_pred = []
    for n, trial in enumerate(trials):
        trial_prev = trials[n - 1] if n > 0 else trials[n + 1]
        true_prev = df_1[df_1['Trial'] == trial_prev].reset_index(drop=True)
        true = df_1[df_1['Trial'] == trial].reset_index(drop=True)
        pred = df_2[df_2['Trial'] == trial].reset_index(drop=True)
        rates_true_prev, rates_true, rates_pred = get_rates_trial(true_prev, intervals), get_rates_trial(true, intervals), get_rates_trial(pred, intervals)
        
        corr_trials_true = calc_corr_psth(rates_true, rates_true_prev)
        corr_trials_pred = calc_corr_psth(rates_true, rates_pred)

        data_boxplot_true.append(np.array(corr_trials_true).flatten())
        data_boxplot_pred.append(np.array(corr_trials_pred).flatten())

    return data_boxplot_true, data_boxplot_pred, corr_trials_true, corr_trials_pred

def plot_error_bar(x, n, color):
    """
    databoxplot_true, databoxplot_pred, corr_trials_true, corr_trials_pred = get_boxplot_data(df_1, df_2, intervals, n_trial)
    
    plot_error_bar(corr_trials_true, n, true_color)
    plot_error_bar(corr_trials_pred, n, pred_color)
    """
    mins = x.min()
    maxes = x.max()
    means = x.mean()
    std = x.std()

    # plt.errorbar(n, means, std, fmt='ok', lw=3)
    # plt.errorbar(n, means, [means - mins, maxes - means],
    #             fmt='.k', ecolor='gray', lw=1)
    # plt.xlim(-1, 8)
    
    green_diamond = dict(markerfacecolor=color, marker='o')
    # fig3, ax3 = plt.subplots()
    # ax3.set_title('Changed Outlier Symbols')
    ax.boxplot(x, flierprops=green_diamond)


def fancy_boxplot(fig, ax1, data, color):
    
    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                alpha=0.5)

    # ax1.set(
    #     axisbelow=True,  # Hide the grid behind plot objects
    #     title='Comparison of IID Bootstrap Resampling Across Five Distributions',
    #     xlabel='Distribution',
    #     ylabel='Value',
    # )

    # Now fill the boxes with desired colors
    # box_colors = ['darkkhaki', 'royalblue']
    # box_colors = sns.dark_palette("#69d", len(data), reverse=True)
    box_colors = [color]
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        box_x = []
        box_y = []
        for j in range(5):
            box_x.append(box.get_xdata()[j])
            box_y.append(box.get_ydata()[j])
        box_coords = np.column_stack([box_x, box_y])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[0]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            ax1.plot(median_x, median_y, 'k')
        medians[i] = median_y[0]
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    # ax1.set_xlim(0.5, num_boxes + 0.5)
    # top = 40
    # bottom = -5
    # ax1.set_ylim(bottom, top)
    # ax1.set_xticklabels(np.repeat(random_dists, 2),
    #                     rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ['bold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], .95, upper_labels[tick],
                transform=ax1.get_xaxis_transform(),
                horizontalalignment='center', size='x-small',
                weight=weights[k], color=box_colors[0])
    fig.supxlabel('Trials')
    fig.supylabel('Pearson Correlation (P)')
    fig.suptitle('Inter-Neuron Correlation Across Trials')
    plt.tight_layout()


def plot_intertrial_corr(corr_true, corr_pred, trial):
    
    def scatter_hist(x, y, ax, ax_histy):
        # no labels
        # ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        # ax.scatter(x, y)
        # bins = 250

        # now determine nice limits by hand:
        # binwidth = 0.25
        # xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        # lim = (int(xymax/binwidth) + 1) * binwidth

        # bins = np.arange(-lim, lim + binwidth, binwidth)
        # ax_histx.hist(x, bins=bins)
        ax_hist = sns.distplot(y,  hist=False, ax=ax_histy, vertical=True) # (x, y, bins=10, orientation='horizontal')
        ax_hist.set(xlabel=None)

    # sns.distplot(top_corr, hist=False, ax=ax_histy, vertical=True)
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005


    rect_scatter = [left, bottom, width, height]
    # rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(15, 15))

    ax = fig.add_axes(rect_scatter)
    # ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(np.array(corr_true.index), corr_true, ax, ax_histy)
    scatter_hist(np.array(corr_pred.index), corr_pred, ax, ax_histy)
    ax.grid(lw=0.8, alpha=0.7, color='gray')
    ax.scatter(corr_true.index, corr_true, label=f'Trial {trial} vs. 1', alpha=0.4)
    ax.scatter(corr_pred.index, corr_pred, label=f'Trial {trial} vs. Pred', alpha=0.5)
    ax.set_title('Pair-wise Correlation Between Trials', fontsize=25)
    ax.set_xlabel('Neuron ID', fontsize=20)
    ax.set_ylim(-0.1, 0.6)
    plt.ylabel('Pearson Correlation (p)')
    ax.legend(fontsize=20, title_fontsize=20)
    plt.show()


def V1_AL_sep_hist(atts_V1_AL, atts_V1_AL_rand, corrs_V1_AL, corrs_V1_AL_rand, V1_n, AL_n):
    set_plot_white()

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    labelfont = 15
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)
    set_plot_white()

    plt.rcParams['font.size'] = '4'

    n_scale = 5
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex=True, squeeze=False)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    V1_att_V1 = atts_V1_AL['V1'][V1_n]
    V1_att_AL = atts_V1_AL['V1'][AL_n]
    norm_val = np.max([V1_att_V1.max(), V1_att_AL.max()])
    # V1_att_V1 = V1_att_V1 / norm_val
    # V1_att_AL = V1_att_AL / norm_val

    AL_att_V1 = atts_V1_AL['AL'][V1_n]
    AL_att_AL = atts_V1_AL['AL'][AL_n]
    norm_val = np.max([AL_att_V1.max(), AL_att_AL.max()])
    # AL_att_V1 = AL_att_V1 / norm_val
    # AL_att_AL = AL_att_AL / norm_val

    V1_corrs_V1 = corrs_V1_AL['V1'][V1_n]
    V1_corrs_AL = corrs_V1_AL['V1'][AL_n]
    norm_val = np.max([V1_corrs_V1.max(), V1_corrs_AL.max()])
    # V1_corrs_V1 = V1_corrs_V1 / norm_val

    AL_corrs_V1 = corrs_V1_AL['AL'][V1_n]
    AL_corrs_AL = corrs_V1_AL['AL'][AL_n]
    norm_val = np.max([AL_corrs_V1.max(), AL_corrs_AL.max()])
    # AL_corrs_V1 = AL_corrs_V1 / norm_val

    V1_att_V1_mean = V1_att_V1.mean()
    V1_att_AL_mean = V1_att_AL.mean()

    V1_att_V1_rand = atts_V1_AL_rand['V1'][V1_n]
    V1_att_AL_rand = atts_V1_AL_rand['V1'][AL_n]
    norm_val = np.max([V1_att_V1_rand.max(), V1_att_AL_rand.max()])

    AL_att_V1_rand = atts_V1_AL_rand['AL'][V1_n]
    AL_att_AL_rand = atts_V1_AL_rand['AL'][AL_n]
    norm_val = np.max([AL_att_V1_rand.max(), AL_att_AL_rand.max()])

    V1_corrs_V1_rand = corrs_V1_AL_rand['V1'][V1_n]
    V1_corrs_AL_rand = corrs_V1_AL_rand['V1'][AL_n]
    norm_val = np.max([V1_corrs_V1_rand.max(), V1_corrs_AL_rand.max()])
    # V1_corrs_V1_rand = V1_corrs_V1_rand / norm_val

    Al_corrs_V1_rand = corrs_V1_AL_rand['AL'][V1_n]
    Al_corrs_AL_rand = corrs_V1_AL_rand['AL'][AL_n]
    norm_val = np.max([Al_corrs_V1_rand.max(), Al_corrs_AL_rand.max()])
    # Al_corrs_V1_rand = Al_corrs_V1_rand / norm_val

    # V1_att_V1_mean /= norm_val
    # V1_att_AL_mean /= norm_val

    plt.suptitle('Area Seperability')

    def plot_bar(data, ax, x, label):
        ax.bar(x, data.mean(), label=label)
        ax.errorbar(x, data.mean(), yerr=data.std(), c='black', fmt='o')
        

    ax[0, 0].set_title('V1', fontsize=20, y=0.8)
    ax[0, 0].set_ylabel('Average Attention')
    ax[0, 0].bar(0, V1_att_V1_mean)
    ax[0, 0].errorbar(0, V1_att_V1_mean, yerr=V1_att_V1.std(), c='black', fmt='o')
    ax[0, 0].bar(1, V1_att_AL_mean, label='AL')
    ax[0, 0].errorbar(1, V1_att_AL_mean, yerr=V1_att_AL.std(), c='black', fmt='o')
    ax[0, 0].bar(2, V1_att_V1_rand.mean(), label='Rand')
    ax[0, 0].errorbar(2, V1_att_V1_rand.mean(), yerr=V1_att_V1_rand.std(), c='black', fmt='o')

    AL_att_V1_mean = AL_att_V1.mean()
    AL_att_AL_mean = AL_att_AL.mean()
    norm_val = np.max([AL_att_V1_mean, AL_att_AL_mean])
    # AL_att_V1_mean /= norm_val
    # AL_att_AL_mean /= norm_val

    ax[0, 1].set_title('AL', fontsize=20, y=0.8)
    ax[0, 1].bar(0, AL_att_V1_mean, label='V1')
    ax[0, 1].errorbar(0, AL_att_V1_mean, yerr=AL_att_V1.std(), c='black', fmt='o')
    ax[0, 1].bar(1, AL_att_AL_mean, label='AL')
    ax[0, 1].errorbar(1, AL_att_AL_mean, yerr=AL_att_AL.std(), c='black', fmt='o')
    ax[0, 1].bar(2, AL_att_V1_rand.mean(), label='Rand')
    ax[0, 1].errorbar(2, AL_att_V1_rand.mean(), yerr=AL_att_V1_rand.std(), c='black', fmt='o')

    V1_corrs_V1_mean = V1_corrs_V1.mean()
    V1_corrs_AL_mean = V1_corrs_AL.mean()
    norm_val = np.max([V1_corrs_V1_mean, V1_corrs_AL_mean])
    # V1_corrs_V1_mean /= norm_val
    # V1_corrs_AL_mean /= norm_val
    ax[1, 0].set_ylabel('Pearson Correlation (p)')
    ax[1, 0].bar(0, V1_corrs_V1_mean)
    ax[1, 0].errorbar(0, V1_corrs_V1_mean, yerr=V1_corrs_V1.std(), c='black', fmt='o')
    ax[1, 0].bar(1, V1_corrs_AL_mean)
    ax[1, 0].errorbar(1, V1_corrs_AL_mean, yerr=V1_corrs_AL.std(), c='black', fmt='o')
    ax[1, 0].bar(2, V1_corrs_V1_rand.mean())
    ax[1, 0].errorbar(2, V1_corrs_AL_rand.mean(), yerr=V1_corrs_V1_rand.std(), c='black', fmt='o')

    AL_corrs_V1_mean = AL_corrs_V1.mean()
    AL_corrs_AL_mean = AL_corrs_AL.mean()
    norm_val = np.max([AL_corrs_V1_mean, AL_corrs_AL_mean])
    # AL_corrs_V1_mean /= norm_val
    # AL_corrs_AL_mean /= norm_val
    ax[1, 1].bar(0, AL_corrs_V1_mean, label='V1')
    ax[1, 1].errorbar(0, AL_corrs_V1_mean, yerr=AL_corrs_V1.std(), c='black', fmt='o')
    ax[1, 1].bar(1, AL_corrs_AL_mean)
    ax[1, 1].errorbar(1, AL_corrs_AL_mean, yerr=AL_corrs_AL.std(), c='black', fmt='o')
    ax[1, 1].bar(2, Al_corrs_AL_rand.mean(), label='Rand')
    ax[1, 1].errorbar(2, Al_corrs_AL_rand.mean(), yerr=Al_corrs_V1_rand.std(), c='black', fmt='o')


    ax[0, 0].get_shared_y_axes().join(ax[0, 0], ax[0, 1])
    ax[1, 0].get_shared_y_axes().join(ax[1, 0], ax[1, 1])


    x_ticklabels = ['V1', 'AL', 'Rand']
    for axes in ax.flat:
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
        axes.spines['left'].set_visible(False)
        axes.xaxis.set_tick_params(labelsize=labelfont)
        axes.yaxis.set_tick_params(labelsize=labelfont)

    ax[0, 1].spines['left'].set_visible(False)
    ax[1, 1].spines['left'].set_visible(False)
    # tick_labels = [0.5 * i for i in range(5)]
    # plt.setp(ax, yticks=tick_labels, ylim=[0, 1.3])
    plt.setp(ax, xticks=np.arange(3), xticklabels=x_ticklabels)

    ax[0, 1].set_yticklabels([])
    # ax[0, 1].ax.axes.yaxis.set_ticks([])
    ax[1, 1].set_yticklabels([])

    # ax[0, 0].get_shared_y_axes().join(ax[0, 0], ax[0, 1])

    score_list = [[V1_att_V1, V1_att_AL], [AL_att_V1, AL_att_AL], [V1_corrs_V1, V1_corrs_AL], [AL_corrs_V1, AL_corrs_AL]]

    for axes in ax.flat:
        axes.grid(alpha=0.5)

    
    return V1_att_V1, V1_att_AL, AL_att_V1, AL_att_AL, V1_corrs_V1, V1_corrs_AL, AL_corrs_V1, AL_corrs_AL, V1_att_V1_rand, V1_att_AL_rand, AL_att_V1_rand, AL_att_AL_rand, V1_corrs_V1_rand, V1_corrs_AL_rand
    # w = 0.5
    # for n, score in enumerate(score_list):
    #     axis = ax.flat[n]
    #     for i, val in enumerate(score):
    #         axis.scatter(i + np.random.random(val.size) * w - w / 2, val, color='black', alpha=0.2)


    # ax[1, 0].legend(loc='upper right', fontsize=20)


    # plt.savefig('/Users/antonis/projects/slab/neuroformer/neuroformer/plots/area_seperability/area_seperability_shuffled.svg')
    # ax.legend()

def plot_distribution(df_1, df_2, save_path=None):
    plt.figure(figsize=(30,20))
    freq_true = df_1.groupby(['ID']).size()
    freq_pred = df_2.groupby(['ID']).size()
    plt.bar(freq_pred.index, freq_pred, label='predicted', alpha=0.5)
    plt.bar(freq_true.index, freq_true, label='true', alpha=0.5)
    print(len(freq_true.index))
    plt.title('Neuron Firing Distribution', fontsize=40)
    plt.legend(fontsize=30)
    plt.xlabel('Neuron ID (n)', fontsize=30)
    plt.ylabel('Count (N)', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    if save_path:
        plt.savefig(save_path)
    plt.show()



def plot_psth(df_1, df_2, ax, xlims=None):
    marker = '|'
    ms = 100
    trials = df_1['Trial'].unique()
    n_trials = len(trials)
    for i, trial in enumerate(trials):
        df_1_trial = df_1[df_1['Trial'] == trial]
        df_2_trial = df_2[df_2['Trial'] == trial]
        ax.scatter(df_1_trial['Time'], [2 + (i + n_trials)] * len(df_1_trial), s=ms, c='black', alpha=1, marker=marker)
        ax.scatter(df_2_trial['Time'], [1 + i] * len(df_2_trial), s=ms, c='red', alpha=1, marker=marker)
    
    if xlims is not None:
        ax.set_xlim(xlims)
    
    ax.set_yticks([])


def plot_hippocampus(ax, embedding, label, gray = False, idx_order = (0,1,2)):
    """
    embedding: (N x 3) (n example, n_embd_clip)
    label: (N x 3) (example, [location, right dir, left dir])
    """
    l_ind = label[:,1] == 1
    r_ind = label[:,2] == 1
    
    if not gray:
        r_cmap = 'cool'
        l_cmap = 'viridis'
        r_c = label[r_ind, 0]
        l_c = label[l_ind, 0]
    else:
        r_cmap = None
        l_cmap = None
        r_c = 'gray'
        l_c = 'gray'
    
    idx1, idx2, idx3 = idx_order
    r=ax.scatter(embedding [r_ind,idx1], 
               embedding [r_ind,idx2], 
               embedding [r_ind,idx3], 
               c=r_c,
               cmap=r_cmap, s=0.5)
    l=ax.scatter(embedding [l_ind,idx1], 
               embedding [l_ind,idx2], 
               embedding [l_ind,idx3], 
               c=l_c,
               cmap=l_cmap, s=0.5)
    
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
        
    return ax


def plot_regression(y_true, y_pred, mode, model_name, r, p, 
                    ax=None, axis_limits=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    ax.scatter(y_true, y_pred, s=100, c='k', alpha=0.5)

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    s_f = 0.8
    combined_limits = [min(xlims[0], ylims[0]) * s_f, max(xlims[1], ylims[1]) * s_f]
    ax.plot(combined_limits, combined_limits, 'k--', color='red')

    ax.set_xlabel(f'True {mode}', fontsize=20)
    ax.set_ylabel(f'Predicted {mode}', fontsize=20)
    ax.set_title(f'{model_name}, Regression', fontsize=20)
    ax.text(0.05, 0.9, 'r = {:.2f}'.format(r), fontsize=20, transform=ax.transAxes)
    ax.text(0.05, 0.8, 'p < 0.001'.format(p), fontsize=20, transform=ax.transAxes)

    if axis_limits is not None:
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)

    if save_path:
        plt.savefig(os.path.join(save_path, 'regression.pdf'), dpi=300, bbox_inches='tight')


"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from neuroformer.visualize import set_plot_white

IS_3D = False

def plot_data(data, is_3d=True):
    set_plot_white()

    # Creating a color map based on the keys of the data
    cmap = cm.get_cmap('viridis')  # Can choose different colormap here
    norm = plt.Normalize(min(data.keys()), max(data.keys()))  # Normalize speed values

    fig = plt.figure(figsize=(10, 10))
    
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel('Z')
    else:
        ax = fig.add_subplot(111)
    
    # Iterating through the dictionary
    for speed in data.keys():
        speed_data = torch.stack(data[speed], dim=0)
        # Convert list of vectors to a numpy array
        vectors = np.array(speed_data)
        
        if is_3d:
            ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], color=cmap(norm(speed)))
        else:
            ax.scatter(vectors[:, 0], vectors[:, 1], color=cmap(norm(speed)))

    # Setting labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # show colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm)
    plt.title('Latent Space, Alignment (Color = Speed)', fontsize=20)


plot_data(latents['id'], is_3d=IS_3D)
save_pth = "./plots/dim_reduction"
# plt.savefig(os.path.join(save_pth, '2D_speed_neurons.png'))
# plt.savefig(os.path.join(save_pth, '2D_speed_neurons.pdf'))
"""

"""

import os
import pathlib
import torch
import pandas as pd
from joblib import Parallel, delayed
from neuroformer.utils import predict_raster_recursive_time_auto, process_predictions
from neuroformer.analysis import compute_scores


CONFIG = {
    "PARALLEL": True,
    "PLOT_PROBS": False,
    "TOP_P": 0.75,
    "TOP_P_T": 1.3,
    "TEMP": 0.75,
    "TEMP_T": 1.3,
    "TRUE_PAST": False
}


def process_trial(trial_number, trial, model, train_dataset, df, stoi, itos_dt, itos, window, window_prev, config):
    print(f"-- No. {trial_number} Trial: {trial} --")
    df_trial = df[df['Trial'] == trial]
    trial_dataset = train_dataset.copy(df_trial)
    results_trial = predict_raster_recursive_time_auto(
        model, trial_dataset, window, window_prev, stoi, itos_dt, itos=itos,
        sample=True, top_p=config["TOP_P"], top_p_t=config["TOP_P_T"], temp=config["TEMP"], temp_t=config["TEMP_T"],
        frame_end=0, get_dt=True, gpu=False, pred_dt=True, plot_probs=config["PLOT_PROBS"], true_past=config["TRUE_PAST"]
    )
    df_trial_pred, df_trial_true = process_predictions(results_trial, stoi, itos, window)
    print(f"pred: {df_trial_pred.shape}, true: {df_trial_true.shape}")
    return df_trial_pred, df_trial_true


def run_analysis(model, model_path, test_data, train_dataset, df, stoi, itos_dt, itos, window, window_prev, config):
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)

    trials = sorted(test_data['Trial'].unique())
    if config["PARALLEL"]:
        results = Parallel(n_jobs=-1)(
            delayed(process_trial)(n, trial, model, train_dataset, df, stoi, itos_dt, itos, window, window_prev, config)
            for n, trial in enumerate(trials)
        )
    else:
        results = [
            process_trial(n, trial, model, train_dataset, df, stoi, itos_dt, itos, window, window_prev, config)
            for n, trial in enumerate(trials)
        ]

    df_pred = None
    df_true = None
    for n, (df_trial_pred, df_trial_true) in enumerate(results):
        print(f"-- No. {n} Trial --")
        if df_pred is None:
            df_pred = df_trial_pred
            df_true = df_trial_true
        else:
            df_pred = pd.concat([df_pred, df_trial_pred])
            df_true = pd.concat([df_true, df_trial_true])

    df_true = df[df['Trial'].isin(trials)]
    scores = compute_scores(df_true, df_pred)
    print(scores)
    print(f"ID unique: pred: {len(df_pred['ID'].unique())}, true: {len(df_true['ID'].unique())}")
    print(f"len pred: {len(df_pred)}, len true: {len(df_true)}")

    title = f"top_p: {config['TOP_P']}, top_p_t: {config['TOP_P_T']}, temp: {config['TEMP']}, temp_t: {config['TEMP_T']}/true_past:{config['TRUE_PAST']}"
    dir_name = os.path.dirname(model_path)
    save_path = os.path.join(dir_name, f"predictions_{title}.csv")
    # Optionally save the results
    # df_pred.to_csv(os.path.join(dir_name, 'df_pred.csv'))
    # df_true.to_csv(os.path.join(dir_name, 'df_true.csv'))


if __name__ == "__main__":
    # Assuming model, model_path, test_data, train_dataset, df, stoi, itos_dt, itos, window, and window_prev are defined.
    run_analysis(model, model_path, test_data, train_dataset, df, stoi, itos_dt, itos, window, window_prev, CONFIG)

"""