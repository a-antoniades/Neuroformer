import sys
sys.path.append('./neuroformer')

import itertools
import torch
import numpy as np
import pandas as pd
import pickle
import os
import mat73

from neuroformer.data_utils import bin_spikes, get_df_visnav, round_n


def split_data_by_interval(intervals, r_split=0.8, r_split_ft=0.1):
    # Generate random indices for training set
    chosen_idx = np.random.choice(len(intervals), int(len(intervals) * r_split), replace=False)
    # Create a mask for all indices
    mask = np.zeros(len(intervals), dtype=bool)
    mask[chosen_idx] = True
    
    # Split intervals using the mask
    train_intervals = intervals[mask]
    test_intervals = intervals[~mask]
    
    # Further split the training intervals to get finetune intervals
    finetune_intervals = np.array(train_intervals[:int(len(train_intervals) * r_split_ft)])
    
    return train_intervals, test_intervals, finetune_intervals

def combo3_V1AL_callback(frames, frame_idx, n_frames, **args):
    """
    Shape of frames: [3, 640, 64, 112]
                     (3 = number of stimuli)
                     (0-20 = n_stim 0,
                      20-40 = n_stim 1,
                      40-60 = n_stim 2)
    frame_idx: the frame_idx in question
    n_frames: the number of frames to be returned
    """
    trial = kwargs['trial']
    if trial <= 20: n_stim = 0
    elif trial <= 40: n_stim = 1
    elif trial <= 60: n_stim = 2
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    f_idx_0 = max(0, frame_idx - n_frames)
    f_idx_1 = f_idx_0 + n_frames
    chosen_frames = frames[n_stim, f_idx_0:f_idx_1].type(torch.float32).unsqueeze(0)
    return chosen_frames

def visnav_callback(frames, frame_idx, n_frames, **args):
    """
    frames: [n_frames, 1, 64, 112]
    frame_idx: the frame_idx in question
    n_frames: the number of frames to be returned
    """
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    f_idx_0 = max(0, frame_idx - n_frames)
    f_idx_1 = f_idx_0 + n_frames
    chosen_frames = frames[f_idx_0:f_idx_1].type(torch.float32).unsqueeze(0)
    return chosen_frames

def download_data():
    print(f"Creating directory ./data and storing datasets!")
    print("Downloading data...")
    import gdown
    url = "https://drive.google.com/drive/folders/1O6T_BH9Y2gI4eLi2FbRjTVt85kMXeZN5?usp=sharing"
    gdown.download_folder(id=url, quiet=False, use_cookies=False, output="./data")

def load_V1AL(config, stimulus_path=None, response_path=None, top_p_ids=None):
    if not os.path.exists("./data"):
        download_data()

    if stimulus_path is None:
        # stimulus_path = "/home/antonis/projects/slab/git/slab/transformer_exp/code/data/SImNew3D/stimulus/tiff"
        stimulus_path = "data/Combo3_V1AL/Combo3_V1AL_stimulus.pt"
    if response_path is None:
        response_path = "data/Combo3_V1AL/Combo3_V1AL.pkl"
    
    data = {}
    data['spikes'] = pickle.load(open(response_path, "rb"))
    data['stimulus'] = torch.load(stimulus_path).transpose(1, 2).squeeze(1)

    intervals = np.arange(0, 31, config.window.curr)
    trials = list(set(data['spikes'].keys()))
    combinations = np.array(list(itertools.product(intervals, trials)))
    train_intervals, test_intervals, finetune_intervals = split_data_by_interval(combinations, r_split=0.8, r_split_ft=0.01)

    return (data, intervals,
           train_intervals, test_intervals, 
           finetune_intervals, combo3_V1AL_callback)

def load_visnav(version, config, selection=None):
    if not os.path.exists("./data"):
        download_data()
    if version not in ["medial", "lateral"]:
        raise ValueError("version must be either 'medial' or 'lateral'")
    
    if version == "medial":
        data_path = "./data/VisNav_VR_Expt/MedialVRDataset/"
    elif version == "lateral":
        data_path = "./data/VisNav_VR_Expt/LateralVRDataset/"

    spikes_path = f"{data_path}/NF_1.5/spikerates_dt_0.01.npy"
    speed_path = f"{data_path}/NF_1.5/behavior_speed_dt_0.05.npy"
    stim_path = f"{data_path}/NF_1.5/stimulus.npy"
    phi_path = f"{data_path}/NF_1.5/phi_dt_0.05.npy"
    th_path = f"{data_path}/NF_1.5/th_dt_0.05.npy"

    data = dict()
    data['spikes'] = np.load(spikes_path)
    data['speed'] = np.load(speed_path)
    data['stimulus'] = np.load(stim_path)
    data['phi'] = np.load(phi_path)
    data['th'] = np.load(th_path)

    if selection is not None:
        selection = np.array(pd.read_csv(os.path.join(data_path, f"{selection}.csv"), header=None)).flatten()
        data['spikes'] = data['spikes'][selection - 1]

    spikes = data['spikes']
    intervals = np.arange(0, spikes.shape[1] * config.resolution.dt, config.window.curr)
    train_intervals, test_intervals, finetune_intervals = split_data_by_interval(intervals, r_split=0.8, r_split_ft=0.01)

    return data, intervals, train_intervals, test_intervals, finetune_intervals, visnav_callback

def generate_intervals(start, end, dt):
    intervals = []
    current = start
    while current < end:
        intervals.append(current)
        current = round_n(current + dt, dt)
    return intervals

def split_data_by_trial_intervals(trialsummary, dt, max_time,
                                  r_split=0.8, r_split_ft=0.1,
                                  blackout_intervals=None):
    # Get the number of trials
    n_trials = len(trialsummary)

    # Generate random indices and shuffle them
    indices = np.arange(n_trials)
    np.random.shuffle(indices)

    # Split indices for train and test sets
    n_train = int(n_trials * r_split)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Further split the train indices to get finetune indices
    n_finetune = int(n_train * r_split_ft)
    finetune_indices = train_indices[:n_finetune]

    # Split the trialsummary based on the shuffled indices
    train_intervals = trialsummary[train_indices]
    test_intervals = trialsummary[test_indices]
    finetune_intervals = trialsummary[finetune_indices]

    # Generate intervals from start to end for each selected trial
    def fill_intervals(trials):
        intervals = []
        for trial in trials:
            trial_idx, start, outcome = trial
            # Assuming the end is one unit after start, modify as needed
            end = start + 1  # Replace with the correct end time logic
            intervals.extend(generate_intervals(start, end, dt))
        return np.array(intervals)

    train_intervals_filled = fill_intervals(train_intervals)
    test_intervals_filled = fill_intervals(test_intervals)
    finetune_intervals_filled = fill_intervals(finetune_intervals)

    # truncate the intervals to the max time
    train_intervals_filled = train_intervals_filled[train_intervals_filled < max_time]
    test_intervals_filled = test_intervals_filled[test_intervals_filled < max_time]
    finetune_intervals_filled = finetune_intervals_filled[finetune_intervals_filled < max_time]

    if blackout_intervals is not None:
        train_intervals_filled = train_intervals_filled[~np.isin(train_intervals_filled, blackout_intervals)]
        test_intervals_filled = test_intervals_filled[~np.isin(test_intervals_filled, blackout_intervals)]
        finetune_intervals_filled = finetune_intervals_filled[~np.isin(finetune_intervals_filled, blackout_intervals)]

    return train_intervals_filled, test_intervals_filled, finetune_intervals_filled


def load_visnav_2(version, config, selection=None):
    if version == "medial":
        data_path = "./data/VisNav_VR_Expt/MedialVRDataset/"
    elif version == "lateral":
        data_path = "./data/VisNav_VR_Expt/LateralVRDataset/"
    elif version == "visnav_tigre":
        data_path = "./data/tigre613_p2s23"
    
    mat_file = mat73.loadmat(os.path.join(data_path, "experiment_data.mat"))['neuroformer']
    data = dict()
    data['spikes'] = bin_spikes(mat_file['spiketimes']['spks'], config.resolution.dt) # get_df_visnav(mat_file['spiketimes']['spks'], dt_vars=0.05)
    data['speed'] = mat_file['speed']
    data['stimulus'] = mat_file['vid_sm']
    data['phi'] = mat_file['phi']
    data['th'] = mat_file['th']
    data['depth'] = mat_file['depth']
    data['trialsummary'] = mat_file['trialsummary']
    data['vid_sm'] = mat_file['vid_sm']

    # add any other data that needs to be loaded
    if hasattr(config, 'modalities'):
        for modality_type_name, modality_type in vars(config.modalities).items():
            for modality_name, modality_details in vars(modality_type.variables).items():
                if modality_name in mat_file:
                    data[modality_name] = mat_file[modality_name]

    # Assuming blackout_indices is available in the data
    if 'blackout_indices' in mat_file:
        # blackout_indices = mat_file['blackout_indices'].astype(bool)
        # for key in data:
        #     shape = data[key].shape
        #     if shape[0] == len(blackout_indices):
        #         data[key] = data[key][~blackout_indices]
        #     elif len(shape) > 1 and shape[1] == len(blackout_indices):
        #         data[key] = data[key][:, ~blackout_indices]
        #     elif len(shape) > 2 and shape[2] == len(blackout_indices):
        #         data[key] = data[key][:, :, ~blackout_indices]
        blackout_intervals = mat_file['blackout_indices'] * 0.05

    max_time = data['stimulus'].shape[0] * 0.05
    intervals = np.arange(0, max_time * config.resolution.dt, config.window.curr)
    train_intervals, test_intervals, finetune_intervals = split_data_by_trial_intervals(data['trialsummary'], 
                                                                                        r_split=0.8, r_split_ft=0.01,
                                                                                        dt=0.05, max_time=max_time,
                                                                                        blackout_intervals=blackout_intervals)

    return data, intervals, train_intervals, test_intervals, finetune_intervals, visnav_callback


# def load_visnav_2(version, config, selection=None):
#     if version == "medial":
#         data_path = "./data/VisNav_VR_Expt/MedialVRDataset/"
#     elif version == "lateral":
#         data_path = "./data/VisNav_VR_Expt/LateralVRDataset/"
#     elif version == "visnav_tigre":
#         data_path = "./data/tigre613_p2s23"
    
#     mat_file = mat73.loadmat(os.path.join(data_path, "experiment_data.mat"))['neuroformer']
#     data = dict()
#     data['spikes'] = bin_spikes(mat_file['spiketimes']['spks'], config.resolution.dt) # get_df_visnav(mat_file['spiketimes']['spks'], dt_vars=0.05)
#     data['speed'] = mat_file['speed']
#     data['stimulus'] = mat_file['vid_sm']
#     data['phi'] = mat_file['phi']
#     data['th'] = mat_file['th']
#     data['depth'] = mat_file['depth']

#     # Assuming blackout_indices is available in the data
#     if 'blackout_indices' in mat_file:
#         blackout_indices = mat_file['blackout_indices'].astype(bool)
#         for key in data:
#             shape = data[key].shape
#             if shape[0] == len(blackout_indices):
#                 data[key] = data[key][~blackout_indices]
#             elif len(shape) > 1 and shape[1] == len(blackout_indices):
#                 data[key] = data[key][:, ~blackout_indices]
#             elif len(shape) > 2 and shape[2] == len(blackout_indices):
#                 data[key] = data[key][:, :, ~blackout_indices]

#     intervals = np.arange(0, data['spikes'].shape[1] * config.resolution.dt, config.window.curr)
#     train_intervals, test_intervals, finetune_intervals = split_data_by_interval(intervals, r_split=0.8, r_split_ft=0.01)

#     return data, intervals, train_intervals, test_intervals, finetune_intervals, visnav_callback




