import sys
sys.path.append('./neuroformer')

import itertools
import torch
import numpy as np
import pandas as pd
import pickle
import os


def split_data_by_interval(intervals, r_split=0.8, r_split_ft=0.1):
    chosen_idx = np.random.choice(len(intervals), int(len(intervals) * r_split))
    train_intervals = intervals[chosen_idx]
    test_intervals = intervals[~chosen_idx]
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
