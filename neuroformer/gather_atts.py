import os
import glob
import collections

import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append("neuroformer")
sys.path.append("/home/antonis/projects/slab/git/neuroformer/neuroformer/")

import pandas as pd
import numpy as np

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from torch.utils.data.dataloader import DataLoader

import math
from torch.utils.data import Dataset

from neuroformer.attention_vis import AttentionVis
from trainer import Trainer, TrainerConfig
from utils import set_seed


from scipy import io as scipyio
from scipy.special import softmax
import skimage
import skvideo.io
from utils import print_full
from scipy.ndimage.filters import gaussian_filter, uniform_filter


import matplotlib.pyplot as plt
from utils import *
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"


# R3D: (3 x T x H x W)

from SpikeVidUtils import image_dataset

def nearest(n, x):
  u = n % x > x // 2
  return n + (-1)**(1 - u) * abs(x * u - n % x)

train_path = "/Users/antonis/projects/slab/neuroformer/neuroformer/data/stimulus/Naturalmovie"
vid_paths = sorted(glob.glob(train_path + '/*.tif'))
vid_list = [skimage.io.imread(vid)[::3] for vid in vid_paths]
video_stack = [torch.nan_to_num(image_dataset(vid)).transpose(1, 0) for vid in vid_list]

# 0.5
# df = pd.read_csv("/content/drive/MyDrive/Antonis/data/SimNeu/NaturalStim/NaturalStim_all.csv").iloc[:, 1:]
# 0.25
df = pd.read_csv("/Users/antonis/projects/slab/neuroformer/neuroformer/data/simulations/simNeu3D/response/Naturalmovie/NaturalStim_SimNeu3D_0.5dt.csv")
# df = pd.read_csv("/content/simNeu_3D_Combo4_1000Rep.csv")

# for OneCombo
# video_stack = torch.load("/content/OneCombo3_(2,3)_stimuli.pt")

window = 0.5
dt = 0.05
# n_dt = sorted((df['Interval_dt'].unique()).round(2)) 
dt_range = math.ceil(window / dt) + 1  # add first / last interval for SOS / EOS'
n_dt = [round(dt * n, 3) for n in range(dt_range)]
df['Time'] = df['Time'].round(3)

from SpikeVidUtils import SpikeTimeVidData

## qv-vae feats
# frames = torch.load(parent_path + "code/data/SImNew3D/stimulus/vq-vae_code_feats-24-05-4x4x4.pt").numpy() + 2
# frame_feats = torch.load(parent_path + "code/data/SImNew3D/stimulus/vq-vae_embed_feats-24-05-4x4x4.pt").numpy()
# frame_block_size = frames.shape[-1] - 1

## resnet3d feats
kernel_size = (20, 8, 8)
T_FRAME = 20 // kernel_size[0]
n_embd = 256
n_embd_frames = 64
frame_feats = video_stack

frame_block_size = (20 // kernel_size[0] * 64 * 112) // (n_embd_frames)
# frame_block_size = 560
prev_id_block_size = 70
id_block_size = prev_id_block_size   # 95
block_size = frame_block_size + id_block_size + prev_id_block_size # frame_block_size * 2  # small window for faster training
frame_memory = 20   # how many frames back does model see
window = window

neurons = sorted(list(set(df['ID'])))
id_stoi = { ch:i for i,ch in enumerate(neurons) }
id_itos = { i:ch for i,ch in enumerate(neurons) }

# translate neural embeddings to separate them from ID embeddings
# frames = frames + [*id_stoi.keys()][-1] 
neurons = [i for i in range(df['ID'].min(), df['ID'].max() + 1)]
# pixels = sorted(np.unique(frames).tolist())
feat_encodings = neurons + ['SOS'] + ['EOS'] + ['PAD']  # + pixels 
stoi = { ch:i for i,ch in enumerate(feat_encodings) }
itos = { i:ch for i,ch in enumerate(feat_encodings) }
stoi_dt = { ch:i for i,ch in enumerate(n_dt) }
itos_dt = { i:ch for i,ch in enumerate(n_dt) }
max(list(itos_dt.values()))

# train_len = round(len(df)*(4/5))
# test_len = round(len(df) - train_len)

# train_data = df[:train_len]
# test_data = df[train_len:train_len + test_len].reset_index().drop(['index'], axis=1)

n = [i for i in range(900, 999)]
# n = [i for i in range(16, 20)]
train_data = df[~df['Trial'].isin(n)].reset_index(drop=True)
test_data = df[df['Trial'].isin(n)].reset_index(drop=True)
small_data = df[df['Trial'].isin([7])].reset_index(drop=True)
small_data = small_data[small_data['Interval'].isin(small_data['Interval'].unique()[:100])]

from SpikeVidUtils import SpikeTimeVidData2

# train_dat1aset = spikeTimeData(spikes, block_size, dt, stoi, itos)

train_dataset = SpikeTimeVidData2(train_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False)
test_dataset = SpikeTimeVidData2(test_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False)
dataset = SpikeTimeVidData2(df, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False)
# single_batch = SpikeTimeVidData(df[df['Trial'].isin([5])], None, block_size, frame_block_size, prev_id_block_size, window, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats)
small_dataset = SpikeTimeVidData2(small_data, None, block_size, id_block_size, frame_block_size, prev_id_block_size, window, dt, frame_memory, stoi, itos, neurons, stoi_dt, itos_dt, frame_feats, pred=False)


print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')

# for isconv in [True, False]:
from neuroformer.model_neuroformer_ import GPT, GPTConfig, neuralGPTConfig, Decoder
# initialize config class and model (holds hyperparameters)
# for is_conv in [True, False]:    
conv_layer = True
mconf = GPTConfig(train_dataset.population_size, block_size,    # frame_block_size
                        id_vocab_size=train_dataset.id_population_size,
                        frame_block_size=frame_block_size,
                        id_block_size=id_block_size,  # frame_block_size
                        prev_id_block_size=prev_id_block_size,
                        sparse_mask=False, p_sparse=0.25, sparse_topk_frame=None, sparse_topk_id=None,
                        n_dt=len(n_dt),
                        data_size=train_dataset.size,
                        class_weights=None,
                        pretrain=False,
                        n_state_layers=1, n_state_history_layers=0, n_stimulus_layers=6,
                        n_layer=10, n_head=4, n_embd=n_embd, n_embd_frames=n_embd_frames,
                        contrastive=True, clip_emb=1024, clip_temp=0.5,
                        conv_layer=conv_layer, kernel_size=kernel_size,
                        temp_emb=True, pos_emb=False,
                        id_drop=0.2, im_drop=0.2)
model = GPT(mconf)
# model_path = "/Users/antonis/Downloads/dict_simNeu3D_correct_norel_sparse_(None, None)__dt__True_perceiver_1.0_0.5_0.05_(1, 0, 4)_8_256-2.pt"
# model_path = "/Users/antonis/Downloads/conv_3d_3Demb_simNeu3D_correct_norelTrue_sparse_(None, None)__dt__True_perceiver_1.0_0.5_0.05_(1, 0, 5)_4_256.pt"
# model_path = "/Users/antonis/Downloads/3Demb_simNeu3D_correct_norelFalse_sparse_(None, None)__dt__True_perceiver_1.0_0.5_0.05_(1, 0, 5)_4_256-2.pt"
model_path = "/Users/antonis/Downloads/contro_3_decoder_conv_3d_(20, 8, 8)_diff_embd_3Demb_simNeu3D_correct_norelTrue_sparse_(None, None)__dt__True_perceiver_1.0_0.5_0.05_(1, 0, 6)_2_256.pt"
model.load_state_dict(torch.load(model_path, map_location='cpu'))


from neuroformer.attention_vis import AttentionVis as AV

loader = DataLoader(train_dataset, shuffle=True, pin_memory=False,
                                  batch_size=1, num_workers=1)
stimulus_atts_scores_agg = AV().att_interval_frames(model, model.neural_visual_transformer.neural_state_stimulus_blocks,
                                loader, model.config.n_stimulus_layers, mconf.id_block_size, rollout=True, pad_key='pad', 
                                agg=True, stoi=stoi, max_it=75, n_layer=5)

# stimulus_atts_scores = att_interval(model, model.neural_visual_transformer.neural_state_block,
#                              loader, model.config.n_stimulus_layers, mconf.id_block_size, 'pad', agg=True)
# B, L, H, N, F = stimulus_atts_scores.shape
print(stimulus_atts_scores_agg.shape)

path = "/Users/antonis/projects/slab/neuroformer/neuroformer/notebooks/centroids/stim_atts_contrastive"
paths = sorted(glob.glob(path + '/*.pt'))

# for p in paths:
#     att = torch.load(p)
#     if stimulus_atts_scores_agg is None:
#         stimulus_atts_scores_agg = att
#     else:
#         stimulus_atts_scores_agg = np.concatenate((stimulus_atts_scores_agg, att), axis=0)

torch.save(stimulus_atts_scores_agg, path + f'/stimulus_atts_scores_agg{len(paths)}.pt')

