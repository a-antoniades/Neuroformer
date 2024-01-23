# %%
import os
import pickle

import sys
import glob
from pathlib import Path, PurePath
path = Path.cwd()
parent_path = path.parents[1]
sys.path.append(str(PurePath(parent_path, 'neuroformer')))
sys.path.append('neuroformer')

import pandas as pd
import numpy as np

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data.dataloader import DataLoader

from neuroformer.model_neuroformer import load_model_and_tokenizer
from neuroformer.utils import get_attr
from neuroformer.utils import (set_seed, running_jupyter, 
                                 all_device, recursive_print,
                                 create_modalities_dict)
from neuroformer.datasets import load_visnav, load_V1AL
from neuroformer.simulation import generate_spikes, decode_modality
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

from neuroformer.default_args import DefaultArgs, parse_args

if running_jupyter(): # or __name__ == "__main__":
    print("Running in Jupyter")
    args = DefaultArgs()
    # args.dataset = "medial"
    # args.ckpt_path = "./models/NF.15/Visnav_VR_Expt/medial/Neuroformer/predict_all_behavior/(state_history=6,_state=6,_stimulus=6,_behavior=6,_self_att=6,_modalities=(n_behavior=25))/25"
    
    args.dataset = "lateral"
    args.ckpt_path = "./models/predict_all_behavior/(state_history=6,_state=6,_stimulus=6,_behavior=6,_self_att=6,_modalities=(n_behavior=25))/25"
    args.predict_modes = ['speed', 'phi', 'th']
else:
    print("Running in terminal")
    args = parse_args()


# SET SEED - VERY IMPORTANT
set_seed(args.seed)

print(f"CONTRASTIUVEEEEEEE {args.contrastive}")
print(f"VISUAL: {args.visual}")
print(f"PAST_STATE: {args.past_state}")


config, tokenizer, model = load_model_and_tokenizer(args.ckpt_path)

# %%
""" 

-- DATA --
neuroformer/data/OneCombo3_V1AL/
df = response
video_stack = stimulus
DOWNLOAD DATA URL = https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=sharing


"""
print(f"DATASET: {args.dataset}")
if args.dataset in ["lateral", "medial"]:
    data, intervals, train_intervals, \
    test_intervals, finetune_intervals, \
    callback = load_visnav(args.dataset, config, 
                           selection=config.selection if hasattr(config, "selection") else None)
elif args.dataset == "V1AL":
    data, intervals, train_intervals, \
    test_intervals, finetune_intervals, \
    callback = load_V1AL(config)

spikes = data['spikes']
stimulus = data['stimulus']

# %%
window = config.window.curr
window_prev = config.window.prev
dt = config.resolution.dt


# -------- #

spikes_dict = {
    "ID": data['spikes'],
    "Frames": data['stimulus'],
    "Interval": intervals,
    "dt": config.resolution.dt,
    "id_block_size": config.block_size.id,
    "prev_id_block_size": config.block_size.prev_id,
    "frame_block_size": config.block_size.frame,
    "window": config.window.curr,
    "window_prev": config.window.prev,
    "frame_window": config.window.frame,
}

""" structure:
{
    type_of_modality:
        {name of modality: {'data':data, 'dt': dt, 'predict': True/False},
        ...
        }
    ...
}
"""



# %%
from neuroformer.data_utils import NFDataloader

modalities = create_modalities_dict(data, config.modalities)
frames = {'feats': stimulus, 'callback': callback, 'window': config.window.frame, 'dt': config.resolution.dt}

train_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                             frames=frames, intervals=train_intervals, modalities=modalities)
test_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                            frames=frames, intervals=test_intervals, modalities=modalities)
finetune_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                                frames=frames, intervals=finetune_intervals, modalities=modalities)

    
# print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')
iterable = iter(train_dataset)
x, y = next(iterable)
recursive_print(x)

# Update the config
config.id_vocab_size = tokenizer.ID_vocab_size

# Create a DataLoader
loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)
iterable = iter(loader)
x, y = next(iterable)
recursive_print(y)
preds, features, loss = model(x, y)

# Set training parameters
MAX_EPOCHS = 300
BATCH_SIZE = 32 * 5
SHUFFLE = True

if config.gru_only:
    model_name = "GRU"
elif config.mlp_only:
    model_name = "MLP"
elif config.gru2_only:
    model_name = "GRU_2.0"
else:
    model_name = "Neuroformer"

CKPT_PATH = args.ckpt_path

# Define the parameters
sample = True
top_p = 0.95
top_p_t = 0.95
temp = 1.
temp_t = 1.
frame_end = 0
true_past = args.true_past
get_dt = True
gpu = True
pred_dt = True

# Run the prediction function
# let's just run this on the finetune dataset
# just so it takes less time.
results_trial = generate_spikes(model, finetune_dataset, window, 
                                window_prev, tokenizer, 
                                sample=sample, top_p=top_p, top_p_t=top_p_t, 
                                temp=temp, temp_t=temp_t, frame_end=frame_end, 
                                true_past=true_past,
                                get_dt=get_dt, gpu=gpu, pred_dt=pred_dt,
                                plot_probs=False)

# Create a filename string with the parameters
filename = f"results_trial_sample-{sample}_top_p-{top_p}_top_p_t-{top_p_t}_temp-{temp}_temp_t-{temp_t}_frame_end-{frame_end}_true_past-{true_past}_get_dt-{get_dt}_gpu-{gpu}_pred_dt-{pred_dt}.pkl"

# Save the results in a pickle file
save_inference_path = os.path.join(CKPT_PATH, "inference")
if not os.path.exists(save_inference_path):
    os.makedirs(save_inference_path)

print(f"Saving inference results in {os.path.join(save_inference_path, filename)}")

with open(os.path.join(save_inference_path, filename), "wb") as f:
    pickle.dump(results_trial, f)


# %%
# model.load_state_dict(torch.load(os.path.join(CKPT_PATH, f"_epoch_speed.pt"), map_location=torch.device('cpu')))
model.load_state_dict(torch.load(os.path.join(CKPT_PATH, f"model.pt"), map_location=torch.device('cpu')))
args.predict_modes = ['speed', 'phi', 'th']
behavior_preds = {}
if args.predict_modes is not None:
    block_type = 'behavior'
    block_config = get_attr(config.modalities, block_type).variables
    for mode in args.predict_modes:
        mode_config = get_attr(block_config, mode)
        behavior_preds[mode] = decode_modality(model, test_dataset, modality=mode, 
                                          block_type=block_type, objective=get_attr(mode_config, 'objective'))
        filename = f"behavior_preds_{mode}.csv"
        save_inference_path = os.path.join(CKPT_PATH, "inference")
        if not os.path.exists(save_inference_path):
            os.makedirs(save_inference_path)
        print(f"Saving inference results in {os.path.join(save_inference_path, filename)}")
        behavior_preds[mode].to_csv(os.path.join(save_inference_path, filename))

# %%
def plot_regression(y_true, y_pred, mode, model_name, r, p, color='black', 
                    ax=None, axis_limits=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    ax.scatter(y_true, y_pred, s=100, alpha=0.5, color=color)

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    s_f = 0.8
    combined_limits = [min(xlims[0], ylims[0]) * s_f, max(xlims[1], ylims[1]) * s_f]
    ax.plot(combined_limits, combined_limits, 'k--', color='black')

    ax.set_xlabel(f'True {mode}', fontsize=20)
    ax.set_ylabel(f'Predicted {mode}', fontsize=20)
    # ax.set_title(f'{model_name}, Regression', fontsize=20)
    ax.text(0.05, 0.9, 'r = {:.2f}'.format(r), fontsize=20, transform=ax.transAxes)
    ax.text(0.05, 0.8, 'p < 0.001'.format(p), fontsize=20, transform=ax.transAxes)

    if axis_limits is not None:
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)

    if save_path:
        plt.savefig(os.path.join(save_path, 'regression_2.pdf'), dpi=300, bbox_inches='tight')

# %%
fig, ax = plt.subplots(figsize=(17.5, 5), nrows=1, ncols=len(args.predict_modes))
plt.suptitle(f'Visnav {args.dataset} Multitask Decoding - Speed + Eye Gaze (phi, th)', fontsize=20, y=1.01)
colors = ['limegreen', 'royalblue', 'darkblue']  # Define your colors here

for n, mode in enumerate(args.predict_modes):
    behavior_preds_mode = behavior_preds[mode]
    x_true, y_true = behavior_preds_mode['cum_interval'][:200], behavior_preds_mode['true'][:200]  # Limit to 200 examples
    x_pred, y_pred = behavior_preds_mode['cum_interval'][:200], behavior_preds_mode[f'behavior_{mode}_value'][:200]  # Limit to 200 examples
    r, p = pearsonr([float(y) for y in y_pred], [float(y) for y in y_true])
    axis = ax[n]
    plot_regression(y_true, y_pred, mode, model_name, r, p, ax=axis, color=colors[n], save_path=args.ckpt_path)  # Use color

# %%



