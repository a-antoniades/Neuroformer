# %%

import glob
import os

import sys
import glob
from pathlib import Path, PurePath
path = Path.cwd()
parent_path = path.parents[1]
sys.path.append(str(PurePath(parent_path, 'neuroformer')))
sys.path.append('neuroformer')
sys.path.append('.')
sys.path.append('../')

import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataloader import DataLoader

import math

from neuroformer.model_neuroformer import Neuroformer, NeuroformerConfig
from neuroformer.utils import get_attr
from neuroformer.trainer import Trainer, TrainerConfig
from neuroformer.utils import (set_seed, update_object, running_jupyter, 
                                 all_device, load_config, 
                                 dict_to_object, object_to_dict, recursive_print,
                                 create_modalities_dict)
from neuroformer.visualize import set_plot_params
from neuroformer.data_utils import round_n, Tokenizer, NFDataloader
from neuroformer.datasets import load_visnav, load_V1AL

parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"
import wandb

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
else:
    print("Running in terminal")
    args = parse_args()

# SET SEED - VERY IMPORTANT
set_seed(args.seed)

print(f"CONTRASTIUVEEEEEEE {args.contrastive}")
print(f"VISUAL: {args.visual}")
print(f"PAST_STATE: {args.past_state}")

# Use the function
if args.config is None:
    config_path = "./configs/NF_1.5/VisNav_VR_Expt/gru2_only_cls/mconf.yaml"
else:
    config_path = args.config
config = load_config(config_path)  # replace 'config.yaml' with your file path


# %%
""" 

-- DATA --
neuroformer/data/OneCombo3_V1AL/
df = response
video_stack = stimulus
DOWNLOAD DATA URL = https://drive.google.com/drive/folders/1jNvA4f-epdpRmeG9s2E-2Sfo-pwYbjeY?usp=sharing


"""

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

""" 
 - see mconf.yaml "modalities" structure:

modalities:
  behavior:
    n_layers: 4
    window: 0.05
    variables:
      speed:
        data: speed
        dt: 0.05
        predict: true
        objective: regression
      phi:
        data: phi
        dt: 0.05
        predict: true
        objective: regression
      th:
        data: th
        dt: 0.05
        predict: true
        objective: regression


Modalities: any additional modalities other than spikes and frames
    Behavior: the name of the <modality type>
        Variables: the name of the <modality>
            Data: the data of the <modality> in shape (n_samples, n_features)
            dt: the time resolution of the <modality>, used to index n_samples
            Predict: whether to predict this modality or not.
                     If you set predict to false, then it will 
                     not be used as an input in the model,
                     but rather to be predicted as an output. 
            Objective: regression or classification

"""

frames = {'feats': stimulus, 'callback': callback, 'window': config.window.frame, 'dt': config.resolution.dt}


def configure_token_types(config, modalities):
    max_window = max(config.window.curr, config.window.prev)
    dt_range = math.ceil(max_window / dt) + 1
    n_dt = [round_n(x, dt) for x in np.arange(0, max_window + dt, dt)]

    token_types = {
        'ID': {'tokens': list(np.arange(0, data['spikes'].shape[0] if isinstance(data['spikes'], np.ndarray) \
                                    else data['spikes'][1].shape[0]))},
        'dt': {'tokens': n_dt, 'resolution': dt},
        **({
            modality: {
                'tokens': sorted(list(set(eval(modality)))),
                'resolution': details.get('resolution')
            }
            # if we have to classify the modality, 
            # then we need to tokenize it
            for modality, details in modalities.items() if config.modalities is not None
            if details.get('predict', False) and details.get('objective', '') == 'classification'
        } if modalities is not None else {})
    }
    return token_types

modalities = create_modalities_dict(data, config.modalities) if get_attr(config, 'modalities', None) else None
token_types = configure_token_types(config, modalities)
tokenizer = Tokenizer(token_types)


# %%
if modalities is not None:
    for modality_type, modality in modalities.items():
        for variable_type, variable in modality.items():
            print(variable_type, variable)


# %%
train_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                             frames=frames, intervals=train_intervals, modalities=modalities)
test_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                            frames=frames, intervals=test_intervals, modalities=modalities)
finetune_dataset = NFDataloader(spikes_dict, tokenizer, config, dataset=args.dataset, 
                                frames=frames, intervals=finetune_intervals, modalities=modalities)

    
# print(f'train: {len(train_dataset)}, test: {len(test_dataset)}')
iterable = iter(train_dataset)
x, y = next(iterable)
print(x['id'])
print(x['dt'])
recursive_print(x)

# Update the config
config.id_vocab_size = tokenizer.ID_vocab_size
model = Neuroformer(config, tokenizer)

# Create a DataLoader
loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)
iterable = iter(loader)
x, y = next(iterable)
recursive_print(y)
preds, features, loss = model(x, y)

# Set training parameters
MAX_EPOCHS = 250
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

CKPT_PATH = f"./models/NF.15/Visnav_VR_Expt/{args.dataset}/{model_name}/{args.title}/{str(config.layers)}/{args.seed}"
CKPT_PATH = CKPT_PATH.replace("namespace", "").replace(" ", "_")

if os.path.exists(CKPT_PATH):
    counter = 1
    print(f"CKPT_PATH {CKPT_PATH} exists!")
    while os.path.exists(CKPT_PATH + f"_{counter}"):
        counter += 1

if args.resume is not None:
    model.load_state_dict(torch.load(args.resume),
                           strict=False)

if args.sweep_id is not None:
    # this is for hyperparameter sweeps
    from neuroformer.hparam_sweep import train_sweep
    print(f"-- SWEEP_ID -- {args.sweep_id}")
    wandb.agent(args.sweep_id, function=train_sweep)
else:
    # Create a TrainerConfig and Trainer
    tconf = TrainerConfig(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=1e-4, 
                          num_workers=16, lr_decay=True, patience=3, warmup_tokens=8e7, 
                          decay_weights=True, weight_decay=1.0, shuffle=SHUFFLE,
                          final_tokens=len(train_dataset)*(config.block_size.id) * (MAX_EPOCHS),
                          clip_norm=1.0, grad_norm_clip=1.0,
                          show_grads=False,
                          ckpt_path=CKPT_PATH, no_pbar=False, 
                          dist=args.dist, save_every=0, eval_every=5, min_eval_epoch=50,
                          use_wandb=True, wandb_project="neuroformer", 
                          wandb_group=f"1.5.1_visnav_{args.dataset}", wandb_name=args.title)

    trainer = Trainer(model, train_dataset, test_dataset, tconf, config)
    trainer.train()