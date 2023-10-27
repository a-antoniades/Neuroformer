import wandb
import torch
from neuroformer.trainer import TrainerConfig, Trainer

def train_sweep():
    run = wandb.init(project="neuroformer", 
                     group=f"visnav_{DATASET}_windows")
    
    # Update the config with the hyperparameters from the sweep
    if get_attr(run.config, 'window_frame') is not None:
        config.window.frame = run.config.window_frame
    if get_attr(run.config, 'window_curr') is not None:
        config.window.curr = run.config.window_curr
    if get_attr(run.config, 'window_prev') is not None:
        config.window.prev = run.config.window_prev
    if get_attr(run.config, 'window_speed') is not None:
        config.window.speed = run.config.window_speed
    
    config.id_vocab_size = tokenizer.ID_vocab_size
    model = GPT(config, tokenizer)

    # %%
    loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
    iterable = iter(loader)
    x, y = next(iterable)
    # x = all_device(x, 'cuda')
    # y = all_device(y, 'cuda')
    recursive_print(y)
    preds, features, loss = model(x, y)

    # %%
    MAX_EPOCHS = 150
    BATCH_SIZE = 32 * 4
    SHUFFLE = True

    if config.gru_only:
        model_name = "GRU"
    elif config.mlp_only:
        model_name = "MLP"
    elif config.gru2_only:
        model_name = "GRU_2.0"
    else:
        model_name = "Neuroformer"

    CKPT_PATH = f"/share/edc/home/antonis/neuroformer/models/NF.15/Visnav_VR_Expt/sweeps/{DATASET}/{model_name}/speed_{modalities['all']['speed']['objective']}/{str(config.layers)}/{SEED}"
    CKPT_PATH = CKPT_PATH.replace("namespace", "").replace(" ", "_")

    tconf = TrainerConfig(max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, learning_rate=7e-5, 
                        num_workers=16, lr_decay=False, patience=3, warmup_tokens=8e7, 
                        decay_weights=True, weight_decay=1.0, shuffle=SHUFFLE,
                        final_tokens=len(train_dataset)*(config.block_size.id) * (MAX_EPOCHS),
                        clip_norm=1.0, grad_norm_clip=1.0,
                        show_grads=False,
                        ckpt_path=CKPT_PATH, no_pbar=False, 
                        dist=DIST, save_every=0, eval_every=5, min_eval_epoch=50,
                        use_wandb=True, wandb_project="neuroformer", wandb_group=f"visnav_{DATASET}",)

    trainer = Trainer(model, train_dataset, test_dataset, tconf, config)
    trainer.train()

sweep_id = "woanderers/neuroformer/h896vm04"
print(f"-- SWEEP_ID -- {sweep_id}")
wandb.agent(sweep_id, function=train_sweep)
