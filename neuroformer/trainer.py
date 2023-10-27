import math
import logging
import pickle

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# from utils import set_plot_params
# set_plot_params()
# plt.rcParams["figure.figsize"] = (20,20)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import wandb
from datetime import datetime
now = datetime.now()
logger = logging.getLogger(__name__)

import collections
import omegaconf
from omegaconf import OmegaConf
import os
parent_path = os.path.dirname(os.path.dirname(os.getcwd())) + "/"

from utils import object_to_dict, save_yaml, all_device


# from torch.nn.parallel import DistributedDataParallell as dist
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from utils import extract_latents
from utils import get_attr, save_config



class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6   # these two numbers came from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    # plot gradient flow
    show_grads = False
    shuffle = True
    score_metrics = ['precision', 'recall', 'F1'] #, 'precision_top5', 'recall_top5', 'F1_top5']
    no_pbar = True
    dist = False
    save_every = 0
    loss_bprop = None
    get_latents = False
    use_wandb = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, omegaconf.dictconfig.DictConfig):
                print("DictConfig")
                for name, value in v.items():
                    print(name, value)
                    setattr(self, name, value)
            else:
                setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, mconf=None):
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mconf = mconf if mconf is not None else self.model.config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.train_losses = []
        self.test_losses = []

        logdir = self.config.ckpt_path[:-3] + "/"
        # self.writer = SummaryWriter(logdir)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            # self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            # self.criterion = self.criterion.to(self.device)

            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                rank = int(os.environ["RANK"])
                world_size = int(os.environ['WORLD_SIZE'])
                local_rank = int(os.environ['LOCAL_RANK'])
                dist.init_process_group(backend='nccl',
                                        init_method='env://', 
                                        world_size=world_size,
                                        rank=rank)
                torch.cuda.set_device(local_rank)
                dist.barrier()

                self.device = torch.device('cuda', torch.cuda.current_device())
                self.model = self.model.to(self.device)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                       device_ids=[torch.cuda.current_device()],
                                                                       output_device=torch.cuda.current_device(),
                                                                       find_unused_parameters=True)
                            
            else:
                self.device = torch.device('cuda', torch.cuda.current_device())
                self.model = self.model.to(self.device)
                # self.model = torch.nn.DataParallel(self.model)
                # self.writer.add_scalar(f"model/no_parameters", sum(p.numel() for p in model.parameters()))
        
        print(f"-- USE WANDB: {config.use_wandb} --")
        if config.use_wandb:
            wandb.init(project="neuroformer", 
                       group=config.wandb_group,
                       name=config.wandb_name)
            wandb.watch(self.model) 


    def save_checkpoint(self, loss, epoch=None):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if os.path.exists(self.config.ckpt_path) is False:
            os.makedirs(self.config.ckpt_path)
        if epoch is not None:
            save_pth = os.path.join(self.config.ckpt_path, f"_epoch_{epoch}.pt")
            logger.info("saving %s", save_pth)
            torch.save(raw_model.state_dict(), save_pth)
        else:
            save_pth = os.path.join(self.config.ckpt_path, f"model.pt")
            logger.info("saving %s", self.config.ckpt_path)
            torch.save(raw_model.state_dict(), save_pth)
        
        # mconf = object_to_dict(self.mconf)
        mconf = self.mconf
        tconf = object_to_dict(self.config)
        # dconf = object_to_dict(self.train_dataset)
        print(mconf)

        # save_yaml(mconf, os.path.join(self.config.ckpt_path, "mconf.yaml"))
        # save yaml
        save_config(mconf, os.path.join(self.config.ckpt_path, "mconf.yaml"))
        save_yaml(tconf, os.path.join(self.config.ckpt_path, "tconf.yaml"))
        # save_yaml(dconf, os.path.join(self.config.ckpt_path, "dconf.yaml"))

        if hasattr(raw_model, "tokenizer"):
            with open(os.path.join(self.config.ckpt_path, "tokenizer.pkl"), 'wb') as f:
                pickle.dump(raw_model.tokenizer, f)

    def save_latents(self, model, dataset, save_file):
        print(f"Saving latents to {save_file}")
        feats, latents = extract_latents(model, dataset)
        embeddings = torch.stack(latents['id']).detach().cpu().numpy()
        labels = np.stack(feats['behavior'])[:, :, 0, 0]
        if not os.path.exists(save_file):
            os.makedirs(save_file)
        np.save(os.path.join(save_file, "embeddings.npy"), embeddings)
        np.save(os.path.join(save_file, "labels.npy"), labels)
                    
    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                if p.grad is None:
                    ave_grads.append(0)
                    max_grads.append(0)
                else:
                    ave_grads.append(p.grad.abs().mean().to('cpu'))
                    max_grads.append(p.grad.abs().max().to('cpu'))
                plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
                plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
                plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
                plt.xticks(range(0,len(ave_grads), 1), [layer for layer in layers], size=20, rotation="vertical")
                plt.xlim(left=0, right=len(ave_grads))
                plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
                plt.xlabel("Layers", size=20)
                plt.ylabel("average gradient", size=20)
                plt.title("Gradient flow", size=30)
                plt.grid(True)
                plt.legend([Line2D([0], [0], color="c", lw=4),
                            Line2D([0], [0], color="b", lw=4),
                            Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
                plt.tight_layout()
        plt.show()
        # plt.savefig('grad_flow.png')

    def train(self):
        model, config, mconf = self.model, self.config, self.mconf
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-3, min_lr=1e-6, 
                                                        #  factor=0.3, patience=config.patience, verbose=True)
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            sampler = DistributedSampler(data, shuffle=True) if config.dist else None
            loader = DataLoader(data, pin_memory=False,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers, sampler=sampler)

            scores = collections.defaultdict(list)
            losses = collections.defaultdict(list)
            pbar = tqdm(enumerate(loader), total=len(loader), disable=self.config.no_pbar) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                # place data on the correct device
                x = all_device(x, self.device)
                y = all_device(y, self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    preds, features, loss = model(x, y)
                    total_loss = 0
                    
                    for key, value in loss.items():
                        # print(key)
                        value = value.mean()
                        loss[key] = value
                        if config.loss_bprop is not None \
                            and key not in config.loss_bprop:
                                continue
                        else:
                            total_loss += value
                        losses[key].append(value.item())
            
                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    total_loss.backward()
                    
                    if config.show_grads is True:
                        self.plot_grad_flow(model.named_parameters())
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    lr = optimizer.param_groups[0]['lr']
                    # report progress
                    precision = preds['precision']
                    # pbar.set_description(f"epoch {epoch+1} iter {it}: frame_loss: {loss['frames'].item():.5f} id_loss {loss['id'].item():.5f} dt_loss: {loss['dt'].item():.5f}   total_loss: {total_loss.item():.5f}. lr {lr:e}")
                    pbar.set_description(f'epoch {epoch+1}  ' + ''.join([f'{str(key)}_{str(split)}: {value:.5f}  ' for key, value in loss.items()]) + \
                                         f'total_loss: {total_loss.mean():.5f}' + f' lr {lr:e}' + ' ' + f'precision: {precision.mean():.5f}')
            
                    #  linear warmup
                    lr_mult = 1
                    self.tokens += (y['id']>=0).sum() # number of tokens processed this step (i.e label is not -100)
                    if self.tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                    else:
                        if config.lr_decay:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    # Wandb logging if specified
                    if config.use_wandb:
                        for loss_name, loss_value in loss.items():
                            wandb.log({f"Loss/{split}_{str(loss_name)}": loss_value})
                        for score in config.score_metrics:
                            wandb.log({f"Score/{split}_{str(score)}": preds[score].mean()})
                
                # if it % 100 == 0:
                #     self.save_checkpoint(it, np.array(scores['F1']).mean())

                for score in config.score_metrics:
                    scores[score].append(preds[score])

                if config.save_every > 0 and it % config.save_every == 0 and it > 0:
                    self.save_checkpoint(total_loss.cpu().detach().numpy(), it)
                    
            # tensorboard
            av_losses = collections.defaultdict(list)
            total_losses = 0
            for key, value in losses.items():
                av_losses[key] = np.array(value).mean()
                if config.loss_bprop is not None \
                    and key not in config.loss_bprop:
                        continue
                else:
                    total_losses += av_losses[key]
                # self.writer.add_scalar(f"Loss/{split}_{str(key)}", av_losses[key], epoch)
                if not is_train:
                    if config.use_wandb:
                        wandb.log({f"Loss/{split}_{str(key)}": av_losses[key]})
            # self.writer.add_scalar(f"Loss/{split}_total", total_losses, epoch)
            if config.use_wandb:
                wandb.log({f"Loss/{split}_total": total_losses})
            for score in config.score_metrics:
                # self.writer.add_scalar(f"Score/{split}_{str(score)}", preds[score].mean(), epoch)
                if config.use_wandb:
                    wandb.log({f"Score/{split}_{str(score)}": preds[score].mean()})
             
            if not is_train:
                # for score in config.score_metrics:
                    # scores[score] = np.array(scores[score].cpu()).mean()
                # scores['F1'] = 2 * scores['precision'] * scores['recall'] / (scores['precision'] + scores['recall'])
                logger.info('  '.join([f'{str(key)}_{str(split)}: {value:.5f}  ' for key, value in av_losses.items()]))
                # logger.info('  '.join([f'{str(key)}_{str(split)}: {value:.5f}  ' for key, value in scores.items() if key in config.score_metrics]))
            
                return total_losses.item(), av_losses, scores

        best_loss = float('inf')
        best_decoder_loss = float('inf')
        best_clip_loss = float('inf')
        best_f1 = 0
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            if self.train_dataset is not None:
                if hasattr(raw_model.config, "epoch"):
                    raw_model.config.epoch += 1
                else:
                    raw_model.config.epoch = 1
                run_epoch('train')
            if self.test_dataset is not None:
                if epoch > get_attr(self, 'min_eval_epoch', 0) and \
                    (epoch % get_attr(self, 'eval_every', 1) == 0 or epoch == config.max_epochs - 1):
                    test_loss, av_losses, scores = run_epoch('test')
                else:
                    self.save_checkpoint(epoch)
                    continue
                # if config.lr_decay:
                #     scheduler.step(test_loss)
            if self.config.get_latents:
                self.save_latents(self.model, self.train_dataset, os.path.join(os.path.dirname(self.config.ckpt_path), 'latents', 'train', f"epoch_{epoch}"))

            # supports early stopping based on the test loss, or just save always if no test set is provided
            if self.test_dataset is None:
                self.save_checkpoint(epoch)
            else:
                good_model = test_loss < best_loss
                if good_model:
                    best_loss = test_loss
                    self.save_checkpoint(test_loss)
            if self.test_dataset is not None:
                # save best checkpoint for each loss
                for loss_type in av_losses.keys():
                    good_model = av_losses[loss_type] < best_loss
                    if good_model:
                        best_loss = av_losses[loss_type]
                        self.save_checkpoint(av_losses[loss_type], epoch=loss_type)
                
                # F1_score = torch.tensor(scores['F1']).mean()
                # if F1_score > best_f1:
                #     self.save_checkpoint(scores['F1'], epoch=f"F1:{F1_score}_epoch:{epoch}")
                #     best_f1 = F1_score
                # best_precision = scores['F1']
                # self.save_checkpoint(epoch, scores['F1'])
