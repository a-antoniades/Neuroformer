# from code.transformer_vid.utils import convert_weights
# import rotary_embedding_torch
from torch.nn.modules.activation import GELU, ReLU
# from data.OneCombo3.trainer import TrainerConfig
import math
import numpy as np
import itertools
import pickle
import logging
from inspect import isfunction

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import copy
from torchvision.models.video import r3d_18
# from ResNet3D import r3d_18
# from torchvision.models._utils import IntermediateLayerGetter
import torchmetrics


from scipy.optimize import linear_sum_assignment
# from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

from einops import rearrange
from einops.layers.torch import Rearrange

import timm
import omegaconf

from modules import (PositionalEmbedding, PositionalEncoding2D, 
                     PositionalEncodingPermute2D, PositionalEncoding3D,
                     TemporalEmbedding, LearntTemporalEmbedding,
                     contrastive_loss, clip_loss, topk_metrics, separate_eos_tokens)
import collections
from utils import get_attr, object_to_dict, load_config

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(path):
    config = load_config(path + "/mconf.yaml")
    tokenizer = pickle.load(open(path + "/tokenizer.pkl", "rb"))
    model = Neuroformer(config, tokenizer)
    print(f" ///// <=----- Loading model from {path} -----=> \\\\\\")
    model.load_state_dict(torch.load(path + "/model.pt"))
    return config, tokenizer, model

def get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        else:
            return default
    elif hasattr(obj, name):
        return getattr(obj, name)
    else:
        return default

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):    # nn.Conv3d,
            l.weight.data = l.weight.data.half()
            if l.bias != None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor != None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr != None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def freeze_weights(model, keep_param):
    for name, param in model.named_parameters():
        if keep_param in name:
            print(f"keeping param {name}")
            param.requires_grad = True
        else:
            param.requires_grad = False

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class NeuroformerConfig:
    """ base Neuroformer config, params common to all Neuroformer versions """
    embd_pdrop = 0.2
    resid_pdrop = 0.2 
    attn_pdrop = 0.2
    pos_pdrop = 0.2
    temp_pdrop = 0.2
    pos_emb = True
    temp_emb = True
    start_prune = 30
    epoch = 0
    sparse_topk = None
    conv_layer = False
    self_att_layers = 0
    resnet_backbone = False
    vit_encoder = True
    freeze_weights = None
    mlp_only = False
    wave_emb = True

    eos_loss = False

    def __init__(self, vocab_size, block_size, **kwargs):
        self.id_vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            if isinstance(v, omegaconf.dictconfig.DictConfig):
                for name, value in v.items():
                    setattr(self, name, value)
            else:
                setattr(self, k, v)

class NeuroformerConfig:
    """ base Neuroformer config, params common to all Neuroformer versions """
    n = 0.4
    im_drop = 0.4
    id_drop = n
    embd_pdrop = n
    resid_pdrop = n
    attn_pdrop = n
    pos_pdrop = n
    temp_pdrop = n
    pos_emb = True
    temp_emb = True
    epoch = 0
    step = 0
    sparse_topk = None
    conv_layer = False
    self_att_layers = 0
    sparse_topk = None

    def __init__(self, vocab_size, block_size, **kwargs):
        self.id_vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class Neuroformer1Config(NeuroformerConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768



class Resnet3DBackbone(nn.Module):
    """ 
    R3D: (3 x T x H x W)
    H, W = 112
    """
    
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(*(list(r3d_18(pretrained=True).children())[:4]))
        convert_weights(self.backbone)
        # freeze backbone
        for k, v in self.backbone.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # B = Batch, T, C, Fm, H, W
        features = self.backbone(x)     # (B, C, T, H, W)
        B, C, T, H, W = features.shape
        features = rearrange(features, 'B C T H W -> B T H W C')
        return features


class VideoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_layer = config.conv_layer
        self.f_emb = int(config.n_embd_frames ** (1/2))
        
        if self.conv_layer:
            kernel_size = config.kernel_size
            stride_size = config.stride_size if hasattr(config, 'stride_size') else kernel_size
            padding_size = config.padding_size if hasattr(config, 'padding_size') else 0
            self.conv_block = torch.nn.Sequential(
                    nn.Conv3d(1, config.n_embd, kernel_size=kernel_size, stride=stride_size, padding=padding_size),
                    Rearrange('b e t h w -> b t h w e'),
                    nn.LayerNorm([config.n_embd]),
                    nn.ReLU()
        )

        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c t (h p1) (w p2) -> b t h w (p1 p2 c)', p1=self.f_emb, p2=self.f_emb),
                nn.Linear(config.n_embd_frames, config.n_embd),
                nn.ReLU()
            )

    def forward(self, x):
        if self.conv_layer:
            x = self.conv_block(x)
        else:
            x = self.to_patch_embedding(x)

        return x

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:3]))
        # self.embedding = model[0:2]
        self.backbone = model[2][:2]
    
    def forward(self, x):
        x = rearrange(x, 'b t h w c -> b (t h w) c')
        features = []
        # for frame in x:
            # frame = self.embedding(frame)
        x = self.backbone(x)
        # features.append(x)
        return x

class MultiheadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    
    """

    def __init__(self, config, kv_embd=None, sparse_topk=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config

        kv_embd = kv_embd if kv_embd != None else config.n_embd
        # key, query, value projections for all heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(kv_embd, config.n_embd)
        self.value = nn.Linear(kv_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.dropout.attn)
        self.resid_drop = nn.Dropout(config.dropout.resid)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # self.register_buffer("mask", self.build_mask(config.block_size.id))  
        self.n_head = config.n_head
        self.sparse_topk = config.sparse.topk if sparse_topk is None else sparse_topk

        self.att = None
        self.T = config.block_size

        # self.rotary_embedding = RotarySpatioTemporalEmbedding(config)
    
    def build_mask(self, block_size):
        mask = torch.tril(torch.ones((block_size, block_size)),
                                     ).view(1, 1, block_size, block_size)
        return mask
    
    def generate_sparse_mask(self, att, p, config):
        """
        Generate a sparse mask according to p.
        """
        assert p >= 0 and p <= 1, "p should be in [0, 1]"
        T = config.block_size
        mask = torch.rand((1, T)) < p
        mask = mask.repeat(T, 1)
        
        mask[0, 0] = False  # don't mask 1st step
        # check if any step is fully masked and umask it
        idx_all_true = (True == torch.all(mask, dim=0)).nonzero()
        for step in idx_all_true:
            sampler = torch.distributions.Uniform(low=0, high=step.item()+1)
            idx_false = sampler.sample((1,1)).long()
            mask[step, idx_false] = False

        # mask = mask.repeat(T, 1)
        mask = mask.view(1, 1, T, T).cuda() if att.is_cuda else mask.view(1, 1, T, T)
        att = att.masked_fill(mask, float('-inf'))
        return att

    def forward(self, q, k=None, v=None, tgt_mask=None, pad=None, dtx=None, sparse_topk=None):
        assert torch.equal(k, v) is True, f"Keys {k.shape} and Values must be the same {v.shape}"
        if None not in (k, v):
            assert k.size() == v.size(), "Keys and Values must be of same size"
            # assert q.size(-1) == k.size(-1) == v.size(-1), "Embedding dims must be of same size"
        else:
            k, v = q, q

        Bt, Tt, Ct = q.size()
        Bs, Ts, Cs = k.size()

        # calculate query, key, values for all head in batch and move head forward to the batch dim]
        q = self.query(q).view(Bt, Tt, self.n_head, Ct // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(k).view(Bs, Ts, self.n_head, Ct // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(v).view(Bs, Ts, self.n_head, Ct // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Normalize value
        # v = F.normalize(v, p=2.0, dim=-1)

        # # apply rotary embeddings
        # if dtx != None:
        #     q, k = self.rotary_embedding(q, k, dtx)

        # causal self-attention; Self-attend: (B, nh, Tt, hs) x (B, nh, hs, Ts) -> (B, nh, Tt, Ts)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if tgt_mask != None:
            att = att.masked_fill(tgt_mask[:,:,:Tt,:Tt] == 0, float('-inf'))
            # if self.training:
            #     att = self.generate_sparse_mask(att, 0.25, self.config)
            # if pad != None and self.training:
            #     for idx, i in enumerate(pad):
            #         att[idx, :, :, Tt - i:] = float('-inf')   # only able to see first padding token
        

        # Explicit Sparse Attention - Use top-K qk values
        if self.sparse_topk != None and self.sparse_topk < att.shape[-1]:
            top, _ = torch.topk(att, self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(att)
            mask = att < vk
            att.masked_fill_(mask, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        
        # self.att = att
        att = self.attn_drop(att)
        self.att = att
        y = att @ v # (B, nh, Tt, Ts) x (B, nh, Ts, hs) -> (B, nh, Tt, hs)
        y = y.transpose(1, 2).contiguous().view(Bt, Tt, Ct) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class ProjectNorm(nn.Module):

    def __init__(self, feat_size, target_size):
        super().__init__()
        self.ln = nn.LayerNorm(feat_size)
        self.mlp = nn.Sequential(
            nn.Linear(feat_size, math.floor(2 * feat_size), bias=False),
            nn.GELU(),
            nn.Linear(math.floor(2 * feat_size), target_size, bias=False),
        )

    def forward(self, x):
        return self.mlp(self.ln(x))


class TimeProjection(nn.Module):
    
    def __init__(self, seq_size, id_seq_size, feat_size, target_size):
        super().__init__()
        self.mlp_seq = nn.Sequential(
            nn.Linear(seq_size, id_seq_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(id_seq_size, id_seq_size)
        )
        self.mlp_t = nn.Sequential(
            nn.Linear(feat_size, feat_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(feat_size // 2, target_size)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # B, T, C -> B, C, T
        x = self.mlp_seq(x)     # B, C, T / 2
        x = x.permute(0, 2, 1)  # B, T / 2, C
        return self.mlp_t(x)    # B, T / 2, 1


class PSTHProjection(nn.Module):
    """Takes Last Output of Block -> (B, C) 
       Builds PSTH table                  
    """
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(config.n_embd * 4, config.id_vocab_size, bias=False)
        )
    
    def forward(self, x):
        return self.mlp(x)


# class PSTHProjection(nn.Module):
    
#     def __init__(self, config):
#         super().__init__()
#         self.mlp_seq = nn.Sequential(
#             nn.Linear(config.id_block_size, config.id_block_size // 2, bias=False),
#             nn.GELU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(config.id_block_size // 2, 1, bias=False)
#         )
#         self.mlp_t = nn.Sequential(
#             nn.Linear(config.n_embd, config.n_embd * 4, bias=False),
#             nn.GELU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(config.n_embd * 4, config.id_vocab_size, bias=False)
#         )
    
#     def forward(self, x):
#         x = x.transpose(-1, -2)  # B, T, C -> B, C, T
#         x = self.mlp_seq(x)     # B, C, 1
#         x = x.transpose(-2, -1)  # B, 1, Vocab_id
#         return self.mlp_t(x)



class DiceLossPSTH(nn.Module):
    def __init__(self, size_average=True, smooth=1):
        super().__init__()
    
    def cross_entropy(self, input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))
    
    def forward(self, logits, targets, smooth=1, class_weights=None):
        total_logits = F.layer_norm(torch.sum(logits, dim=-2), [logits.size()[-1]])
        # probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax(total_logits, dim=-1)
        # logits = F.gelu(logits)
        # probs = logits / (logits.max(dim=-1).values.unsqueeze(-1))
        # flatten label and prediction tensors
        outputs = probs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        labels = torch.zeros_like(outputs)
        labels[targets] = 1 / len(targets)
        # intersection = (outputs * labels).sum()
        # dice = (2. * intersection + smooth) / (outputs.sum() + labels.sum() + smooth)
        return self.cross_entropy(outputs[None, ...], labels[None, ...])


class SetLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def cross_entropy(self, input, target):
        return torch.mean(-torch.sum(target * torch.log(input), 1))
    
    def forward(self, logits, targets):
        targets = targets.contiguous().view(-1)
        loss = 0
        for n_step, n_logits in enumerate(logits):
            n_logits = F.softmax(n_logits, dim=-1)
            n_target = targets[n_step:]
            n_target_dist = torch.zeros_like(n_logits)
            if len(n_target) != 0:
                n_target_dist[n_target] = 1 / len(n_target)
                loss += self.cross_entropy(n_logits[None,...], n_target_dist[None, ...])
        return loss / len(logits)


class TruncatedLoss(nn.Module):

    def __init__(self, q=0.8, k=0.2, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes=None):
        p = F.softmax(logits, dim=-1)
        Yg = torch.gather(p, 2, targets.unsqueeze(2))

        loss = ((1-(Yg**self.q))/self.q) - ((1-(self.k**self.q))/self.q)
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=-1)
        Yg = torch.gather(p, 2, targets.unsqueeze(2))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        
        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


# class PSTHLOSS(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, logits, targets):
#         total_logits = torch.sum(logits, dim=-2)    # sum over sequence dimension
#         probs = F.softmax(total_logits, dim=-1)
#         outptuts


class HungarianMatcher(nn.Module):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, logits, targets, device):
        T, C = logits.size()
        probs = F.softmax(logits, dim=-1)
        cost_id = (1 - probs[:, targets]).cpu().view(T, -1).unsqueeze(0)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_id.split(len(targets), -1))]
        return [(torch.as_tensor(i, dtype=torch.int64, device=device), torch.as_tensor(j, dtype=torch.int64, device=device)) for i, j in indices]

class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.KLdiv = nn.KLDivLoss()
    def forward(self, logits, targets):
        log_probs = self.log_softmax(logits)
        return self.KLdiv(log_probs.long(), targets)


class PoissonCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # self.softmax = nn.Softmax(dim=-1)
        self.nll_poisson = nn.PoissonNLLLoss()
        # self.nll_poisson = nn.NLLLoss()

    def forward(self, logits, targets):
        log_probs = self.log_softmax(logits)
        return self.nll_poisson(log_probs, targets)


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, kv_embd=None, sparse_topk=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttention(config, kv_embd, sparse_topk)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout.resid),
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(self, q, k=None, v=None, mask=None, pad=None, dtx=None):
        x = q
        # x = self.ln1(x + self.attn(x, k, v, mask, pad))
        # x = self.ln2(x + self.mlp(x))
        x = x + self.attn(self.ln1(x), k, v, mask, pad)
        x = x + self.mlp(self.ln2(x))
        return self.ln_f(x)


class BlockSequential(nn.Sequential):
    def forward(self, q, k, v, mask=None, pad=None, dtx=None):
        for module in self._modules.values():
            x = module(q, k, v, mask, pad, dtx)
        return x


# def contrastive_loss(image_features, neural_features, temp=0.5):

#     device = image_features.device

#     Bid, Tid = image_features.size()
#     Bim, Tim = neural_features.size()

#     assert Tid==Tim, "feature sequences not equal"
#     B = Bid = Bim
#     T = Tid = Tim

#     # resize
#     # image_features = image_features.contiguous().view(B * T, -1) # (B x T, C) 
#     # neural_features = neural_features.contiguous().view(B * T, -1) # (B x T, C)

#     # normalize
#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#     neural_features = neural_features / neural_features.norm(dim=-1, keepdim=True)

#     # cosine similarity as logits
#     logits_per_image = temp * image_features @ neural_features.t()
#     logits_per_neurons = temp * neural_features @ image_features.t()

#     # (a)symmetric loss function
#     labels = torch.arange(B).to(device)
#     loss_i = F.cross_entropy(logits_per_image, labels)
#     loss_n = F.cross_entropy(logits_per_neurons, labels)
#     loss = (1/2 * loss_i) + (1/2 * loss_n) 

#     return loss


class CLIP(nn.Module):
    def __init__ (self, config, loss):
        super().__init__()
        self.temp = config.contrastive.clip_temp
        clip_embd = config.contrastive.clip_embd if hasattr(config, 'clip_embd') else config.n_embd
        self.proj = nn.Linear(config.n_embd, clip_embd, bias=False)
        self.loss = loss

    def forward(self, features):

        # frame_feats = rearrange(frame_feats, 'b hw e -> b (hw e)')
        # id_feats = rearrange(id_feats, 'b t e -> b (t e)')
        for mode in features:
            features[mode] = self.proj(features[mode])

        loss = contrastive_loss(features, temp=self.temp)

        return loss, features

class FeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.neural_state_blocks = _get_clones(Block(config, sparse_topk=config.sparse_topk_id), config.n_state_layers)
        if config.vit_encoder:
            self.frame_state_blocks = _get_clones(Block(config, sparse_topk=config.sparse_topk_frame), config.n_stimulus_layers)
        else:
            self.frame_state_blocks = None
        self.register_buffer("mask", self.build_mask(config.block_size.id))  
    
    def build_mask(self, block_size):
        mask = torch.tril(torch.ones((block_size, block_size)),
                                     ).view(1, 1, block_size, block_size)
        return mask
    
    def forward(self, neural, visual):
        for mod in self.neural_state_blocks:
            neural = mod(neural, neural, neural, self.mask)
        if self.frame_state_blocks != None:
            for mod in self.frame_state_blocks:
                visual = mod(visual, visual, visual)

            
        features = dict()
        features['id'] = neural
        features['frames'] = visual
        
        return features
        
class FeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.neural_state_blocks = _get_clones(Block(config, sparse_topk=config.sparse.topk_id), config.layers.state)
        if config.vit_encoder:
            self.frame_state_blocks = _get_clones(Block(config, sparse_topk=config.sparse.topk_frame), config.layers.stimulus)
        else:
            self.frame_state_blocks = None
        self.register_buffer("mask", self.build_mask(config.block_size.id))  
    
    def build_mask(self, block_size):
        mask = torch.tril(torch.ones((block_size, block_size)),
                                     ).view(1, 1, block_size, block_size)
        return mask
    
    def forward(self, neural, visual):
        for mod in self.neural_state_blocks:
            neural = mod(neural, neural, neural, self.mask)
        if self.frame_state_blocks is not None and visual.nelement() > 0:
            for mod in self.frame_state_blocks:
                visual = mod(visual, visual, visual)
            
        features = dict()
        features['id'] = neural
        features['frames'] = visual
        
        return features

class MultimodalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Get modalities from config
        modalities_config = object_to_dict(self.config.modalities) if get_attr(self.config, 'modalities') else {}

        # Initialize dictionary to hold modalities and their corresponding blocks
        self.modalities_blocks = nn.ModuleDict()

        # Iterate over all modalities in config
        for modality_group, modality_config in modalities_config.items():
            self.modalities_blocks[modality_group] = nn.ModuleDict()
            
            # get the number of layers for this modality_group
            n_layers = modality_config.get('n_layers', 1)  # default to 1 if not specified
            
            for modality_group in modalities_config.keys():
                # Initialize blocks for this variable
                self.modalities_blocks[modality_group] = _get_clones(Block(self.config), n_layers)

        # initialize the standard (id) and special (frame) blocks
        self.neural_state_blocks = _get_clones(Block(config, 
                                                     sparse_topk=get_attr(config.sparse, 'topk_id')), 
                                                     config.layers.state)
        self.neural_state_history_blocks = _get_clones(Block(config,
                                                             sparse_topk=get_attr(config.sparse, 'topk_id')), 
                                                             config.layers.state_history)
        self.neural_state_stimulus_blocks = _get_clones(Block(config, 
                                                              sparse_topk=get_attr(config.sparse, 'topk_frame')), 
                                                              config.layers.stimulus)

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.epoch = 0

        # Assuming 'id' modality is always present
        self.register_buffer("mask", self.build_mask(config.block_size.id))

    
    def build_mask(self, block_size):
        mask = torch.tril(torch.ones((block_size, block_size)),
                                     ).view(1, 1, block_size, block_size)
        return mask
    
    def generate_sparse_mask(self, p, T):
        """
        Generate a sparse mask of length T, according to p.
        """
        assert p >= 0 and p <= 1, "p should be in [0, 1]"
        mask = torch.rand((1, T)) < 1 - p
        mask = mask.repeat(T, 1)
        
        mask[0, 0] = True  # don't mask 1st step
        # check if any step is fully masked and umask it
        idx_all_true = (False == torch.all(mask, dim=0)).nonzero()
        for step in idx_all_true:
            sampler = torch.distributions.Uniform(low=0, high=step.item()+1)
            idx_false = sampler.sample((1,1)).long()
            mask[step, idx_false] = True

        # mask = mask.repeat(T, 1)
        mask = torch.tril(mask.long()).view(1, 1, T, T)
        return mask.cuda() if self.mask.is_cuda else mask
    
    def forward(self, features):
        """
        Args:
            neural_state: [batch_size, seq_len_t, hidden_size]
            neural_history: [batch_size, seq_len_t, hidden_size]
            stimulus: [batch_size, seq_len_s, hidden_size]
        """
        self.epoch += 1/48
        if self.epoch == 10: print(self.epoch)
        if self.config.sparse.mask and self.epoch >= 10 and self.training:
            p = 0.4 / (1 + np.exp( -self.epoch / 10))
            mask = self.generate_sparse_mask(p, self.config.block_size.id)
        else:
            mask = self.mask

        neural_state = features['id']
        neural_history = features['id_prev']
        modalities = get_attr(features, 'modalities', [])
        
        x = neural_state
        # state history
        for mod in self.neural_state_history_blocks:
            x = mod(x, neural_history, neural_history)
        
        # stimulus
        for mod in self.neural_state_stimulus_blocks:
            if 'frames' in features:
                if features['frames'] != None:
                    x = mod(x, features['frames'], features['frames'])
        
        # Iterate across the rest of the modalities
        # and their corresponding blocks
        for modality_group, modality_blocks in self.modalities_blocks.items():
            # chek if the modality has actually been given
            # if not, it's okay, just skip the block
            if modality_group not in features.keys():
                continue
            # all features of this modality_group concatenated along block dimesion
            modality_group_feats = features[modality_group]['all']
            blocks = self.modalities_blocks[modality_group]
            # Process feature through each block in the modality
            for block in blocks:
                x = block(x, modality_group_feats, modality_group_feats)

        # masked self-attention
        for mod in self.neural_state_blocks:
            x = mod(x, x, x, mask)
        return x

class NF_GRU(nn.Module):
    def __init__(self, config, num_layers=3):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.grus = nn.ModuleList([nn.GRU(config.n_embd, config.n_embd, batch_first=True) for _ in range(num_layers)])
        logger.info(f"NF_GRU: no. params: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x, h_0=None):
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.config.n_embd).to(x.device)
        else:
            # No need to unsqueeze h_0 or concatenate it with a tensor of zeros.
            # Just ensure that it has the right shape (num_layers, batch_size, n_embd).
            h_0 = h_0.transpose(0, 1).contiguous()

        h_t = []
        for i, gru in enumerate(self.grus):
            x, h = gru(x, h_0[i].unsqueeze(0))
            h_t.append(h)
        h_t = torch.cat(h_t, dim=0)
        return x, h_t

class Tapered_GRU(nn.Module):
    def __init__(self, config, num_layers=3, taper_factor=0.5):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.taper_factor = taper_factor
        self.grus = nn.ModuleList()

        n_embd = config.n_embd
        for _ in range(num_layers):
            self.grus.append(nn.GRU(n_embd, int(n_embd * (1 - self.taper_factor)), batch_first=True))
            n_embd = int(n_embd * (1 - self.taper_factor))  # update for next layer

        logger.info(f"Tapered_GRU: no. params: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x, h_0=None):
        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.config.n_embd).to(x.device)
        else:
            h_0 = h_0.transpose(0, 1).contiguous()

        h_t = []
        for i, gru in enumerate(self.grus):
            x, h = gru(x, h_0[i].unsqueeze(0))
            h_t.append(h)
            if i != len(self.grus) - 1:  # don't apply on last layer
                x = F.interpolate(x, scale_factor=self.taper_factor, mode='nearest')
        h_t = torch.cat(h_t, dim=0)
        return x, h_t
    


class LayerNormGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        h_next = self.gru_cell(x, h)
        h_next = self.ln(h_next)
        return h_next

class ImprovedGRU(nn.Module):
    def __init__(self, config, num_layers=3):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.grus = nn.ModuleList(
            [nn.GRU(config.n_embd, config.n_embd, batch_first=True, bidirectional=True)
             for _ in range(num_layers)]
        )
        self.linear_layers = nn.ModuleList(
            [nn.Linear(config.n_embd * 2, config.n_embd) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(config.n_embd) for _ in range(num_layers)])
        logger.info(f"NF_GRU: no. params: {sum(p.numel() for p in self.parameters())}")

    def forward(self, x, h_0=None):
        if h_0 is None:
            h_0 = torch.zeros(2*self.num_layers, x.size(0), self.config.n_embd).to(x.device)
        else:
            h_0 = h_0.transpose(0, 1).contiguous()

        h_t = []
        for i, (gru, linear, layer_norm) in enumerate(zip(self.grus, self.linear_layers, self.layer_norms)):
            x, h = gru(x, h_0[2*i:2*(i+1)].contiguous())
            x = linear(x)
            x = layer_norm(x)
            if i != self.num_layers - 1:  # apply dropout to all but the last layer
                x = self.dropout(x)
            h_t.append(h)

        h_t = torch.cat(h_t, dim=0)

        if self.num_layers > 1:
            x = self.tanh(x)

        return x, h_t

    def _layer_forward(self, x, h, gru_cell, tapering_weights):
        outputs = []
        seq_len = x.size(1)
        for t in range(seq_len):
            h = gru_cell(x[:, t, :] * tapering_weights[t], h)
            outputs.append(h)
        return torch.stack(outputs, dim=1), h


class Neuroformer(nn.Module):
    """ The full Neuroformer Model
    
    """

    def __init__(self, config, tokenizer):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        self.config = config
        self.modalities_config = object_to_dict(self.config.modalities) if get_attr(self.config, 'modalities') else None
        self.tokenizer = tokenizer
        self.id_vocab_size = tokenizer.ID_vocab_size
        self.dt_vocab_size = tokenizer.dt_vocab_size

        # -- Input Embedding Stem -- #        self.n_embd = config.n_embd
        self.tok_emb = nn.Embedding(self.id_vocab_size, config.n_embd)
        if config.pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size.id, config.n_embd))
            self.pos_emb_prev = nn.Parameter(torch.zeros(1, config.block_size.prev_id, config.n_embd))
        if config.temp_emb:
            """ Choose between learned or sinusoidal temporal embeddings"""
            if hasattr(config, 'wave_emb') and config.wave_emb:
                self.temp_emb = TemporalEmbedding(config.n_embd, config.dropout.id)
                self.temp_emb_prev = TemporalEmbedding(config.n_embd, config.dropout.id)
            else:
                self.temp_emb = LearntTemporalEmbedding(config.n_embd)
                self.temp_emb_prev = LearntTemporalEmbedding(config.n_embd)
        
        # -- Modality Embeddings (if used as input) -- #
        self.modality_embeddings = nn.ModuleDict()
        if self.modalities_config is not None:
            for modality_group, modality_config in self.modalities_config.items():
                if modality_config.get('n_layers', 0) > 0:
                    for variable_name, variable_config in modality_config['variables'].items():
                            if modality_group not in self.modality_embeddings.keys():
                                self.modality_embeddings[modality_group] = nn.ModuleDict()
                            self.modality_embeddings[modality_group][variable_name] = nn.ModuleDict()
                            self.modality_embeddings[modality_group][variable_name]['mlp'] = ProjectNorm(1, config.n_embd)
                            self.modality_embeddings[modality_group][variable_name]['temp_emb'] = TemporalEmbedding(config.n_embd, config.dropout.id)
                    self.modality_embeddings[modality_group]['pos_emb'] = PositionalEncoding2D(config.n_embd)


        # -- Modality Projection Heads (if used as prediction) -- #
        self.modality_projection_heads = nn.ModuleDict()
        if self.modalities_config is not None:
            for modality_type, modality_config in self.modalities_config.items():
                for variable_name, variable_config in modality_config['variables'].items():
                    if variable_config.get('predict', False):
                        if modality_type not in self.modality_projection_heads.keys():
                            self.modality_projection_heads[modality_type] = nn.ModuleDict()
                        if variable_config.get('objective') == 'regression':
                            self.modality_projection_heads[modality_type][variable_name] = nn.Linear(config.n_embd, 1, bias=True)
                        if variable_config.get('objective') == 'classification':
                            self.modality_projection_heads[modality_type][variable_name] = nn.Linear(config.n_embd, variable_config.get('n_classes', 1), bias=False)
                        
        # TODO: add learnt frame embedding
        # frame embeddings
        # if config.conv_layer:
        #     self.n_frames = (20 // config.kernel_size[0])
        #     assert config.frame_block_size % self.n_frames == 0, "frameblocksize not divisible by n_frames"
        # self.pos_emb_frames = nn.Parameter(torch.zeros(1, config.frame_block_size // self.n_frames, config.n_embd))
        # frame_temp_emb = torch.tensor(list(itertools.chain(*[[n * 1] * (config.frame_block_size//self.n_frames) for n in range(self.n_frames)]))).unsqueeze(1)
        # self.register_buffer("frame_temp_emb_seq", frame_temp_emb)
        # self.temp_emb_frames = TemporalEmbedding(config.n_embd, config.pos_pdrop, position=self.frame_temp_emb_seq)
        self.frame_3d_emb = PositionalEncoding3D(config.n_embd)
        self.id_drop = nn.Dropout(config.dropout.id)
        self.im_drop = nn.Dropout(config.dropout.im)
        self.drop = nn.Dropout(config.dropout.embd)

        # -- Visual Backbone -- #
        if hasattr(config.layers, 'stimulus'): 
            if config.layers.stimulus > 0:
                if get_attr(config.frame_encoder, 'resnet_backbone', False) is False:
                    self.video_encoder = VideoEncoder(config.frame_encoder)
                else:
                    self.video_encoder = Resnet3DBackbone()
                    print("Using Resnet Backbone")

        # -- feature encoder -- # 
        self.feature_encoder = FeatureEncoder(config)

        # -- Multimodal Transformer -- #
        if config.contrastive.contrastive:
            self.contrastive_loss = contrastive_loss if not get_attr(config.contrastive, 'clip_loss', False) else clip_loss
            self.clip = CLIP(config, loss=self.contrastive_loss)
        self.neural_visual_transformer = MultimodalTransformer(config)
       
        # -- ID, dt, Logit Projections -- #
        print(config.n_embd, self.dt_vocab_size)
        self.head_id = nn.Linear(config.n_embd, len(self.tokenizer.stoi['ID'].keys()), bias=False)  
        self.head_dt = nn.Linear(config.n_embd, len(self.tokenizer.stoi['dt'].keys()), bias=False)      
        # if self.config.predict_behavior:
        #     self.head_behavior = nn.Linear(config.n_embd, config.n_behavior, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        
        if config.class_weights != None:
            for key in config.class_weights.keys():
                print(f"registering class weights for {key}")
                self.register_buffer(f"class_weights_{key}", config.class_weights[key])
        
        # -- GRU, MLP experiments -- #
        num_layers = 5  # define the number of layers you need
        if config.mlp_only:
            self.mlp = nn.Sequential(*[ProjectNorm(config.n_embd, config.n_embd) for _ in range(num_layers)])
            # print no. parameters:
            logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        elif config.gru_only:
            self.gru = NF_GRU(config)
        # elif config.gru_tapered:
            # self.gru = Tapered_GRU(config)
        elif get_attr(config, "gru2_only"):
            self.gru = ImprovedGRU(config)
            # print no. parameters:
            logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias != None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # if self.config.freeze_weights != None:
        #     # freeze_weights(self, self.config.freeze_weights)
        #     for name, param in self.named_parameters():
        #         if name in self.config.freeze_weights:
        #             print(f"NOT Freezing {name}")
        #             param.requires_grad = True
        #         else:
        #             param.requires_grad = False

    def configure_optimizers(self, train_config):
        """
        Separates parameters into those who will experience weight decay and those that will not
        """
        if train_config.decay_weights:
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                    else: no_decay.add(fpn)

            # special case the position embedding parameter in the root Neuroformer module as not decayed
            black_list_mods = ['pos_emb', 'temp_emb']
            for mods in black_list_mods:
                for name, param in self.named_parameters():
                    if mods in name:
                        print(f"not decaying: {name}")
                        no_decay.add(name)    # also pos_emb
            
            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}
            no_decay -= decay & no_decay
            inter_params = decay & no_decay
            union_params = decay | no_decay

            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
            # optimizer = torch.optim.SGD(optim_groups, lr=train_config.learning_rate, momentum=0.9)
            print(f"weight_decay: {train_config.weight_decay}")
        
        else:
            parameters = self.parameters()
            # optimizer = torch.optim.SGD(parameters, lr=train_config.learning_rate, momentum=0.9)
            optimizer = torch.optim.AdamW(parameters, lr=train_config.learning_rate)
        
        
        return optimizer

    def process_all_modalities(self, x):
        features = collections.defaultdict(lambda: collections.defaultdict())
        for modality_group, group_features in x['modalities'].items():
            for variable_name, variable in group_features.items():
                modality_emb = self.modality_embeddings[modality_group][variable_name]['mlp'](variable['value'])
                if modality_emb.dim() < 4:
                    modality_emb = modality_emb.unsqueeze(-2)
                modality_temp_emb = self.modality_embeddings[modality_group][variable_name]['temp_emb'](variable['dt'].float()) if self.config.temp_emb and 'dt' in variable else 0
                n_vars = variable['value'].size(1)
                modality_temp_emb = modality_temp_emb.repeat(1, n_vars, 1)
                modality_emb = rearrange(modality_emb, 'b t c e -> b (t c) e')
                modality_emb = self.id_drop(modality_emb + modality_temp_emb)
                features[modality_group][variable_name] = modality_emb
            features[modality_group]['all'] = torch.cat([features[modality_group][variable_name] for variable_name in features[modality_group]], dim=1).unsqueeze(-2)
            features[modality_group]['all'] = features[modality_group]['all'] + self.modality_embeddings[modality_group]['pos_emb'](features[modality_group]['all'])
            features[modality_group]['all'] = rearrange(features[modality_group]['all'], 'b t 1 e -> b t e')
        return features

    def process_features(self, x):
        # batch, block_size, feature
        p_idx = x['id_prev']
        idx = x['id']
        dtx = x['dt']
        dtx_prev = x['dt_prev']
        if hasattr(self, 'video_encoder'):
            frames = self.video_encoder(x['frames']) if 'frames' in x else None
        else:
            frames = None
        pad = x['pad']
        features = dict()

        # forward the Neuroformer model
        ''' 
        positional and temporal embeddings implemented in multiple ways, learnt, 
        fourrier decomposition and in the case of time, just passed as is. 
        '''
        # Embeddings
        prev_id_position_embeddings = self.pos_emb_prev if self.config.pos_emb else 0
        prev_id_temporal_embeddings = self.temp_emb_prev(dtx_prev.float()) if self.config.temp_emb else 0
        id_position_embeddings = self.pos_emb if self.config.pos_emb else 0
        id_temporal_embeddings = self.temp_emb(dtx.float()) if self.config.temp_emb else 0
        # frame_position_embeddings = self.pos_emb_frames.repeat(1, self.n_frames, 1)
        # frame_temporal_embeddings = self.temp_emb_frames(self.frame_temp_emb_seq)
        
        # Extract ID features
        prev_token_embeddings = self.id_drop(self.tok_emb(p_idx) + prev_id_temporal_embeddings + prev_id_position_embeddings)
        token_embeddings = self.id_drop(self.tok_emb(idx) + id_temporal_embeddings + id_position_embeddings) # each index maps to a (learnable) vector

        # Extract image features and add time embeddings
        if 'frames' in x and frames != None:
            frame_embeddings = frames    # self.tok_emb(frames)
            im_3d_embeddings = self.frame_3d_emb(frames)
            frame_embeddings = frames + im_3d_embeddings
            frame_embeddings = rearrange(frames, 'b t h w e -> b (t h w) e')
            frame_sos = token_embeddings[:, 0].unsqueeze(1)
            frame_embeddings = torch.cat([frame_sos, frame_embeddings], dim=1)
            # frame_embeddings = frame_embeddings + frame_position_embeddings + frame_temporal_embeddings
            frame_embeddings = self.im_drop(frame_embeddings)   # separate pos emb?
        else:
            frame_embeddings = None
        
        features = self.feature_encoder(token_embeddings, frame_embeddings)
        token_embeddings = features['id']
        frame_embeddings = features['frames'] if 'frames' != None else 0
        if self.modalities_config is not None:
            features = self.process_all_modalities(x)
        
        # Tidy up
        features['id_prev'] = prev_token_embeddings
        features['id'] = token_embeddings
        features['frames'] = frame_embeddings
        features['raw_frames'] = frames
        
        return features, pad

    def predict_var(self, x):
        x_mean = x.mean(dim=1)
        logits = self.head_behavior(x_mean)
        return logits

    def forward(self, x, targets=None):
        idx = x['id']
        pad = x['pad']
        B, t = idx.size()
        
        features, pad = self.process_features(x)
        if get_attr(self.config, 'mlp_only', False):
            x = features['id']
        elif get_attr(self.config, 'gru_only', False):
            x, _ = self.gru(features['id'], features['id_prev'])
        elif get_attr(self.config, 'gru2_only', False):
            x, _ = self.gru(features['id'], features['id_prev'])
        else:
            x = self.neural_visual_transformer(features)
        id_logits = self.head_id(x)
        dt_logits = self.head_dt(x)    # (B, T_id, 1)
        if get_attr(self.config, 'predict_behavior', False):
            behavior_logits = self.predict_var(x)

        # if targets, calculate loss
        # calculate loss on logits up to padding token for each batch
        loss = None
        preds = collections.defaultdict(dict)
        loss_frames = 0
        loss_id = []
        loss_time = []
        precision = []
        recall = []
        F1 = []
        probs_id = []
        logits_modalities = {}
        torch.cuda.empty_cache()
        if targets != None:
            loss = collections.defaultdict(float)
            n = float('inf') if not self.config.contrastive else 4

            
            ## separate ids and dts into the neural data and eos tokens
            eos_id_tok = self.tokenizer.stoi['ID']['EOS']
            eos_dt_tok = self.tokenizer.stoi['dt']['EOS']
            eos_id_logits, eos_dt_logits, \
            id_logits_no_eos, dt_logits_no_eos, \
            targets = separate_eos_tokens(eos_id_tok, eos_dt_tok, 
                                          targets, id_logits, 
                                          dt_logits)

            if self.config.class_weights != None:
                loss_id = F.cross_entropy(id_logits.view(-1, id_logits.size(-1)), targets['id'].view(-1),
                                          ignore_index=self.config.ignore_index_id, weight=self.class_weights_id)
                loss_time = F.cross_entropy(dt_logits.view(-1, dt_logits.size(-1)), targets['dt'].view(-1), 
                                          ignore_index=self.config.ignore_index_dt, weight=self.class_weights_dt)
            elif get_attr(self.config, 'ignore_index_pad', True):
                loss_id =  F.cross_entropy(id_logits.view(-1, id_logits.size(-1)), targets['id'].view(-1), 
                                           ignore_index=self.tokenizer.stoi['ID']['PAD']) 
                loss_time = F.cross_entropy(dt_logits.view(-1, dt_logits.size(-1)), targets['dt'].view(-1), 
                                            ignore_index=self.tokenizer.stoi['dt']['PAD'])
            else:
                loss_id = F.cross_entropy(id_logits.view(-1, id_logits.size(-1)), targets['id'].view(-1))
                loss_time = F.cross_entropy(dt_logits.view(-1, dt_logits.size(-1)), targets['dt'].view(-1))

            # TODO: fix the ambiguity with block_type / modality
            # iterate over modalities required for prediction
            x_mean = x.mean(dim=1)
            if 'modalities'in targets.keys():
                for modality_type, modality_config in self.modalities_config.items():
                    for variable_name, variable_config in modality_config['variables'].items():
                        if variable_config.get('predict', False) and variable_name in targets['modalities'][modality_type]:
                            modality_projection = self.modality_projection_heads[modality_type][variable_name](x_mean)
                            modality_target = targets['modalities'][modality_type][variable_name]['value']
                            if variable_config['objective'] == 'regression':
                                loss_modality = F.mse_loss(modality_projection, modality_target.view(B, -1))
                            elif variable_config['objective'] == 'classification':
                                loss_modality = F.cross_entropy(modality_projection.view(-1, modality_projection.size(-1)), modality_target.view(-1))
                            loss[f'{variable_name}'] = loss_modality
                            preds[f'{variable_name}'] = modality_projection
                            
            
            if self.config.contrastive.contrastive:
                clip_id_feats = []
                for B, P in enumerate(pad):
                    clip_id_feats.append(features['id'][B, t - P])
                clip_id_feats = torch.stack(clip_id_feats)
                # n = 2
                # loss['clip'] = self.clip(features['frames'][:, 0], features['id'][:, -1]) * (1 / n) 
                # loss['clip'] = self.clip(features['frames'][:, 0], clip_id_feats) * (1 / n)
                feats_clip = dict()
                # feed in x (last layer representation)
                feat_contra_id = x.mean(dim=1) # features['id'].mean(dim=1)
                if hasattr(self.config.contrastive, 'vars'):
                    for variable in self.config.contrastive.vars:
                        # mean pool all sequence
                        if isinstance(variable, dict):
                            for key, value in variable.items():
                                feat = features[key][value].mean(dim=1)
                                feats_clip[key] = feat
                        else:
                            if variable == 'id':
                                feats_clip['id'] = feat_contra_id
                            # use features from eos pos
                            elif variable == 'eos':
                                feats_clip['eos'] = clip_id_feats
                            elif variable == 'frames':
                                feats_clip['frames'] = features['frames'].mean(dim=1)
                            else:
                                feats_clip[variable] = features[variable]
                # if clip vars unspecified, use id and frames
                else:
                    feats_clip['frames'] = features['frames'].mean(dim=1)
                    feats_clip['id'] = feat_contra_id
                assert len(feats_clip.keys()) >= 2, "Need at least 2 variables for contrastive loss"
                loss['clip'], features['clip'] = self.clip(feats_clip)
                loss['clip'] = loss['clip'] * (1 / n)
            
            loss['id'] = ((3 / 5) * loss_id) * (1 - 1 / n)   # sum(loss_id) / (b * 2)   # / len(loss_id)
            loss['time'] = ((2 / 5) * loss_time) * (1 - 1 / n)

            t = targets['id'].size(1)
            for B, P in enumerate(pad):                
                id_targets = targets['id'][B, :t - P - 1] # don't include EOS.
                id_logits_ = id_logits[B, :t - P - 1]         
                if len(id_targets) > 0:
                    ## score metrics
                    probs_neurons = F.softmax(id_logits_, dim=-1)
                    _, ix_top_k = torch.topk(probs_neurons, k=1, dim=-1)
                    pred_neurons = ix_top_k.detach().flatten()
                    true_neurons = id_targets.detach().flatten()
                    precision_score = torchmetrics.functional.precision(true_neurons, pred_neurons, task='multiclass', num_classes=self.id_vocab_size).to(self.device)
                    recall_score = torchmetrics.functional.recall(true_neurons, pred_neurons, task='multiclass', num_classes=self.id_vocab_size).to(self.device)
                    F1_score = torchmetrics.functional.f1_score(true_neurons, pred_neurons, task='multiclass', num_classes=self.id_vocab_size).to(self.device)

                    probs_id.append(probs_neurons)
                    if (precision, recall, F1) != None:
                        precision.append(precision_score)
                        recall.append(recall_score)
                        F1.append(F1_score)
                    else:
                        zero_tensor = torch.zeros(1).to(self.device)
                        precision.append(zero_tensor)
                        recall.append(zero_tensor)
                        F1.append(zero_tensor)
            if len(probs_id) > 0:            
                preds['probs_id'] = torch.cat(probs_id)
            else:
                preds['probs_id'] = torch.zeros(1).to(self.device)
        else:
            zero_tensor = torch.zeros(1).to(self.device)
            precision.append(zero_tensor)
            recall.append(zero_tensor)
            F1.append(zero_tensor) 
            
        # check if precision, recall and f1 are all same shape
        if len(precision) > 0 and len(recall) > 0 and len(F1) > 0:
            preds['precision'] = torch.stack(precision).mean()
            preds['recall'] = torch.stack(recall).mean()
            preds['F1'] = torch.stack(F1).mean()
        else:
            preds['precision'] = torch.zeros(1).to(self.device)
            preds['recall'] = torch.zeros(1).to(self.device)
            preds['F1'] = torch.zeros(1).to(self.device)

        preds['id'] = id_logits    # [:, tf:]    # only id logits
        preds['dt'] = dt_logits
        features['last_layer'] = x

        return preds, features, loss