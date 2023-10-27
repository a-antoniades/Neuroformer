import math
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchmetrics


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

def separate_eos_tokens(eos_id_tok, eos_dt_tok, targets, id_logits, dt_logits):
    """
    Separates the logits and targets of the End-Of-Sequence (EOS) tokens 
    from the non-EOS tokens.

    Parameters:
    - eos_id_tok (int): The EOS token index for 'id' targets.
    - eos_dt_tok (int): The EOS token index for 'dt' targets.
    - targets (dict): A dictionary containing the targets 'id' and 'dt'. Both have shape (batch_size, seq_length).
    - id_logits (torch.Tensor): The logits for the 'id' targets. Shape (batch_size, seq_length, num_classes).
    - dt_logits (torch.Tensor): The logits for the 'dt' targets. Shape (batch_size, seq_length, num_classes).

    Returns:
    - eos_id_logits (torch.Tensor): The logits for the EOS tokens in 'id'. Shape (num_eos, num_classes).
    - eos_dt_logits (torch.Tensor): The logits for the EOS tokens in 'dt'. Shape (num_eos, num_classes).
    - id_logits_no_eos (torch.Tensor): The logits for the non-EOS tokens in 'id'. Shape (num_non_eos, num_classes).
    - dt_logits_no_eos (torch.Tensor): The logits for the non-EOS tokens in 'dt'. Shape (num_non_eos, num_classes).
    - targets (dict): The updated targets dictionary with keys 'id_', 'dt_', 'id_eos', and 'dt_eos'.
    """
    # Create masks for EOS tokens
    eos_id_mask = targets['id'] == eos_id_tok
    eos_dt_mask = targets['dt'] == eos_dt_tok

    # Create masks for non-EOS tokens
    non_eos_id_mask = ~eos_id_mask
    non_eos_dt_mask = ~eos_dt_mask

    # Apply the mask to select elements
    eos_id_logits = torch.masked_select(id_logits, eos_id_mask.unsqueeze(-1)).view(-1, id_logits.size(2))
    eos_dt_logits = torch.masked_select(dt_logits, eos_dt_mask.unsqueeze(-1)).view(-1, dt_logits.size(2))

    id_logits_no_eos = torch.masked_select(id_logits, non_eos_id_mask.unsqueeze(-1)).view(-1, id_logits.size(2))
    dt_logits_no_eos = torch.masked_select(dt_logits, non_eos_dt_mask.unsqueeze(-1)).view(-1, dt_logits.size(2))

    # Update targets dictionary
    targets['id_'] = targets['id'][non_eos_id_mask]
    targets['dt_'] = targets['dt'][non_eos_dt_mask]
    targets['id_eos'] = targets['id'][eos_id_mask]
    targets['dt_eos'] = targets['dt'][eos_dt_mask]

    return eos_id_logits, eos_dt_logits, id_logits_no_eos, dt_logits_no_eos, targets
        

def contrastive_loss(features, temp=0.1):
    # Get the names and embeddings of all modalities
    modalities = list(features.keys())
    embeddings = [features[modality] for modality in modalities]

    batch_size = embeddings[0].size(0)
    total_loss = 0.0
    num_pairs = 0

    # Iterate over all pairs of modalities
    for i, j in combinations(range(len(modalities)), 2):
        emb_i, emb_j = embeddings[i], embeddings[j]
        
        #normalize
        emb_i = emb_i / emb_i.norm(dim=-1, keepdim=True)
        emb_j = emb_j / emb_j.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        sim_matrix = torch.matmul(emb_i, emb_j.t()) / temp

        # Calculate the diagonal elements (correct matches)
        pos_sim = torch.diag(sim_matrix)

        # Calculate the loss for the current pair
        pair_loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
        total_loss += pair_loss
        num_pairs += 1

    # Take the average loss over all pairs of modalities
    total_loss /= num_pairs
    return total_loss.mean()

def clip_loss(features, temp=0.1):
    # Get the names and embeddings of all modalities

    modalities = list(features.keys())
    embeddings = [features[modality] for modality in modalities]

    batch_size = embeddings[0].size(0)
    total_loss = 0.0
    num_pairs = 0

    # Iterate over all pairs of modalities
    for i, j in combinations(range(len(modalities)), 2):
        emb_i, emb_j = embeddings[i], embeddings[j]
        
        #normalize
        emb_i = emb_i / emb_i.norm(dim=-1, keepdim=True)
        emb_j = emb_j / emb_j.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        logits_per_i = emb_i @ emb_j.t() / temp
        logits_per_j = emb_j @ emb_i.t() / temp

        # (a)ssymetric loss function
        labels = torch.arange(batch_size, device=emb_i.device)
        loss_i = F.cross_entropy(logits_per_i, labels)
        loss_j = F.cross_entropy(logits_per_j, labels)
        loss = (1/2 * loss_i) + (1/2 * loss_j)

        total_loss += loss
        num_pairs += 1

    # Take the average loss over all pairs of modalities
    total_loss /= num_pairs
    return total_loss    

# def topk_metrics(logits, labels, k=3, ignore_index=-100, device=None):
#     device = logits.device if device is None else device

#     # Get top k predictions
#     topk_preds = logits.topk(k, dim=-1).indices

#     # Create a mask for ignoring padding tokens
#     mask = (labels != ignore_index)

#     # Calculate binary ground truth
#     binary_gt = torch.zeros_like(logits).scatter_(-1, labels.unsqueeze(-1), 1)
#     binary_preds = torch.zeros_like(logits).scatter_(-1, topk_preds, 1)

#     # Calculate precision, recall, and F1-score
#     precision = torchmetrics.Precision(average='macro', threshold=0.5, task='binary').to(device)
#     recall = torchmetrics.Recall(average='macro', threshold=0.5, task='binary').to(device)
#     f1_score = torchmetrics.F1Score(average='macro', threshold=0.5, task='binary').to(device)

#     topk_precision = precision(binary_preds[mask], binary_gt[mask])
#     topk_recall = recall(binary_preds[mask], binary_gt[mask])
#     topk_f1_score = f1_score(binary_preds[mask], binary_gt[mask])

#     return topk_precision, topk_recall, topk_f1_score

def topk_metrics(logits, targets, k, num_classes, ignore_index, device=None):
    device = logits.device if device is None else device
    probs = torch.softmax(logits, dim=-1)
    _, ix_top_k = torch.topk(probs, k=k, dim=-1)
    pred_neurons = ix_top_k.detach().flatten()
    true_neurons = targets.repeat_interleave(k).view(-1)  # Repeat true targets 'k' times and flatten
    
    precision_score = torchmetrics.functional.precision(true_neurons, pred_neurons, task='multiclass',
                                                        ignore_index=ignore_index, num_classes=num_classes).to(device)
    recall_score = torchmetrics.functional.recall(true_neurons, pred_neurons, task='multiclass',
                                                        ignore_index=ignore_index, num_classes=num_classes).to(device)
    F1_score = torchmetrics.functional.f1_score(true_neurons, pred_neurons, task='multiclass',
                                                        ignore_index=ignore_index, num_classes=num_classes).to(device)
    return precision_score, recall_score, F1_score


class PositionalEmbedding(nn.Module):
    """ Implement the PE function. """
    def __init__(self, n_embd, p_drop=0.1, max_len=1500):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2) *
                             -(math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print(f"divterm: {div_term.shape}, pe: {pe.shape}, position: {position.shape}")
        # print(f"posdiv: {(position * div_term).shape}")
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # print(f"x: {x.shape}, pe: {self.pe[:, :x.size(1)].shape}")
        x = Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x).to(x.device)
    

class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = (
            torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc.to(tensor.device)

    
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc.to(tensor.device)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class TemporalEmbedding(nn.Module):
    """ encoding temporal information using fourrier signals """
    def __init__(self, n_embd, p_drop, position=None, max_len=1200):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)

        # Compute the positional encodings once in log space.
        if position is None:
            pe = torch.zeros(max_len, n_embd)
            position = torch.arange(0, max_len).unsqueeze(1)
        else:
            pe = torch.zeros(max_len, n_embd)
        div_term = torch.exp(torch.arange(0, n_embd, 2) *
                             -(math.log(10000.0) / n_embd))
        # div_term = div_term.unsqueeze(0).repeat(batch_size, 1)
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('div_term', div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x.unsqueeze(-1)
        pe = self.pe.repeat(x.size(0), 1, 1)
        div_term = self.div_term
        pe = pe[:, :x.size(1), :]
        pe[:, :, 0::2] = torch.sin(x * div_term)
        pe[:, :, 1::2] = torch.cos(x * div_term)

        x = Variable(pe, requires_grad=False)
        return self.dropout(x).to(x.device)


class LearntTemporalEmbedding(nn.Module):
    """
    Project B x T x 1 time sequence to
            B x T x C
    """
    def __init__(self, n_embd, p_drop=0.2):
        super().__init__()
        self.temp_emb = nn.Sequential(
            nn.Linear(1, n_embd // 2),
            nn.GELU(),
            nn.Linear(n_embd // 2, n_embd),
            nn.Dropout(p_drop)
        )
    
    def forward(self, x):
        return self.temp_emb(x.unsqueeze(-1))


class ClassifierWrapper(nn.Module):
    def __init__(self, model, config, n_classes):
        super().__init__()
        self.model = model
        self.config = config
        # self.classifier = nn.Linear(model.config.n_embd, n_classes)
        self.freeze_model = config.freeze_model
        self.classifier = nn.Sequential(
            nn.Linear(model.config.n_embd, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, n_classes)
        )
    
    def configure_optimizers(self, config):
        optimizer = self.model.configure_optimizers(config)
        # freeze all self.model parameters
        if self.freeze_model:
            print('-- Freezing model parameters --')
            for param in self.model.parameters():
                param.requires_grad = False
        return optimizer

    def forward(self, x, targets=None):
        preds, features, loss = self.model(x, targets)
        ## previously i mean pooled the last layer
        # last_layer = features['last_layer'].mean(dim=1)
        ## but now i just use the SOS token
        last_layer = features['last_layer'][:, 0, :]
        x = self.classifier(last_layer)

        if targets is not None:
            labels = targets['labels'].flatten()
            cls_loss = F.cross_entropy(x, labels)
            probs = F.softmax(x, dim=1)
            _, cls_topk = probs.topk(1, dim=-1)
            cls_topk = cls_topk.detach().flatten()
            preds['precision'] = torchmetrics.functional.precision(labels, cls_topk, task='binary').to(self.model.device)
            preds['recall'] = torchmetrics.functional.recall(labels, cls_topk, task='binary').to(self.model.device)
            preds['f1'] = torchmetrics.functional.f1_score(labels, cls_topk, task='binary').to(self.model.device)

        preds['cls'] = x
        features['cls'] = last_layer
        loss['cls'] = cls_loss
        
        return preds, features, loss 


class MLP(nn.Module):
    def __init__(self, n_in, n_out, p_drop=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_in, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, n_in)
        )
    
    def forward(self, x):
        return self.mlp(x)

