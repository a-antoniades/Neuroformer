from re import L
import numpy as np
from pyrsistent import discard
from sympy import Q
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import math
from scipy.special import softmax
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from sklearn.preprocessing import normalize
from matplotlib.colors import LinearSegmentedColormap
from numpy import linalg as LA
from SpikeVidUtils import get_frame_idx
from utils import top_k_top_p_filtering
from einops import rearrange


from scipy import signal


def convolve_atts_3D(stim_atts):
    '''
    input: (ID, T, Y, X)
    '''
    sigma = 2.0     # width of kernel
    x = np.arange(-3,4,1)   # coordinate arrays -- make sure they contain 0!
    y = np.arange(-3,4,1)
    z = np.arange(-3,4,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))

    for n_id in range(stim_atts.shape[0]):
        stim_atts[n_id] = signal.convolve(stim_atts[n_id], kernel, mode="same")
    return stim_atts


def reshape_features(features, kernel_size, frame_window, dt_frames):
    """
    Reshapes features from (batch_size, time_steps * height * width, features)
    to (batch_size, time_steps, height, width, features).
    
    :param features: Dictionary containing 'raw_frames' and 'frames' numpy arrays
    :param frame_window: Frame window size
    :param dt_frames: Time step size between frames
    :return: Reshaped features array
    """
    
    # Extract dimensions from the input features
    n_frames = int( (frame_window / dt_frames) // kernel_size[0])
    H = int(features['raw_frames'].shape[2])
    W = int(features['raw_frames'].shape[3])
    feats = np.array(features['frames'][:, 1:].detach().cpu().numpy())
    
    # # Ensure that the batch size is 1
    # if feats.shape[0] != 1:
    #     raise ValueError("Batch size must be 1")
    
    # Ensure that the dimensions are consistent
    if n_frames * H * W != feats.shape[1]:
        raise ValueError("The dimensions of the input features are inconsistent")

    # expand first dimension (t)
    feats = np.expand_dims(feats, axis=1)
    
    # Reshape the features using rearrange
    return rearrange(feats, 'b n (t h w) d -> b (t n) h w d', t=n_frames, h=H, w=W)


def rollout_attentions(att):
    ''' Rollout attentions
    Input: (L, H, ID, F)
    '''
    rollout_att = torch.eye(att.shape[-2], att.shape[-1])
    for i in range(att.shape[0]):
        if i==0:
            continue
        I = torch.eye(att.shape[-2], att.shape[-1])
        a = att[i]
        a = a.max(axis=0)[0]
        a = (a + 1.0*I) / 2
        a = a / a.sum(axis=-1, keepdims=True)
        if a.shape[1] == rollout_att.shape[0]:
            rollout_att = a @ rollout_att
        else:
            rollout_att = a * rollout_att
    return rollout_att


def grad_rollout(attentions, gradients, discard_ratio=0.8, idx=None, n_layers=0):
    result = None
    # attentions = [rollout_attentions(torch.cat(attentions))]
    # if len(attentions) > 1:
    #         attentions = [torch.cat(attentions).sum(0)[None, ...]]
    n_layers = len(attentions) if n_layers is None else n_layers
    with torch.no_grad():
        for i, (attention, grad) in enumerate(zip(attentions, gradients)):
            if i <= n_layers:
                continue
            # attention = attention if idx is None else attention[:, :, idx]
            # grad = grad if idx is None else grad[:, :, idx] 
            attention_heads_fused = (grad*attention).mean(axis=1)
            # attention_heads_fused[attenti,on_heads_fused < discard_ratio] = 0

            # Drop the lowest attentions, but
            # don't drop the class token
            # flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            # _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            # #indices = indices[indices != 0]
            # flat[0, indices] = 0
            I = torch.eye(attention_heads_fused.size(-2), attention_heads_fused.size(-1))
            # a = (attention_heads_fused + 1.0*I)/2
            a = attention_heads_fused
            # a = a.clamp(min=0)
            # a = a[:, pos_index]
            if result == None:
                result = a
            else:   
                # print(result.shape, a.shape)
                result = result + a * result      

    # print(result.shape)
    # # Look at the total attention between the class token,
    # # and the image patches
    # mask = result[0, 0 ,pos_index]
    # # In case of 224x224 image, this brings us from 196 to 14
    # width = int(mask.size(-1)**0.5)
    # mask = mask.reshape(width, width).numpy()
    # mask = mask / np.max(mask)
    return result

def grad_att(attentions, gradients, discard_ratio=0.8):
    with torch.no_grad():
        # atts = attentions * gradients
        # return atts
        return attentions


def get_atts(name):
    def hook(model, input, output):
        attentions[name] = output
    return hook


def get_atts(model):
    attentions = {}

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_blocks):
        attentions[f'neural_state_block_{n}'] = mod.attn.att.detach().cpu()

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_history_blocks):
        attentions[f'neural_state_history_block_{n}'] = mod.attn.att.detach().cpu()

    for n, mod in enumerate(model.neural_visual_transformer.neural_state_stimulus_blocks):
        attentions[f'neural_stimulus_block_{n}'] = mod.attn.att.detach().cpu()
    
    return attentions

def accum_atts(att_dict, key=None, n_channels=1000):
    if key is None:
        att_keys = att_dict.keys()
    else:
        att_keys = [k for k in att_dict.keys() if key in k]
    atts = []
    for k in att_keys:
        att = att_dict[k]
        att = att.sum(-3).squeeze(0).detach().cpu()
        reshape_c = att.shape[-1] // n_channels
        assert att.shape[-1] % n_channels == 0, "Attention shape does not match stimulus shape"
        att = att.view(att.shape[0], reshape_c, att.shape[1] // reshape_c)
        att = att.sum(-2)
        atts.append(att)
    return torch.stack(atts)


def interpret(x, y, model, idx=None, n_layer=0):

    def get_attention(module, n_blocks, block_size, pad=0, rollout=False):
        # aggregate attention from n_Blocks
        atts = []
        T = block_size
        for n in range(n_blocks):
            att = module[n].attn.att
            # n_heads = att.size()[1]
            if pad != 0:
                att = att[:, :, T - pad, :,]
            atts.append(att)
        return atts
    
    model.zero_grad(set_to_none=True)
    mconf = model.config
    T_id = mconf.id_block_size - int(x['pad'])
    preds, _, loss = model(x, y)
    logits_id = preds['id']
    category_mask = torch.zeros(logits_id.size()).detach().cpu().numpy()
    y_id = x['id'].flatten()
    y_idx = y_id if idx == None else y_id[idx]
    category_mask[:, torch.arange(len(y_id)), y_idx] = 1
    category_mask = torch.from_numpy(category_mask).requires_grad_(True)
    loss = torch.sum(logits_id * category_mask)
    model.zero_grad()
       
    id_id_att = get_attention(model.neural_visual_transformer.neural_state_blocks, mconf.n_state_layers, mconf.id_block_size)
    id_vis_att = get_attention(model.neural_visual_transformer.neural_state_stimulus_blocks, mconf.n_stimulus_layers, mconf.id_block_size)
    # attentions = get_atts(model)
    # id_vis_att = accum_atts(attentions, key='neural_stimulus_block').mean(0)


    R_id = torch.eye(id_id_att[0].shape[-2], id_id_att[0].shape[-1])
    R_id = R_id[:T_id, :T_id]
    # for blk_att in id_id_att:
    #         grad = torch.autograd.grad(loss, blk_att, retain_graph=True)[0].detach()[:, :, :T_id, :T_id]
    #         blk_att = blk_att.detach()[:, :, :T_id, :T_id]
    #         blk_att = grad * blk_att
    #         blk_att = blk_att.clamp(min=0).mean(dim=1)
    #         R_id = R_id + torch.matmul(blk_att, R_id)
    #         del grad

    R_id_vis = torch.eye(id_vis_att[0].shape[-2], id_vis_att[0].shape[-1])[:T_id]
    # R_id_vis = None
    R_vis = torch.eye(id_vis_att[0].shape[-1], id_vis_att[0].shape[-1])
    for i, blk_att in enumerate(id_vis_att):
        if i != n_layer:
            continue
        grad = torch.autograd.grad(loss, blk_att, retain_graph=True)[0].detach()[:, :, :T_id]
        grad = grad.clamp(0)
        blk_att = blk_att.detach()[:, :, :T_id]
        blk_att = grad * blk_att
        blk_att = blk_att.mean(dim=1)
        # blk_att[blk_att < 0.75] = 0
        R_id_vis = blk_att if R_id_vis is None else R_id_vis + blk_att
        # print(R_id_vis.shape, torch.transpose(R_id[0], -1, -2).shape, blk_att[0].shape, R_vis.shape)
        # print((torch.transpose(R_id[0], -1, -2) @ blk_att[0]).shape)
        # R_id_vis = R_id_vis + torch.transpose(R_id[0], -1, -2) @ blk_att[0] @ R_vis
        # R_id_vis = R_id_vis + R_id @ blk_att @ R_vis
        del grad
    
    if idx is not None:
        R_id_vis = R_id_vis[..., idx, :,].mean(-2).unsqueeze(0)       # R_id_vis.mean(-2)
    
    else:
        R_id_vis = R_id_vis.unsqueeze(0)
    model.zero_grad(set_to_none=True)

    del loss
    del category_mask

    return R_id, R_id_vis


class VITAttentionGradRollout:
    """
    This class is an adaptation of Jacob Gildenblat's implementation: 
    https://github.com/jacobgil/vit-explain

    We calculate Attention Rollou (Abnar, Zuidema, 2020), 
    for stimuluts-state attention, and condition
    it on the gradient of a specific target neuron.

    This way we can get neuron-specific attentions.
    """

    def __init__(self, model, module, attn_layer_name='attn_drop', discard_ratio=0.5):
        self.model = model
        self.module = module
        self.discard_ratio = discard_ratio
        for name, module in self.module.named_modules():
            if attn_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_full_backward_hook(self.get_attention_gradient)
        
        self.attentions = []
        self.attention_gradients = []
    
    def grad_rollout(self, atts, grads, ds=0.1):
        cams = torch.cat(atts)
        grads = torch.cat(grads)

    
    def get_attention(self, module, input, output):
        # output = output if self.idx is None else output[:, :, self.idx]
        self.attentions.append(output.cpu())
        # print(output.shape)
        
    
    def get_attention_gradient(self, module, grad_input, grad_output):
        grad = grad_input[0]
        # grad = grad if self.idx is None else grad[:, :, self.idx]
        self.attention_gradients.append(grad_input[0].cpu())
        # print(grad_input[0].shape)

    def __call__(self, x, y, idx=None, n_layer=0):
        self.model.zero_grad()
        preds, _, loss = self.model(x, y)
        logits_id = preds['id']    # if self.idx==None else preds['id'][:, self.idx]
        # return preds['id']
        category_mask = torch.zeros(logits_id.size()).detach().cpu().numpy()
        y_id = y['id'].flatten()
        y_idx = y_id if idx==None else y_id[idx]
        # y_idx = self.idx 
        # category_mask[:, :, y_idx] = 1
        # category_mask = torch.from_numpy(category_mask).requires_grad_()
        category_mask = torch.tensor([y_idx])
        # loss = (logits_id*category_mask).sum()
        loss = loss['id'] if idx==None else  F.cross_entropy(logits_id[0, idx][None, ...], category_mask)
        loss.backward()
        
        T = x['id'].shape[1]
        cams = torch.cat(self.attentions[n_layer:])
        grads = torch.cat(self.attention_gradients[n_layer:]).clamp(0)
        # print(grads.shape)
        gradcam = (cams * grads)[:, :T - x['pad']]      # .mean(-2) #[..., idx, :,]
        gradcam = gradcam[..., idx, :,] if idx is not None else gradcam
        gradcam = gradcam.sum(dim=0).min(dim=0)[0].unsqueeze(0)
        # gradcam = self.attentions[n_layer:][:, ..., idx, :,].sum(dim=0).mean(dim=0).unsqueeze(0)
        
        # print(len(self.attention_gradients))
        return gradcam.detach()
        # return grad_att(torch.cat(self.attentions), torch.cat(self.attention_gradients))  # grad_rollout(self.attentions, self.attention_gradients, self.discard_ratio)


@torch.no_grad()
def get_attention(module, n_blocks, block_size, pad=0, rollout=False):
    # aggregate attention from n_Blocks
    atts = None
    T = block_size
    # TODO: get index of 166, get attentions up until that stage
    for n in range(n_blocks):
        att = module[n].attn.att
        # n_heads = att.size()[1]
        if pad != 0:
            att = att[:, :, T - pad, :,]
        att = att.detach().squeeze(0).to('cpu').numpy()
        atts = att[None, ...] if atts is None else np.concatenate((atts, att[None, ...]))
    return atts

from scipy.special import softmax

def get_att_neurons(model, module, loader, n_blocks, block_size, pad_key=None, curr_state=True, sum_=False):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    model.to(device)
    mconf = model.config
    model = model.eval()
    T = mconf.block_size
    attention_scores = np.zeros((mconf.id_vocab_size, mconf.id_vocab_size))
    pbar = tqdm(enumerate(loader), total=len(loader))
    for it, (x, y) in pbar:
        pad = x[pad_key] if pad_key is not None else 0
    # place data on the correct device
    for key, value in x.items():
        x[key] = x[key].to(device)
    for key, value in y.items():
        y[key] = y[key].to(device)
    # forward model to calculate attentions
    _, _, _ = model(x)
    # scores = np.array(np.zeros(len(neurons)))
    att = np.zeros((mconf.id_vocab_size, mconf.id_vocab_size))
    score = get_attention(module, n_blocks, T)
    score = rollout_attentions(torch.tensor(score)).numpy()
    real_ids = x['id'][..., :T - pad].flatten()  
    # score = (52, 52)
    t_seq = int(T - x['pad'])
    xid_prev = x['id_prev'][..., :T - x['pad_prev']].flatten().tolist()
    xid = x['id'][..., :t_seq].flatten().tolist()
    yid = y['id'][..., :t_seq].flatten().tolist()
    score = score[:t_seq, :T - x['pad_prev']]
    for step in range(t_seq):
        step_score = score[step] # / (t_seq - step + 1)
        yid_step = yid[step] # get the id we are predicting at this step
        att[yid_step][xid_prev] += step_score
    attention_scores += att
    # if sum_:
    #     attention_scores = attention_scores.sum(axis=0)
    return attention_scores

class AttentionVis:
    '''attention Visualizer'''
    
    # def getAttention(self, spikes, n_Blocks):
    #         spikes = spikes.unsqueeze(0)
    #         b, t = spikes.size()
    #         token_embeddings = self.model.tok_emb(spikes)
    #         position_embeddings = self.model.pos_emb(spikes)
    #         # position_embeddings = self.model.pos_emb(spikes)
    #         x = token_embeddings + position_embeddings

    #         # aggregate attention from n_Blocks
    #         atts = None
    #         for n in n_Blocks:
    #                 attBlock = self.model.blocks[n].attn
    #                 attBlock(x).detach().numpy()    # forward model
    #                 att = attBlock.att.detach().numpy()
    #                 att = att[:, 1, :, :,].squeeze(0)
    #                 atts = att if atts is None else np.add(atts, att)
        
    #         # normalize
    #         atts = atts/len(n_Blocks)
    #         return atts
    
    def visAttention(att):
        plt.matshow(att)
        att_range = att.max()
        cb = plt.colorbar()
        cb.ax.tick_params()
        plt.show()
    
    
    def grad_attentions(self, model, x, y, stoi, n_layer=0):
        grad_attentions = None
        for idx, id_ in enumerate(y['id'].flatten()):
            y_id = y['id'].flatten()
            T = len(y_id)
            y_id = y_id[: T - int(x['pad'])]
            # idx = np.arange(len(y_id))
            _, att = interpret(x, y, model, idx=idx, n_layer=n_layer)
            # grad_attentions = att[None, ...] if grad_attentions is None else torch.cat((grad_attentions, att[None, ...]))
            grad_attentions = att if grad_attentions is None else torch.cat((grad_attentions, att))
            model.zero_grad()
            if id_ >= stoi['SOS']:
                break
        return grad_attentions

    # def grad_attentions(self, model, x, y, stoi, n_layer=0):
    #         grad_attentions = None
    #         y_id = y['id'].flatten()
    #         T = len(y_id)
    #         y_id = y_id[: T - int(x['pad'])]
    #         # idx = np.arange(len(y_id))
    #         _, att = interpret(x, y, model, n_layer=n_layer)
    #         # grad_attentions = att[None, ...] if grad_attentions is None else torch.cat((grad_attentions, att[None, ...]))
    #         grad_attentions = att if grad_attentions is None else torch.cat((grad_attentions, att))
    #         grad_attentions = grad_attentions[0][:T - int(x['pad'])]
    #         model.zero_grad()
    #         return grad_attentions

    
    # @torch.no_grad()
    def att_interval_frames(self, model, module, loader, n_blocks, block_size, 
                rollout=False, pad_key=None, agg=False, stoi=None, max_it=None, n_layer=0):
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        model.to(device)
        mconf = model.config
        model = model.eval()
        T = block_size
        attention_scores = None
        len_loader = len(loader) if max_it is None else max_it
        pbar = tqdm(enumerate(loader), total=len_loader)
        if rollout: grad_rollout = VITAttentionGradRollout(model, module)
        for it, (x, y) in pbar:
            pad = x[pad_key] if pad_key is not None else 0
            # place data on the correct device
            for key, value in x.items():
                x[key] = x[key].to(device)
            for key, value in y.items():
                y[key] = y[key].to(device)
            # att = np.swapaxes(att, -1, -2)
            if rollout:
                # preds, features, loss, = model(x, y)
                # att = AttentionVis.get_attention(module, n_blocks, T)
                # att = self.rollout_attentions(att)
                # grad_rollout = VITAttentionGradRollout(model, module)
                # att = grad_rollout(x, y)[0]

                att = self.grad_attentions(model, x, y, stoi, n_layer=n_layer)
                if att == None:
                    continue


            if not rollout:
                with torch.no_grad():
                    preds, features, loss, = model(x, y)
                    # preds_id = F.softmax(preds['id'] / 0.8, dim=-1).squeeze(0)
                    # ix = torch.multinomial(preds_id, num_samples=1).flatten()
                    att = get_attention(module, n_blocks, T)
                    ## predict iteratively
                    # ix, att = self.predict_iteratively(model, mconf, x, stoi, top_k=0, top_p=0.5, temp=0.5, sample=True, pred_dt=False)
            with torch.no_grad():
                if agg: 
                    t_seq = int(T - x['pad'])
                    # att = att - att.mean(axis=-2, keepdims=True)
                    # att = att - att.mean(axis=(0, 1, 2), keepdims=True)
                    if not rollout:
                        att = np.max(att, axis=1)
                        att = np.mean(att, axis=0)
                        # att = np.sum(att, axis=0)
                        # att = np.max(att, axis=(0, 1))
                    score = np.zeros((mconf.id_vocab_size, mconf.frame_block_size))
                    # score = score.reshape(-1, 20, 8, 14).min(axis=1)
                    xid = x['id'].cpu().flatten().tolist()[:t_seq]
                    yid = y['id'].cpu().flatten().tolist()[:t_seq]
                    # score[ix] = att
                    score[xid] = att[:t_seq]
                    # score[t_seq:] == 0
                else:
                    score = att
                
                if attention_scores is None:
                    attention_scores = score[None, ...]
                else:
                    attention_scores = np.concatenate((attention_scores, score[None, ...]))
                
                if max_it is not None and it == max_it:
                    break

                # att_dict[int(y['id'][:, n])] = step
                # atts[tuple(x['interval'].cpu().numpy().flatten())] = att_dict
        return attention_scores
            # take attentions from last step
    @torch.no_grad()
    def att_models(model, module, loader, n_blocks, block_size, pad_key=None):
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        model.to(device)
        model = model.eval()
        mconf = model.config
        T = block_size
        attention_scores = np.zeros(mconf.id_vocab_size)
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (x, y) in pbar:
            pad = x[pad_key] if pad_key is not None else 0
            # place data on the correct device
            for key, value in x.items():
                x[key] = x[key].to(device)
            for key, value in y.items():
                y[key] = y[key].to(device)
            # forward model to calculate attentions
            _, _, _ = model(x)
            # scores = np.array(np.zeros(len(neurons)))
            att = np.zeros(len(mconf.id_vocab_size))
            score = get_attention(module, n_blocks, T, pad)
            score = np.sum(score, axis=0)   # sum over all heads 
            score = np.sum(score, axis=0)   # sum over all steps
            # take attentions from last step
            # if score.size >= 1: score = score[-1]
            # scores.append(score)
            real_ids = x['id'][..., :T - pad].flatten()
            for idx, code in enumerate(real_ids):
                """ 
                for each code in scores,
                add its score to the array
                """
                code = int(code.item())
                att[code] += score[idx]
            attention_scores = np.vstack((attention_scores, att))
        return attention_scores.sum(axis=0)

    def heatmap2d(self, arr: np.ndarray, ax=None, alpha=0.5, clim=None, blur=0):
        ncolors = 256
        color_array = plt.get_cmap('jet')(range(ncolors))

        # change alpha values
        n = 20
        color_array[:,-1] = [0.0] * n +  np.linspace(0.0,1.0,(ncolors - n)).tolist()

        # create a colormap object
        map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

        # register this new colormap with matplotlib
        if 'rainbow_alpha' not in plt.colormaps():
            plt.register_cmap(cmap=map_object)
        if blur > 0:
            arr = gaussian_filter(arr, blur)
        if ax:
            h = ax.imshow(arr, cmap='rainbow_alpha', alpha=alpha)
        else:
            h = plt.imshow(arr, cmap='rainbow_alpha', alpha=alpha)

        # plt.colorbar()
        # plt.colorbar(mappable=h)
        if clim is not None:
            h.set_clim(clim)
                          
    @torch.no_grad()
    def plot_stim_attention_step(self, dataset, n_embd, video_stack, attention_scores, ix_step=None):
        '''
        In: (S, ID, Frame)
        Out: Attention heatmaps of neurons (y) - frames (x): (S, ID, Frame)
        '''

        # ix_step = [1, 2, 3, 4]
        if ix_step is None:
            ix_step = np.random.choice(len(attention_scores), 1)
        

        for step in ix_step:
            interval_trials = dataset.t

            H, W = video_stack[0].shape[-2], video_stack[0].shape[-1]
            xy_res = int(n_embd ** (1/2))

            # # step, layer, head, row = sorted_att_std # layer, head, 
            # step = ix_step   # 5, 3   # layer, head

            interval_trials = dataset.t
            dataset_step = dataset[step]
            x, y = dataset_step[0], dataset_step[1]
            x_id = x['id'].flatten().tolist()
            x_pad = int(x['pad'].flatten())
            neuron_idx = x_id[: len(x_id) - x_pad]

            ncol = 10
            nrow = len(neuron_idx)
            fig, ax = plt.subplots(figsize=(60, 4 * nrow), nrows=nrow, ncols=ncol)


            # attention_scores[ix_step] /= attention_scores[ix_step].max()
            # att_max, att_min = attention_scores[ix_step].max(), attention_scores[ix_step].min()
            att_step = attention_scores[step]
            print(att_step.shape)
            # att_step = softmax(att_step, axis=0)   # softmax over IDs
            att_mean, att_std = att_step.mean(), att_step.std()
            att_min, att_max = att_step.max(), att_step.min()
            # attention_scores[ix_step] = attention_scores[ix_step] - att_mean / att_std
            for n, idx in enumerate(neuron_idx):
                top_n = n
                att = att_step[idx]
                att_min, att_max = att.min(), att.max()
                att_mean, att_std = att.mean(), att.std()
                # att = (att - att_mean) / att.std()
                # att = softmax(att, axis=-1)
                # att = att / att.max()
                att_im = att.reshape(1, 20, H // xy_res, W // xy_res)
                # att_im = att_im - att_im.mean(axis=1)
                # att_im = (att_im - att_mean) / att_std
                att_im = att_im[-1, :, :, :]
                
                t = interval_trials.iloc[ix_step]
                t_trial = t['Trial'].item()

                # print(n_stim, math.ceil(t['Interval'] * 20))
                frame_idx = get_frame_idx(t['Interval'], 1/20)
                frame_idx = frame_idx if frame_idx >= 20 else 20
                im_interval = x['frames'][0]
                # im_interval = video_stack[n_stim, frame_idx - 20: frame_idx]
                # att_grid =  softmax(att_top_std_im)
                # att_grid = np.repeat(att_im, xy_res, axis=-2)
                # att_grid = np.repeat(att_grid, xy_res, axis=-1)
                att_grid = F.interpolate(torch.as_tensor(att_im[None, ...]), size=(H, W), mode='bilinear', align_corners=True).numpy()[0]

                
                tdx_range = range(10, att_grid.shape[0])
                for tdx in tdx_range:
                    axis = ax[n][tdx - 10]
                    # print(att_grid[tdx, :, :].shape)
                    axis.imshow(im_interval[tdx], cmap='gray')
                    # clim = (att_trials_id[ix_step].min(), att_trials_id[ix_step].max())
                    std_n = 3
                    self.heatmap2d(att_grid[tdx, :, :], ax=axis, alpha=0.85)        # , clim=(att_mean + att_std * std_n, att_mean + att_std * std_n))
                    # axis.axis('off')
                    axis.set_title(str(tdx))
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if tdx == min(tdx_range):
                        axis.set_ylabel(f"ID {idx}", fontsize=40)
                    # fig.suptitle(f'Neuron {idx}', y=0.8)
            
            # fig.supylabel('Neurons', fontsize=nrow * 6)
            # fig.supxlabel('Frames (N)', fontsize=nrow * 6)
            fig.suptitle(f"Interval {int(t['Interval'])} ({ix_step}) Trial {int(t['Trial'])}", y=1.01, fontsize=80)
            # plt.title(f"Interval {int(t['Interval'])} ({n}) Trial {int(t['Trial'])}", y=2, fontsize=80)

            plt.tight_layout()
            # plt.savefig(f"SimNeu3D_Combo4, Interval {int(t['Interval'])} Trial {int(t['Trial'])}.png")
    
    @torch.no_grad()
    def predict_iteratively(self, model, mconf, x, stoi, temp, top_p, top_k, sample=True, pred_dt=True, device='cpu'):
        t = x['id'].shape[-1]
        pad = x['pad'] if 'pad' in x else 0
        x['id_full'] = x['id'][:, 0]
        x['id'] = x['id'][:, 0]
        x['dt_full'] = x['dt'][:, 0]
        x['dt'] = x['dt'][:, 0] if pred_dt else x['dt']
        T_id = mconf.id_block_size
        current_id_stoi = torch.empty(0, device=device)
        current_dt_stoi = torch.empty(0, device=device)
        att_total = None
        for i in range(T_id):
            t_pad = torch.tensor([stoi['PAD']] * (T_id - x['id_full'].shape[-1]), device=device)
            t_pad_dt = torch.tensor([0] * (T_id - x['dt_full'].shape[-1]), device=device)
            x['id'] = torch.cat((x['id_full'], t_pad)).unsqueeze(0).long()
            x['dt'] = torch.cat((x['dt_full'], t_pad_dt)).unsqueeze(0).long()

            logits, features, _ = model(x)
            logits['id'] = logits['id'][:, i] / temp
            if pred_dt:
                logits['dt'] = logits['dt'][:, i] / temp


            att_step = AttentionVis.get_attention(model.neural_visual_transformer.neural_state_stimulus, mconf.n_stimulus_layers, mconf.id_block_size)
            att_step = att_step[:, :, i]
            att_total = att_step[None, ...] if att_total is None else np.concatenate((att_total, att_step[None, ...]))
            # optionally crop probabilities to only the top k / p options
            if top_k or top_p != 0:
                logits['id'] = top_k_top_p_filtering(logits['id'], top_k=top_k, top_p=top_p)
                if pred_dt:
                    logits['dt'] = top_k_top_p_filtering(logits['dt'], top_k=top_k, top_p=top_p)

            # apply softmax to logits
            probs = F.softmax(logits['id'], dim=-1)
            if pred_dt:
                probs_dt = F.softmax(logits['dt'], dim=-1)
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
                if pred_dt:
                    ix_dt = torch.multinomial(probs_dt, num_samples=1)
                # ix = torch.poisson(torch.exp(logits), num_samples=1)
            else:
                # choose highest topk (1) sample
                _, ix = torch.topk(probs, k=1, dim=-1)
                if pred_dt:
                    _, ix_dt = torch.topk(probs_dt, k=1, dim=-1) 
            
            # if ix > stoi['PAD']:
            #     ix = torch.tensor([513])
            
            # convert ix_dt to dt and add to current time
            current_id_stoi = torch.cat((current_id_stoi, ix.flatten()))
            if pred_dt:
                current_dt_stoi = torch.cat((current_dt_stoi, ix_dt.flatten()))
            
            # append true and predicted in lists
            # get last unpadded token
            x['id_full'] = torch.cat((x['id_full'], ix.flatten()))
            if pred_dt:
                x['dt_full'] = torch.cat((x['dt_full'], ix_dt.flatten()))

            if ix == stoi['EOS']: # and dtx == 0.5:    # dtx >= window:   # ix == stoi['EOS']:
            # if len(current_id_stoi) == T_id - x['pad']:
                # if ix != stoi['EOS']:
                #     torch.cat((current_id_stoi, torch.tensor([stoi['EOS']])))
                # if dtx <= window:
                #     torch.cat((current_dt_stoi, torch.tensor([max(list(itos_dt.keys()))])))
                id_prev_stoi = current_id_stoi
                dt_prev_stoi = current_dt_stoi
                break
        return x['id_full'].flatten().tolist()[1:], att_total.transpose(1, 2, 0, 3)
    
    @torch.no_grad()
    def plot_stim_attention_step_realtime(self, model, mconf, dataset, n_embd, video_stack, ix_step=None, rollout=False):
        '''
        In: (S, ID, Frame)
        Out: Attention heatmaps of neurons (y) - frames (x): (S, ID, Frame)
        '''

        # ix_step = [1, 2, 3, 4]
        if ix_step is None:
            ix_step = np.random.choice(len(dataset), 1)

        dataset = dataset
        interval_trials = dataset.t


        H, W = video_stack.shape[-2], video_stack.shape[-1]
        xy_res = int(n_embd ** (1/2))

        # # step, layer, head, row = sorted_att_std # layer, head, 
        # step = ix_step   # 5, 3   # layer, head

        interval_trials = dataset.t
        data_step = dataset[ix_step]
        for key in data_step[0].keys():
            data_step[0][key] = data_step[0][key].unsqueeze(0)
        x = data_step[0]
        x_id = dataset[ix_step][0]['id'].flatten().tolist()
        x_pad = int(dataset[ix_step][0]['pad'].flatten())
        neuron_idx = x_id[: len(x_id) - x_pad]

        print(x.keys())
        
        # model.eval()
        # with torch.no_grad():
        #         preds, features, loss, = model(x)
        # preds_id = F.softmax(preds['id'] / 0.95, dim=-1).squeeze(0)
        # ix = torch.multinomial(preds_id, num_samples=1).flatten().tolist()

        ix, att_step = self.predict_iteratively(model, mconf, x, dataset.stoi, top_k=0, top_p=0.85, temp=0.85, sample=True, pred_dt=False)
        print(f"ix: {ix}, att_step: {att_step.shape}")
        # ix = torch.argmax(preds_id, dim=-1)
        neuron_idx = []
        neuron_idx = []
        for idx in ix:
            neuron_idx.append(idx)
            if idx >= dataset.stoi['EOS']:
                break
        
        no_frames = 6
        ncol = no_frames
        nrow = len(neuron_idx) if len(neuron_idx) > 1 else 2
        nrow = 5
        fig, ax = plt.subplots(figsize=(ncol * 6, 4 * nrow), nrows=nrow, ncols=ncol)

        # attention_scores[ix_step] /= attention_scores[ix_step].max()
        # att_max, att_min = attention_scores[ix_step].max(), attention_scores[ix_step].min()
        # att_step = AttentionVis.get_attention(model.neural_visual_transformer.neural_state_stimulus, mconf.n_stimulus_layers, mconf.id_block_size)
        att_step = att_step.max(axis=0).max(axis=0) if rollout is False else self.rollout_attentions(att_step)
        # att_step = softmax(att_step, axis=0)   # softmax over IDs
        att_mean, att_std = att_step.mean(), att_step.std()
        att_min, att_max = att_step.max(), att_step.min()
        # attention_scores[ix_step] = attention_scores[ix_step] - att_mean / att_std
        for n, idx in enumerate(neuron_idx):
            if n > 4: break
            top_n = n
            att = att_step[n]
            att_min, att_max = att.min(), att.max()
            att_mean, att_std = att.mean(), att.std()
            # att = (att - att_mean) / att.std()
            # att = softmax(att, axis=-1)
            # att = att / att.max()
            att_im = att.reshape(1, 20, H // xy_res, W // xy_res)
            # att_im = (att_im - att_mean) / att_std
            att_im = att_im[-1, :, :, :]
            
            t = interval_trials.iloc[ix_step]
            t_trial = t['Trial'].item()
            if video_stack.shape[0] == 1:
                n_stim = 0
            elif video_stack.shape[0] <= 4:
                if t['Trial'] <= 20: n_stim = 0
                elif t['Trial'] <= 40: n_stim = 1
                elif t['Trial'] <= 60: n_stim = 2
            elif video_stack.shape[0] <= 8:
                n_stim = int(t['Trial'] // 200) - 1

            # print(n_stim, math.ceil(t['Interval'] * 20))
            frame_idx = get_frame_idx(t['Interval'], 1/20)
            frame_idx = frame_idx if frame_idx >= 20 else 20
            frame_idx = frame_idx if frame_idx < video_stack.shape[1] else video_stack.shape[1]
            im_interval = video_stack[n_stim, frame_idx - 20: frame_idx]

            # att_grid =  softmax(att_top_std_im)
            # att_grid = np.repeat(att_im, xy_res, axis=-2)
            # att_grid = np.repeat(att_grid, xy_res, axis=-1)
            print(att_im.shape)
            att_grid = F.interpolate(torch.tensor(att_im[None, ...]), size=(H, W), mode='bilinear', align_corners=False).numpy()[0]

            
            tdx_range = range(10, 10 + no_frames)
            for tdx in tdx_range:
                axis = ax[n][tdx - 10]
                # print(att_grid[tdx, :, :].shape)
                axis.imshow(im_interval[tdx, 0], cmap='gray')
                # clim = (att_trials_id[ix_step].min(), att_trials_id[ix_step].max())
                std_n = 3
                self.heatmap2d(att_grid[tdx, :, :], ax=axis, alpha=0.7, blur=2)
                # axis.axis('off')
                axis.set_title(str(tdx))
                axis.set_xticks([])
                axis.set_yticks([])
                if tdx == min(tdx_range):
                    axis.set_ylabel(f"ID {idx}", fontsize=40)
                # fig.suptitle(f'Neuron {idx}', y=0.8)
        
        # fig.supylabel('Neurons', fontsize=nrow * 6)
        # fig.supxlabel('Frames (N)', fontsize=nrow * 6)
        fig.suptitle(f"Interval {int(t['Interval'])} Trial {int(t['Trial'])}", fontsize=40)
        plt.tight_layout()
        # plt.savefig(f"SimNeu3D_Combo4, Interval {int(t['Interval'])} Trial {int(t['Trial'])}.png")
    
    
    @torch.no_grad()
    def plot_stim_attention_time_agg(self, x, mconf, attention_scores, t_frame, h=4, w=7, ix_step=None):
        '''
        In: (I, ID, Time, Frame)
        Out: Attention heatmaps of neurons (y) - frames (x)
        '''

        # ix_step = [1, 2, 3, 4]
        if ix_step is None:
            ix_step = np.random.choice(len(attention_scores), 1)

        dataset = dataset
        interval_trials = dataset.t


        H, W = video_stack.shape[-2], video_stack.shape[-1]

        # step, layer, head, row = sorted_att_std # layer, head, 
        step = ix_step   # 5, 3   # layer, head

        x_id = dataset[int(ix_step)][0]['id'].flatten().tolist()
        x_pad = int(dataset[int(ix_step)][0]['pad'].flatten())
        neuron_idx = x_id[: len(x_id) - x_pad]

        ncol = t_frame
        nrow = len(neuron_idx)
        fig, ax = plt.subplots(figsize=(60, 4 * nrow), nrows=nrow, ncols=ncol)


        print(neuron_idx)
        for n, idx in enumerate(neuron_idx):
            top_n = n
            att_idx = ix_step, n  # att_idx_1[0], att_idx_1[1], att_idx_1[2], ix
            att = attention_scores[att_idx]
            att = att / att.max()
            att_im = att.reshape(1, attention_scores.shape[-2], h, w)
            att_im = att_im[-1, :, :, :]

            att_grid = np.repeat(att_im, (H // h), axis=-2)
            att_grid = np.repeat(att_grid, (H // h), axis=-1)
            im_interval = x['frames'][0][0]
            for tdx in range(att_grid.shape[0]):
                axis = ax[n][tdx]
                # print(att_grid[tdx, :, :].shape)
                axis.imshow(im_interval[tdx], cmap='gray')
                # clim = (att_trials_id[ix_step].min(), att_trials_id[ix_step].max())
                self.heatmap2d(att_grid[tdx, :, :], ax=axis, alpha=0.6, clim=None)
                axis.axis('off')
                axis.set_title(str(tdx))
                axis.set_ylabel(f"Neuron {idx}")
                # fig.suptitle(f'Neuron {idx}', y=0.8)

        fig.suptitle(f"Interval {int(t['Interval'])} Trial {int(t['Trial'])}", fontsize=30, y=0.9)
        # plt.savefig(f"SimNeu3D_Combo4, Interval {int(t['Interval'])} Trial {int(t['Trial'])}.png")

    
    def plot_stim_att_layer_head(self, x, mconf, attention_scores, t_frame=1, h=8, w=14, ix_step=None, save_path=None, layer_no=None):
        """
        In: (I, Layer, Head, ID, Frame)
        Out: Attention heatmaps for neurons
        """
        n_embd = mconf.n_embd
        attention_scores = attention_scores[..., 1:]

        # # ix_step = [1, 2, 3, 4]
        if ix_step is None:
            ix_step = np.random.choice(len(attention_scores), 1)


        ncol = mconf.n_head
        nrow = mconf.n_stimulus_layers if layer_no is None else 1

        H, W = x['frames'].shape[-2], x['frames'].shape[-1]

        # sorted_att_std = np.unravel_index(np.argsort(-att_trials_id_std.ravel()), att_trials_id_std.shape)
        # step, layer, head, row = sorted_att_std # layer, head, 
        # step = ix_step   # 5, 3   # layer, head
        images, attentions = [], []
        for step in ix_step:
            xy_res = int(n_embd ** (1/2))

            # # step, layer, head, row = sorted_att_std # layer, head, 
            # step = ix_step   # 5, 3   # layer, head

            x_id = x['id'].flatten().tolist()
            x_pad = int(x['pad'].flatten())
            neuron_idx = x_id[: len(x_id) - x_pad]

            fig, ax = plt.subplots(figsize=(100, 4 * nrow), nrows=nrow, ncols=ncol, squeeze=False)
            for n, idx in enumerate([ix_step]):
                print(idx)
                xid_n = np.random.choice(range(len(neuron_idx)), 1)
                # att_n = attention_scores[int(idx), :, :, int(xid_n)]
                att_n = attention_scores[:, :, int(xid_n)]
                n_plot = 0
                for layer in range(att_n.shape[0]):
                    if layer_no is not None:
                            if layer != layer_no:
                                continue
                    for head in range(att_n.shape[1]):
                        att_l_h = att_n[layer, head]
                        att_l_h = att_l_h / att_l_h.max()
                        att_im = att_l_h.reshape(1, t_frame, h, w)
                        att_im = att_im[-1, :, :, :]
                        
                        # print(n_stim, math.ceil(t['Interval'] * 20))
                        im_interval = x['frames'][0][0]
                        # att_grid =  softmax(att_top_std_im)
                        att_grid = F.interpolate(torch.as_tensor(att_im[None, ...]), size=(H, W), mode='bilinear', align_corners=True).numpy()[0]
                        axis = ax if nrow and ncol == 1 else ax[n_plot][head]
                        # plt.subplot(nrow, ncol, n + layer + head + 1)
                        f_idx = t_frame // 2
                        axis.imshow(im_interval[f_idx], cmap='gray')
                        self.heatmap2d(att_grid[f_idx, :, :], ax=axis, alpha=0.6, blur=0, clim=None)
                        axis.axis('off')
                        axis.set_title(f'Layer {layer}, Head {head}', fontsize=15)
                        images.append(im_interval[f_idx])
                        attentions.append(att_grid[f_idx, :, :])
                    n_plot += 1
                plt.suptitle(f"Interval {float(x['interval'])}, Neuron {neuron_idx[int(xid_n)]}", y=0.97, fontsize=30)
        if save_path is not None:
                plt.savefig(f"{save_path}/svg/SimNeu_att_layer_head_{neuron_idx[int(xid_n)]}_interval.svg")
                plt.savefig(f"{save_path}/png/SimNeu_att_layer_head_{neuron_idx[int(xid_n)]}_interval.png")
        return images, attentions
    
    def export_att_frames(self, model, module, mconf, loader, video_stack, xy_res, path):
        """
        Input: 
        Attentions Scores of shape (S, L, H, ID, F)
        (where S = Steps, L = Layers, H = Heads, ID = Neurons, F = Frames)
        Video Stack of shape (T_idx, 1, H, W)
        (where T_idx = Frame Idx, 1 = Channels, H = Height, W = Width)

        Ouput:
        Attention heatmaps overlayed on stimulus
        """
        n_blocks = mconf.n_stimulus_layers
        T = mconf.id_block_size

        H, W = video_stack.shape[-2], video_stack.shape[-1]
        counter = 0
        for it, (x, y) in enumerate(loader):
            # forward model to calculate attentions
            _, _, _ = model(x)
            # scores = np.array(np.zeros(len(neurons)))
            score = AttentionVis.get_attention(module, n_blocks, T)
            # att = self.rollout_attentions(score).sum(axis=0)
            att = score.mean(axis=0).sum(axis=0).sum(axis=0)
            # att = softmax(att, axis=-1)
            att = att.reshape(20, H // xy_res, W // xy_res)
            att_grid = np.repeat(att, (H // xy_res), axis=-2)
            att_grid = np.repeat(att_grid, (H // xy_res), axis=-1)
            att_grid = softmax(att_grid, axis=-1)
            t_trial = x['trial'].item()
            t_interval = math.ceil(x['interval'] * 20)
            video_interval = x['frames'][0][0, 5:15]
            if len(video_interval) < 10:
                continue
            for frame in range(len(att_grid[8:11])):
                plt.imshow(video_interval[frame], cmap='gray')
                self.heatmap2d(att_grid[frame], alpha=0.7, blur=2.5)
                plt.savefig(f"{path}/natstim{str(counter).zfill(5)}.png")
                plt.close()
                counter += 1




from scipy.special import softmax

def get_att_neurons(model, module, loader, n_blocks, block_size, pad_key=None, curr_state=True, sum_=False):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model = model.eval()
    T = block_size
    attention_scores = np.zeros((mconf.id_vocab_size, mconf.id_vocab_size))
    pbar = tqdm(enumerate(loader), total=len(loader))
    for it, (x, y) in pbar:
        pad = x[pad_key] if pad_key is not None else 0
        # place data on the correct device
        for key, value in x.items():
            x[key] = x[key].to(device)
        for key, value in y.items():
            y[key] = y[key].to(device)
        # forward model to calculate attentions
        _, _, _ = model(x)
        # scores = np.array(np.zeros(len(neurons)))
        att = np.zeros((mconf.id_vocab_size, mconf.id_vocab_size))
        score = get_attention(module, n_blocks, T)
        score = rollout_attentions(torch.tensor(score)).numpy()
        # score = np.sum(score, axis=0) / score.shape[0]   # sum over all layers
        # score = np.sum(score, axis=0) / score.shape[0]   # sum over all heads
        # score = softmax(score, axis=-1)
        # score = np.sum(score, axis=0)   # sum over all steps
        # take attentions from last step
        # if score.size >= 1: score = score[-1]
        # scores.append(score)
        real_ids = x['id'][..., :T - pad].flatten()
        # score: (52)
        # for idx in range(T - pad):
        #     """ 
        #     for each code in scores,
        #     add its score to the array
        #     """

        # # for current_state:
        # t_seq = T - pad
        # xid = x['id'][..., :T - pad].flatten().tolist()
        # yid = y['id'][..., :T - pad].flatten().tolist()
        # score = score[:T - pad, :T - pad]
        # for step in range(T - pad):
        #     step_score = score[step] / (T - pad - t_seq)
        #     yid_step = yid[step] # get the id we are predicting at this step
        #     att[yid_step][xid] += step_score
        # attention_scores += att
        
        # score = (52, 52)
        t_seq = int(T - x['pad'])
        xid_prev = x['id_prev'][..., :T - x['pad_prev']].flatten().tolist()
        xid = x['id'][..., :t_seq].flatten().tolist()
        yid = y['id'][..., :t_seq].flatten().tolist()
        score = score[:t_seq, :T - x['pad_prev']]
        for step in range(t_seq):
            step_score = score[step] # / (t_seq - step + 1)
            yid_step = yid[step] # get the id we are predicting at this step
            att[yid_step][xid_prev] += step_score
        attention_scores += att
        # if sum_:
        #     attention_scores = attention_scores.sum(axis=0)
    return attention_scores
            


"""

from attentionVis import AttentionVis

loader = DataLoader(test_dataset, shuffle=False, pin_memory=False,
                             batch_size=1, num_workers=1)
iterable = iter(loader)

state_inter_atts = get_att_neurons(model, model.neural_visual_transformer.neural_state_blocks,
                                   loader, model.config.n_state_layers, mconf.id_block_size, 'pad')
history_inter_atts = get_att_neurons(model, model.neural_visual_transformer.neural_state_history_blocks,
                          loader, model.config.n_state_history_layers, mconf.id_block_size, 'pad_prev') 

# print(state_inter_atts.shape)

"""