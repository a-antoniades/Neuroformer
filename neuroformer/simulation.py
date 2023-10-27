import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import all_device

@torch.no_grad()
def top_k_top_p_filtering(logits, top_k=0, top_p=0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits

@torch.no_grad()
def model_ensemble(models, x):
    """
    Ensemble of models
    """
    logits_total = dict()
    for model in models:
        model.eval()
        logits, _, _ = model(x)
        logits_total['id'] = logits['id'] + logits_total['id'] if 'id' in logits_total else logits['id']
        logits_total['dt'] = logits['dt'] + logits_total['dt'] if 'dt' in logits_total else logits['dt']
    return logits 

@torch.no_grad()
def generate_spikes(model, dataset, window, window_prev, tokenizer,
                    get_dt=False, sample=False, top_k=0, top_p=0, top_p_t=0, temp=1, temp_t=1, 
                    frame_end=0, gpu=False, pred_dt=True, true_past=False,
                    p_bar=False, plot_probs=False):    
    """
    Generates spike predictions using a trained model.

    Parameters:
    model (torch.nn.Module): The trained model to use for prediction.
    dataset (torch.utils.data.Dataset): The dataset to use for prediction.
    window (int): The size of the window for prediction.
    window_prev (int): The size of the previous window for prediction.
    tokenizer (Class Tokenizer): The tokenizer to use for encoding and decoding.
    get_dt (bool, optional): If True, the function will return the time difference. Default is False.
    sample (bool, optional): If True, sampling is used for prediction. If False, argmax is used. Default is False.
    top_k (int, optional): The number of top probabilities to consider for top-k filtering. If 0, no top-k filtering is applied. Default is 0.
    top_p (float, optional): The cumulative probability threshold for nucleus (top-p) filtering. If 0, no top-p filtering is applied. Default is 0.
    top_p_t (float, optional): The cumulative probability threshold for nucleus (top-p) filtering for time. If 0, no top-p filtering is applied. Default is 0.
    temp (float, optional): The temperature to use for softmax. Default is 1.
    temp_t (float, optional): The temperature to use for softmax for time. Default is 1.
    frame_end (int, optional): The end frame for prediction. Default is 0.
    gpu (bool, optional): If True, use GPU for prediction. If False, use CPU. Default is False.
    pred_dt (bool, optional): If True, predict the time difference. If False, do not predict the time difference. Default is True.
    true_past (bool, optional): If True, use true past for prediction. If False, use predicted past. Default is False.
    p_bar (bool, optional): If True, display a progress bar. If False, do not display a progress bar. Default is False.
    plot_probs (bool, optional): If True, plot the probabilities. If False, do not plot the probabilities. Default is False.

    Returns:
    dict: A dictionary containing the predicted IDs, time differences, trials, intervals, and true values.

    The function first prepares the model and data for prediction. It then iterates over the dataset, generating predictions for the specified modality. If the objective is 'classification', it applies top-k and/or top-p filtering if specified, and uses either sampling or argmax to generate predictions. If the objective is 'regression', it directly uses the logits for prediction. The function finally organizes the predictions, intervals, trials, and true values into a DataFrame and returns it.
    """
    
    def pad_x(x, length, pad_token):
        """
        pad x with pad_token to length
        """
        if torch.is_tensor(x):
            x = x.tolist()
            
        pad_n = length - len(x)
        if pad_n < 0:
            x = x[-(length + 1):]
        if pad_n > 0:
            x = x + [pad_token] * pad_n
        x = torch.tensor(x, dtype=torch.long, device=device)
        return x
    

    def aggregate_dt(dt):
        agg_dt = []
        for i in range(len(dt)):
            # curr_dt = agg_dt[i] if i > 0 else 0 
            prev_dt = agg_dt[i - 1] if i > 1 else 0
            # agg_dt.append(curr_dt + dt[i])
            if i==0:
                agg_dt.append(dt[i])
            elif dt[i] == 0:
                if prev_dt == 0:
                    agg_dt.append(0)
                elif prev_dt > 0:
                    agg_dt.append(prev_dt + 1)
            elif dt[i] == dt[i - 1]:
                agg_dt.append(agg_dt[i - 1])
            elif dt[i] < dt[i - 1]:
                tot = agg_dt[i - 1] + dt[i] + 1
                agg_dt.append(tot)
            elif dt[i] > dt[i - 1]:
                diff = agg_dt[i - 1] + (dt[i] - dt[i - 1])
                agg_dt.append(dt[i - 1] + diff)
            else:
                return ValueError 
            # assert agg_dt[i] >= 0, f"agg_dt[{i}] = {agg_dt[i]}, dt[{i}] = {dt[i]}, dt[{i - 1}] = {dt[i - 1]}"

        assert len(agg_dt) == len(dt)
        # assert max(agg_dt) <= max(list(itos_dt.keys()))
        return agg_dt

    
    def add_sos_eos(x, sos_token=None, eos_token=None, idx_excl=None):
        """
        add sos and eos tokens to x
        """
     
        if sos_token is not None:
            idx_excl = []
            x_clean = []
            for n, i in enumerate(x):
                if i not in (eos_token, sos_token):
                    x_clean.append(i)
                else:
                    idx_excl.append(n)
        else:
            x_clean = [i for n, i in enumerate(x) if n not in idx_excl]
            sos_token, eos_token = min(list(itos_dt.keys())), max(list(itos_dt.keys()))
        x = torch.tensor([sos_token] + x_clean + [eos_token], dtype=torch.long, device=device)
        return x, idx_excl
    
    def sort_ids(id, dt):
        """
        sort ids and dt by dt
        (on last axis)
        """
        dt, idx = torch.sort(dt, dim=-1, descending=True)
        id = torch.gather(id, dim=-1, index=idx)
        return id, dt

    device = 'cpu' if not gpu else torch.cuda.current_device() # torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    model = [model_n.to(device) for model_n in model] if isinstance(model, list) else model.to(device) 
    model = [model_n.eval() for model_n in model] if isinstance(model, list) else model.eval()
    stoi, itos, itos_dt = tokenizer.stoi['ID'], tokenizer.itos['ID'], tokenizer.itos['dt']
    tf = 0
    mconf = model[0].config if isinstance(model, list) else model.config
    T_id = mconf.block_size.id
    T_id_prev = mconf.block_size.prev_id
    
    context = [] # torch.tensor(0, device=device).unsqueeze(0)
    data = dict()
    data['true'] = []
    data['ID'] = context
    data['time'] = []
    data['dt'] = context
    data['Trial'] = context
    data['Interval'] = context

    loader = DataLoader(dataset, shuffle=False, pin_memory=False)
    pbar = tqdm(enumerate(loader), total=len(loader), disable=p_bar)
    for it, (x, y) in pbar:
        xid_prev_real = x['id_prev'].flatten()
        pad_real = x['pad_prev'].flatten() - 2

        xdt_prev_real = x['dt_prev'].flatten()
        # if it > 2:
        #     break
        # print(f"it = {it}, interval: {x['interval']}, window_prev: {window_prev}, window: {window}")

        x = all_device(x, device)
        y = all_device(y, device)
        
        # feed predicted IDs from buffer into past state
        # count how many steps 
        if true_past is False:
            if it > (window_prev / window):
                df = {k: v for k, v in data.items() if k in ('ID', 'dt', 'Trial', 'Interval', 'Time')}
                df = pd.DataFrame(df)

                # filter all instances of ['SOS', 'EOS' and 'PAD'] from all columns of pandas dataframe:
                df = df[(df != 'SOS').all(axis=1)].reset_index(drop=True)
                df = df[(df != 'EOS').all(axis=1)].reset_index(drop=True)
                df = df[(df != 'PAD').all(axis=1)].reset_index(drop=True)
                df['Time'] = df['dt'] + df['Interval'] - window

                # store results in dict
                prev_id_interval, current_id_interval = dataset.calc_intervals(x['interval'])
                x['id_prev'], x['dt_prev'], pad_prev = dataset.get_interval(prev_id_interval, float(x['trial']), T_id_prev, data=df)
                x['id_prev'] = torch.tensor(x['id_prev'][:-1], dtype=x['id'].dtype).unsqueeze(0).to(device)
                x['dt_prev'] = torch.tensor(x['dt_prev'][:-1], dtype=x['id'].dtype).unsqueeze(0).to(device)
                x['pad_prev'] = torch.tensor(pad_prev, dtype=x['id_prev'].dtype).unsqueeze(0).to(device)

                # sort ids according to dt
                x['id_prev'], x['dt_prev'] = sort_ids(x['id_prev'], x['dt_prev'])

                x_id_prev = x['id_prev'].flatten()
                x_dt_prev = x['dt_prev'].flatten()
                x_pad_prev = x['pad_prev'].flatten() - 2
                # print(f"real: {xid_prev_real[:len(xid_prev_real) - pad_real]}")
                # print(f"pred: {x_id_prev[:len(x_id_prev) - x_pad_prev]}")
                # print(f"real_dt: {xdt_prev_real[:len(xdt_prev_real) - pad_real]}")
                # print(f"pred_dt: {x_dt_prev[:len(x_dt_prev) - x_pad_prev]}")
                # print("---------------------------------")

            
        pad = x['pad'] if 'pad' in x else 0
        x['id_full'] = x['id'][:, 0]
        # x['id'] = x['id'][:, 0]
        x['dt_full'] = x['dt'][:, 0] if pred_dt else x['dt']
        # x['dt'] = x['dt'][:, 0] if pred_dt else x['dt']

        current_id_stoi = torch.empty(0, device=device)
        current_dt_stoi = torch.empty(0, device=device)
        for i in range(T_id - 1):   # 1st token is SOS (already there)
            t_pad = torch.tensor([stoi['PAD']] * (T_id - x['id_full'].shape[-1]), device=device)
            t_pad_dt = torch.tensor([0] * (T_id - x['dt_full'].shape[-1]), device=device)

            # forward model, if list of models, then ensemble
            if isinstance(model, list):
                logits = model_ensemble(model, x)
            else:
                logits, features, _ = model(x, y)
            
            logits = all_device(logits, device)
            features = all_device(features, device)
            
            logits['id'] = logits['id'][:, i]
            logits['dt'] = logits['dt'][:, i]
            # optionally crop probabilities to only the top k / p options
            if top_k or top_p != 0:
                logits['id'] = top_k_top_p_filtering(logits['id'], top_k=top_k, top_p=top_p)
                logits['dt'] = top_k_top_p_filtering(logits['dt'], top_k=top_k, top_p=top_p_t)
            
            logits['id'] = logits['id'] / temp
            logits['dt'] = logits['dt'] / temp_t

            # apply softmax to logits
            probs = F.softmax(logits['id'], dim=-1)
            probs_dt = F.softmax(logits['dt'], dim=-1)
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
                ix_dt = torch.multinomial(probs_dt, num_samples=1)
                # ix = torch.poisson(torch.exp(logits), num_samples=1)
            else:
                # choose highest topk (1) sample
                _, ix = torch.topk(probs, k=1, dim=-1)
                _, ix_dt = torch.topk(probs_dt, k=1, dim=-1)
            
            # print(f"Step {it}, i: {i} ix: {ix}, x_true: {y['id'][0, i]}")

            if plot_probs:
                probs_n = np.array(probs)[0]
                xaxis = np.arange(len(probs_n))
                topk=5
                topk_indices = np.argpartition(probs_n, -topk)[-topk:]
                topk_probs = probs_n[topk_indices]
                plt.figure()
                plt.title(f"ID t={i}, indices: {topk_indices}")
                plt.axvline(x=ix, color='b', linestyle='--')
                plt.bar(xaxis, probs_n)
                plt.show()
                print(x['id'])
                print(f"Step {it}, i: {i} ix: {ix}, x_true: {y['id'][0, i]}")
                print(f"topk_ix: {topk_indices}")
                print(f"topk_probs: {topk_probs}")

                # plot dt probs
                probs_dt_n = np.array(probs_dt)[0]
                xaxis = np.arange(len(probs_dt_n))
                topk=5
                topk_indices = np.argpartition(probs_dt_n, -topk)[-topk:]
                topk_probs = probs_dt_n[topk_indices]
                plt.figure()
                plt.title(f"DT t={i}, indices: {topk_indices}")
                plt.axvline(x=ix_dt, color='b', linestyle='--')
                plt.bar(xaxis, probs_dt_n)
                plt.show()
                print(x['dt'])
                print(f"Step {it}, i: {i} ix_dt: {ix_dt}, dt_true: {y['dt'][0, i]}")
                print(f"topk_ix_dt: {topk_indices} \
                        topk_dt_probs: {topk_probs}")
            
            # convert ix_dt to dt and add to current time
            # print(f"ix: {ix}, x_true: {y['id'][0, i]} ")
            current_id_stoi = torch.cat((current_id_stoi, ix.flatten()))
            current_dt_stoi = torch.cat((current_dt_stoi, ix_dt.flatten()))
            dtx_itos = [itos_dt[int(ix_dt.flatten())]]

            x['id'][:, i + 1] = ix.flatten()
            x['dt'][:, i + 1] = ix_dt.flatten() if pred_dt else x['dt']
           
            if ix.flatten() >= stoi['EOS']:  # ix >= stoi['EOS']:   # or i > T_id - int(x['pad']):   # or len(current_id_stoi) == T_id: # and dtx == 0.5:    # dtx >= window:   # ix == stoi['EOS']:
                # print(f"n_regres_block: {i}")
                break
                        
            try:
                ix_itos = torch.tensor(itos[int(ix.flatten())]).unsqueeze(0)
            except:
                TypeError(f"ix: {ix}, itos: {itos}")
            
            data['ID'] = data['ID'] + ix_itos.tolist()    # torch.cat((data['ID'], ix_itos))
            data['dt'] = data['dt'] + dtx_itos          # torch.cat((data['dt'], dtx_itos))
            data['Trial'] = data['Trial'] + x['trial'].tolist()            # torch.cat((data['Trial'], x['trial']))
            data['Interval'] = data['Interval'] + x['interval'].tolist() # torch.cat((data['Interval'], x['interval']))

            # x['id_full'] = torch.cat((x['id_full'], ix.flatten()))
            # x['dt_full'] = torch.cat((x['dt_full'], ix_dt.flatten())) if pred_dt else x['dt']

        dty_itos = [itos_dt[int(dt)] for dt in y['dt'][:, :T_id - pad].flatten()]
        data['time'] = data['time'] + dty_itos
        # data['true'] = torch.cat((data['true'], y['id'][:, :T_id - pad].flatten()))
        data['true'] = data['true'] + list(y['id'][:, :T_id - pad].flatten())
        pbar.set_description(f"len pred: {len(data['ID'])}, len true: {len(data['true'])}")

        
    return data


@torch.no_grad()
def decode_modality(model, dataset, modality, 
                     block_type='modalities', objective='classification', 
                     sample=False, top_k=0, top_p=0):
    
    """
    Generates predictions for a specified behavioral variable (modality) using a trained model.

    Parameters:
    model (torch.nn.Module): The trained model to use for prediction.
    dataset (torch.utils.data.Dataset): The dataset to use for prediction.
    modality (str): The behavioral variable to predict.
    block_type (str, optional): The type of block to use for prediction. Default is 'modalities'.
    objective (str, optional): The objective of the prediction, either 'classification' or 'regression'. Default is 'classification'.
    sample (bool, optional): If True, sampling is used for prediction. If False, argmax is used. Default is False.
    top_k (int, optional): The number of top probabilities to consider for top-k filtering. If 0, no top-k filtering is applied. Default is 0.
    top_p (float, optional): The cumulative probability threshold for nucleus (top-p) filtering. If 0, no top-p filtering is applied. Default is 0.

    Returns:
    pandas.DataFrame: A DataFrame containing the predicted values, intervals, trials, true values, and cumulative intervals for the specified modality.

    The function first prepares the model and data for prediction. It then iterates over the dataset, generating predictions for the specified modality. If the objective is 'classification', it applies top-k and/or top-p filtering if specified, and uses either sampling or argmax to generate predictions. If the objective is 'regression', it directly uses the logits for prediction. The function finally organizes the predictions, intervals, trials, and true values into a DataFrame and returns it.
    """
    
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    tokenizer = model.tokenizer
    block_type_modality = f'{block_type}_{modality}_value'
    
    model.eval()
    model.to(device)
    
    loader = DataLoader(dataset, batch_size=50, shuffle=False, pin_memory=False)
    pbar = tqdm(enumerate(loader), total=len(loader))
    
    modality_preds = []
    intervals = []
    trials = []
    modality_true = []
    for it, (x, y) in pbar:
        x = all_device(x, device)
        y = all_device(y, device)
        logits, features, loss = model(x, y)
        # take everything back to cpu
        logits = all_device(logits, 'cpu')
        features = all_device(features, 'cpu')
        loss = all_device(loss, 'cpu')
        
        if objective == 'classification':
            if top_k or top_p != 0:
                logits[modality] = top_k_top_p_filtering(logits[modality], top_k=top_k, top_p=top_p)
            probs = F.softmax(logits[modality], dim=-1)
            if sample:
                ix = torch.multinomial(probs, 1).flatten()
            else:
                ix = torch.argmax(probs, dim=-1).flatten()
        
            ix_itos = tokenizer.decode(ix, modality)
            y_modality = tokenizer.decode(y[block_type][modality]['value'], modality)
        elif objective == 'regression':
            ix = logits[modality].flatten()
            ix_itos = ix
            y_modality = y["modalities"][block_type][modality]['value'].flatten()
        
        interval = x['interval'].flatten().tolist()
        trial = x['trial'].flatten().tolist()
        modality_preds.extend([float(ix_t) for ix_t in ix_itos])
        intervals.extend(interval)
        trials.extend(trial)
        modality_true.extend([float(y_t) for y_t in y_modality])

    # make modality preds, intervals etc into dataframe=
    modality_preds = pd.DataFrame(modality_preds)
    modality_preds.columns = [block_type_modality]
    modality_preds['interval'] = intervals
    modality_preds['trial'] = trials
    modality_preds['true'] = modality_true

    # make cum interval
    modality_preds['cum_interval'] = modality_preds['interval'].copy()
    prev_trial = None
    for trial in modality_preds['trial'].unique():
        if prev_trial is None:
            prev_trial = trial
            continue
        else:
            max_interval = modality_preds[modality_preds['trial'] == prev_trial]['interval'].max()
            modality_preds.loc[modality_preds['trial'] >= trial, 'cum_interval'] += max_interval

    return modality_preds