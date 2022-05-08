""" 
    Function:   Define some evaluation metrics
	Author:     Bing Yang
    Copyright Bing Yang
"""

import os
import torch
import scipy.io
import numpy as np
from common.config import dirconfig
from pesq import pesq
from pesq import PesqError
from pypesq import pesq as nb_pesq
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources

def doa_estimate(w, rtf, rtf_template, DOA_candidate, mode='all'): 
    """ Function: DOA estimation (from DP-RTF predictions) 
        Args:       w               - estimated time-frequency weight (nbatch, nf, nt)
                    rtf             - DP-RTF predictions (nbatch, 3nf, nt)
                    rtf_template    - DP-RTF template (nbatch, 3nf, nt, ndoa)
                    DOA_candidate   - DOA candidates corresponding to DP-RTF template
                    mode            - mode for similarity spectrum
                                     'all': one similarity spectrum for all TF bins
                                     'tf': one simlarity spectrum for each TF bin
        Returns:    doa             - DOA estimates (nbatch, 1)
                    sim             - similarity spectrum (nbatch, ndoa)
    """
    nbatch = rtf_template.shape[0]
    nf = w.shape[1]
    nt = rtf_template.shape[2]
    ndoa = rtf_template.shape[3]

    # sim_tf = torch.zeros(nbatch, nf, nt, ndoa, requires_grad=True).cuda()
    # sim_sum = torch.zeros(nbatch, ndoa, requires_grad=True).cuda()
    sim_tf = torch.zeros(nbatch, nf, nt, ndoa).cuda()
    sim_sum = torch.zeros(nbatch, ndoa).cuda()
    for doa_idx in range(0, ndoa, 1):
        if rtf.shape[1] == 128 * 3:
            pow_temp = torch.pow(rtf - rtf_template[:, :, :, doa_idx], 2)
            sim_tf[:, :, :, doa_idx] = (pow_temp[:, 0:128, :]+pow_temp[:, 128:128*2, :]+pow_temp[:, 128*2:128*3, :]) * w
            sim_sum[:, doa_idx] = torch.sum(torch.sum(sim_tf[:, :, :, doa_idx], dim=2), dim=1)/nt
        elif rtf.shape[1] == 128 * 2:
            pow_temp = torch.pow(rtf - rtf_template[:, 128:128*3, :, doa_idx], 2)
            sim_tf[:, :, :, doa_idx] = (pow_temp[:, 0:128, :] + pow_temp[:, 128:128 * 2, :]) * w
            sim_sum[:, doa_idx] = torch.sum(torch.sum(sim_tf[:, :, :, doa_idx], dim=2), dim=1) / nt
        elif rtf.shape[1] == 128:
            pow_temp = torch.pow(rtf - rtf_template[:, 0:128, :, doa_idx], 2)
            sim_tf[:, :, :, doa_idx] = pow_temp * w
            sim_sum[:, doa_idx] = torch.sum(torch.sum(sim_tf[:, :, :, doa_idx], dim=2), dim=1) / nt

    doa_idx_set = torch.argmin(sim_sum, dim=1) #.long()
    doa_idx_set = doa_idx_set[:, np.newaxis]
    doa = DOA_candidate[doa_idx_set]

    if mode == 'all':
        return doa, sim_sum
    elif mode == 'tf':
        return doa, sim_tf

def doa_evaluate(doa, doa_gt, threshold, mode, vad=None):  
    """ Function: Evaluation of DOA estimation in terms of MAE and accuracy
        Args:       doa         - DOA estimates (nbatch, nsources)
                    doa_gt      - DOA ground truths (nbatch, nsources)
                    threshold   - DOA error threshold for successful localization
        Returns:    mae         - MAE (1)
                    acc         - accuracy (1)
    """
    nbatch = doa.shape[0]

    if mode == 's':
        if vad == None:
            mae_set = torch.abs(doa-doa_gt).float()
            mae = torch.sum(mae_set)/nbatch
            acc_set = (mae_set<threshold)+0.0
            acc = torch.sum(acc_set.float())/nbatch
        else:
            nact = torch.sum(vad)
            mae_set = torch.abs(doa-doa_gt).float()
            mae = torch.sum(mae_set*vad)/nact
            acc_set = (mae_set<threshold) + 0.0
            acc = torch.sum(acc_set.float()*vad)/nact

        return mae, acc

    elif mode == 'two':
        ns = doa.shape[1]
        mae_set1 = (torch.abs(doa[:,0]-doa_gt[:,0]).float()+torch.abs(doa[:,1]-doa_gt[:,1]).float())/ns
        mae_set2 = (torch.abs(doa[:,1]-doa_gt[:,0]).float()+torch.abs(doa[:,0]-doa_gt[:,1]).float())/ns
        mae_set = torch.min(mae_set1,mae_set2)
        mae = torch.mean(mae_set, dim=0)
        acc_set = (mae_set<threshold) + 0.0
        acc = torch.sum(acc_set.float(), dim=0)/nbatch/ns

        return mae, acc

def run_estimate_evaluate(w_est, rtf_est, rtf_template=[], doa_gt_idx=[], doa_candidate = list(range(-90, 91, 5)),
                          mic_idx=[], mic_type = 'CIPIC', threshold=10):  
    """ Function: DOA estimation and evaluation (from DP-RTF predictions)
        Args:       w_est           - estimated time-frequency weight (nbatch, nf, nt)
                    rtf_est         - DP-RTF predictions (nbatch, 3nf, nt) 
                    rtf_template    - DP-RTF template (nbatch, 3nf, nt, ndoa)
                    doa_gt_idx      - the index of ground-truth DOA in candidates (nbatch, 1)
                    doa_candidate   - DOA candidates 
                    mic_idx         - the index of microphone, e.g., 21, 165 in CIPIC (nbatch, 1)
                    mic_type        - the type of microphone array
                    threshold       - DOA error threshold for successful localization
        Returns:    mae             - MAE (1)
                    acc             - accuracy (1)
    """

    doa_candidate = torch.tensor(list(doa_candidate)).cuda()
    nbatch, rtf_dim, nt = rtf_est.shape

    if rtf_template==[]:
        ger_dirs = dirconfig()
        rtf_dir = ger_dirs['rtf']
        if mic_type == 'MIT':
            path = os.path.join(rtf_dir, 'rtf_dp_Hkemar.mat')  # test507030_dist140
            rtf = scipy.io.loadmat(path)['rtf_dp']  # (3nf, ndoa)
            rtf = torch.from_numpy(rtf.astype('float32')).cuda()
            rtf_template = rtf[np.newaxis, :, np.newaxis, :].expand(nbatch, rtf_dim, nt, rtf.shape[1])
        elif mic_type == 'CIPIC':
            path = os.path.join(rtf_dir, 'rtf_dp_set_ori.mat')
            rtf_dp_set = scipy.io.loadmat(path)['rtf_dp']  # (3nf, ndoa, nhead)
            rtf_dp_set = torch.from_numpy(rtf_dp_set.astype('float32')).cuda()
            mic_idx_set = scipy.io.loadmat(path)['mic_idx']  # (1, nhead)
            mic_idx_set = torch.from_numpy(mic_idx_set.astype('float32')).cuda()
            mic_idx = mic_idx.expand(nbatch, mic_idx_set.shape[1])
            mic_idx_set = mic_idx_set.expand(nbatch, mic_idx_set.shape[1])
            eq_result = mic_idx.eq(mic_idx_set).float()
            rtf = torch.matmul(rtf_dp_set, eq_result.permute([1, 0])).permute([2, 0, 1]) # (nbatch, 3nf, ndoa)
            rtf_template = rtf[:, :, np.newaxis, :].repeat(1, 1, nt,1)

    doa_gt_set = doa_candidate[doa_gt_idx.long()]
    doa_est_set, _ = doa_estimate(w=w_est, rtf=rtf_est, rtf_template=rtf_template, DOA_candidate=doa_candidate)
    mae, acc = doa_evaluate(doa=doa_est_set, doa_gt=doa_gt_set, threshold=threshold, mode='s')

    return mae, acc

def run_estimate_evaluate_onehot(onehot, doa_gt_idx=[], doa_candidate = list(range(-90, 91, 5)), threshold=10, vad = None):
    """ Function: DOA estimation and evaluation (from onehot predictions)
        Args:       onehot          - the onehot vector that indicates the source presence probability (nbatch, ndoa) 
                    doa_gt_idx      - the index of ground-truth DOA in candidates (nbatch, 1)
                    doa_candidate   - DOA candidates 
                    threshold       - DOA error threshold for successful localization
        Returns:    mae             - MAE (1)
                    acc             - accuracy (1)
    """

    doa_candidate = torch.tensor(doa_candidate).cuda()

    doa_idx_set = torch.argmax(onehot, dim=1)  # .long()
    doa_idx_set = doa_idx_set[:, np.newaxis]
    doa_est_set = doa_candidate[doa_idx_set]*1
    doa_gt_set = doa_candidate[doa_gt_idx.long()]*1

    mae, acc = doa_evaluate(doa=doa_est_set, doa_gt=doa_gt_set, threshold=threshold, mode='s', vad = vad)

    return mae, acc

""" Function: Evaluation metrics for speech enhancement 
    References: 
        https://github.com/ludlows/python-pesq
        https://github.com/dakenan1/Speech-measure-SDR-SAR-STOI-PESQ
        https://github.com/mpariente/pystoi
        bss_eval_sources http://bass-db.gforge.inria.fr/bss_eval/ https://hal.inria.fr/inria-00564760/PDF/PI-1706.pdf
"""

def STOI(ref, est, sr=16000):
    return stoi(ref, est, sr, extended=False) # return 1e-5 if not valid

def WB_PESQ(ref, est, sr=16000):
    return pesq(sr, ref, est, "wb", on_error= PesqError.RETURN_VALUES) # return -1 if not valid

def NB_PESQ(ref, est, sr=16000):
    return nb_pesq(ref, est, sr)

def SDR_MY(reference, estimation, sr=16000, EPS=1e-8):

    estimation, reference = np.broadcast_arrays(estimation, reference)
    distortion = estimation - reference

    ratio = np.sum(reference ** 2, axis=-1) / (np.sum(distortion ** 2, axis=-1) + EPS)
    sdr = 10 * np.log10(ratio + EPS)

    return sdr

def SI_SDR(reference, estimation, sr=16000, EPS=1e-8):
    """ Function: Compute scale-invariant signal-to-distortion ratio (SI-SDR)
        Args:   reference   - numpy.ndarray, [..., T]
                estimation  - numpy.ndarray, [..., T]
        Returns: SI-SDR
        References: SDR– Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """

    estimation, reference = np.broadcast_arrays(estimation, reference)

    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy
    reference_scaled = optimal_scaling * reference
    distortion = estimation - reference_scaled

    ratio = np.sum(reference_scaled ** 2, axis=-1) / (np.sum(distortion ** 2, axis=-1) + EPS)
    si_sdr = 10 * np.log10(ratio + EPS)

    return si_sdr

def SD_SDR(reference, estimation, sr=16000, EPS=1e-8): 
    """ Function: Compute scale-dependent signal-to-distortion ratio (SI-SDR)
        Args:   reference   - numpy.ndarray, [..., T]
                estimation  - numpy.ndarray, [..., T]
        Returns: SD-SDR
        References： SDR– Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """

    estimation, reference = np.broadcast_arrays(estimation, reference)

    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy
    reference_scaled = optimal_scaling * reference

    distortion = estimation - reference

    ratio = np.sum(reference_scaled ** 2, axis=-1) / (np.sum(distortion ** 2, axis=-1) + EPS)
    sd_sdr = 10 * np.log10(ratio + EPS)

    return sd_sdr

def SDR(reference, estimation, sr=16000):
    sdr, _, _, _ = bss_eval_sources(reference[None, :], estimation[None, :])
    return sdr


if __name__ == '__main__':

    onehot = torch.zeros(2, 37).cuda()
    onehot[0, 0] = 1
    onehot[1, 1] = 1
    doa_gt_idx = torch.ones(2, 1).cuda()
    mae, acc = run_estimate_evaluate_onehot(onehot, doa_gt_idx=doa_gt_idx, threshold=10)
    print(mae, acc)