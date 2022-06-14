""" 
    Function:   Define some basic operations
	Author:     Bing Yang
    Copyright Bing Yang
"""

import os
import scipy.signal
import numpy as np
import torch

def DoSTFT(signal, win_len, win_shift_ratio, nfft, fre_used_ratio, win='hann', ver='np'):
    """ Function: Get STFT coefficients of microphone signals (numpy / torch version)
        Args:       signal          - the microphone signals in time domain (nsample, nch) / (nbatch, nsample, nch)
                    win_len         - the length of frame / window
                    win_shift_ratio - the ratio between frame shift and frame length
                    nfft            - the number of fft points
                    fre_used_ratio  - the ratio between used frequency and valid frequency
                    win             - window type 
                                    'boxcar': a rectangular window (equivalent to no window at all)
                                    'hann': a Hann window
                    ver             - version
                                    'np': numpy version
                                    'torch': torch version
        Returns:    stft            - STFT coefficients (nf, nt, nch) / (nbatch, nf, nt, nch)
    """
    # short, zero padding; long, discarding extra points
    # if win_len<nfft:
    #     signal = np.concatenate((signal, np.zeros((nfft-win_len,signal.shape[1]))),axis=0)
    #     # print('a smaller window size!')
    # elif win_len>nfft:
    #     signal = signal[0: nfft, :]
    #     print('a larger window size!')
    # win_len = nfft

    nsample = signal.shape[-2]
    nch = signal.shape[-1]
    win_shift = int(win_len * win_shift_ratio)
    nfft_valid = int(nfft / 2)+1
    nf = int(nfft_valid * fre_used_ratio)

    if ver == 'np':
        nt = int((nsample - win_len) / win_shift) + 1
        stft = np.zeros((nf, nt, nch), dtype=np.complex)
        window = scipy.signal.get_window(window=win, Nx=win_len)
        for ch_idx in range(0, nch, 1):
            fre, _, stft_temp = scipy.signal.stft(signal[:, ch_idx], fs=16000, window = window, nperseg = win_len,
                                noverlap = win_len - win_shift, nfft = nfft, boundary = None, padded = True) # more similar to MATLAB results
            # Difference with MATLAB: MATLAB gets nfft coefficients (from 0hz), python gets nfft/2+1 coefficients (from 0hz)
            # stft_temp = librosa.stft(sensor_signal[:, ch_idx], n_fft=nfft, win_length=win_len, hop_length=win_shift, center = False) # may be some problem
            # a = librosa.util.frame(sensor_signal[:, ch_idx], frame_length=win_len, hop_length=win_shift)
            stft[:, :, ch_idx] = stft_temp[0:nf, 0:nt]

    elif ver == 'torch':
        nb = signal.shape[0]
        nt = int((nsample) / win_shift) + 1 # for iSTFT
        # nt = np.floor((nsample - win_len) / win_shift + 1).astype(int)
        stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64)
        if win == 'hann':
            window = torch.hann_window(window_length=win_len, device=signal.device)
        for ch_idx in range(0, nch, 1):
            stft_temp = torch.stft(signal[:, :, ch_idx], n_fft = nfft, hop_length = win_shift, win_length = win_len,
                                   window = window, center = True, normalized = False, return_complex = True)  # for iSTFT
            # stft_temp = torch.stft(signal[:, :, ch_idx], n_fft=nfft, hop_length=win_shift, win_length=win_len,
            #                        window=window, center=False, normalized=False, return_complex=True)

            stft[:, :, :, ch_idx] = stft_temp[:, 0:nf, 0:nt]

    else:
        assert False, 'Version of DoSTFT is not specified ~'

    return stft

def DoISTFT(stft, win_len, win_shift_ratio, nfft, ver='np'):
    """ Function: Get inverse STFT (numpy / torch version) 
        Args:   stft            - STFT coefficients (nf, nt, nch) / (nbatch, nf, nt, nch)
                    win_len         - the length of frame / window
                    win_shift_ratio - the ratio between frame shift and frame length
                    nfft            - the number of fft points
                    ver             - version
                                    'np': numpy version
                                    'torch': torch version
        Returns:    sig_time        - time-domain microphone signals (nsample, nch) / (nbatch, nsample, nch)
    """
    nf = stft.shape[-3]
    nt = stft.shape[-2]
    nch = stft.shape[-1]
    win_shift = int(win_len * win_shift_ratio)

    if ver == 'np':
        nsample = (nt - 1) * win_shift + win_len
        sig_time = np.zeros((nsample, nch))
        for ch_idx in range(0, nch, 1):
            _, sig_time_temp = scipy.signal.istft(stft[:,:,ch_idx], fs = 16000, nperseg= win_len,
                                        noverlap = win_len - win_shift, nfft = nfft, boundary = None)
            sig_time[:, ch_idx] = sig_time_temp[0: nsample]

    elif ver == 'torch':
        nb = stft.shape[0]
        nsample = (nt - 1) * win_shift
        sig_time = torch.zeros((nb, nsample, nch))
        for ch_idx in range(0, nch, 1):
            sig_time_temp = torch.istft(stft[:, :, :, ch_idx], n_fft=nfft, hop_length=win_shift, win_length=win_len,
                                        center=True, normalized=False, return_complex=False)
            # sig_time_temp = torch.istft(stft[:, :, :, ch_idx], n_fft=nfft, hop_length = win_shift, win_length = win_len,
            #                        center = False, normalized = False, return_complex = False)
            sig_time[:, :, ch_idx] = sig_time_temp[:, 0:nsample]

    else:
        assert False, 'Version of DoISTFT is not specified ~'

    return sig_time

def ATF2RTF(atf):
    """ Function: Convert acoustic transfer function (ATF) to relative transfer function (RTF) (numpy version)
        Args:        atf - ATF (nf, nt, nch) (nch = 2) 
        Returns:     rtf - RTF (3nf, nt)
    """ 
    if (np.sum(atf[:, :, 0] == 0) == 0):
        eps = 0
    else:
        eps = 0.0000000001
        print(eps)
    rtf_complex = atf[:, :, 1] / (atf[:, :, 0] + eps)
    rtf_mag = np.abs(rtf_complex)
    rtf_phase = np.angle(rtf_complex)
    if (np.sum(rtf_mag == 0) == 0):
        eps = 0
    else:
        eps = 0.0000000001
        print(eps)
    ILD = np.log10(rtf_mag + eps)
    IPD_sin = np.sin(rtf_phase)
    IPD_cos = np.cos(rtf_phase)
    rtf_real = np.vstack((ILD, IPD_sin, IPD_cos)) 
    return rtf_complex, rtf_real

def LoadDPRTF(dir, mic_type, mic_idx, doa_idx = None, full_fre = False):
    """ Function:  load specific direct-path relative transfer functions (torch version)
        Args:       dir         - the directory of the set of direct-path relative transfer functions
                    mic_type    - the type of microphone array
                    mic_idx     - the index of microphone, e.g., 21, 165 in CIPIC (nbatch, )
                    doa_idx     - the index of DOA (nbatch, )
                    full_fre    - whether use full frequency or not
        Returns:    dp_rtf      - direct-path relative transfer function of specific DOA (and microphone index) (nbatch, 3nf)
    """ 
    if mic_type == 'MIT':
        if full_fre:
            path = os.path.join(dir, 'GerData/RTF/rtf_dp_fullfre_Hkemar.mat')
        else:
            path = os.path.join(dir, 'GerData/RTF/rtf_dp_Hkemar.mat')
        dp_rtf_ndoa = scipy.io.loadmat(path)['rtf_dp'].astype('float32')  # 3nf*ndoa
        dp_rtf_ndoa = torch.from_numpy(dp_rtf_ndoa).cuda()
        if doa_idx == None:
            return dp_rtf_ndoa
        else:
            dp_rtf = dp_rtf_ndoa[:, doa_idx].transpose(1, 0)
            return dp_rtf

    elif mic_type == 'CIPIC':
        if full_fre:
            path = os.path.join(dir, 'GerData/RTF/rtf_dp_fullfre_set.mat')
        else:
            path = os.path.join(dir, 'GerData/RTF/rtf_dp_set.mat')
        rtf_dp_ndoa_nmic = scipy.io.loadmat(path)['rtf_dp'].astype('float32')  # (3nf, ndoa, nmic)
        rtf_dp_ndoa_nmic = torch.from_numpy(rtf_dp_ndoa_nmic).cuda()
        mic_idx_nmic = scipy.io.loadmat(path)['mic_idx']  # (1, nmic)
        mic_idx_nmic = torch.from_numpy(mic_idx_nmic.astype('float32')).cuda()

        if mic_idx == None:
            return rtf_dp_ndoa_nmic
        else:
            _, ndoa, nmic = rtf_dp_ndoa_nmic.shape
            nbatch = mic_idx.shape[0]
            mic_idx = mic_idx[:, np.newaxis].expand(nbatch, nmic)
            mic_idx_nmic = mic_idx_nmic.expand(nbatch, nmic)
            eq_result = mic_idx.eq(mic_idx_nmic).float() # (nbatch, nmic)

            # check eq_result to determine whether mic_idx exists in mic_idx_nmic or not
            eq_result_check = torch.sum(eq_result, dim=1)
            if torch.sum(eq_result_check)!=eq_result.shape[0]:
                print('mic_idx does not exist in mic_idx_nmic~')

            dp_rtf_ndoa = torch.matmul(rtf_dp_ndoa_nmic, eq_result.permute([1, 0])).permute([2, 0, 1])  # (nb, 3nf, ndoa)

            if doa_idx == None:
                return dp_rtf_ndoa
            else:
                doa_idx = doa_idx[:, np.newaxis].expand(nbatch, ndoa)
                doa_idx_set = torch.arange(ndoa).long()
                doa_idx_set = doa_idx_set[np.newaxis, :].expand(nbatch, ndoa).cuda()
                eq_result = doa_idx.eq(doa_idx_set).float()
                eq_result = eq_result[:, :, np.newaxis]     # (nbatch, ndoa, 1)
                dp_rtf = torch.bmm(dp_rtf_ndoa, eq_result)  # (nbatch, 3nf, 1)
                dp_rtf = dp_rtf.squeeze() # (nbatch, 3nf)
                return dp_rtf

def GerDPRTF(source_doa, ndoa_candidate, mic_location, nf=257, fre_max=8000, speed=343.0):
    """ Function: load specific normalized direct-path relative transfer functions (torch version)
        Args:       source_doa      - the DOAs of sources for given microphone signals (ntime, nsource, 2)
                    ndoa_candidate  - the number of candidate DOAs of source (2,) [nele, nazi]
                    mic_location    - the positions of microphones relative to the array center (nmic, 3)
                    nf              - the number of fft points
                    fre_max         - the maximum frequency
                    speed           - the speed of sound propagation
        Returns:    dp_rtf          - complex direct-path relative transfer function (indeed neglect magnitude)
                                     (nele, nazi, nf, nmic, nmic) / (ntime, nsource, nf, nmic, nmic)
    """
    nmic = mic_location.shape[-2]

    # Given ndoa_candidate (nele, nazi)
    if (source_doa is None) & (ndoa_candidate is not None):
        nele = ndoa_candidate[0]
        nazi = ndoa_candidate[1]
        ele_candidate = np.linspace(0, np.pi, nele)
        azi_candidate = np.linspace(-np.pi, np.pi, nazi)
        ITD = np.empty((nele, nazi, nmic, nmic)) # time differences, floats
        IPD = np.empty((nele, nazi, nf, nmic, nmic))  # phase differences
        fre_range = np.linspace(0.0, fre_max, nf)
        for m1 in range(nmic):
            for m2 in range(nmic):
                r = np.stack([np.outer(np.sin(ele_candidate), np.cos(azi_candidate)),
                              np.outer(np.sin(ele_candidate), np.sin(azi_candidate)),
                              np.tile(np.cos(ele_candidate), [nazi, 1]).transpose()], axis=2)
                ITD[:, :, m1, m2] = np.dot( r, mic_location[m2,:]- mic_location[m1, :] ) / speed
                IPD[:, :, :, m1, m2] = -2 * np.pi * np.tile(fre_range[np.newaxis, np.newaxis, :], [nele, nazi, 1]) * \
                                       np.tile(ITD[:, :, np.newaxis, m1, m2], [1, 1, nf])
        dprtf_complex = np.exp(1j*IPD) # (nele, nazi, nf, nmic, nmic)

    # Given source_doa (ntime, nsource, 2)
    elif (source_doa is not None) & (ndoa_candidate is None):
        nsource = source_doa.shape[-2]
        ntime = source_doa.shape[-3]
        ITD = np.empty((ntime, nsource, nmic, nmic))  # time differences, floats
        IPD = np.empty((ntime, nsource, nf, nmic, nmic))  # phase differences
        fre_range = np.linspace(0.0, fre_max, nf)

        for m1 in range(nmic):
            for m2 in range(nmic):
                print(np.sin(source_doa[:,:,0]).shape, )
                r = np.stack([np.sin(source_doa[:,:,0])*np.cos(source_doa[:,:,1]),
                              np.sin(source_doa[:,:,0])*np.sin(source_doa[:,:,1]),
                              np.cos(source_doa[:,:,0])], axis=2)
                ITD[:, :, m1, m2] = np.dot(r, mic_location[m2, :] - mic_location[m1, :]) / speed
                IPD[:, :, :, m1, m2] = -2*np.pi*np.tile(fre_range[np.newaxis, np.newaxis, :],[ntime, nsource, 1])*\
                            np.tile(ITD[:, :, np.newaxis, m1, m2], [1, 1, nf])

        dprtf_complex = np.exp(1j*IPD)  # (ntime, nsource, nf, nmic, nmic)

    else:
        assert False, 'Error exists in DPRTF Generation ~'

    return dprtf_complex

def detect_infnan(data):
    """ Function: check whether there is inf/nan in the element of data or not
    """ 
    import numpy as np
    inf_flag = np.isinf(data)
    nan_flag = np.isnan(data)
    if (True in inf_flag):
        print('INF exists in data ~')
    if (True in nan_flag):
        print('NAN exists in data ~')

def set_seed(seed):
    """ Function: fix random seed
    """ 
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # cudnn.benchmark = False
    # cudnn.enabled = False

def get_learningrate(optimizer):
    """ Function: get learning rates from optimizer
    """ 
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

if __name__ == "__main__":
    import torch
    import numpy as np
    import scipy.io

    ## Generate normalized DP-RTF
    source_doa = np.ones((100, 2, 2))
    source_doa[0, 0, :] = np.array([np.pi/2, 0.00])
    source_doa[0, 1, :] = np.array([np.pi/2, np.pi/2])
    ndoa_candidate = [32, 64]
    mic_location = np.array(((1.00, 0.00, 0.00), (-1.00, 0.00, 0.00)))
    nf = 257
    dprtf_complex = GerDPRTF(source_doa=None, ndoa_candidate=ndoa_candidate, mic_location=mic_location, nf=nf, fre_max=8000, speed=343.0)
    dprtf = np.concatenate((dprtf_complex[:, :, :, 0, 1:, np.newaxis].real, dprtf_complex[:, :, :, 0, 1:, np.newaxis].imag), axis=4)
    scipy.io.savemat('dprtf.mat', {'dprtf_complex': dprtf_complex[:,:,:,0,:]})

    ## Load specific DP-RTFs
    # from common.config import dirconfig
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(7)
    # ger_dirs = dirconfig()
    # data_dir = ger_dirs['data']
    # mic_idx = torch.ones(2).cuda() * 1
    # LoadDPRTF(dir=data_dir, mic_type='CIPIC', mic_idx=mic_idx, doa_idx=None, full_fre=False)

    ## Generate STFT coefficients (torch)
    # signal = torch.randn((1, 1*16000, 2)).cuda()
    # stft_torch = DoSTFT(signal, win_len=512, win_shift_ratio=0.5, nfft=512, fre_used_ratio=1, win='hann', ver='torch')
    # signal_torch_istft = DoISTFT(stft_torch, win_len=512, win_shift_ratio=0.5, nfft=512, ver='torch')
    # stft_torch_np = stft_torch[0,:,:].cpu().numpy()
    # signal_torch_np_istft = signal_torch_istft[0, :, :].cpu().numpy()
    # signal = signal[0, :, :].cpu().numpy()
    # scipy.io.savemat('torch.mat', {'signal': signal, 'signal_torch_np_istft': signal_torch_np_istft})
    # # Note: signal_torch_np_istft/signal = 0.5
    
    ## Generate STFT coefficients (np)
    # stft_np = DoSTFT(signal, win_len=512, win_shift_ratio=0.5, nfft=512, fre_used_ratio=1, win='hann', ver='np')
    # signal_np_istft = DoISTFT(stft_np, win_len=512, win_shift_ratio=0.5, nfft=512, ver='np')
    # scipy.io.savemat('np.mat', {'signal': signal, 'signal_np_istft': signal_np_istft})
 
