"""
	Function:   Define some operations for generating multi-channel audio data 
	Author:     Bing Yang
    Copyright Bing Yang
"""

import os
import random
import librosa
import scipy.io
import numpy as np
import soundfile as sf
from common.config import room_exp_config

def SouConvRIR(source_signal, rir):
    """ Function: Perform convolution between source signal and room impulse reponses (RIRs)
        Args:       source_signal   - source signal (nsample, )
                    rir             - multi-channel RIR (nrirsample, nch)
        Returns:    sensor_signal   - multi-channel microphone signals (nsample, nch)
    """ 
    nsample = len(source_signal)
    _, nch = rir.shape
    sensor_signal = np.zeros((nsample, nch))

    for ch_idx in range(0, nch, 1):
        sensor_signal_temp = np.convolve(source_signal, rir[:, ch_idx], 'full')
        sensor_signal[:, ch_idx] = sensor_signal_temp[0: nsample]
    return sensor_signal

def AddNoise(sensor_signal_won, noise_signal, snr):
    """ Function: Add noise to clean microphone signals with a given signal-to-noise ratio (SNR)
        Args:       sensor_signal_won   - clean microphone signals without any noise (nsample, nch)
                    noise_signal        - noise signals (nsample, nch)
                    snr                 - specific SNR
        Returns:    sensor_signal       - microphone signals with noise (nsample, nch)
    """ 
    nsample, _ = sensor_signal_won.shape
    av_pow = np.mean(np.sum(sensor_signal_won**2, axis=0)/nsample, axis=0) 	# average mic power across all received signals
    av_noisepow = np.mean(np.sum(noise_signal**2, axis=0)/nsample, axis=0)
    noise_wsnr = pow(av_pow / (10 ** (snr / 10)), 0.5)/ pow(av_noisepow,0.5) * noise_signal
    sensor_signal = sensor_signal_won + noise_wsnr
    return sensor_signal

def GetMicIdxFromRoomType(room_type):
    """ Function: Determine the microphone array type according to room type
        Args:       room_type   - the name of room type (str), e.g., 'H021_507030'
        Returns:    mic_idx     - the ID of microphone type (int), e.g, 21
    """ 
    H_idx = room_type.find('H')
    if H_idx != -1:
        mic_idx = room_type[H_idx + 1:(H_idx + 4)]
        mic_idx = int(mic_idx)
    else:
        mic_idx = -1

    return mic_idx

def PreLoad(set_flag, room_setup, exp_setup, dirs, save_dir = None):
    """ Function: Pre-load room impulse reponses (RIRs), noise signals, names of source signals according to the predefined setups 
        Args:       set_flag                - the flag for stage
                                             'train': training
                                             'val': validation 
                                             'test': test
                    room_setup              - room acoustic setup
                    exp_setup               - experimental setup
                    dirs                    - directories
                    save_dir                - data save directory (None is for not saving data)
        Returns:    rir_set                 - the set of RIRs 
                    noise_signal_set        - the set of noise signals
                    source_signal_path_set  - the set of source signal paths 
    """ 
    rir_dir = dirs['rir']
    sousig_dir = dirs['sousig']
    noisig_dir = dirs['noisig']
    sample_rate = 16000

    ## Pre-load files of rirs
    rir_set = {}
    room_type_set = exp_setup['room'][set_flag]
    for room_type in room_type_set:
        rt_set = room_setup[room_type]['rt']
        doa_all_set = room_setup[room_type]['doa_all']
        doa_set = room_setup[room_type]['doa']
        dist_set = room_setup[room_type]['dist']
        for rt in rt_set:
            for dist in dist_set:
                for doa in doa_set:
                    doa_idx_in_all = doa_all_set.index(doa)
                    doa_idx = doa_set.index(doa)
                    if 'MIT' in rir_dir:
                        rir_name = 'R' + str(rt) + '_S' + str(doa_idx_in_all + 1) + '.mat'
                        rir_path = os.path.join(rir_dir + room_type + '/D' + dist, rir_name)

                    elif 'CIPIC' in rir_dir:
                        dist_idx = dist_set.index(dist)
                        doa_idx_temp = len(doa_all_set) * dist_idx + doa_idx_in_all
                        rir_name = 'R' + str(rt) + '_S' + str(doa_idx_temp + 1) + '.mat'
                        rir_path = os.path.join(rir_dir + room_type + '/', rir_name)

                    rir_temp = scipy.io.loadmat(rir_path)['data'].transpose(1,0)  # (nch, nsample)
                    sr_temp = scipy.io.loadmat(rir_path)['Fs'][0,0]
                    rir_temp = librosa.resample(rir_temp, orig_sr=sr_temp, target_sr=sample_rate)
                    rir_set[room_type + str(rt) + dist + str(doa_idx)] = rir_temp.transpose(1,0)

        # without reverberation (and noise)
        rt = 0
        for dist in dist_set:
            for doa in doa_set:
                doa_idx_in_all = doa_all_set.index(doa)
                doa_idx = doa_set.index(doa)
                if 'MIT' in rir_dir:
                    rir_name = 'R' + str(rt) + '_S' + str(doa_idx_in_all + 1) + '.mat'
                    rir_path = os.path.join(rir_dir + room_type + '/D' + dist, rir_name)

                elif 'CIPIC' in rir_dir:
                    dist_idx = dist_set.index(dist)
                    doa_idx_temp = len(doa_all_set) * dist_idx + doa_idx_in_all
                    rir_name = 'R' + str(rt) + '_S' + str(doa_idx_temp + 1) + '.mat'
                    rir_path = os.path.join(rir_dir + room_type + '/', rir_name)

                rir_temp = scipy.io.loadmat(rir_path)['data'].transpose(1, 0)  # (nch, nsample)
                sr_temp = scipy.io.loadmat(rir_path)['Fs'][0,0]
                rir_temp = librosa.resample(rir_temp, orig_sr=sr_temp, target_sr=sample_rate)
                rir_set[room_type + str(rt) + dist + str(doa_idx)] = rir_temp.transpose(1, 0)

    ## Pre-load files of noise signals
    noise_signal_set = {}
    if (set_flag=='train'):
        st = exp_setup['noise']['train'][0]
        ed = exp_setup['noise']['train'][1]
    elif (set_flag=='val'):
        offset = exp_setup['noise']['train'][1] - exp_setup['noise']['train'][0] + 1
        st = exp_setup['noise']['val'][0] + offset
        ed = exp_setup['noise']['val'][1] + offset
    else:
        offset = exp_setup['noise']['val'][1] - exp_setup['noise']['val'][0] + 1 + \
                 exp_setup['noise']['train'][1] - exp_setup['noise']['train'][0] + 1
        st = exp_setup['noise']['test'][0] + offset
        ed = exp_setup['noise']['test'][1] + offset

    for room_type in room_type_set:
        noise_type_set = room_setup[room_type]['n_type']
        head_name = room_type[:-7]
        for noise_type in noise_type_set:
            noise_file_path = noisig_dir + '/noisesignal_' + noise_type + '_' + head_name + '.mat'
            noise_signal_temp = scipy.io.loadmat(noise_file_path)['noise']  # (nsample, nch)
            noise_signal_set[head_name + noise_type] = noise_signal_temp[st*sample_rate:(ed+1)*sample_rate, :]

    ## Pre-load file names of source signals
    if (set_flag=='train'):
        file_dir = sousig_dir + 'train'
        st = exp_setup['source']['train'][0]
        ed = exp_setup['source']['train'][1]
    elif (set_flag=='val'):
        file_dir = sousig_dir + 'train'
        offset = exp_setup['source']['train'][1] - exp_setup['source']['train'][0] + 1
        st = exp_setup['source']['val'][0] + offset
        ed = exp_setup['source']['val'][1] + offset
    else:
        file_dir = sousig_dir + 'test'
        st = exp_setup['source']['test'][0]
        ed = exp_setup['source']['test'][1]
    file_set = librosa.util.find_files(file_dir, ext='wav')

    source_signal_path_set = []
    idx = 0
    for fname in file_set:
        if (idx>=st)&(idx<=ed):
            source_signal_path_set.append(fname)
        idx = idx + 1

    ## Save (source_signal_path_set, rir_set)
    if save_dir != None:
        exist_temp = os.path.exists(save_dir)
        if exist_temp == False:
            os.makedirs(save_dir)
            print('make dir: ' + save_dir)
        idx = 0
        source_signal_path_set = []
        for fname in file_set:
            if (idx >= st) & (idx <= ed):
                rootpath = os.path.abspath('..')
                re_path = fname.replace(rootpath, '~')
                source_signal_path_set.append(re_path)
            idx = idx + 1

        filename = save_dir + 'source_signal_path_set_' + set_flag + '.txt'
        f = open(filename, 'w', encoding='utf - 8')
        for idx in range(0, len(source_signal_path_set), 1):
            f.write(str(source_signal_path_set[idx]))
            f.write('\n')
        f.close()

        if 'MIT' in rir_dir:
            filename = save_dir + 'MIT_rir_set_' + set_flag + '.txt'
        elif 'CIPIC' in rir_dir:
            filename = save_dir + 'CIPIC_rir_set_' + set_flag + '.txt'
        f = open(filename, 'w', encoding='utf - 8')
        for key in rir_set:
            f.write(str(key))
            f.write('\n')
        f.close()

    return rir_set, noise_signal_set, source_signal_path_set

def GenerateSensig(rir_set, noise_signal_set, source_signal_path_set,
              mic_type, set_flag, room_setup=None, exp_setup=None, sample_rate=16000):
    """ Function: Generate random sensor signals
        Args:       rir_set                 - the set of RIRs 
                    noise_signal_set        - the set of noise signals
                    source_signal_path_set  - the set of source signal paths 
                    mic_type                - the type of microphone array 
                                              'MIT': array on the MIT KEMAR dummy head
                                              'CIPIC': array on the CIPIC head subject
                    set_flag                - the flag for stage
                                              'train': training
                                              'val': validation 
                                              'test': test
                    room_setup              - room acoustic setup
                    exp_setup               - experimental setup
                    sample_rate             - sampling rate
        Returns:    sensor_signal           - microphone signals with noise and reverberation
                    sensor_signal_won       - microphone signals without noise but with reverberation
                    sensor_signal_wonr      - microphone signals without noise and reverberation
                    doa_label               - the index of ground-truth DOA in DOA candidates, e.g., [0, 24] for CIPIC
                    mic_idx                 - the ID of microphone type (int), e.g, 21
    """ 
    if (room_setup==None)&(exp_setup==None):
        room_setup, exp_setup = room_exp_config(mic_type)

    ## Get random index -  set_flag, room_setup, exp_setup
    if (set_flag == 'train'):

        room_type_set = exp_setup['room'][set_flag]

        room_type_idx = random.randint(0, len(room_type_set) - 1)
        room_type = room_type_set[room_type_idx]

        doa_set = room_setup[room_type]['doa']
        dist_set = room_setup[room_type]['dist']
        rt_set = room_setup[room_type]['rt']
        snr_set = room_setup[room_type]['snr']
        noise_type_set = room_setup[room_type]['n_type']
        noise_st_ed = exp_setup['noise'][set_flag] # in seconds
        source_st_ed = exp_setup['source'][set_flag] # in files

        doa_idx = random.randint(0, len(doa_set) - 1)

        dist_idx = random.randint(0, len(dist_set) - 1)
        dist = dist_set[dist_idx]

        rt_random_idx = random.randint(0, len(rt_set) - 1)
        rt = rt_set[rt_random_idx]

        snr_random_idx = random.randint(0, len(snr_set) - 1)
        snr = snr_set[snr_random_idx]

        noise_type_idx = random.randint(0, len(noise_type_set) - 1)
        noise_type = noise_type_set[noise_type_idx]

        # noise_idx = random.randint(noise_st_ed[0], noise_st_ed[1])

        source_idx = random.randint(source_st_ed[0], source_st_ed[1])

    elif (set_flag == 'val') | (set_flag == 'test'):
        print('Unprocessed for val/test data!') 

    ## Load data - room_type, rt, snr, noise_type, doa_idx, dist, noise_idx, source_idx
    time_duration = 1

    # rir & rir_dp - room_type, doa_idx, dist, rt
    rir = rir_set[room_type + str(rt) + dist + str(doa_idx)]
    rir_dp = rir_set[room_type + str(0) + dist + str(doa_idx)]

    # source signal - source_idx
    sousig_name = source_signal_path_set[source_idx]
    source_signal_temp, sample_rate_temp = sf.read(sousig_name)
    if sample_rate != sample_rate_temp:
        source_signal_temp = librosa.resample(source_signal_temp, orig_sr = sample_rate_temp, target_sr = sample_rate)

    nsample = source_signal_temp.shape[0]
    if nsample < sample_rate * time_duration:
        sample_st = 0
        sample_ed = sample_st + sample_rate * time_duration
        source_signal = np.concatenate((source_signal_temp[sample_st:nsample], source_signal_temp[0:sample_ed-nsample]),axis=0)
    else:
        sample_st = random.randint(0, nsample - sample_rate * time_duration)
        sample_ed = sample_st + sample_rate * time_duration
        source_signal = source_signal_temp[sample_st:sample_ed] * 1

    # noise signal - room_type, noise_type, noise_st_ed
    head_name = room_type[:-7]
    noise_signal_temp = noise_signal_set[head_name + noise_type]
    sample_st = random.randint(sample_rate * noise_st_ed[0], sample_rate * noise_st_ed[1] - sample_rate * time_duration)
    sample_ed = sample_st + sample_rate * time_duration
    noise_signal = noise_signal_temp[sample_st:sample_ed]

    ## Generate sensor signal - snr
    sensor_signal_wonr = SouConvRIR(source_signal, rir_dp)
    sensor_signal_won = SouConvRIR(source_signal, rir)
    sensor_signal = AddNoise(sensor_signal_won, noise_signal, snr)

    ## Obtain DOA label and array index
    doa_label = doa_idx
    mic_idx = GetMicIdxFromRoomType(room_type)

    return sensor_signal, sensor_signal_won, sensor_signal_wonr, doa_label, mic_idx

if __name__ == "__main__":
    source_signal = np.random.rand(1600,)
    rir = np.random.rand(10,1)
    sensor_signal = SouConvRIR(source_signal, rir)
    print(sensor_signal.shape)