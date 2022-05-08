"""
	Function:   Generate audio data for sound source localization
    Descrition: Specific data paths in path(); Traverse all set_flags and generate_modes to generate desired data

    Reference:  B. Yang, H. Liu, and X. Li. "Learning deep direct-path relative transfer function for binaural sound source localization," 
    IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), vol. 29,  pp. 3491 - 3503, 2021.
	Author:     Bing Yang
    History:    2022-04-25 - Initial version
    Copyright Bing Yang
"""

import os
import random
import librosa
import scipy.io
import numpy as np
import soundfile as sf
import argparse 
from common.config import dirconfig, room_exp_config 
from common.utils import DoSTFT, ATF2RTF
from common.utils_room_acoustics import SouConvRIR, AddNoise, PreLoad

def room_random(set_flag, room_setup, exp_setup, save_dir=None):
    """ Function: Generate random room acoustics 
        Args:       set_flag        - the flag for stage
                                    'train': training
                                    'val': validation 
                                    'test': test
                    room_setup      - room acoustic setup
                    exp_setup       - experimental setup
                    save_dir        - data save directory (None is for not saving data)
        Returns:    room_setup_set  - the set of room acoustics setups
    """ 
    room_type_set = exp_setup['room'][set_flag]

    for room_type in room_type_set:
        room_setup_set = []
        num_idx = 0
        rt_set = room_setup[room_type]['rt']
        snr_set = room_setup[room_type]['snr']
        noise_type_set = room_setup[room_type]['n_type']
        doa_set = room_setup[room_type]['doa']
        dist_set = room_setup[room_type]['dist']
        ins_num = int(room_setup[room_type]['num']*exp_setup['num_scale'][set_flag])

        # Generate ramdom numbers
        if (set_flag=='train')|(set_flag=='val'):
            for ins_idx in range(0, ins_num, 1):
                for doa in doa_set:
                    doa_idx = doa_set.index(doa)

                    noise_type_idx = random.randint(0, len(noise_type_set)-1)
                    noise_type = noise_type_set[noise_type_idx]

                    dist_idx = random.randint(0, len(dist_set)-1)
                    dist = dist_set[dist_idx]

                    rt_random_idx = random.randint(0, len(rt_set) - 1)
                    rt = rt_set[rt_random_idx]

                    snr_random_idx = random.randint(0, len(snr_set) - 1)
                    snr = snr_set[snr_random_idx]

                    st_ed = exp_setup['noise'][set_flag]
                    noise_idx = random.randint(st_ed[0], st_ed[1])

                    st_ed = exp_setup['source'][set_flag]
                    source_idx = random.randint(st_ed[0], st_ed[1])

                    temp = [num_idx, room_type, rt, snr, noise_type, doa_idx, dist, noise_idx, source_idx]
                    room_setup_set.append(temp)

                    num_idx = num_idx + 1

        elif set_flag == 'test':
            for ins_idx in range(0, ins_num, 1):
                for noise_type in noise_type_set:
                    for dist in dist_set:
                        for doa in doa_set:
                            doa_idx = doa_set.index(doa)
                            for rtsnr_idx in range(0, len(rt_set), 1):
                                rt = rt_set[rtsnr_idx]
                                snr = snr_set[rtsnr_idx]

                                st_ed = exp_setup['noise'][set_flag]
                                noise_idx = random.randint(st_ed[0], st_ed[1])

                                st_ed = exp_setup['source'][set_flag]
                                source_idx = random.randint(st_ed[0], st_ed[1])

                                temp = [num_idx, room_type, rt, snr, noise_type, doa_idx, dist, noise_idx, source_idx]
                                room_setup_set.append(temp)

                                num_idx = num_idx+1

        # Save room setups
        if save_dir!=None:
            exist_temp = os.path.exists(save_dir)
            if exist_temp==False:
                os.makedirs(save_dir)
                print('make dir: ' + save_dir)
            filename = save_dir + 'setting_' + set_flag + room_type + '.txt'
            f = open(filename, 'w', encoding='utf - 8')
            for setup_idx in range(0, len(room_setup_set), 1):
                f.write(str(room_setup_set[setup_idx]))
                f.write('\n')
            f.close()

    return room_setup_set

def read_room_setup_set(filename):
    """ Function: Get the setups of room acoustics from saved files
    """
    f = open(filename, 'r', encoding="utf-8")
    room_setup_set = []
    for line in f:
        ele = line.strip('[').strip(']\n').split(', ')
        int_range = [0,2,3,5,7,8]
        str_range = [1,4,6]
        for i in int_range:
            ele[i] = int(ele[i])
        for i in str_range:
            ele[i] = ele[i].strip('\'') #.strip(' \'')
        room_setup_set.append(ele)

    return room_setup_set

def generate_save_all_rtf_dp_from_hrir(exp_setup, read_dir, save_dir, fre_used_ratio = 0.5, sample_rate = 16000):
    """ Function: Generate and save all direct-path relative transfer functions (DP-RTFs) according to head-related impulse responses (HRIRs)
        Args:   exp_setup       - experimental setup
                read_dir        - the directories of HRIRs
                save_dir        - the directories of DP-RTFs
                fre_used_ratio  - the ratio between used frequency and valid frequency
                                  fre_used_ratio = 0.5 → half frequency
                                  fre_used_ratio = 1 → full frequency
                sample_rate     - sampling rate
    """
    doa_set = exp_setup['doa']
    win_len_sec = 0.032
    nfft = int(sample_rate * win_len_sec)
    nf = int(nfft*0.5*fre_used_ratio) # desired frequencies 
    file_names = os.listdir(read_dir)
    for fname in file_names:
        # Load head-related impulse responses (hrirs)
        rir_path = read_dir + fname
        rir_set = scipy.io.loadmat(rir_path)['data']  # sample*nch*ndoa
        sr = scipy.io.loadmat(rir_path)['Fs'][0, 0]

        rtf_dp = np.empty((nf*3, 0))
        rtf_dp_c = np.empty((nf, 0))
        for doa in doa_set:
            doa_idx = doa_set.index(doa)

            # Resample hrirs to a desired sample rate
            rir_temp =  rir_set[:, :, doa_idx].transpose(1, 0)
            rir_dp = librosa.resample(rir_temp, orig_sr=sr, target_sr=sample_rate)
            rir_dp = rir_dp.transpose(1, 0)

            # Transform hrirs into TF domain from time domain 
            rir_len = rir_dp.shape[0]
            if rir_len < nfft:
                rir_dp = np.concatenate((rir_dp, np.zeros((nfft - rir_len, rir_dp.shape[1]))), axis=0)
                # print('a smaller window size!')
            elif rir_len > nfft:
                rir_dp = rir_dp[0: nfft, :]
                print('a larger window size!')
            rir_len = rir_dp.shape[0]
            stft_rir = DoSTFT(rir_dp, win_len=rir_len, win_shift_ratio=1, nfft=nfft, fre_used_ratio=fre_used_ratio)

            # Generate relative transfer function from hrtfs
            rtf_dp_c_temp, rtf_dp_temp = ATF2RTF(stft_rir)
            rtf_dp = np.hstack((rtf_dp, rtf_dp_temp))
            rtf_dp_c = np.hstack((rtf_dp_c, rtf_dp_c_temp))

        # Save rtfs
        exist_temp = os.path.exists(save_dir)
        if exist_temp==False:
            os.makedirs(save_dir)
            print('make dir: ' + save_dir)
        rtf_name = save_dir + 'rtf_dp_' + fname 
        # rtf_name = save_dir + 'rtf_dp_fullfre_' + fname
        scipy.io.savemat(rtf_name, {'rtf_dp': rtf_dp, 'rtf_dp_c': rtf_dp_c})

def generate_save_all_rtf_dp_set(read_dir, save_dir):
    """ Function: Generate and save the set of direct-path relative transfer functions (DP-RTFs)
        Args:   read_dir        - the directories of DP-RTFs
                save_dir        - the directories of the set of DP-RTFs
    """
    head_set = ['003','010','018','020','021','027','028','033','040','044',
                '048','050','051','058','059','060','061','065','119','124',
                '126','127','131','133','134','135','137','147','148','152',
                '153','154','155','156','162','163','165','008','009','011',
                '012','015','017','019','158'] # rank with the order that the head width is given or not

    rtf_dp = []
    mic_idx = []
    for head_name in head_set:
        # idx = head_set.index(head_name)
        file_name = 'rtf_dp_H' + head_name + '.mat' 
        # file_name = 'rtf_dp_fullfre_H' + head_name + '.mat'
        rtf_dp += [scipy.io.loadmat(read_dir + file_name)['rtf_dp']]
        mic_idx += [int(head_name)]
    rtf_dp = np.array(rtf_dp).transpose(1,2,0)  # rtf_dp: (nf*3, 25, len(head_set))
    mic_idx = np.array(mic_idx)[np.newaxis,:]   # mic_idx: (1, len(head_set))

    exist_temp = os.path.exists(save_dir)
    if exist_temp==False:
        os.makedirs(save_dir)
        print('make dir: ' + save_dir)
    rtf_name = save_dir + 'rtf_dp_set.mat'
    # rtf_name = save_dir + 'rtf_dp_fullfre_set.mat' 
    scipy.io.savemat(rtf_name, {'rtf_dp': rtf_dp, 'mic_idx': mic_idx})


def generate_sensor_signal(setup_idx, room_setup_set, rir_set, noise_signal_set, source_signal_path_set, sample_rate = 16000):
    """ Function: Generate sensor signals corresponding setup_idx
        Args:       setup_idx               - the index of setups
                    room_setup_set          - the set of room acoustics setups
                    rir_set                 - the set of RIRs 
                    noise_signal_set        - the set of noise signals
                    source_signal_path_set  - the set of source signal paths 
                    sample_rate             - sampling rate
        Returns:    sensor_signal           - microphone signals
                    sample_rate             - sampling rate
    """
    ## Determine acoustics
    _, room_type, rt, snr, noise_type, doa_idx, dist, noise_idx, source_idx = room_setup_set[setup_idx]

    ## Load rir
    rir = rir_set[room_type + str(rt) + dist + str(doa_idx)]
    rir_dp = rir_set[room_type + str(0) + dist + str(doa_idx)]

    ## Generate source & noise signals
    time_duration = 1
    source_signal_sample_offset = 1000

    # source signal
    sousig_name = source_signal_path_set[source_idx]
    source_signal_temp, sample_rate_temp = sf.read(sousig_name)
    if sample_rate != sample_rate_temp:
        source_signal_temp = librosa.resample(source_signal_temp, orig_sr = sample_rate_temp, target_sr = sample_rate)

    nsample = source_signal_temp.shape[0]
    sample_st = source_signal_sample_offset
    sample_ed = sample_rate * time_duration + source_signal_sample_offset
    if nsample < sample_ed:
        source_signal = np.concatenate((source_signal_temp[sample_st:nsample], source_signal_temp[0:sample_ed-nsample]),axis=0)
    else:
        source_signal = source_signal_temp[sample_st:sample_ed]*1

    # noise signal
    sample_st = noise_idx * sample_rate * time_duration
    sample_ed = (noise_idx+1) * sample_rate * time_duration
    head_name = room_type[:-7]
    noise_signal = noise_signal_set[head_name+noise_type][sample_st:sample_ed,:]*1

    ## Generate sensor signal
    sensor_signal_wonr = SouConvRIR(source_signal, rir_dp)
    sensor_signal_won = SouConvRIR(source_signal, rir)
    sensor_signal_w = AddNoise(sensor_signal_won, noise_signal, snr)
    sensor_signal = {}
    sensor_signal['w'] = sensor_signal_w
    sensor_signal['won'] = sensor_signal_won
    sensor_signal['wonr'] = sensor_signal_wonr

    return sensor_signal, sample_rate

def generate_save_all_sensor_signal(set_flag, room_setup_set, room_type, rir_set, noise_signal_set, source_signal_path_set, dirs):
    """ Function: Generate and save all sensor signals
        Args:       set_flag                - the flag for stage
                                              'train': training
                                              'val': validation 
                                              'test': test
                    room_setup_set          - the set of room acoustics setups
                    room_type               - the name of room type (str), e.g., 'H021_507030'
                    rir_set                 - the set of RIRs
                    noise_signal_set        - the set of noise signals
                    source_signal_path_set  - the set of source signal paths 
                    dirs                    - directories
    """
    for setup_idx in range(0, len(room_setup_set), 1):
        if setup_idx % 1000 == 0:
            print('Instance: {}/{} [{:.0f}%]'.format(setup_idx, len(room_setup_set), 100.0*setup_idx / len(room_setup_set)))

        sensor_signal, sample_rate = generate_sensor_signal(setup_idx, room_setup_set, rir_set,
                                                                           noise_signal_set, source_signal_path_set)
        # Save
        sensig_dir = dirs['sensig'] + set_flag + '/' + room_type + '/'
        exist_temp = os.path.exists(sensig_dir)
        if exist_temp==False:
            os.makedirs(sensig_dir)
            print('make dir: ' + sensig_dir)
        sensig_name = sensig_dir + str(setup_idx) + '.wav'
        sf.write(sensig_name, sensor_signal['w'], sample_rate)

        # sensig_dir = dirs['sensigclean'] + set_flag + '/' + room_type + '/'
        # exist_temp = os.path.exists(sensig_dir)
        # if exist_temp == False:
        #     os.makedirs(sensig_dir)
        #     print('make dir: ' + sensig_dir)
        # sensig_name = sensig_dir + str(setup_idx) + '.wav'
        # sf.write(sensig_name, sensor_signal['wonr'], sample_rate)


def path(mic_type):
    """ Function: Define optional paths for data generation
    """
    dirs = dirconfig()
    dirs['hrir'] = dirs['hrir'] + mic_type + '/'
    dirs['rir'] = dirs['rir'] + mic_type +'/'
    # data_dir = dirs['data']
    # dirs = {}
    # dirs['hrir'] = data_dir + 'HRIR/' + mic_type + '/'
    # dirs['rir'] = data_dir + 'RIR/'+ mic_type +'/'
    # dirs['sousig'] = data_dir + 'SouSig/TIMIT/'
    # dirs['noisig'] = data_dir + 'NoiSig/DiffNoise/'
    # gerdata_dir = data_dir + 'GerData2/'
    # dirs['ger'] = gerdata_dir
    # dirs['sensig'] = gerdata_dir + 'SenSig/'
    # # dirs['sensig'] = gerdata_dir + 'SenSigEnhanced/'
    # dirs['sensigclean'] = gerdata_dir + 'SenSigClean/'
    # dirs['sensigclean_n'] = gerdata_dir + 'SenSigClean_n/'
    # dirs['rtf'] = gerdata_dir + 'RTF/'
    # dirs['setting'] = gerdata_dir + 'Setting/'

    return dirs

def parse():
    """ Function: Define optional arguments for data generation
    """
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('--stage', type=str, default='train', metavar='stage', help='stage (train, val, test)')
    parser.add_argument('--data', type=str, default='sousig_rir_list', metavar='data', help='data type (sousig_rir_list, dprtf, room_setting, sensig)')
    parser.add_argument('--mic', type=str, default='CIPIC', metavar='mic', help='mic type (CIPIC, MIT)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    ## Options 
    args = parse()
    set_flag = args.stage
    generate_mode = args.data
    mic_type = args.mic
    dirs = path(mic_type = mic_type)
    os.chdir(dirs['data'])

    ## Fix random seed
    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    ## Get settings
    room_setup, exp_setup = room_exp_config(mic_type)

    if (generate_mode == 'sousig_rir_list'):
        # Generate and save path lists of source signals and acoustic settings of rirs ((head id, room size, rt, dist, doa_idx)
        print(set_flag+'- pre-loading of path lists and acoustic settings!')
        rir_set, noise_signal_set, source_signal_path_set = PreLoad(set_flag, room_setup, exp_setup, dirs=dirs, save_dir = dirs['ger'])

    if (generate_mode == 'dprtf'):
        # Generate and save rtf_dp from HRIR
        print('direct-path rtf generating!')
        generate_save_all_rtf_dp_from_hrir(exp_setup, read_dir=dirs['hrir'], save_dir = dirs['rtf'])
        generate_save_all_rtf_dp_set(read_dir = dirs['rtf'], save_dir = dirs['rtf'])

    elif (generate_mode == 'room_setting'):
        # Generate and save all random acoustic settings of room
        print(set_flag + '- room setting generating!')
        room_setup_set = room_random(set_flag, room_setup, exp_setup, save_dir = dirs['setting'])

    else:
        # Pre-load
        print(set_flag + ': pre-loading of path lists and acoustic settings!')
        rir_set, noise_signal_set, source_signal_path_set = PreLoad(set_flag, room_setup, exp_setup, dirs = dirs, save_dir = None)

        if (generate_mode == 'sensig'):
            # Generate and save sensor signals 
            print(set_flag+'- sensor signal generating!')
            room_type_set = exp_setup['room'][set_flag]
            for room_type in room_type_set:
                print('----room' + room_type + '----')
                filename = dirs['setting'] + 'setting_' + set_flag + room_type + '.txt'
                room_setup_set = read_room_setup_set(filename)
                generate_save_all_sensor_signal(set_flag, room_setup_set, room_type, rir_set, noise_signal_set, source_signal_path_set, dirs = dirs)