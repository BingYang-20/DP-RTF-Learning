""" 
    Function:   Define datasets
	Author:     Bing Yang
    Copyright Bing Yang
"""

import os
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from common.getData import read_room_setup_set
from common.config import dirconfig, room_exp_config 
from common.utils import DoSTFT 
from common.utils_room_acoustics import PreLoad, GenerateSensig, GetMicIdxFromRoomType

def GetInOut(stft, stft_wonr, doa_label, mic_idx):
    """ Function: Get the for input and target of DNN
        Args:       stft        - stft coefficients of microphone signals (nf, nt, 2)
                    stft_wonr   - stft coefficients of microphone signals that without noise and reveration (nf, nt, 2) 
                    doa_label   - doa index in doa candidates (1)
        Returns:    input_nn    - input of DNN
                    target_nn   - target of DNN
    """ 
        
    # Magnitude and phase
    input_mag = np.abs(stft)
    input_phase = np.angle(stft)

    # Time-frequency weight 
    # target_wtf = pow(pow(abs(stft_wonr), 2) / (pow(abs(stft_wonr), 2) + pow(abs(stft[:, :, 0:2] - stft_wonr), 2)), 0.5)  # ideal ratio mask (IRM)
    # target_wtf  = np.maximum(0, target_wtf * np.real(stft_wonr / stft / abs(stft_wonr / stft))) # phase-sensitive mask (PSM)
    # target_wtf  = abs(stft_wonr) / (abs(stft) + 0.00000001)  # ratio mask (RM)
    # target_wtf = np.maximum(1, abs(stft_wonr) / (abs(stft) + 0.00000001))  # ratio mask (RM)

    # Magnitude and phase
    # target_mag = np.log10(np.abs(stft_wonr) + 0.00000001)
    # target_phase = np.angle(stft_wonr)

    # Microphone arrya index
    mic_idx = mic_idx*np.ones((stft_wonr.shape[0], stft_wonr.shape[1], 1))
    mic_idx = mic_idx.astype('float32')

    # DOA label
    doa_label = doa_label*np.ones((1, stft_wonr.shape[1]))

    # Input & Target
    input_nn = np.concatenate((input_mag, input_phase, mic_idx), axis=2).transpose(2, 0, 1)
    target_nn = np.vstack((doa_label, ))

    return input_nn, target_nn

def GetSTFT(signal, sample_rate = 16000):
    """ Function: Get STFT coefficients
        Args:       signal      - microphone array signals (nsample, nch)/(nbatch, nsample, nch)
                    sample rate - sampling rate of microphone signals
        Returns:    stft        - STFT coefficients (nf, nt, nch)/(nbatch, nf, nt, nch)
    """

    win_len_sec = 0.032
    win_len = int(sample_rate * win_len_sec)
    win_shift_ratio = 0.5
    nfft = win_len
    fre_used_ratio = 4000 / (sample_rate / 2)
    stft = DoSTFT(signal, win_len, win_shift_ratio, nfft, fre_used_ratio)
    return stft

class TrainDataSet(Dataset):
    def __init__(self, set_flag, nt_context, mic_type, mic_match=False, data_random=False):
        super(TrainDataSet, self).__init__()
        
        self.dirs = dirconfig()
        self.dirs['hrir'] = self.dirs['hrir'] + mic_type + '/'
        self.dirs['rir'] = self.dirs['rir'] + mic_type +'/'
        self.set_flag = set_flag
        self.context = nt_context
        self.ndime = int(nt_context/nt_context)
        self.mic_type = mic_type
        self.data_random = data_random

        room_setup, exp_setup = room_exp_config(mic_type)

        if mic_type == 'MIT':
            if (set_flag == 'train'):
                room_type_set = ['708050', '606035', '405530', '383025']
                num = 50000  # extreme large
            elif (set_flag == 'val'):
                room_type_set = ['708050', '606035', '405530', '383025']
                num = 50000  # extreme large
            elif (set_flag == 'test'):
                room_type_set = ['608038', '507030', '404027']
                num = 50000  # extreme large

        elif mic_type == 'CIPIC':
            if mic_match:
                exp_setup['room'] =  {
                    'train': [
                    'H021_707040', 'H021_806536', 'H021_506028', 'H021_456031', 'H021_708050', 'H021_679046',
                    'H021_405530', 'H021_538038', 'H021_383025', 'H021_503229',
                    # 'H040_707040', 'H040_806536', 'H040_506028', 'H040_456031', 'H040_708050', 'H040_679046',
                    # 'H040_405530', 'H040_538038', 'H040_383025', 'H040_503229',
                    ],
                    'val': [
                        'H021_606035', 'H021_406032',
                        # 'H040_606035', 'H040_406032',
                    ],
                    'test': [
                        'H021_507030', 'H021_608038', 'H021_404027',
                        # 'H040_507030', 'H040_608038', 'H040_404027',
                    ],
                }
            else:
                exp_setup['room'] = {
                    'train': [
                        'H010_707040', 'H028_707040', 'H124_707040', 'H011_806536', 'H012_806536', 'H165_806536',
                        'H044_506028', 'H127_506028', 'H156_506028', 'H015_456031', 'H017_456031', 'H018_456031',
                        'H048_708050', 'H050_708050', 'H051_708050', 'H058_679046', 'H059_679046', 'H060_679046',
                        'H134_405530', 'H135_405530', 'H137_405530', 'H147_538038', 'H148_538038', 'H152_538038',
                        'H153_383025', 'H154_383025', 'H155_383025', 'H158_503229', 'H162_503229', 'H163_503229',
                    ],
                    'val': [
                        'H061_606035', 'H065_606035', 'H119_606035',
                        'H126_406032', 'H131_406032', 'H133_406032',
                    ],
                    'test': [
                        'H021_507030', 'H003_507030', 'H040_507030',
                        'H008_608038', 'H009_608038', 'H033_608038',
                        'H019_404027', 'H020_404027', 'H027_404027',
                        # 'H021_608038', 'H021_404027',
                        # 'H040_608038', 'H040_404027',
                    ],
                }
            self.room_setup = room_setup
            self.exp_setup = exp_setup

            if (set_flag == 'train'):
                room_type_set = exp_setup['room'][set_flag]
                num = 120000 / len(room_type_set)
            elif (set_flag == 'val'):
                room_type_set = exp_setup['room'][set_flag]
                num = 24000 / len(room_type_set)
            else:
                room_type_set = exp_setup['room'][set_flag]
                num = 50000  # A extreme large number

        if not self.data_random:
            self.file_paths = []
            self.mic_idxs = []
            self.doa_labels = []
            for room_type in room_type_set:
                file_name = self.dirs['setting'] + 'setup_' + set_flag + room_type + '.txt'
                room_setup_set = read_room_setup_set(file_name)

                file_dir = self.dirs['sensig'] + set_flag + '/' + room_type
                file_names = os.listdir(file_dir)
                for fname in file_names:
                    if int(fname[:-4])<num: # if re.search(r'N\d',fname):
                        self.file_paths.append((os.path.join(file_dir, fname)))

                        setup_idx = int(os.path.splitext(fname)[0])
                        _, _, _, _, _, doa_label, _, _, _ = room_setup_set[setup_idx]
                        self.doa_labels.append(doa_label)

                        mic_idx = GetMicIdxFromRoomType(room_type)
                        self.mic_idxs.append(mic_idx)

            self.nins = len(self.file_paths)    

        else:  # only used in training
            self.rir_set, self.noise_signal_set, self.source_signal_path_set = \
                PreLoad(set_flag, room_setup, exp_setup, self.dirs, save_dir=None)
            self.nins = 120000

    def __getitem__(self, idx):
        if (self.set_flag == 'train')|(self.set_flag == 'val'):
            if self.context == 31:
                offset = 20
            elif self.context == 61:
                offset = 0
        elif self.set_flag == 'test':
            offset = 0
        dim_idx = idx % self.ndime + offset

        if not self.data_random: 
            file_idx = idx // self.ndime

            file_path = self.file_paths[file_idx]
            sensor_signal, _ = sf.read(file_path)

            file_path_wonr = file_path.replace("SenSig", "SenSigClean")
            sensor_signal_wonr, _ = sf.read(file_path_wonr)

            doa_label = self.doa_labels[file_idx]
            mic_idx = self.mic_idxs[file_idx]

        else:    
            sensor_signal, sensor_signal_won, sensor_signal_wonr, doa_label, mic_idx = \
                GenerateSensig(rir_set=self.rir_set, noise_signal_set=self.noise_signal_set, source_signal_path_set=self.source_signal_path_set,
                    mic_type=self.mic_type, set_flag=self.set_flag, room_setup=self.room_setup, exp_setup=self.exp_setup) # doa_label (1), mic_idx (1)

        stft_temp = GetSTFT(sensor_signal)
        stft = stft_temp.astype('complex64')[:, dim_idx:dim_idx + self.context, :]
        stft_wonr_temp = GetSTFT(sensor_signal_wonr)
        stft_wonr = stft_wonr_temp.astype('complex64')[:, dim_idx:dim_idx + self.context, :]

        data_nn, target_nn = GetInOut(stft, stft_wonr, doa_label, mic_idx)

        return data_nn, target_nn

    def __len__(self):
        return self.nins * self.ndime

class TestDataSet(Dataset):
    def __init__(self, room_type, name_set, nt_context):
        super(TestDataSet, self).__init__()

        dirs = dirconfig()

        file_name = dirs['setting'] + 'setup_test' + room_type + '.txt'
        room_setup_set = read_room_setup_set(file_name)

        file_dir = dirs['sensig'] + 'test' + '/' + room_type
        self.file_paths = []
        self.mic_idxs = []
        self.doa_labels = []
        for setup_idx in name_set:
            fname = str(setup_idx) + '.wav'
            self.file_paths.append((os.path.join(file_dir, fname)))

            _, _, _, _, _, doa_label, _, _, _ = room_setup_set[setup_idx]
            self.doa_labels.append(doa_label)

            mic_idx = GetMicIdxFromRoomType(room_type)
            self.mic_idxs.append(mic_idx)

        self.context = nt_context
        self.ndime = int(nt_context/nt_context)

    def __getitem__(self, idx):
        file_idx = idx // self.ndime
        dim_idx = idx % self.ndime + 0
        if self.context == 5:
            dim_idx = 6
            print(dim_idx)
        if self.context == 10:
            dim_idx = 5
            print(dim_idx)
        if self.context == 12:
            dim_idx = 5
            print(dim_idx)
        if self.context == 18:
            dim_idx = 5
            print(dim_idx)

        file_path = self.file_paths[file_idx]
        sensor_signal, _ = sf.read(file_path)

        file_path_wonr = file_path.replace("SenSig", "SenSigClean")
        sensor_signal_wonr, _ = sf.read(file_path_wonr)
        doa_label = self.doa_labels[file_idx]
        mic_idx = self.mic_idxs[file_idx]

        stft_temp = GetSTFT(sensor_signal)
        stft = stft_temp.astype('complex64')[:, dim_idx:dim_idx + self.context, :]
        stft_wonr_temp = GetSTFT(sensor_signal_wonr)
        stft_wonr = stft_wonr_temp.astype('complex64')[:, dim_idx:dim_idx + self.context, :]

        data_nn, target_nn = GetInOut(stft, stft_wonr, doa_label, mic_idx)

        return data_nn, target_nn

    def __len__(self):
        return len(self.file_paths) * self.ndime

class TestDataSet_SE(Dataset):
# test dataset for speech enhancement
    def __init__(self, room_type, name_set, nt_context, data_mode):
        super(TestDataSet_SE, self).__init__()

        dirs = dirconfig()

        file_name = dirs['setting'] + 'setup_test' + room_type + '.txt'
        room_setup_set = read_room_setup_set(file_name)

        file_dir = dirs['sensig'] + 'test' + '/' + room_type
        self.file_paths = []
        self.mic_idxs = []
        self.doa_labels = []
        for setup_idx in name_set:
            fname = str(setup_idx) + '.wav'
            self.file_paths.append((os.path.join(file_dir, fname)))

            _, _, _, _, _, doa_label, _, _, _ = room_setup_set[setup_idx]
            self.doa_labels.append(doa_label)

            mic_idx = GetMicIdxFromRoomType(room_type)
            self.mic_idxs.append(mic_idx)

        self.context = nt_context
        self.data_mode = data_mode
        self.ndime = int(nt_context / nt_context)

    def __getitem__(self, idx):
        file_idx = idx // self.ndime
        dim_idx = idx % self.ndime + 0

        mic_idx = self.mic_idxs[file_idx]

        file_path = self.file_paths[file_idx]
        sensor_signal, _ = sf.read(file_path)

        file_path_en = file_path.replace("SenSig", "SenSigEnhanced_wonr_cirm0315")
        # file_path_en = file_path.replace("SenSig", "SenSigEnhanced_wonr_logmag")
        sensor_signal_en, _ = sf.read(file_path_en)

        file_path_wonr = file_path.replace("SenSig", "SenSigClean")
        sensor_signal_wonr, _ = sf.read(file_path_wonr)

        if self.data_mode == 'time':
            data_nn = np.concatenate((sensor_signal, sensor_signal_en), axis=1)
            target_nn = sensor_signal_wonr * 1

        elif self.data_mode == 'time-frequency':
            stft_temp = GetSTFT(sensor_signal)
            stft = stft_temp.astype('complex64')[:, dim_idx:dim_idx + self.context, :]
            stft_en_temp = GetSTFT(sensor_signal_en)
            stft_en = stft_en_temp.astype('complex64')[:, dim_idx:dim_idx + self.context, :]
            stft_wonr_temp = GetSTFT(sensor_signal_wonr)
            stft_wonr = stft_wonr_temp.astype('complex64')[:, dim_idx:dim_idx + self.context, :]

            mic_idx = mic_idx * np.ones((stft_wonr.shape[0], stft_wonr.shape[1], 1))
            mic_idx = mic_idx.astype('float32')

            data_nn = np.concatenate((stft, stft_en, mic_idx), axis=2)
            target_nn = stft_wonr*1

        return data_nn, target_nn

    def __len__(self):
        return len(self.file_paths)


if __name__ == '__main__':
    nt = 31
    nf = 128
    ns = 1
    dt = TrainDataSet('train', nt_context=31, mic_type='CIPIC')
    for i in range(100):
        k, v = dt[i]
        print('TrainDataset',i, k.shape, type(k), v.size, type(v))

    dt = TrainDataSet('val', nt_context=31, mic_type='CIPIC')
    for i in range(100):
        k, v = dt[i]
        print('ValDataset',i, k.shape, type(k), v.size, type(v))

    dt = TrainDataSet('test', nt_context=31, mic_type='CIPIC')
    for i in range(100):
        k, v = dt[i]
        print('TestDataset',i, k.shape, type(k), v.size, type(v))
