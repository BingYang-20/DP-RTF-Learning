""" 
    Function:   Define some optional configurations
	Author:     Bing Yang
    Copyright Bing Yang
"""

import os

def dirconfig():
    """ Function: get directories of code, data and experimental results
    """ 
    work_dir = r'~'
    work_dir = os.path.abspath(os.path.expanduser(work_dir))
    dirs = {}

    dirs['code'] = work_dir + '/code/'
    dirs['data'] = work_dir + '/data/'
    dirs['exp'] = work_dir + '/exp/'

    dirs['hrir'] = dirs['data'] + 'HRIR/'
    dirs['rir'] = dirs['data'] + 'RIR/'
    dirs['sousig'] = dirs['data'] + 'SouSig/TIMIT/'
    dirs['noisig'] = dirs['data'] + 'NoiSig/DiffNoise/'

    gerdata_dir = dirs['data'] + 'GerData/'
    dirs['ger'] = gerdata_dir
    dirs['sensig'] = gerdata_dir + 'SenSig/'
    # dirs['sensig'] = gerdata_dir + 'SenSigEnhanced/'
    dirs['sensigclean'] = gerdata_dir + 'SenSigClean/'
    dirs['sensigclean_n'] = gerdata_dir + 'SenSigClean_n/'
    dirs['setting'] = gerdata_dir + 'Setting/'
    dirs['rtf'] = gerdata_dir + 'RTF/'

    return dirs

def room_exp_config(mic_type):
    """ Function: get room and experimental configurations
        Args:       mic_type    - the type of microphone array
                                 'MIT': array on the MIT KEMAR dummy head
                                 'CIPIC': array on the CIPIC head subject
        Returns:    room_setup  - room acoustic setup
                    exp_setup   - experimental setup
        
    """ 
    if mic_type == 'MIT':
        doa_all = list(range(-90, 91, 5))
        doa =  list(range(-90, 91, 5))
        n_type = ['Nw', 'Nb', 'Nf1']
        snr_train = list(range(-5, 21, 5))
        snr_test = [15, 10, 5, 0, -5, 5, 5, 5]
        rt_test = [600, 600, 600, 600, 600, 200, 400, 800]

        room_setup = {
            '708050': {'rt': [0,170,340,510,680,850],   'snr': snr_train,       'n_type': n_type,
                       'doa_all': doa_all, 'doa': doa,  'dist':['150','200','250','300','340'],   'num':1 },
            '606035': {'rt': [0,220,440,660,880],       'snr': snr_train,       'n_type': n_type,
                       'doa_all': doa_all, 'doa': doa,  'dist':['175','225'],   'num':1 },
            '405530': {'rt': [0,250,500,750],           'snr': snr_train,       'n_type': n_type,
                       'doa_all': doa_all, 'doa': doa,  'dist':['050','100'],   'num':1 },
            '383025': {'rt': [0,300,600,900],           'snr': snr_train,       'n_type': n_type,
                       'doa_all': doa_all, 'doa': doa,  'dist':['075','125'],   'num':1 },
            '608038': {'rt': rt_test, 'snr': snr_test,  'n_type': n_type,
                       'doa_all': doa_all, 'doa': doa,  'dist': ['060', '150', '240', '330'], 'num': 1},
            '507030': {'rt': rt_test,  'snr': snr_test, 'n_type': n_type,
                       'doa_all': doa_all, 'doa': doa,  'dist': ['070','140','210'],         'num':1  },
            '404027': {'rt': rt_test,  'snr': snr_test, 'n_type': n_type,
                       'doa_all': doa_all, 'doa': doa,  'dist': ['080','130'],               'num':1  },
            }
        exp_setup = {
            'room': {'train': ['708050', '606035', '405530', '383025'],
                     'val': ['708050', '606035', '405530', '383025'],
                     'test': ['608038', '507030', '404027']},
            'noise': {'train': [0, 59], 'val': [60, 74], 'test': [75, 114]}, # 115=60+15+40
            'source': {'train': [0, 3629], 'val': [3630, 4620-1], 'test': [4620, 4620 + 1680-1]}, 
            'num_scale': {'train': 900, 'val': 180, 'test': 20},
            'doa': doa,
        }

    elif mic_type == 'CIPIC':
        """ Description of some parameters: 
            'H021_707040' - the head ID is 21, and the room size is 7m x 7m x 4m
            doa_all     - DOA candidates for RIR files (some candidates are not valid)
            doa         - DOA candidates for loaded files
            n_type      - noise type: 'Nw' for white noise, 'Nb' for babble noise, 'Nf' for factory noise
            snr_train   - SNR range for training
            snr_test    - SNR range for test
            rt_test     - RT60 range for test
            'dist'      - distance between sources and mirophone array
        """ 
        doa_all = list(range(-90, 91, 5)) 
        doa = list([-80, -65, -55]) + list(range(-45, 46, 5)) + list([55, 65, 80]) 
        n_type = ['Nw', 'Nb', 'Nf'] 
        snr_train = list(range(-5, 21, 5)) 
        snr_test = [15, 10, 5, 0, -5, 5, 5, 5] 
        rt_test = [600, 600, 600, 600, 600, 200, 400, 800] 

        room_setup = {
                'H021_707040': {'rt': [0, 180, 360, 540, 720, 900], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['100', '200', '300'], 'num': 3},
                'H040_707040': {'rt': [0, 180, 360, 540, 720, 900], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['100', '200', '300'], 'num': 3},
                'H010_707040': {'rt': [0, 180, 360, 540, 720, 900], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['100', '200', '300'], 'num': 1},
                'H028_707040': {'rt': [0, 180, 360, 540, 720, 900], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['100', '200', '300'], 'num': 1},
                'H124_707040': {'rt': [0, 180, 360, 540, 720, 900], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['100', '200', '300'], 'num': 1},

                'H021_806536': {'rt': [0, 270, 540, 810], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['150', '290'], 'num': 3},
                'H040_806536': {'rt': [0, 270, 540, 810], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['150', '290'], 'num': 3},
                'H011_806536': {'rt': [0, 270, 540, 810], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['150', '290'], 'num': 1},
                'H012_806536': {'rt': [0, 270, 540, 810], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['150', '290'], 'num': 1},
                'H165_806536': {'rt': [0, 270, 540, 810], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['150', '290'], 'num': 1},

                'H021_506028': {'rt': [0, 210, 420, 630, 840], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['050', '150', '250'], 'num': 3},
                'H040_506028': {'rt': [0, 210, 420, 630, 840], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['050', '150', '250'], 'num': 3},
                'H044_506028': {'rt': [0, 210, 420, 630, 840], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['050', '150', '250'], 'num': 1},
                'H127_506028': {'rt': [0, 210, 420, 630, 840], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['050', '150', '250'], 'num': 1},
                'H156_506028': {'rt': [0, 210, 420, 630, 840], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['050', '150', '250'], 'num': 1},

                'H021_456031': {'rt': [0, 260, 520, 780], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['080', '220'], 'num': 3},
                'H040_456031': {'rt': [0, 260, 520, 780], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['080', '220'], 'num': 3},
                'H015_456031': {'rt': [0, 260, 520, 780], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['080', '220'], 'num': 1},
                'H017_456031': {'rt': [0, 260, 520, 780], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['080', '220'], 'num': 1},
                'H018_456031': {'rt': [0, 260, 520, 780], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['080', '220'], 'num': 1},

                'H021_708050': {'rt': [0, 170, 340, 510, 680, 850], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['150','200','250','300','340'], 'num': 3},
                'H040_708050': {'rt': [0, 170, 340, 510, 680, 850], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['150','200','250','300','340'], 'num': 3},
                'H048_708050': {'rt': [0, 170, 340, 510, 680, 850], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['150','200','250','300','340'], 'num': 1},
                'H050_708050': {'rt': [0, 170, 340, 510, 680, 850], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['150','200','250','300','340'], 'num': 1},
                'H051_708050': {'rt': [0, 170, 340, 510, 680, 850], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['150','200','250','300','340'], 'num': 1},

                'H021_679046': {'rt': [0, 280, 560, 840], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['300', '360'], 'num': 3},
                'H040_679046': {'rt': [0, 280, 560, 840], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['300', '360'], 'num': 3},
                'H058_679046': {'rt': [0, 280, 560, 840], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['300', '360'], 'num': 1},
                'H059_679046': {'rt': [0, 280, 560, 840], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['300', '360'], 'num': 1},
                'H060_679046': {'rt': [0, 280, 560, 840], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['300', '360'], 'num': 1},

                'H021_606035': {'rt': [0, 220, 440, 660, 880], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['175', '225'], 'num': 3},
                'H040_606035': {'rt': [0, 220, 440, 660, 880], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['175', '225'], 'num': 3},
                'H061_606035': {'rt': [0, 220, 440, 660, 880], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['175', '225'], 'num': 1},
                'H065_606035': {'rt': [0, 220, 440, 660, 880], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['175', '225'], 'num': 1},
                'H119_606035': {'rt': [0, 220, 440, 660, 880], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['175', '225'], 'num': 1},

                'H021_406032': {'rt': [0, 240, 480, 720], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['075', '125'], 'num': 3},
                'H040_406032': {'rt': [0, 240, 480, 720], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['075', '125'], 'num': 3},
                'H126_406032': {'rt': [0, 240, 480, 720], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['075', '125'], 'num': 1},
                'H131_406032': {'rt': [0, 240, 480, 720], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['075', '125'], 'num': 1},
                'H133_406032': {'rt': [0, 240, 480, 720], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['075', '125'], 'num': 1},

                'H021_405530': {'rt': [0, 250, 500, 750], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['050', '100'], 'num': 3},
                'H040_405530': {'rt': [0, 250, 500, 750], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['050', '100'], 'num': 3},
                'H134_405530': {'rt': [0, 250, 500, 750], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['050', '100'], 'num': 1},
                'H135_405530': {'rt': [0, 250, 500, 750], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['050', '100'], 'num': 1},
                'H137_405530': {'rt': [0, 250, 500, 750], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['050', '100'], 'num': 1},

                'H021_538038': {'rt': [0, 230, 460, 690, 920], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['180', '240'], 'num': 3},
                'H040_538038': {'rt': [0, 230, 460, 690, 920], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['180', '240'], 'num': 3},
                'H147_538038': {'rt': [0, 230, 460, 690, 920], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['180', '240'], 'num': 1},
                'H148_538038': {'rt': [0, 230, 460, 690, 920], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['180', '240'], 'num': 1},
                'H152_538038': {'rt': [0, 230, 460, 690, 920], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['180', '240'], 'num': 1},

                'H021_383025': {'rt': [0, 300, 600, 900], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['075', '125'], 'num': 3},
                'H040_383025': {'rt': [0, 300, 600, 900], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['075', '125'], 'num': 3},
                'H153_383025': {'rt': [0, 300, 600, 900], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['075', '125'], 'num': 1},
                'H154_383025': {'rt': [0, 300, 600, 900], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['075', '125'], 'num': 1},
                'H155_383025': {'rt': [0, 300, 600, 900], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['075', '125'], 'num': 1},

                'H021_503229': {'rt': [0, 190, 380, 570, 760, 950], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['060', '120'], 'num': 3},
                'H040_503229': {'rt': [0, 190, 380, 570, 760, 950], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['060', '120'], 'num': 3},
                'H158_503229': {'rt': [0, 190, 380, 570, 760, 950], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['060', '120'], 'num': 1},
                'H162_503229': {'rt': [0, 190, 380, 570, 760, 950], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['060', '120'], 'num': 1},
                'H163_503229': {'rt': [0, 190, 380, 570, 760, 950], 'snr': snr_train, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['060', '120'], 'num': 1},

                'H021_608038': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['060', '150', '240', '330'], 'num': 1},
                'H040_608038': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['060', '150', '240', '330'], 'num': 1},
                'H008_608038': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['060', '150', '240', '330'], 'num': 1},
                'H009_608038': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['060', '150', '240', '330'], 'num': 1},
                'H033_608038': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['060', '150', '240', '330'], 'num': 1},

                'H021_507030': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['070', '140', '210'], 'num': 1},
                'H003_507030': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['070', '140', '210'], 'num': 1},
                'H040_507030': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['070', '140', '210'], 'num': 1},

                'H021_404027': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['080', '130'], 'num': 1},
                'H040_404027': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['080', '130'], 'num': 1},
                'H019_404027': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['080', '130'], 'num': 1},
                'H020_404027': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['080', '130'], 'num': 1},
                'H027_404027': {'rt': rt_test, 'snr': snr_test, 'n_type': n_type,
                                'doa_all': doa_all, 'doa': doa, 'dist': ['080', '130'], 'num': 1},
            }

        exp_setup = {
                'room': {'train': [
                            # 'H021_707040', 'H021_806536', 'H021_506028', 'H021_456031', 'H021_708050', 'H021_679046',
                            # 'H021_405530', 'H021_538038', 'H021_383025', 'H021_503229',
                            # 'H040_707040', 'H040_806536', 'H040_506028', 'H040_456031', 'H040_708050', 'H040_679046',
                            # 'H040_405530', 'H040_538038', 'H040_383025', 'H040_503229',
                            'H010_707040', 'H028_707040', 'H124_707040', 'H011_806536', 'H012_806536', 'H165_806536',
                            'H044_506028', 'H127_506028', 'H156_506028', 'H015_456031', 'H017_456031', 'H018_456031',
                            'H048_708050', 'H050_708050', 'H051_708050', 'H058_679046', 'H059_679046', 'H060_679046',
                            'H134_405530', 'H135_405530', 'H137_405530', 'H147_538038', 'H148_538038', 'H152_538038',
                            'H153_383025', 'H154_383025', 'H155_383025', 'H158_503229', 'H162_503229', 'H163_503229',
                            ],
                         'val': [
                             # 'H021_606035', 'H021_406032',
                             # 'H040_606035', 'H040_406032',
                             'H061_606035', 'H065_606035', 'H119_606035',
                             'H126_406032', 'H131_406032', 'H133_406032',
                            ],
                         'test': [
                            'H008_608038', 'H009_608038', 'H033_608038',
                            'H021_507030', 'H003_507030', 'H040_507030',
                            'H019_404027', 'H020_404027', 'H027_404027',
                            # 'H021_608038', 'H021_404027',
                            # 'H040_608038', 'H040_404027',
                            ],
                         },
                'noise': {'train': [0, 59], 'val': [0, 14], 'test': [0, 39]},   # 115=60+15+40
                'source': {'train': [0, 3629], 'val': [0, 990 - 1], 'test': [0, 1680 - 1]},  # TIMIT, train: 0-3629, 3630-4619, test: 0-1680
                'num_scale': {'train': 240, 'val': 240, 'test': 20},
                'doa': doa,
            }    
            # instance number of training and validation = nroom * nhead/room * ndoa * num_scale * num
            # instance number of test = (nroom,ndist) * nhead/room * nrtsnr * ndoa * nnoise_type * num_scale * num
    
    return room_setup, exp_setup

if __name__ == '__main__':
    dirs = dirconfig()
    room_setup, exp_setup = room_exp_config(mic_type='MIT')
    print(dirs, room_setup, exp_setup)
