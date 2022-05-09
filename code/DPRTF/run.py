"""
	Function: Run training and test processes for source source localization on simulated dataset

    Reference:  B. Yang, H. Liu, and X. Li. "Learning deep direct-path relative transfer function for binaural sound source localization," 
    IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP), vol. 29,  pp. 3491 - 3503, 2021.
	Author:     Bing Yang
    History:    2022-04-25 - Initial version
    Copyright Bing Yang
"""

import os
# import cudnn
import scipy.io
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F 
from tensorboardX import SummaryWriter

from arguments import parse
from model import CRNN
from loss import Loss
from dataset import TrainDataSet, TestDataSet, TestDataSet_SE
from common.getData import read_room_setup_set
from common.config import dirconfig, room_exp_config 
from common.evaluation_metrics import run_estimate_evaluate_onehot, run_estimate_evaluate, STOI, WB_PESQ, SI_SDR, SDR_MY
from common.utils import LoadDPRTF, set_seed, get_learningrate
# from tqdm import tqdm

def train_epoch(args, model, loss, device, loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(loader): # tqdm(loader)
        data, target = data.to(device), target.to(device)
        output, rtf, rtf_gt_set = model(data)
        train_loss = loss(output, rtf, rtf_gt_set, target)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        global_step = (epoch - 1) * len(loader) + batch_idx
        writer['train'].add_scalar('loss', train_loss, global_step)

        output = F.softmax(output, dim=1)
        mae, acc = run_estimate_evaluate_onehot(onehot=output, doa_gt_idx=target[:, 0:1, 0], doa_candidate=args.doa_candidate, threshold=10)
        _, acc_class = run_estimate_evaluate_onehot(onehot=output, doa_gt_idx=target[:, 0:1, 0], doa_candidate=args.doa_candidate, threshold=5)
        writer['train'].add_scalar('mae', mae, global_step)
        writer['train'].add_scalar('acc', acc, global_step)
        writer['train'].add_scalar('acc_class', acc_class, global_step)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}, [{}/{} ({:.0f}%)], \tLoss: {:.6f}, \tlr: {}'.format(
                epoch, batch_idx * len(data), len(loader.dataset), 100. * batch_idx / len(loader), train_loss.item(), get_learningrate(optimizer)))

    writer['train'].add_scalar('epoch', epoch, global_step)
    writer['train'].add_scalar('lr', get_learningrate(optimizer), global_step)

def val_epoch(args, model, loss, device, loader, epoch, writer, val_test_mode, global_step):
    model.eval()
    val_loss = 0

    mae = 0
    acc = 0
    acc_class = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output, rtf, rtf_gt_set = model(data)
            loss_temp = loss(output, rtf, rtf_gt_set, target)

            val_loss += loss_temp
            output = F.softmax(output, dim=1)
            mae_temp, acc_temp = run_estimate_evaluate_onehot(onehot=output, doa_gt_idx=target[:, 0:1, 0], doa_candidate=args.doa_candidate, threshold=10)
            _, acc_class_temp = run_estimate_evaluate_onehot(onehot=output, doa_gt_idx=target[:, 0:1, 0], doa_candidate=args.doa_candidate, threshold=5)

            mae += mae_temp
            acc += acc_temp
            acc_class += acc_class_temp

    val_loss /= len(loader)
    mae /= len(loader)
    acc /= len(loader)
    acc_class /= len(loader)

    writer[val_test_mode].add_scalar('loss', val_loss, global_step)

    print('\nValidation Epoch: {}, \tLoss: {:.4f}, \tMAE: {:.1f}, ACC: {:.1f}%, ACC_Class: {:.1f}%\n'.format(epoch, val_loss, mae, acc*100, acc_class*100))

    writer[val_test_mode].add_scalar('mae', mae, global_step)
    writer[val_test_mode].add_scalar('acc', acc, global_step)
    writer[val_test_mode].add_scalar('acc_class', acc_class, global_step)

def test_epoch(args, model, device, loader):
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output, rtf, rtf_gt_set = model(data)
    return output, rtf, rtf_gt_set  # only one batch

def run_train_val(args, model, loss, device, kwargs):
    nt_context = args.time_frame

    data_train = TrainDataSet('train', nt_context, mic_type=args.mic_type, mic_match=args.mic_match, data_random=args.data_random)
    data_val = TrainDataSet('val', nt_context, mic_type=args.mic_type, mic_match=args.mic_match, data_random=False)
    data_test = TrainDataSet('test', nt_context, mic_type=args.mic_type, mic_match=args.mic_match, data_random=False)

    train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(dataset=data_val, batch_size=args.val_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.val_batch_size, shuffle=False, **kwargs)

    params = list(model.parameters()) + list(loss.parameters()) # for loss with learned parameters, e.g., center loss
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=0.0001)  # betas=(0.9, 0.999)

    train_writer = SummaryWriter(exp_dir + 'our_{}/train/'.format(args.time), 'train')
    val_writer = SummaryWriter(exp_dir + 'our_{}/val/'.format(args.time), 'val')
    test_writer = SummaryWriter(exp_dir + 'our_{}/test/'.format(args.time), 'test')
    writer = {'train': train_writer, 'val': val_writer, 'test': test_writer}

    for epoch in range(1, args.epochs + 1):
        if epoch <= 10:
            args.lr = 0.0001 * 5
        elif (epoch > 10):
            args.lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

        train_epoch(args, model, loss, device, train_loader, optimizer, epoch, writer)
        global_step = epoch * len(train_loader) - 1
        val_epoch(args, model, loss, device, val_loader, epoch, writer, 'val', global_step)
        val_epoch(args, model, loss, device, test_loader, epoch, writer, 'test', global_step)

        if (epoch % 2 == 0) & (epoch >=  8):
            torch.save(model.state_dict(), exp_dir + 'our_{}/model_{}.pth'.format(args.time, epoch))

def run_test(args, model, device, kwargs, EPS = 1e-10):
    ################ parameters #################
    """
        time_flag + epoch   - define the evaluated model
        use_ins             - define the evaluated instances - 'all' instances or 'specific' instance
        test_mode           - define the test model for evaluation
                              'DOAEst': DOA estimation 
                              'DOAEst_En': DOA estimation with/without speech enhancement
                              'En_Mo': Speech enhancement (momaural metrics)
                              'En_Bi': Speech enhancement (binaural metrics)
    """
    time_flag = '00000001' 
    epoch = 52 
    use_ins = 'all' 
    test_mode = 'DOAEst' 
    #############################################

    print('Test mode: ' + test_mode)
    nt_context = args.time_frame
    set_flag = 'test'
    
    room_setup, exp_setup = room_exp_config(mic_type=args.mic_type)
    if args.mic_match: # array-matched condition
        room_type_set = [
            'H021_507030', 'H021_608038', 'H021_404027',
            'H040_507030', 'H040_608038', 'H040_404027',
        ] 
    else: # array-mismatched condition
        room_type_set = [
            'H021_507030', 'H003_507030', 'H040_507030',
            'H008_608038', 'H009_608038', 'H033_608038',
            'H019_404027', 'H020_404027', 'H027_404027',
        ]

    if test_mode == 'DOAEst':
        print('Load Model:', time_flag, epoch)
        model.load_state_dict(torch.load(exp_dir + 'our_{}/model_{}.pth'.format(time_flag, epoch), map_location=device))

    ################### parameters ####################
    if  use_ins == 'specific':
        room_type_set = [room_type_set[0]]  # 0: 'H021_507030', 2: 'H040_507030'
        dist_idx_set = [2]      # 1: 150, 2: 210
        doa_idx_set = [18]      # 12: 0, 18: 30, 24:80
        rtsnr_idx_set = [2]     # 2: 600ms 5dB, 3: 600ms 0dB, 4: 600ms -5dB, 7: 800ms 5dB
        noise_type_idx_set = [1]  # 1: babble
    ###################################################

    for room_type in room_type_set:
        print('** Room Type - {} **'.format(room_type))
        noise_type_set = room_setup[room_type]['n_type']
        dist_set = room_setup[room_type]['dist']
        doa_set = room_setup[room_type]['doa']
        rt_set = room_setup[room_type]['rt']
        snr_set = room_setup[room_type]['snr']
        ins_num = int(room_setup[room_type]['num'] * exp_setup['num_scale'][set_flag])

        n_noise_type = len(noise_type_set)
        n_dist = len(dist_set)
        n_doa = len(doa_set)
        n_rt = len(rt_set)

        if use_ins == 'all':
            dist_idx_set = range(n_dist)
            doa_idx_set = range(n_doa)
            rtsnr_idx_set = range(n_rt)
            noise_type_idx_set = range(n_noise_type)

        accmae = np.zeros((3, n_dist, n_doa, n_rt, n_noise_type)) #[acc_10, acc_5, mae]
        pesqstoisdr = np.zeros((8, n_dist, n_doa, n_rt, n_noise_type)) # [pesq_ori, stoi_ori, sisdr_ori, sdrmy_ori, pesq_en, stoi_en, sisdr_en, sdrmy_en]
        rtferror = np.zeros((6, n_dist, n_doa, n_rt, n_noise_type)) # [rtf_ori, iid_ori, ipd_ori, rtf_en, iid_en, ipd_en]

        # Classify test instances into different acoustic conditions
        filename = setup_dir + 'setup_' + set_flag + room_type + '.txt'
        room_setup_set = read_room_setup_set(filename)

        setup_detailed = np.zeros((n_dist, n_doa, n_rt, n_noise_type, ins_num), dtype='uint32')
        setup_ins_counter = np.ones((n_dist, n_doa, n_rt, n_noise_type), dtype='uint16') * (-1)
        for setup_idx in range(0, len(room_setup_set), 1):

            # determine acoustic condition
            _, _, rt, snr, noise_type, doa_idx, dist, _, _ = room_setup_set[setup_idx]
            for index, value in enumerate(rt_set):
                if (snr_set[index] == snr) & (rt_set[index] == rt):
                    rtsnr_idx = index

            noise_type_idx = noise_type_set.index(noise_type)
            dist_idx = dist_set.index(dist)

            # collect the indexes of settings
            setup_ins_counter[dist_idx, doa_idx, rtsnr_idx, noise_type_idx] = setup_ins_counter[dist_idx, doa_idx, rtsnr_idx, noise_type_idx] + 1
            ins_idx = setup_ins_counter[dist_idx, doa_idx, rtsnr_idx, noise_type_idx]
            setup_detailed[dist_idx, doa_idx, rtsnr_idx, noise_type_idx, ins_idx] = setup_idx

        # Evaluation for each acoustic condition
        # use the instances belonging to the same acoustic condition as a batch
        for dist_idx in dist_idx_set:
            print('distance_id: {}'.format(dist_idx))
            for doa_idx in doa_idx_set:
                for rtsnr_idx in rtsnr_idx_set:
                    for noise_type_idx in noise_type_idx_set:
                        setup_idx_used_set = setup_detailed[dist_idx, doa_idx, rtsnr_idx, noise_type_idx, :]

                        if test_mode == 'DOAEst':
                            data_test = TestDataSet(room_type=room_type, name_set=setup_idx_used_set, nt_context=nt_context)

                            test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=len(setup_idx_used_set), shuffle=False, **kwargs)
                            output_gpu, rtf_gpu, rtf_gt_gpu = test_epoch(args, model, device, test_loader)

                            doa_gt_idx_set = doa_idx * torch.ones((output_gpu.shape[0], 1)).cuda()
                            mae, acc = run_estimate_evaluate_onehot(onehot=output_gpu,
                                    doa_gt_idx=doa_gt_idx_set, doa_candidate=args.doa_candidate, threshold=10)  # onehot: (nbatch, ndoa)
                            _, acc_5 = run_estimate_evaluate_onehot(onehot=output_gpu,
                                    doa_gt_idx=doa_gt_idx_set, doa_candidate=args.doa_candidate, threshold=5)
                        
                        elif test_mode == 'DOAEst_En':
                            data_test = TestDataSet_SE(room_type=room_type, name_set=setup_idx_used_set, nt_context=nt_context, data_mode='time-frequency')

                            test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=len(setup_idx_used_set), shuffle=False, **kwargs)

                            for data, target in test_loader:
                                data, target = data.to(device), target.to(device)
                                mic_idx = data[:, 0, 0, -1]
                                nb, nf, nt, _ = data.shape
                                doa_idx_ = doa_idx * torch.ones(nb).cuda()
                                rtf_gt = LoadDPRTF(dir=data_dir, mic_type=args.mic_type, mic_idx=mic_idx, doa_idx=doa_idx_, full_fre=False)  # (nb, 3nf)
                                rtf_gt_gpu = rtf_gt[:, :, np.newaxis].expand(nb, nf * 3, nt)  # (nb, 3nf, nt)

                                stft_ori = data[:, :, :, 0:2]
                                stft_en = data[:, :, :, 2:4]
                                stft_wonr = target[:, :, :, 0:2]

                                stft = stft_en
                                logmag = torch.log10(torch.abs(stft) + EPS)
                                phase = torch.angle(stft)
                                iid = logmag[:, :, :, 1] - logmag[:, :, :, 0]
                                ipd = torch.cat((torch.sin(phase[:, :, :, 1] - phase[:, :, :, 0]), torch.cos(phase[:, :, :, 1] - phase[:, :, :, 0])), axis=1)
                                rtf_gpu = torch.cat((iid, ipd), axis=1)  # (nb,3nf,nt)

                                ##################################################
                                # logmag = torch.log10(torch.abs(stft_ori) + EPS)
                                # logmag_en = torch.log10(torch.abs(stft_en) + EPS)
                                # logmag_clean = torch.abs(stft_wonr) + EPS
                                # file_name = exp_dir + 'withME/logmag_room{}_doa{}_rtsnr{}{}{}_dist{}_nt{}.mat'.format(room_type,\
                                #     doa_set[doa_idx], rt_set[rtsnr_idx], snr_set[rtsnr_idx], noise_type_set[noise_type_idx], \
                                #     dist_set[dist_idx], nt_context)
                                # scipy.io.savemat(file_name, {'logmag': logmag.cpu().numpy(),
                                #                              'logmag_en':logmag_en.cpu().numpy(),
                                #                              'logmag_clean':logmag_clean.cpu().numpy()})
                                # print(a)
                                ##################################################

                            doa_gt_idx_set = doa_idx * torch.ones((rtf_gpu.shape[0], 1)).cuda()
                            mic_idx_set = data[:rtf_gpu.shape[0], -1:, 0, 0]
                            w_est_set = torch.ones((rtf_gpu.shape[0], int(rtf_gpu.shape[1]/3), nt_context)).cuda()
                            mae, acc = run_estimate_evaluate(w_est=w_est_set, rtf_est=rtf_gpu,
                                               doa_gt_idx=doa_gt_idx_set, doa_candidate=args.doa_candidate,
                                               mic_idx=mic_idx_set, mic_type=args.mic_type, threshold=10)  # rtf_est_set: nb*nf*nt
                            _, acc_5 = run_estimate_evaluate(w_est=w_est_set, rtf_est=rtf_gpu,
                                               doa_gt_idx=doa_gt_idx_set, doa_candidate=args.doa_candidate,
                                               mic_idx=mic_idx_set, mic_type=args.mic_type, threshold=5)

                        elif (test_mode == 'En_Mo'):
                            data_test = TestDataSet_SE(room_type=room_type, name_set=setup_idx_used_set, nt_context=nt_context, data_mode = 'time')

                            test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=len(setup_idx_used_set), shuffle=False, **kwargs)

                            for data, target in test_loader:
                                data, target = data.to(device), target.to(device)
                                sig_clean_set = target[:,:,0:2].cpu().numpy() # (nb, nsample, nch)
                                sig_ori_set = data[:,:,0:2].cpu().numpy()
                                sig_en_set = data[:,:,2:4].cpu().numpy()

                            nb = data.shape[0]
                            pesqstoisdr_temp = np.zeros((nb, 8)).astype('float')
                            # [pesq_ori, stoi_ori, sisdr_ori, sdrmy_ori, pesq_en, stoi_en, sisdr_en, sdrmy_en,]
                            valid_flag = np.zeros((nb, 8))
                            for b_idx in range(0, nb, 1):
                                ch_idx = 0
                                sig_clean = sig_clean_set[b_idx, :, ch_idx]
                                sig_ori = sig_ori_set[b_idx, :, ch_idx]
                                sig_en = sig_en_set[b_idx, :, ch_idx]

                                pesq_ori = WB_PESQ(ref=sig_clean, est=sig_ori)
                                stoi_ori = STOI(ref=sig_clean, est=sig_ori)
                                sisdr_ori = SI_SDR(reference=sig_clean, estimation=sig_ori)
                                sdrmy_ori = SDR_MY(reference=sig_clean, estimation=sig_ori)
                                pesq_en = WB_PESQ(ref=sig_clean, est=sig_en)
                                stoi_en = STOI(ref=sig_clean, est=sig_en)
                                sisdr_en = SI_SDR(reference=sig_clean, estimation=sig_en)
                                sdrmy_en = SDR_MY(reference=sig_clean, estimation=sig_en)
                                if (pesq_ori == -1) | (pesq_en == -1):
                                    print('warning pesq ~')
                                elif (stoi_ori == 1e-5) | (stoi_en == 1e-5):
                                    print('warning stoi ~')
                                else:
                                    valid_flag[b_idx, :] = 1
                                    pesqstoisdr_temp[b_idx, :] = [pesq_ori, stoi_ori, sisdr_ori, sdrmy_ori, pesq_en, stoi_en, sisdr_en, sdrmy_en]

                            pesqstoisdr[:, dist_idx, doa_idx, rtsnr_idx, noise_type_idx] = \
                                np.sum(pesqstoisdr_temp * valid_flag, axis=0) / (np.sum(valid_flag, axis=0)+0.0000001)

                        elif (test_mode == 'En_Bi'):
                            data_test = TestDataSet_SE(room_type=room_type, name_set=setup_idx_used_set, nt_context=nt_context, mic_type=args.mic_type, data_mode = 'time-frequency')

                            test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=len(setup_idx_used_set), shuffle=False, **kwargs)

                            for data, target in test_loader:
                                data, target = data.to(device), target.to(device)
                                mic_idx = data[:, 0, 0, -1]
                                nb, nf, nt, _ = data.shape
                                doa_idx_ = doa_idx*torch.ones(nb).cuda()
                                rtf_gt = LoadDPRTF(dir=data_dir, mic_type=args.mic_type, mic_idx=mic_idx, doa_idx=doa_idx_, full_fre=False) # (nb, 3nf)
                                rtf_gt = rtf_gt[:,:,np.newaxis].expand(nb, nf*3, nt) # (nb, 3nf, nt)

                                stft_ori = data[:,:,:,0:2]
                                stft_en = data[:,:,:,2:4]
                                stft_wonr = target[:,:,:,0:2]

                                iid_gt = rtf_gt[:, 0:nf, :]
                                ipd_gt = rtf_gt[:, nf:nf * 3, :]

                                logmag_ori = torch.log10(torch.abs(stft_ori)+EPS)
                                phase_ori = torch.angle(stft_ori)
                                iid_ori = logmag_ori[:, :, :, 1] - logmag_ori[:, :, :, 0]
                                ipd_ori = torch.cat((torch.sin(phase_ori[:, :, :, 1] - phase_ori[:, :, :, 0]),
                                                     torch.cos(phase_ori[:, :, :, 1] - phase_ori[:, :, :, 0])), axis=1)
                                rtf_ori = torch.cat((iid_ori, ipd_ori), axis=1) # (nb,3nf,nt)

                                logmag_en = torch.log10(torch.abs(stft_en)+EPS)
                                phase_en = torch.angle(stft_en)
                                iid_en = logmag_en[:, :, :, 1] - logmag_en[:, :, :, 0]
                                ipd_en = torch.cat((torch.sin(phase_en[:, :, :, 1] - phase_en[:, :, :, 0]),
                                                    torch.cos(phase_en[:, :, :, 1] - phase_en[:, :, :, 0])), axis = 1)
                                rtf_en = torch.cat((iid_en, ipd_en), axis=1) # (nb,3nf,nt)
                                th = 1e-5
                                tf_valid_flag = (torch.mean(torch.abs(stft_wonr), dim=3) > th) * 1 # (nb,nf,nt)
                                tf_valid_flag2 = torch.cat((tf_valid_flag, tf_valid_flag),axis=1)
                                tf_valid_flag3 = torch.cat((tf_valid_flag, tf_valid_flag2),axis=1)
                                num_tf_valid = torch.sum(tf_valid_flag)
                                rtf_diff = [ # (rtf_ori, iid_ori, ipd_ori, rtf_en, iid_en, ipd_en)
                                    torch.sum(torch.pow(rtf_ori - rtf_gt,2)*tf_valid_flag3)/num_tf_valid,
                                    torch.sum(torch.pow(iid_ori - iid_gt,2)*tf_valid_flag)/num_tf_valid,
                                    torch.sum(torch.pow(ipd_ori - ipd_gt,2)*tf_valid_flag2)/num_tf_valid,
                                    torch.sum(torch.pow(rtf_en - rtf_gt,2)*tf_valid_flag3)/num_tf_valid,
                                    torch.sum(torch.pow(iid_en - iid_gt,2)*tf_valid_flag)/num_tf_valid,
                                    torch.sum(torch.pow(ipd_en - ipd_gt,2)*tf_valid_flag2)/num_tf_valid
                                ] # MSE

                                rtferror[:, dist_idx, doa_idx, rtsnr_idx, noise_type_idx] = rtf_diff

                        if (test_mode == 'DOAEst') | (test_mode == 'DOAEst_En'):
                            rtf_cpu = rtf_gpu.cpu().numpy()
                            rtf_gt_cpu = rtf_gt_gpu[0,:,:].cpu().numpy()
                            mae = mae.cpu().numpy()
                            acc = acc.cpu().numpy()
                            acc_5 = acc_5.cpu().numpy()
                            accmae[:, dist_idx, doa_idx, rtsnr_idx, noise_type_idx] = [acc, acc_5, mae]

        if use_ins == 'all':
            if test_mode == 'DOAEst':
                file_name = exp_dir + 'our_{}/maeacc_model_{}_room_{}.mat'.format(time_flag, epoch, room_type)
            elif test_mode == 'DOAEst_En':
                file_name = exp_dir + 'withME/maeacc_room_{}.mat'.format(room_type)
                # file_name = exp_dir + 'withME/wo_maeacc_room_{}.mat'.format(room_type)
            elif (test_mode == 'En_Mo'):
                file_name = exp_dir + 'withME/pesqstoisdr_room_{}.mat'.format(room_type)
                exist_temp = os.path.exists(file_name)
                if exist_temp==False:
                    os.makedirs(file_name)
                    print('make dir: ' + file_name)
                scipy.io.savemat(file_name, {'pesqstoisdr': pesqstoisdr})
                print('Room-{}: pesq_ori [{:.2f}], stoi_ori [{:.2f}%], sisdr_ori [{:.2f}dB], sdrmy_ori [{:.2f}dB], '
                      'pesq_en [{:.2f}], stoi_en [{:.2f}%], sisdr_en [{:.2f}dB], sdrmy_en [{:.2f}dB], '.format(room_type,
                       np.mean(pesqstoisdr[0, :, :, :, :]), np.mean(pesqstoisdr[1, :, :, :, :] * 100),
                       np.mean(pesqstoisdr[2, :, :, :, :]), np.mean(pesqstoisdr[3, :, :, :, :]),
                       np.mean(pesqstoisdr[4, :, :, :, :]), np.mean(pesqstoisdr[5, :, :, :, :] * 100),
                       np.mean(pesqstoisdr[6, :, :, :, :]), np.mean(pesqstoisdr[7, :, :, :, :]), ))
            elif (test_mode == 'En_Bi'):
                file_name = exp_dir + 'withME/rtferror_room_{}.mat'.format(room_type)
                exist_temp = os.path.exists(file_name)
                if exist_temp==False:
                    os.makedirs(file_name)
                    print('make dir: ' + file_name)
                scipy.io.savemat(file_name, {'rtferror': rtferror})
                print('Room-{}: rtf [{:.2f}], iid [{:.2f}], ipd [{:.2f}], rtf_en [{:.2f}], iid_en [{:.2f}], ipd_en [{:.2f}]]'.format(room_type,
                      np.mean(rtferror[0, :, :, :, :]), np.mean(rtferror[1, :, :, :, :]), np.mean(rtferror[2, :, :, :, :]),
                      np.mean(rtferror[3, :, :, :, :]), np.mean(rtferror[4, :, :, :, :]), np.mean(rtferror[5, :, :, :, :]) ))

            if (test_mode == 'DOAEst') | (test_mode == 'DOAEst_En'):
                scipy.io.savemat(file_name, {'accmae': accmae})
                print('Room-{}: acc-10 [{:.2f}%], acc-5 [{:.2f}%], mae [{:.2f} deg]'.format(room_type, np.mean(
                    accmae[0, :, :, :, :]) * 100, np.mean(accmae[1, :, :, :, :]) * 100, np.mean(accmae[2, :, :, :, :])))
        
        elif use_ins == 'specific':
            if test_mode == 'DOAEst':
                file_name = exp_dir + 'our_{}/rtf_room{}_doa{}_rtsnr{}{}{}_dist{}_nt{}.mat'.format(time_flag, room_type,\
                        doa_set[doa_idx], rt_set[rtsnr_idx], snr_set[rtsnr_idx], noise_type_set[noise_type_idx], \
                        dist_set[dist_idx], nt_context)
            elif test_mode == 'DOAEst_En':
                file_name = exp_dir + 'withME/wo_rtf_room{}_doa{}_rtsnr{}{}{}_dist{}_nt{}.mat'.format(room_type, \
                         doa_set[doa_idx], rt_set[rtsnr_idx], snr_set[rtsnr_idx], noise_type_set[noise_type_idx], \
                         dist_set[dist_idx], nt_context)

            if (test_mode == 'DOAEst') | (test_mode == 'DOAEst_En'):
                scipy.io.savemat(file_name, {'rtf_est': rtf_cpu, 'rtf_gt': rtf_gt_cpu, 'doa_idx': doa_idx})
                print('doa:{}, rt:{}, snr:{}, noise_type:{}, distance:{}'.format(doa_set[doa_idx], \
                            rt_set[rtsnr_idx], snr_set[rtsnr_idx], noise_type_set[noise_type_idx], dist_set[dist_idx]))
                print('Room-{}: acc-10 [{:.2f}%], acc-5 [{:.2f}%], mae [{:.2f} deg]'.format(room_type, np.mean(
                    acc) * 100, np.mean(acc_5) * 100, np.mean(mae)))


if __name__ == "__main__":
    
    ger_dirs = dirconfig()
    code_dir = ger_dirs['code']
    data_dir = ger_dirs['data']
    exp_dir = ger_dirs['exp']
    setup_dir = ger_dirs['setting']
    os.chdir(code_dir)
    args = parse()

    if args.mic_type == 'MIT':
        args.doa_candidate = list(range(-90, 91, 5))
    if args.mic_type == 'CIPIC':
        args.doa_candidate =  list([-80, -65, -55]) + list(range(-45, 46, 5)) + list([55, 65, 80])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 20, 'pin_memory': True}  if use_cuda else {}

    set_seed(args.seed)

    model = CRNN(mic_type=args.mic_type).to(device)
    loss = Loss(mic_type=args.mic_type, mode='mse')

    if (not args.test):
        print('Training stage: ', 'MIC type: '+args.mic_type, ', MIC match: '+ str(args.mic_match), ', # Parameters:', sum(param.numel() for param in model.parameters()))
        run_train_val(args, model, loss, device, kwargs)
    elif args.test:
        print('Test stage: ', 'MIC type: '+args.mic_type, ', MIC match: '+ str(args.mic_match), ',# Parameters:', sum(param.numel() for param in model.parameters()))
        run_test(args, model, device, kwargs)

