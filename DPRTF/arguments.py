""" 
    Function:   Define some optional arguments
	Author:     Bing Yang
    Copyright Bing Yang
"""

import argparse
import time

def parse():
    """ Function: Define optional arguments for sound source localization
    """
    parser = argparse.ArgumentParser(description='Sound source localization')

    parser.add_argument('--gpu-id', type=int, default=7, metavar='GPU', help='GPU ID (default: 7)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='TrainBatch', help='input batch size for training (default: 100)')
    parser.add_argument('--val-batch-size', type=int, default=100 , metavar='ValBatch', help='input batch size for validating (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='Epoch', help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default:0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='Seed', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='LogInterval', help='number of batches between training logging (default: 100)')

    parser.add_argument('--mic-type', type=str, default='CIPIC', metavar='MicType', help='type of microphone array (default: CIPIC)')
    parser.add_argument('--mic-match', action='store_true', default=False, help='microphone array matching or not for training and test (default: False)')
    parser.add_argument('--data-random', action='store_true', default=False, help='random condition for training data (default: False)')
    parser.add_argument('--time-frame', type=int, default=31, metavar='TimeFrame', help='number of time frames (default: 31)')
    parser.add_argument('--test', action='store_true', default=False, help='change to test stage (default: False)')

    parser.add_argument('--meth', type=str, default='our', metavar='Method', help='adopted method') 

    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    str_time = time.strftime('%m%d%H%M', local_time)
    parser.add_argument('--time', type=str, default=str_time, help='local time (default: local time when running)')

    args = parser.parse_args()

    return args
