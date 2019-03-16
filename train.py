import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

import json
import time

import math
import numpy as np

from netsettings import net_archs, net_classes, net_prefixes, net_params

import argparse

# main training code
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Neural net training',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_name', type=str, help="""Name of the model to be trained. 
    Supported models: '""" + "', '".join(list(net_archs.keys())) + "'")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="""Learning rate to use for the Adam optimizer.""")
    parser.add_argument('--hidden_units', type=int, default=2,
                        help="""Number of hidden units for the new classifier. 
                        Recommended values: 2 or 3.""")
    parser.add_argument('--training_epochs', type=int, default=15,
                        help="""Number of epochs to train the model.""")
    parser.add_argument('--save_filename', type=str,
                        help="""Name for the checkpoint file.""")
    args = parser.parse_args()
    
    model_name = args.model_name
    learning_rate = args.learning_rate
    classifier_hidden = args.hidden_units
    epochs = args.training_epochs
    if args.save_filename is None:
        save_filename = net_prefixes[model_name] + 'checkpoint.pth'
    else:
        save_filename = args.save_filename
    #parser.print_help()