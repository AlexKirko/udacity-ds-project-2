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

# main training code
if __name__ == '__main__':
    print(net_archs)