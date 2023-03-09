import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import pickle
import rdkit
import time

import numpy as np

print("Test")

l = np.linspace(1, 500, 500)

for x in l:
    print(x)
