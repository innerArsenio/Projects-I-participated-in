#can be deleted, just made for myself to make evrything clean
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import torch
from torch import nn
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import random
from argparse import Namespace
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#seeding can be used when training
def seed_e(seed_value):
  pl.seed_everything(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value) 
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

re_dict = {
    "path_dataset_encoder": "/home/aldeka/senior_sound/",
    "input_size_encoder": 46960,
    "d_encoder": 100,
    "lr_encoder": 1e-3,
}

hparams = Namespace(**re_dict)
