# region Imports
import os
import argparse
import datetime
import sys
import cv2
import glob
import gdcm
import json
import shutil
import pydicom
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from distutils.util import strtobool
from sklearn.metrics import roc_curve, auc, roc_auc_score

import torch
import torch.nn.functional as F
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.types import DALIDataType
from pydicom.filebase import DicomBytesIO
from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type

import dicomsdl
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from mmengine.runner import load_checkpoint
from mmengine.registry import MODELS
import mmengine
from mmcls.utils import register_all_modules
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# change
from training import *
from constants import *
#endregion

register_all_modules()
torch.backends.cudnn.benchmark = True
torch.jit.enable_onednn_fusion(True)
torch.cuda.empty_cache()

def train(args, exp_name):

    if args.learned_loss_attnuation:
        print('Training with learned loss attenuation....')
        train_model_loss_attenuation(args, INPUT_CHK_PT, THRESHOLD, exp_name)
    else:
        print('Training ....')
        train_model(args, INPUT_CHK_PT, THRESHOLD, exp_name)

    # # clean-up
    if args.remove_processed_pngs.lower() == 'true':
        print('WARNING. DELETING ' + args.save_img_root_dir_pth)
        shutil.rmtree(args.save_img_root_dir_pth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploying RSNA model on input data')
    parser.add_argument('--exp_name', type=str, help='name of the run')
    parser.add_argument('--train_csv_pth', type=str, help='Path to training CSV file')
    parser.add_argument('--valid_csv_pth', type=str, help='Path to tuning CSV file')
    parser.add_argument('--dcm_root_dir_pth', type=str, help='Root directory of DICOM images')
    parser.add_argument('--save_img_root_dir_pth', type=str, help='Directory to save processed PNG images')
    parser.add_argument('--remove_processed_pngs', type=str, default = True, help = 'Choose if you want to save generated pngs (True or False)')
    parser.add_argument('--data', type=str, default = 'RSNA', choices = ['RSNA', 'EMBED'], help = 'Choose dataset')
    parser.add_argument('--out_dir', type=str, help='Output root directory')
    parser.add_argument('--fine_tuning', default='full', help="options: 'full' or 'partial'")
    parser.add_argument('--upto_freeze', type=int, default=0, help="options: provide the layer number upto which to freeze, maximum of 42 layers")
    parser.add_argument('--batch_size', type=int, default=8, help='batch size.')
    parser.add_argument('--threads', type=int, default=4, help='num. of threads.')
    parser.add_argument('--num_epochs', type=int, default=100, help='num. of epochs.')
    parser.add_argument('--optimizer', default = 'adam', choices = ['adam', 'sgd', 'adamW'], help = 'Choose optimizer')
    parser.add_argument('--decay_every_N_epoch', type=int, default=5, help='Drop the learning rate every N epochs')
    parser.add_argument('--decay_multiplier', type=float, default=0.95, help='Decay multiplier')
    parser.add_argument('--start_learning_rate', type=float, default=1e-4, help='starting learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
    parser.add_argument('--save_every_N_epochs', type=int, default=1, help='save checkpoint every N number of epochs')
    parser.add_argument('--bsave_valid_results_at_epochs', type=lambda x: bool(strtobool(x)), default=False, help='save validation results csv at every epoch, True/False')
    parser.add_argument('--random_state', type=int, default=None)
    parser.add_argument('--dropout_rate', type=float, default=None)
    parser.add_argument('--finite_sample_rate', type=float, default=None)
    parser.add_argument('--learned_loss_attnuation', default = False, type=lambda x: bool(strtobool(x)), help = 'Choose if you want apply learned loss attnuation')
    parser.add_argument('--training_augment', default = False, type=lambda x: bool(strtobool(x)), help = 'Choose if apply cutout augmentation during training')
    parser.add_argument('--t_number', type=int, default=10, help='num. of t Monte Carlo samples drawn during learned loss attenuation.')
    parser.add_argument('--dataset_partition', default=False, type=lambda x: bool(strtobool(x)), help='Choose if experiment has to be conducted on a smaller dataframe')
    parser.add_argument('--rows_experiment', default=5000, type=int)
    parser.add_argument('--prefetch_factor', default=4, type=int)
    
    
    args = parser.parse_args()
    print(args)
    # # save the args
    os.makedirs(args.out_dir, exist_ok=True)

    exp_name =  (args.exp_name) + '__'  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    args_log_file = os.path.join(args.out_dir, exp_name)

    print(f"********************************************************")
    print(args_log_file)
    print(f"********************************************************")

    
    with open(args_log_file, 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    # # call train
    train(args, exp_name)
    print('.... COMPLETED')