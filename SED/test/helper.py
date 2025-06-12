import os
import random
import numpy as np
import torch


class Config:
    
    num_workers = 4
    seed = 42
    print_freq = 100

    cfg_file = 'train/inputs/2.txt'
    train_datadir = 'train/inputs/train_audio'
    train_csv = 'train/inputs/train.csv'
    test_soundscapes = 'train/inputs/train_audio/train_soundscapes'
    submission_csv = 'train/inputs/sample_submission.csv'
    taxonomy_csv = 'train/inputs/taxonomy.csv'
    #model_files = ['train/outputs/sedmodel.pth'
    #              ]
    model_files = ['train/outputs/sed_seresnext26t/pytorch/train_ce/epochepoch=14.ckpt'
                  ]
    model_name = 'seresnext26t_32x4d'  
    pretrained = False
    target_duration = 5
    train_duration = 10
    in_channels = 1
    SR = 32000
    device = 'cpu'

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)