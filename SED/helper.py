import os
import random
import numpy as np
import torch


class CFG:
    
    seed = 42
    print_freq = 100
    num_workers = 4

    stage = 'train_bce'

    train_datadir = 'SED/inputs/train_audio'
    train_csv = 'SED/inputs/train.csv'
    test_soundscapes = 'SED/inputs/train_audio/train_soundscapes'
    submission_csv = 'SED/inputs/sample_submission.csv'
    taxonomy_csv = 'SED/inputs/taxonomy.csv'
    #model_files = ['SED/outputs/sedmodel.pth'
    #              ]
    model_files = ['SED/outputs/sed_seresnext26t/pytorch/train_ce/epochepoch=14.ckpt'
                  ]
    cfg_file = 'SED/inputs/2.txt'
    model_name = 'seresnext26t_32x4d'  
    pretrained = False
    in_channels = 1
    SR = 32000
    target_duration = 5
    train_duration = 10
    device = 'cpu'


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False