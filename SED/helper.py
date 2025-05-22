import os
import random
import numpy as np
import torch


class CFG:
    
    seed = 42
    print_freq = 100
    num_workers = 4

    stage = 'train_bce'

    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_files = ['/kaggle/input/bird2025-sed-ckpt/sedmodel.pth'
                  ]
 
    model_name = 'seresnext26t_32x4d'  
    pretrained = False
    in_channels = 1

    
    SR = 32000
    target_duration = 5
    train_duration = 10
    
    
    device = 'cpu'


def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False