import json
import time
import logging
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import soundfile as sf

import torch
from tqdm.auto import tqdm
from glob import glob
import concurrent.futures as cf

from model import *
from helper import *

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

def initialize():
    configuration = Config()
    print(f"Device selected: {configuration.device}")
    print("Loading taxonomy labels...")
    label_df = pd.read_csv(configuration.taxonomy_csv)
    target_labels = label_df['primary_label'].tolist()
    set_seed(configuration.seed)
    return configuration, target_labels

def chunk_audio(file_path, config):
    audio_data, sample_rate = sf.read(file_path, dtype='float32')
    total_len = config.SR * config.target_duration
    segments = []
    audio_data = np.concatenate([audio_data] * 3)

    for offset in range(total_len, len(audio_data) + total_len, total_len):
        start = max(0, offset - total_len)
        end = start + total_len
        if end <= len(audio_data):
            segments.append((start, end))

    output_clips = []
    for idx, (start, end) in enumerate(segments):
        pad = int(config.SR * (config.train_duration - config.target_duration) / 2)
        true_start = start - pad + len(audio_data)//3
        true_end = end + pad + len(audio_data)//3
        segment = audio_data[true_start:true_end].astype(np.float32)

        if idx == 0:
            segment[:pad] = 0
        elif idx == len(segments) - 1:
            segment[-pad:] = 0

        output_clips.append(segment)

    return output_clips

def discover_models(config) -> List[str]:
    model_files = []
    root = Path(config.model_path)
    for m in root.glob('**/*.pth'):
        model_files.append(str(m))
    return model_files

def prepare_models(config, class_count):
    all_models = []
    model_paths = config.model_files
    if not model_paths:
        print(f"No model files in: {config.model_path}")
        return all_models

    print(f"Loading {len(model_paths)} model(s)...")

    for m_path in model_paths:
        try:
            print(f"Initializing model from: {m_path}")
            checkpoint = torch.load(m_path, map_location=config.device)

            with open(config.cfg_file) as f:
                local_cfg = json.load(f)

            local_cfg['device'] = config.device
            local_cfg['taxonomy_csv'] = config.taxonomy_csv

            net = BirdCLEFModel(local_cfg)
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            net.to(config.device)
            net.eval()
            net.zero_grad()
            net.half().float()
            all_models.append(net)
        except Exception as err:
            print(f"Failed loading {m_path}: {err}")

    return all_models

def infer_audio(file_path, model_list, config):
    file_path = str(file_path)
    print(f"Running inference on: {Path(file_path).stem}")

    audio_segments = chunk_audio(file_path, config)
    results = []
    row_ids = []

    for idx, segment in enumerate(audio_segments):
        time_tag = (idx + 1) * config.target_duration
        row_ids.append(f"{Path(file_path).stem}_{time_tag}")
        tensor_segment = torch.tensor(segment).unsqueeze(0).unsqueeze(0).to(config.device)

        with torch.no_grad():
            if len(model_list) == 1:
                prediction = model_list[0].infer(tensor_segment).squeeze()
            else:
                pred_pool = [m.infer(tensor_segment).squeeze() for m in model_list]
                prediction = sum(pred_pool) / len(pred_pool)

        results.append(prediction.cpu().numpy())

    return row_ids, results

def bulk_infer(config, model_group):
    files = list(Path(config.test_soundscapes).glob('*.ogg'))
    if not files:
        files = sorted(glob(str(Path('/kaggle/input/birdclef-2025/train_soundscapes') / '*.ogg')))[:10]

    print(f"Identified {len(files)} test files")
    all_ids = []
    all_preds = []

    with cf.ThreadPoolExecutor(max_workers=4) as exec_pool:
        futures = [exec_pool.submit(infer_audio, f, model_group, config) for f in files]
        for future in tqdm(cf.as_completed(futures), total=len(futures)):
            ids, preds = future.result()
            all_ids.extend(ids)
            all_preds.extend(preds)

    return all_ids, all_preds

def compose_submission(ids, preds, label_pool, config):
    print("Constructing submission DataFrame...")
    submit = {'row_id': ids}
    for i, label in enumerate(label_pool):
        submit[label] = [p[i] for p in preds]

    submission = pd.DataFrame(submit).set_index('row_id')
    ref = pd.read_csv(config.submission_csv, index_col='row_id')

    missing = set(ref.columns) - set(submission.columns)
    for col in missing:
        submission[col] = 0.0

    submission = submission[ref.columns].reset_index()
    return submission

def temporal_smooth(file_path):
    print("Applying smoothing to predictions...")
    df = pd.read_csv(file_path)
    label_cols = df.columns[1:]
    file_groups = df['row_id'].str.rsplit('_', n=1).str[0].values
    unique_files = np.unique(file_groups)

    for uf in unique_files:
        idx = np.where(file_groups == uf)[0]
        data_slice = df.iloc[idx]
        pred = data_slice[label_cols].values.copy()

        if len(pred) > 1:
            pred[0] = pred[0] * 0.8 + pred[1] * 0.2
            pred[-1] = pred[-1] * 0.8 + pred[-2] * 0.2
            for i in range(1, len(pred) - 1):
                pred[i] = pred[i - 1] * 0.2 + pred[i] * 0.6 + pred[i + 1] * 0.2
        df.iloc[idx, 1:] = pred

    df.to_csv(file_path, index=False)
    print(f"Smoothed predictions saved at {file_path}")

def main():
    start = time.time()
    print("=== BirdCLEF 2025 Inference Pipeline Start ===")

    config, labels = initialize()
    models = prepare_models(config, len(labels))

    if not models:
        print("Model loading failed. Check configuration.")
        return

    print(f"Using {'ensemble' if len(models) > 1 else 'single'} model configuration")

    ids, preds = bulk_infer(config, models)
    final_df = compose_submission(ids, preds, labels, config)

    output_csv = 'submission.csv'
    final_df.to_csv(output_csv, index=False)
    print(f"Raw submission stored at {output_csv}")

    temporal_smooth(output_csv)
    print(f"Total time: {(time.time() - start) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
