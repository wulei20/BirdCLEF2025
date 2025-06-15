import argparse
import json
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data_utils import load_dataset, audio_transformations, create_data_loaders
from model import initialize_model


def parse_arguments():
    parser = argparse.ArgumentParser(description='Audio Bird Classification Training')
    # 训练设置
    parser.add_argument('--train_phase', choices=["pretrain_bce", "train_ce"], help='训练阶段')
    parser.add_argument('--use_pseudo_labels', action='store_true', help='是否使用伪标签')
    # 训练参数
    parser.add_argument('--random_seed', type=int, default=0, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--data_directory', type=str, default="train", help='训练数据目录')
    args = parser.parse_args()
    # 模型参数
    args.model_architecture = 'sed_seresnext26t'
    args.sample_rate = 32000
    args.duration = 5
    args.mel_spectrogram_params = {
        'n_mels': 128,
        'f_min': 50,
        'f_max': 14000,
        'hop_length': 512,
        'n_fft': 2048
    }
    # 训练参数
    args.learning_rates = {"pretrain_bce": 1e-3, "train_ce": 3e-4}
    args.training_epochs = {"pretrain_bce": 40, "train_ce": 60}
    args.weight_decay = 1e-5
    args.label_smoothing_epsilon = 0.1
    args.mixup_hyperparameters = {
        'beta': 0.5,
        'probability': 0.5,
        'double': 0.3
    }
    return args


def main():
    training_arguments = parse_arguments()
    training_phase = training_arguments.train_phase
    use_pseudo_labels = training_arguments.use_pseudo_labels

    training_dataframe = pd.read_csv(training_arguments.data_directory)
    bird_species_list = training_dataframe['primary_label'].unique().tolist()
    training_arguments.target_bird_species = bird_species_list

    pl.seed_everything(training_arguments.random_seed, workers=True)

    train_dataframe, validation_dataframe, train_labels, validation_labels, weight_samples = load_dataset(training_arguments)

    pseudo_labels = None
    if use_pseudo_labels:
        with open('pseudo_labels.json') as pseudo_file:
            pseudo_labels = json.loads(pseudo_file.read())

    train_loader, validation_loader = create_data_loaders(
        train_dataframe,
        validation_dataframe,
        train_labels,
        validation_labels,
        weight_samples,
        training_arguments,
        pseudo_labels,
        audio_transformations
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{training_phase}/",
        filename='birdclef',
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        verbose=True
    )

    bird_classification_model = initialize_model(training_arguments, training_arguments.train_phase)

    trainer = pl.Trainer(
        devices=2,
        val_check_interval=1.0,
        deterministic=True,
        max_epochs=training_arguments.training_epochs[training_phase],
        callbacks=[checkpoint_callback],
        precision=32,
        accelerator="auto"
    )

    print(f"Starting {training_phase} training phase...")
    trainer.fit(
        bird_classification_model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader
    )
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
