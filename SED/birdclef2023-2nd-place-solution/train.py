import argparse
import importlib
from modules.preprocess import preprocess,prepare_cfg
from modules.dataset import get_train_dataloader
from modules.model import load_model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning, EarlyStopping
import torch
import os
import gc
import json

def make_parser():
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--stage', choices=["pretrain_ce","pretrain_bce","train_ce","train_bce","finetune"])
    parser.add_argument('--model_name', choices=["sed_v2s",'sed_b3ns','sed_seresnext26t','cnn_v2s','cnn_resnet34d','cnn_b3ns','cnn_b0ns'])
    parser.add_argument('--use_pseudo', action='store_true')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    stage = args.stage
    model_name = args.model_name
    use_pseudo = args.use_pseudo
    cfg = importlib.import_module(f'configs.{model_name}').basic_cfg
    cfg = prepare_cfg(cfg,stage)
    os.environ['WANDB_API_KEY'] = cfg.WANDB_API_KEY

    pl.seed_everything(cfg.seed[stage], workers=True)

    df_train, df_valid, df_label_train, df_label_valid, sample_weight, transforms = preprocess(cfg)

    pseudo = None

    if use_pseudo:
        # =========================================================
        with open('/content/birdclef2023-2nd-place-solution/inputs/pseudo_label/pseudo.json') as f:
            pseudo = json.loads(f.read())

        with open('/content/birdclef2023-2nd-place-solution/inputs/hand_label/hand_label.json') as f:
            hand_label = json.loads(f.read())

        for version in hand_label['pred'].keys():
            for filename in hand_label['pred'][version].keys():
                for label in hand_label['pred'][version][filename].keys():
                    for second in hand_label['pred'][version][filename][label].keys():
                        for i in range(len(pseudo['subset1']['pseudo'])):
                            if second in pseudo['subset1']['pseudo'][i]['pred'][version][filename][label].keys():
                                pseudo['subset1']['pseudo'][i]['pred'][version][filename][label][second] = hand_label['pred'][version][filename][label][second]
        # =========================================================

    dl_train, dl_val, ds_train, ds_val = get_train_dataloader(
        df_train,
        df_valid,
        df_label_train,
        df_label_valid,
        sample_weight,
        cfg,
        pseudo,
        transforms
    )

    logger = WandbLogger(project='BirdClef-2025', name=f'{model_name}_{stage}')

    best_model_ckpt = ModelCheckpoint(
        monitor='train_loss',
        dirpath=cfg.output_path[stage],
        filename='best-epoch{epoch}-val_loss{val_loss:.4f}',
        save_top_k=1,
        mode='min',
        save_weights_only=True,
        verbose=True
    )
    last_model_ckpt = ModelCheckpoint(
        dirpath=cfg.output_path[stage],
        filename='last',
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        verbose=True
    )
    periodic_ckpt = ModelCheckpoint(
        dirpath=cfg.output_path[stage],
        filename='epoch{epoch}',
        every_n_epochs=5,
        save_top_k=-1,  # 保存所有周期模型
        save_weights_only=True,
        verbose=True
    )

    callbacks_to_use = [best_model_ckpt, last_model_ckpt, periodic_ckpt]
    model = load_model(cfg,stage)
    #torch.save({"state_dict": model.state_dict()}, "ckpt.pth")
    trainer = pl.Trainer(
        devices=2,
        val_check_interval=1.0,
        deterministic=None,
        max_epochs=cfg.epochs[stage],
        logger=logger,
        callbacks=callbacks_to_use,
        precision=cfg.PRECISION, accelerator="auto",
    )

    print("Running trainer.fit")
    trainer.fit(model, train_dataloaders = dl_train, val_dataloaders = dl_val)

    # gc.collect()
    torch.cuda.empty_cache()
    return

if __name__=='__main__':
    main()
