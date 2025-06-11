import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()
        self.activation = activation
        self.att = nn.Conv1d(in_features, out_features, 1, 1, 0, bias=True)
        self.cla = nn.Conv1d(in_features, out_features, 1, 1, 0, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        attn = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla_out = self.nonlinear_transform(self.cla(x))
        return torch.sum(attn * cla_out, dim=2), attn, cla_out

    def nonlinear_transform(self, x):
        return x if self.activation == "linear" else torch.sigmoid(x)


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = len(pd.read_csv(cfg['taxonomy_csv']))
        self.bn0 = nn.BatchNorm2d(cfg['n_mels'])

        backbone = timm.create_model(
            cfg['model_name'], pretrained=False,
            in_chans=cfg['in_channels'], drop_rate=0.2, drop_path_rate=0.2
        )

        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.fc1 = nn.Linear(backbone.fc.in_features, backbone.fc.in_features, bias=True)
        self.att_block = AttBlockV2(backbone.fc.in_features, self.num_classes, activation="sigmoid")

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg['SR'], hop_length=cfg['hop_length'],
            n_mels=cfg['n_mels'], f_min=cfg['f_min'], f_max=cfg['f_max'],
            n_fft=cfg['n_fft'], pad_mode="constant", norm="slaney",
            onesided=True, mel_scale="htk"
        )
        self.melspec_transform = mel.cuda() if cfg['device'] == "cuda" else mel.cpu()
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def extract_feature(self, inp):
        inp = inp.permute((0, 1, 3, 2))
        t = inp.shape[2]
        inp = inp.transpose(1, 3)
        inp = self.bn0(inp)
        inp = inp.transpose(1, 3).transpose(2, 3)
        feat = self.encoder(inp)
        feat = torch.mean(feat, dim=2)
        max_pool = F.max_pool1d(feat, kernel_size=3, stride=1, padding=1)
        avg_pool = F.avg_pool1d(feat, kernel_size=3, stride=1, padding=1)
        feat = max_pool + avg_pool
        feat = F.dropout(feat, p=0.5, training=self.training)
        feat = F.relu_(self.fc1(feat.transpose(1, 2))).transpose(1, 2)
        feat = F.dropout(feat, p=0.5, training=self.training)
        return feat, t

    @torch.cuda.amp.autocast(enabled=False)
    def transform_to_spec(self, audio):
        audio = audio.float()
        spec = self.db_transform(self.melspec_transform(audio))
        if self.cfg['normal'] == 80:
            return (spec + 80) / 80
        elif self.cfg['normal'] == 255:
            return spec / 255
        else:
            raise NotImplementedError

    def forward(self, x):
        with torch.no_grad():
            x = self.transform_to_spec(x)
        feat, _ = self.extract_feature(x)
        out, attn, seg_out = self.att_block(feat)
        _ = torch.sum(attn * self.att_block.cla(feat), dim=2)
        _ = self.att_block.cla(feat).transpose(1, 2)
        _ = seg_out.transpose(1, 2)
        return torch.logit(out)

    def infer(self, x, tta_delta=2):
        with torch.no_grad():
            x = self.transform_to_spec(x)
        feat, _ = self.extract_feature(x)
        attn = torch.tanh(self.att_block.att(feat))
        T = feat.size(-1)
        d = self.cfg['infer_duration'] / self.cfg['duration_train']
        s = int(T / 2 - T * d / 2)
        e = int(s + T * d)

        pred = self.attention_infer(s, e, feat, attn)
        pred_minus = self.attention_infer(max(0, s - tta_delta), e - tta_delta, feat, attn)
        pred_plus = self.attention_infer(s + tta_delta, min(T, e + tta_delta), feat, attn)

        return 0.5 * pred + 0.25 * pred_minus + 0.25 * pred_plus

    def attention_infer(self, s, e, feat, attn):
        x_seg = feat[:, :, s:e]
        seg_pred = torch.sigmoid(self.att_block.cla(x_seg))
        return seg_pred.max(dim=2)[0]
