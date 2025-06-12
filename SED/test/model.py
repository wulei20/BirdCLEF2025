import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio


class AttentionBlock(nn.Module):
    def __init__(self, in_feature_num: int, out_feature_num: int, act_type="linear"):
        super().__init__()
        self.act_type = act_type
        self.att = nn.Conv1d(in_feature_num, out_feature_num, 1, 1, 0, bias=True)
        self.cla = nn.Conv1d(in_feature_num, out_feature_num, 1, 1, 0, bias=True)
        self.init_model_weight()

    def single_layer_init(layer):
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data.fill_(0.0)

    def init_model_weight(self):
        self.single_layer_init(self.att)
        self.single_layer_init(self.cla)

    def forward(self, x):
        attn = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla_out = self.nonlinear_transform(self.cla(x))
        return torch.sum(attn * cla_out, dim=2), attn, cla_out

    def nonlinear_transform(self, x):
        return x if self.act_type == "linear" else torch.sigmoid(x)

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bn0 = nn.BatchNorm2d(cfg['n_mels'])
        self.num_classes = len(pd.read_csv(cfg['taxonomy_csv']))

        ref_net = timm.create_model(
            cfg['model_name'], pretrained=False,
            in_chans=cfg['in_channels'], drop_rate=0.2, drop_path_rate=0.2
        )

        self.encoder = nn.Sequential(*list(ref_net.children())[:-2])
        self.fc1 = nn.Linear(ref_net.fc.in_feature_num, ref_net.fc.in_feature_num, bias=True)
        self.att_block = AttentionBlock(ref_net.fc.in_feature_num, self.num_classes, act_type="sigmoid")

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg['SR'], hop_length=cfg['hop_length'],
            n_mels=cfg['n_mels'], f_min=cfg['f_min'], f_max=cfg['f_max'],
            n_fft=cfg['n_fft'], pad_mode="constant", norm="slaney",
            onesided=True, mel_scale="htk"
        )
        self.melspec_transform = mel.cuda() if cfg['device'] == "cuda" else mel.cpu()
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def feature_extract(self, inp):
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
    def transform_audio_to_spec(self, audio):
        spec_graph = self.db_transform(self.melspec_transform(audio.float()))
        if self.cfg['normal'] == 80:
            return (spec_graph + 80) / 80
        elif self.cfg['normal'] == 255:
            return spec_graph / 255
        else:
            raise NotImplementedError

    def forward(self, x):
        with torch.no_grad():
            x = self.transform_audio_to_spec(x)
        feat, _ = self.feature_extract(x)
        out, _, _ = self.att_block(feat)
        return torch.logit(out)

    def infer(self, x, offset=2):
        with torch.no_grad():
            x = self.transform_audio_to_spec(x)
        feat, _ = self.feature_extract(x)
        attn = torch.tanh(self.att_block.att(feat))
        T = feat.size(-1)
        d = self.cfg['infer_duration'] / self.cfg['duration_train']
        s = int(T / 2 - T * d / 2)
        e = int(s + T * d)

        pred = self.infer_attention(s, e, feat, attn)
        pred_minus = self.infer_attention(max(0, s - offset), e - offset, feat, attn)
        pred_plus = self.infer_attention(s + offset, min(T, e + offset), feat, attn)

        return 0.5 * pred + 0.25 * pred_minus + 0.25 * pred_plus

    def infer_attention(self, s, e, feat, attn):
        x_seg = feat[:, :, s:e]
        seg_pred = torch.sigmoid(self.att_block.cla(x_seg))
        return seg_pred.max(dim=2)[0]
