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
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)

def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        taxonomy_df = pd.read_csv('/kaggle/input/birdclef-2025/taxonomy.csv')
        self.num_classes = len(taxonomy_df)

        self.bn0 = nn.BatchNorm2d(cfg['n_mels'])
        
        self.backbone = timm.create_model(
            cfg['model_name'],
            pretrained=False,
            in_chans=cfg['in_channels'],
            drop_rate=0.2,
            drop_path_rate=0.2,
        )

        layers = list(self.backbone.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        
        if "efficientnet" in self.cfg['model_name']:
            backbone_out = self.backbone.classifier.in_features
        elif "eca" in self.cfg['model_name']:
            backbone_out = self.backbone.head.fc.in_features
        elif "res" in self.cfg['model_name']:
            backbone_out = self.backbone.fc.in_features
        else:
            backbone_out = self.backbone.num_features
            
        
        self.fc1 = nn.Linear(backbone_out, backbone_out, bias=True)
        self.att_block = AttBlockV2(backbone_out, self.num_classes, activation="sigmoid")

        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg['SR'],
            hop_length=self.cfg['hop_length'],
            n_mels=self.cfg['n_mels'],
            f_min=self.cfg['f_min'],
            f_max=self.cfg['f_max'],
            n_fft=self.cfg['n_fft'],
            pad_mode="constant",
            norm="slaney",
            onesided=True,
            mel_scale="htk",
        )
        if self.cfg['device'] == "cuda":
            self.melspec_transform = self.melspec_transform.cuda()
        else:
            self.melspec_transform = self.melspec_transform.cpu()

        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )


    def extract_feature(self,x):
        x = x.permute((0, 1, 3, 2))
        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # if self.training:
        #    x = self.spec_augmenter(x)
        
        x = x.transpose(2, 3)
        # (batch_size, channels, freq, frames)
        x = self.encoder(x)
        
        # (batch_size, channels, frames)
        x = torch.mean(x, dim=2)
        
        # channel smoothing
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        return x, frames_num
        
    @torch.cuda.amp.autocast(enabled=False)
    def transform_to_spec(self, audio):

        audio = audio.float()
        
        spec = self.melspec_transform(audio)
        spec = self.db_transform(spec)

        if self.cfg['normal'] == 80:
            spec = (spec + 80) / 80
        elif self.cfg['normal'] == 255:
            spec = spec / 255
        else:
            raise NotImplementedError
                
        if self.cfg['in_channels'] == 3:
            #spec = image_delta(spec)
            pass
        return spec

    def forward(self, x):

        with torch.no_grad():
            x = self.transform_to_spec(x)

        x, frames_num = self.extract_feature(x)
        
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        return torch.logit(clipwise_output)

    def infer(self, x, tta_delta=2):
        with torch.no_grad():
            x = self.transform_to_spec(x)
        x,_ = self.extract_feature(x)
        time_att = torch.tanh(self.att_block.att(x))
        feat_time = x.size(-1)
        start = (
            feat_time / 2 - feat_time * (self.cfg['infer_duration'] / self.cfg['duration_train']) / 2
        )
        end = start + feat_time * (self.cfg['infer_duration'] / self.cfg['duration_train'])
        start = int(start)
        end = int(end)
        pred = self.attention_infer(start,end,x,time_att)

        start_minus = max(0, start-tta_delta)
        end_minus=end-tta_delta
        pred_minus = self.attention_infer(start_minus,end_minus,x,time_att)

        start_plus = start+tta_delta
        end_plus=min(feat_time, end+tta_delta)
        pred_plus = self.attention_infer(start_plus,end_plus,x,time_att)

        pred = 0.5*pred + 0.25*pred_minus + 0.25*pred_plus
        return pred
        
    def attention_infer(self,start,end,x,time_att):
        feat = x[:, :, start:end]
        # att = torch.softmax(time_att[:, :, start:end], dim=-1)
        #             print(feat_time, start, end)
        #             print(att_a.sum(), att.sum(), time_att.shape)
        framewise_pred = torch.sigmoid(self.att_block.cla(feat))
        framewise_pred_max = framewise_pred.max(dim=2)[0]
        # clipwise_output = torch.sum(framewise_pred * att, dim=-1)
        #logits = torch.sum(
        #    self.att_block.cla(feat) * att,
        #    dim=-1,
        #)

        # return clipwise_output
        return framewise_pred_max