import torch
import torch.nn as nn
import time
from speakerlab.models.talknet.audioEncoder import audioEncoder
from speakerlab.models.talknet.visualEncoder import visualFrontend, visualTCN, visualConv1D
from speakerlab.models.talknet.attentionLayer import attentionLayer

class talkNetModel(nn.Module):
    """
    TalkNet model for active speaker detection task.
    Reference:
    - Is someone talking? TalkNet: Audio-visual active speaker detection Model.
    - https://github.com/TaoRuijie/TalkNet-ASD
    """
    def __init__(self):
        super(talkNetModel, self).__init__()
        # Visual Temporal Encoder
        self.visualFrontend = visualFrontend() # Visual Frontend
        self.visualTCN = visualTCN() # Visual Temporal Network TCN
        self.visualConv1D = visualConv1D() # Visual Temporal Network Conv1d

        # Audio Temporal Encoder
        self.audioEncoder = audioEncoder(layers = [3, 4, 6, 3], num_filters = [16, 32, 64, 128])

        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = 128, nhead = 8)
        self.crossV2A = attentionLayer(d_model = 128, nhead = 8)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = 256, nhead = 8)

        # Classifier
        self.fcAV = nn.Linear(256, 2)
        self.fcA = nn.Linear(128, 2)
        self.fcV = nn.Linear(128, 2)

    def visual_frontend(self, x):
        B, T, W, H = x.shape
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)
        x = x.transpose(1,2)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x

    def audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)
        return x

    def cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src = x1, tar = x2)
        x2_c = self.crossV2A(src = x2, tar = x1)
        return x1_c, x2_c

    def audio_visual_backend(self, x1, x2):
        x = torch.cat((x1,x2), 2)
        x = self.selfAV(src = x, tar = x)
        return x

    def forward(self, audioX, visualX):
        audioX = self.audio_frontend(audioX)
        visualX = self.visual_frontend(visualX)
        audioX, visualX = self.cross_attention(audioX, visualX)
        audio_visualX = self.audio_visual_backend(audioX, visualX)

        return self.fcAV(audio_visualX), self.fcA(audioX), self.fcV(visualX)
