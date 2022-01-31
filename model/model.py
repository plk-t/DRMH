import torch.nn as nn
import torch
from .self_attention import SelfAttention, SingleAttention


class TestModel(nn.Module):
    def __init__(self, num_classes=100, hash_code_length=16):
        super(TestModel, self).__init__()
        self.box_updim = nn.Sequential(
            nn.Linear(4, 768),
        )
        self.img_downdim = nn.Sequential(
            nn.Linear(2048, 768)
        )
        self.sattention = nn.Sequential(
            SingleAttention(768, 768),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hash_code_length, num_classes)
        )
        self.hashlayer = nn.Sequential(
            nn.Linear(768, hash_code_length),
        )

    def forward(self, img, box):
        box = self.box_updim(box)
        img = self.img_downdim(img)
        out = (img + box) / 2
        out = self.sattention(out)
        out = out.mean(1)
        out1 = self.hashlayer(out)
        out1 = torch.tanh(out1)
        out2 = self.classifier(out1)
        return out1, out2
