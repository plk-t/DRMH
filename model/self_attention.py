import torch
from torch import nn
import math
from .attention import BasicAttention


class SelfAttention(BasicAttention):
    def __init__(self, embd_size, num_heads=1, **kwargs):
        q_embd_size = embd_size
        k_embd_size = embd_size
        v_embd_size = embd_size
        super().__init__(q_embd_size, k_embd_size, v_embd_size, num_heads=num_heads, **kwargs)

    def forward(self, embd, mask=None):
        q_embd = embd
        k_embd = embd
        v_embd = embd
        return super().forward(q_embd, k_embd, v_embd, mask)


class SingleAttention(nn.Module):
    def __init__(self, embd_size, out_size):
        super(SingleAttention, self).__init__()
        self.embd_size = embd_size
        self.out_size = out_size
        self.attWeight = nn.Linear(self.embd_size, self.out_size)
        # self.drop = nn.Dropout(0.2)

    def forward(self, embd):
        c = self.attWeight(embd)
        temp = torch.bmm(c, c.permute(0, 2, 1))
        score = torch.div(temp, math.sqrt(self.embd_size))

        score = nn.functional.softmax(score, dim=-1)
        # score = self.drop(score)
        output = torch.bmm(score, c)
        # out = nn.functional.relu(out)
        return output


if __name__ == "__main__":
    # att = SelfAttention(768)
    matrix = torch.rand(4, 5, 768)
    print(matrix.shape)
    matrix.permute(0, 2, 1)
    print(matrix.shape)
    # result = att(matrix)
    # print(result)
    # print(result.shape)