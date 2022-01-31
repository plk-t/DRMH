import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class HashLoss(nn.Module):
    def __init__(self, num_classes, hash_code_length):
        super().__init__()
        self.num_classes = num_classes
        self.hash_code_length = hash_code_length
        self.classify_loss_fun = nn.BCELoss()

    def calculate_similarity(self, label1):
        temp = torch.einsum('ij,jk->ik', label1, label1.t())
        L2_norm = torch.norm(label1, dim=1, keepdim=True)
        fenmu = torch.einsum('ij,jk->ik', L2_norm, L2_norm.t())
        sim = temp / fenmu
        return sim

    def hash_NLL_my(self, out, s_matrix):
        hash_bit = out.shape[1]
        cos = torch.tensor(cosine_similarity(out.detach().cpu(), out.detach().cpu())).cuda()
        w = torch.abs(s_matrix - (1 + cos) / 2)
        inner_product = torch.einsum('ij,jk->ik', out, out.t())

        L = w * ((inner_product + hash_bit) / 2 - s_matrix * hash_bit) ** 2

        diag_matrix = torch.tensor(np.diag(torch.diag(L.detach()).cpu())).cuda()
        loss = L - diag_matrix
        count = (out.shape[0] * (out.shape[0] - 1) / 2)

        return loss.sum() / 2 / count

    def quanti_loss(self, out):
        b_matrix = torch.sign(out)
        temp = torch.einsum('ij,jk->ik', out, out.t())
        temp1 = torch.einsum('ij,jk->ik', b_matrix, b_matrix.t())
        q_loss = temp - temp1
        q_loss = torch.abs(q_loss)
        loss = torch.exp(q_loss / out.shape[1])

        return loss.sum() / out.shape[0] / out.shape[0]

    def forward(self, out2, out_class, label):
        classify_loss = self.classify_loss_fun(torch.sigmoid(out_class), label)
        sim_matrix = self.calculate_similarity(label)
        hash_loss = self.hash_NLL_my(out2, sim_matrix)
        quanti_loss = self.quanti_loss(out2)
        return classify_loss, hash_loss, quanti_loss

