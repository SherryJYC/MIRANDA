"""
* Rank loss adapted from Rank-n-Contrast paper, from https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py
* CORAL loss adapted from https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/coral.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs] distance
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        elif self.distance_type == 'cos':
            labels = F.normalize(labels, dim=-1)
            return - labels @ labels.T # B x B
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs] similarity
        if self.similarity_type == 'l1':
            return - torch.abs(features[:, None, :] - features[None, :, :]).sum(dim=-1)
        elif self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'cos':
            features = F.normalize(features, dim=-1)
            return features @ features.T # B x B
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

        print(f">> using temperature = {self.t} in rank loss")

    def forward(self, features, labels, log=False):
        # features: [bs, feat_dim]
        # labels: [bs, label_dim]

        if features.shape[0] > labels.shape[0]:
            labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        if log:
            print(f"features={features.mean()}, logits={logits.mean()}")
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss


class DomainContrastiveMemory:
    def __init__(self, dim, size=128, momentum=0.5, device='cuda'):
        self.size = size
        self.dim = dim
        self.device = device
        self.momentum = momentum
        self.ptr = 0

        self.memory_source = {
            "features": torch.zeros((size, dim), device=device),
            "labels": torch.zeros((size, 1), device=device)
        }
        self.memory_target = {
            "features": torch.zeros((size, dim), device=device),
            "labels": torch.zeros((size, 1), device=device)
        }

        self.ptr_source = 0
        self.ptr_target = 0
        self.filled_source = 0
        self.filled_target = 0

    @torch.no_grad()
    def update(self, features, labels, domain):
        """
        features: [B, D]
        domain: 'source' or 'target'
        """

        if domain == "source":
            self._insert_into_bank(features, labels, "source")
        elif domain == "target":
            self._insert_into_bank(features, labels, "target")
        else:
            raise ValueError("Domain must be 'source' or 'target'")

    @torch.no_grad()
    def _insert_into_bank(self, features, labels, domain):
        bs = features.size(0)
        bank = getattr(self, f"memory_{domain}")
        ptr = getattr(self, f"ptr_{domain}")
        filled = getattr(self, f"filled_{domain}")

        if ptr + bs <= self.size:
            bank["features"][ptr:ptr+bs] = features
            bank["labels"][ptr:ptr+bs] = labels
            ptr = (ptr + bs) % self.size
        # overflow, then do FIFO
        else:
            overflow = ptr + bs - self.size
            bank["features"][ptr:] = features[:bs - overflow]
            bank["features"][:overflow] = features[bs - overflow:]
            bank["labels"][ptr:] = labels[:bs - overflow]
            bank["labels"][:overflow] = labels[bs - overflow:]
            ptr = overflow

        filled = min(self.size, filled + bs)

        setattr(self, f"ptr_{domain}", ptr)
        setattr(self, f"filled_{domain}", filled)

    def get_bank(self, domain):
        bank = getattr(self, f"memory_{domain}")
        filled = getattr(self, f"filled_{domain}")
        return {"features": bank["features"][:filled], "labels": bank["labels"][:filled]}


def coral(source, target):
    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(1, n, device=device) #torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c