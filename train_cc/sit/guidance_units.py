import torch
import torch.nn as nn
from typing import Callable
import numpy as np





class QVLoRAUnits(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(QVLoRAUnits, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # LoRA weight matrices
        self.lora_QA = nn.Parameter(torch.randn(in_features, rank))
        self.lora_QB = nn.Parameter(torch.randn(rank, out_features))

        self.lora_VA = nn.Parameter(torch.randn(in_features, rank))
        self.lora_VB = nn.Parameter(torch.randn(rank, out_features))


    def forward(self, q,v):
        q_l = q @ (self.lora_QA @ self.lora_QB)
        v_l = v @ (self.lora_VA @ self.lora_VB)
        return q_l, v_l



