import os
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torchvision.models import vit_b_16, ViT_B_16_Weights
import math


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
    
    

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-2)  

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_weights = self.softmax(attention_scores)
        weighted_features = torch.matmul(attention_weights, V)
        return weighted_features, attention_weights


class Modifiedvit(nn.Module):
    def __init__(self, num_classes1, num_classes2, num_classes3, num_classes4, num_classes5):
        super(Modifiedvit, self).__init__()
        self.vitb1 = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vitb1.heads = MultiTaskHead(num_classes1, num_classes2, num_classes3, num_classes5)  # 修改fc层输出类别数量
       
        self.vitb2 = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
  
        self.vitb2.heads = MultiTaskHead2(num_classes1, num_classes2, num_classes3, num_classes4, num_classes5)
 
    def forward(self, x, p=True):
        x1 = 0
        x2 = 0
        x3 = 0
        x4 = torch.zeros((x.shape[0],768))
        x5 = 0
        task1 = 0
        task2 = 0
        task3 = 0
        task4 = torch.zeros((x.shape[0],2))
        task5 = 0
        if p:
            x1, x2, x3, x5, task1, task2, task3, task5 = self.vitb1(x)
        else:
            x1, x2, x3, x4, x5, task1, task2, task3, task4, task5 = self.vitb2(x)
        return x1, x2, x3, x4, x5, task1, task2, task3, task4, task5
    


if __name__ == "__main__":
    model = Modifiedvit()
    print(model)