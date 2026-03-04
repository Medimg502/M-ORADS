import os
import numpy as np
import cv2
import torch
import pandas as pd
import shutil
import csv
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import random
import math
from collections import Counter
from torch.utils.data import DataLoader
import torch.utils.data as Data
from torchvision.models import vit_b_16, ViT_B_16_Weights




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




  






        
    






       
  
