#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import math
from math import sqrt
from itertools import product as product

class default:
    
    def __init__(self, image_size=300, feature_maps=[19, 10, 5, 3, 2, 1], min_sizes=[]):
        super(default, self).__init__()
        self.image_size = 300
        # number of priors for feature map location (either 4 or 6)
        self.feature_maps = feature_maps
        self.min_sizes = [16, 30, 60, 100, 150, 300]  #0.2, 0.34, 0.48, 0.62, 0.76, 0.9?
        self.max_sizes = [30, 60, 100, 150, 300, 300]
        self.steps = [19,10,5,3,2,1] # 이미지 그리드로 나눈 개수
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = True

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.steps[k] # 그리드 개수
                # default 박스 중점
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # aspect_ratio: 1
                # default 박스 공식이 아닌 임의로 설정 그리드 크기를 기준으로 나눔
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, 0.7] # s_k

                # aspect_ratio: 1
                # s_k*s_k+1
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                
                #print(cx,cy,s_k_prime,self.max_sizes[k]/self.image_size)
                mean += [cx, cy, s_k_prime, 0.7] #s_k

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), 0.7] # s_k/sqrt(ar)
                    mean += [cx, cy, s_k/sqrt(ar), 0.7]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
    def cxcy_to_xy(self, tensor_default_box):
        return tensor_default_box*self.image_size
    
    def xy_to_cxcy(self, tensor_default_box):
        return tensor_default_box/self.image_size

