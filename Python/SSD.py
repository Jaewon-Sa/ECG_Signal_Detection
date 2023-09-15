#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
from torchvision import models

class SSD(nn.Module):
    def __init__(self, backbone, n_class=3, default_box_n=[4,6,6,6,4,4], state = "Train"):
        
        super().__init__()
        
        self.input_layer = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        
        self.n_class = n_class
        self.default_box_n = default_box_n
        self.state = state
        
        self.softmax = nn.Softmax(dim=-1)
            
        #가중치 초기화 인자
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 112, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)
        
        self.relu = nn.ReLU()
        #backbone
        
        self.backbone_layer = nn.Sequential(
            backbone.features,
        )
        self.backbone = backbone
        
        #extra layer
        self.extra_layer_1 = self.extra_layers(960,512,4)
        self.extra_layer_2 = self.extra_layers(512,256,4)
        self.extra_layer_3 = self.extra_layers(256,256,2)
        self.extra_layer_4 = self.extra_layers(256,128,2)
        
        self.extra_layers = [self.extra_layer_1, 
                            self.extra_layer_2, 
                            self.extra_layer_3, 
                            self.extra_layer_4]
        
        
        #detection output
        
        self.cls_layers = nn.ModuleList([
                                        nn.Conv2d(672, default_box_n[0]*(self.n_class), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(960, default_box_n[1]*(self.n_class), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(512, default_box_n[2]*(self.n_class), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(256, default_box_n[3]*(self.n_class), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(256, default_box_n[4]*(self.n_class), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(128, default_box_n[5]*(self.n_class), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                        ])
        
        self.loc_layers = nn.ModuleList([
                                        nn.Conv2d(672, default_box_n[0]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(960, default_box_n[1]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(512, default_box_n[2]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(256, default_box_n[3]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(256, default_box_n[4]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.Conv2d(128, default_box_n[5]*4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                        ])
        
        
    
    def init_conv2d(self):
        #가중치 초기화 함수
        
        #모델학습이 순조롭지않다면 향후 추가 예정
        return
    
    def extra_layers(self, input_size, output_size, div):
        layer = nn.Sequential(
            #conv2D 해상도낮추기
            nn.Conv2d(input_size, output_size, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(output_size, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.Hardswish(),
            
            #Inverted Residual (mobilev2 + squeeze)
            
            #depthwise 
            # kernel size 5 고려해보기
            nn.Conv2d(output_size, output_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=output_size, bias=False),
            nn.BatchNorm2d(output_size, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.Hardswish(),
            
            #SqueezeExcitation
            nn.Conv2d(output_size, output_size//div, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_size//div, output_size, kernel_size=(1, 1), stride=(1, 1)),
            
            #Point-wise
            nn.Conv2d(output_size, output_size, kernel_size=(1, 1), stride=(1, 1)),
            nn.Hardswish(),
            nn.Identity()
            
        ) 
        return layer
    
    def forward(self, x):
        f_maps=[]
        for n, layer in enumerate(self.backbone.features):#12 16
            if n==13: 
                #L2 norm
                #norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()     # (N, 1, 19, 19)
                #featureMap_1 = x / norm                                  # (N, 112, 19, 19)
                #featureMap_1= conv4_3 * self.rescale_factors            # (N, 512, 19, 19)
                
                #13계층 bottleneck까지만
                seq_layer = next(layer.children())
                for seq_n in range(4):
                    x = seq_layer[seq_n](x)
                    if seq_n==0:
                        f_maps.append(x)
                continue
                
            x = layer(x)
            #print("size : {0}, number = {1}".format(x.size(), n))
            if n==16:
                f_maps.append(x) 
               
        #extra layer
        mini_batch_n = x.size(0)
        for extra_layer in self.extra_layers:
            for idx, layer in enumerate(extra_layer.children()):
                if mini_batch_n ==1 and (idx == 1 or idx == 4):
                    continue
                x = layer(x)
            
            f_maps.append(x)
            
        cls = []
        loc = []
        for f_map, cls_layer, loc_layer in zip(f_maps,self. cls_layers, self.loc_layers):
            output_cls = cls_layer(f_map)
            output_loc = loc_layer(f_map)
            
            cls.append(output_cls.permute(0, 2, 3, 1).contiguous())
            loc.append(output_loc.permute(0, 2, 3, 1).contiguous())
            #cls.append(output_cls)
            #loc.append(output_loc)
            
        cls = torch.cat([o.view(o.size(0), -1) for o in cls], 1)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)

        loc = loc.view(loc.size(0), -1, 4)
        if self.state == "Train":
            cls = cls.view(cls.size(0), -1, self.n_class)
            #cls = self.softmax(cls.view(cls.size(0), -1, self.n_class))
        else:
            cls = self.softmax(cls.view(cls.size(0), -1, self.n_class))
            
            
        return cls, loc


# In[1]:


def build_model(inference_type, input_channels=1, is_freeze=True):

    backbone = models.mobilenet_v3_large(pretrained=True)
    if is_freeze:
        for name, layers in backbone.named_parameters():
            if name.split(".")[1]!="0":
                layers.requires_grad = False
            else:
                pass

    #for name, layers in backbone.named_children():
    #    for param in layers.parameters():
    #        param.requires_grad = False

    #for name, module in backbone.named_children():
    #    print(name)
    if input_channels == 1:
        for n, x in enumerate(backbone.features.children()):
            if n==0:
                seq=x.children()
                seq=next(seq)

                prev_weight = seq.weight
                new_weight = prev_weight[:, :1, :, :]
                seq.weight = nn.Parameter(new_weight)
                seq.in_channels = 1
            if n==13:
                seq=x.children()
                seq=next(seq)
    model = SSD(backbone, n_class=3, state = inference_type)
    return model


# In[ ]:




