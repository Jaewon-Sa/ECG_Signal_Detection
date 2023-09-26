#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import time
import torch
from torch import nn
from torch import optim
from defaultbox import default
from loss import MultiBoxLoss
from eval_step import *
DIR_PATH = "./objectdetection_model"     

import math
from torch.optim.lr_scheduler import _LRScheduler

#
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

            
def train_step(model, test_model, train_Data_loader, valid_Data_loader, epoch_num, batchsize, 
               optim_type="SGD", lr=2e-3, device="cpu", model_name="BASE", 
               is_wandb=False, is_freeze=True):
    
    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)
        print(f"make '{DIR_PATH}' DIR path")
    else:
        print(f"Already '{DIR_PATH}' DIR path")
            
    if is_wandb==True:
        wandb.init(
            # set the wandb project where this run will be logged
            project="Heart_Signal_Detection",

            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "architecture": "Mobilenet_v3 + SSD",
            "dataset": "circor-heart-sound",
            "epochs": epoch_num,
            "batch" : batchsize 
            }
        )
        
    d=default()
    tensor_d = d.forward()
    
    if optim_type =="SGD":
        optimizer = optim.SGD(model.parameters(), lr=1e-4)
    elif optim_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=2, eta_max=lr,  T_up=3, gamma=0.5)
    
    loss_func = MultiBoxLoss(device=device)
    
    epoch_train_loss = 0.0 # 에포크 손실 합
    #epoch_val_loss = 0.0
    
    for epoch in range(epoch_num):
        epoch_start = time.time()
        iter_start = time.time()
        print("Epoch : {0} / {1}".format(epoch+1,epoch_num))
        
        #targets=np.array([[[1,2,3,4,5],[1,2,3,4,5]],[[1,2,3,4,5],[1,2,3,4,5]]])
        total_batch_size = len(train_Data_loader)
        for idx, data in enumerate(train_Data_loader):
            
            #img=torch.zeros((32,3,300,300),dtype=torch.float)
            model.to(device)
            images = data[0].to(device)
            
            labels = [label.to(device) for label in data[1]]
            
            optimizer.zero_grad()#이전 값 들에 대한 가중치 기울기 초기화
            with torch.set_grad_enabled(True):
                cls, loc = model(images)
                tensor_d = tensor_d.to(device)
                loss_l, loss_c = loss_func((cls, loc), labels, tensor_d)
                
                loss = loss_l + loss_c
                loss.backward()
                
                #nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                        
                
                optimizer.step() # 파라미터 갱신
            if(idx % 50 == 0):
                iter_end = time.time()
                
                print(f'Current Batch {idx} / {total_batch_size} '
                      f'learning rate : {scheduler.get_lr()} | Cls Loss : {loss_l.item():.3f},'
                      f'Loc Loss : {loss_c.item():.3f}, Total Loss : {loss.item():.3f} |'
                      f'50 iter time {iter_end - iter_start:.4f}: ')
                
                if is_wandb==True:
                    wandb.log({"total_loss": loss.item(),
                               "Cls_loss": loss_c.item(),
                               "Loc_loss": loss_l.item()})
                
                iter_start =time.time()
                
            epoch_train_loss+=loss.item()
            
        epoch_end = time.time() 
        scheduler.step()
        if (epoch + 1) % 3 == 0 and  (epoch + 1) >= 15:
            train_parameters = model.state_dict()
            test_model.load_state_dict(train_parameters)
            test_model.eval()
            
            torch.cuda.empty_cache()
            
            eval_start = time.time()
            mRecall, mPrecison, mAP = test_step(test_model, valid_Data_loader, device = device)

            eval_end = time.time()
            print((f'Epoch : {epoch+1} / {epoch_num} | Total Loss : {epoch_train_loss:.3f}' 
                   f'| 1 epoch update time : {epoch_end-epoch_start:.2f}s | learning rate : {scheduler.get_lr()} |' 
                   f'mRecall : {mRecall:.2%} , mPrecison : {mPrecison:.2%}, mAP: {mAP:.3f}, eval_time : {eval_end - eval_start:.2f}s'))
            torch.cuda.empty_cache()
            
        else:
            print("Epoch : {0} / {1} of Total Loss : Total Loss : {2:.3f} | 1 epoch update time : {3:.2f}s"
                  .format(epoch+1, epoch_num, epoch_train_loss,epoch_end-epoch_start))
        print("-----------------------------------------------")
        epoch_train_loss=0
        
        torch.save(model, 
                f"{DIR_PATH}/{model_name}_{epoch+1}_{batchsize}_{optim_type}_{is_freeze}.pth")
        torch.save(model.state_dict(), 
                f"{DIR_PATH}/{model_name}_weight_{epoch+1}_{batchsize}_{optim_type}_{is_freeze}.pth")
    if is_wandb==True:
        wandb.finish()


# In[ ]:




