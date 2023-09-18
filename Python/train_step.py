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
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optim_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_func = MultiBoxLoss(device=device)
    
    epoch_train_loss = 0.0 # 에포크 손실 합
    #epoch_val_loss = 0.0
    
    for epoch in range(epoch_num):
        epoch_start = time.time()
        iter_start = time.time()
        print("Epoch : {0} / {1}".format(epoch+1,epoch_num))
        
        #targets=np.array([[[1,2,3,4,5],[1,2,3,4,5]],[[1,2,3,4,5],[1,2,3,4,5]]])
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
                
                print("Current Batch {0} / {1} | Cls Loss : {2:.3f}, Loc Loss : {3:.3f}, Total Loss : {4:.3f} | 50 iter time {5:.4f}: "
                      .format(idx, len(train_Data_loader), loss_l.item(), loss_c.item(), loss.item(), iter_end - iter_start))
                
                if is_wandb==True:
                    wandb.log({"total_loss": loss.item(),
                               "Cls_loss": loss_c.item(),
                               "Loc_loss": loss_l.item()})
                
                iter_start =time.time()
                
            epoch_train_loss+=loss.item()
            
        epoch_end = time.time() 
        
        if (epoch + 1) % 3 == 0 and  (epoch + 1) >= 15:
            train_parameters = model.state_dict()
            test_model.load_state_dict(train_parameters)
            test_model.eval()
            
            torch.cuda.empty_cache()
            
            eval_start = time.time()
            mRecall, mPrecison, mAP = test_step(test_model, valid_Data_loader, device = device)

            eval_end = time.time()
            print((f'Epoch : {epoch+1} / {epoch_num} | Total Loss : {epoch_train_loss:.3f}' 
                   f'| 1 epoch update time : {epoch_end-epoch_start:.2f}s |' 
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




