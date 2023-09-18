#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import numpy as np
import torch
from torch import nn
from torch import optim
from defaultbox import default
from Detect import Detect

from matplotlib import pyplot as plt
import time

def get_iou(a,b):
    if len(a)!=4 or len(b)!=4:
        return 0
    a_area = (a[2]-a[0])*(a[3]-a[1])
    b_area = (b[2]-b[0])*(b[3]-b[1])
    
    union_area = a_area + b_area
    
    x1 = max(a[0], b[0])
    x2 = min(a[2], b[2])
    
    y1 = max(a[1], b[1])
    y2 = min(a[3], b[3])
    
    iter_area=0
    w = x2-x1
    h = y2-y1
    if w > 0 and h > 0:
        iter_area = w*h
    
    return iter_area / (union_area-iter_area)

def get_correct_ration(det_bboxes, target_bboxes, batch, n_class, iou_threshold = 0.2):
    '''
    det_bbox 
    type:list
    (cls, batch) data= numpy(검출한 객체개수, 5) xmin, ymin, xmax, ymax, cls(confidence)
    
    target_bbox
    type : list
    (batch,객체개수,5) xmin, ymin, xmax, ymax, cls
    정규화 되어있음
    '''
    # confidence type iou index
    TPFN=[[[-1, "FN", -1, -1] for _ in t_bboxes] for t_bboxes in target_bboxes]  #tp, fn 저장용 실측 기준
    TPFP=[[[[b_cls_bbox[-1], "FP", -1] for b_cls_bbox in b_cls_bboxes] for b_cls_bboxes in cls_bboxes] for cls_bboxes in det_bboxes]  #tp, fp 저장용 예측 기준

    for i in range(batch):
        for cl in range(1,n_class):
            t_bboxes = np.array(target_bboxes[i])#객체개수, 5
            d_bboxes = det_bboxes[cl][i]
            for t_i, t_bbox in enumerate(t_bboxes):
                if t_bbox[-1]==cl:
                    for d_i, d_bbox in enumerate(d_bboxes):
                        '''
                        1. 한 gt에 여러개 dbox 존재하는경우
                        -> 가장 높은 iou dbox 채택
                        2. 한 dbox에 높은 iou 를 가진 gt가 2개 존재하는경우
                        -> dbox 가 이미 tp이면 넘어감
                        '''
                        iou = get_iou(d_bbox[:-1],t_bbox[:-1])
                        
                        if TPFP[cl][i][d_i][1] == 'TP' and TPFP[cl][i][d_i][-1] != t_i: # 탐색과정에서 dbox가 다른 gt에 tp 일 경우
                            continue
 
                        if iou >= iou_threshold:
                            if TPFN[i][t_i][1]=='TP' and iou > TPFN[i][t_i][2]: # 가장 높은 iou를 가진 값을 채택
                                past_d_i =  TPFN[i][t_i][-1]
                                TPFN[i][t_i] = [d_bbox[-1], "TP", iou, d_i]
                                TPFP[cl][i][d_i] = [d_bbox[-1], "TP", iou, t_i]
                                TPFP[cl][i][past_d_i][1] = "FP"
                                
                            elif TPFN[i][t_i][1]=='FN':
                                TPFN[i][t_i] = [d_bbox[-1],"TP", iou, d_i]
                                TPFP[cl][i][d_i] = [d_bbox[-1], "TP", iou, t_i]                            
                else:
                    continue
            
    return TPFN, TPFP

def get_ap(tp_n, fp_n, fn_n, detect_list):
    detect_list.sort(key = lambda i : -i[0])
    recall = [] 
    precison = []
    tp_count = 0.
    fp_count = 0.
    ap = 0
    
    for conf, _type in detect_list:
        if _type == "FP":
            fp_count += 1
            
            recall_v = tp_count / (tp_n + fn_n)
            precison_v = tp_count / (tp_count + fp_count)
            
            recall.append(recall_v)
            precison.append(precison_v)
            
        elif _type == "TP": 
            tp_count += 1
            
            recall_v = tp_count / (tp_n + fn_n)
            precison_v = tp_count / (tp_count + fp_count)
            
            recall.append(recall_v)
            precison.append(precison_v)
            
    recall = np.array(recall)
    precison = np.array(precison)
    '''        
    11-point interpolation 방식 
    해당 코드 연산비용은 all point 방식과 비슷, 연습겸 작성한 코드
    '''

    recallRange = [x / 10. for x in range(0,11,1)]
    area = []
    for r in recallRange:
        Recalls = np.argwhere(recall >= r)
        pmax = 0

        if Recalls.size != 0:
            pmax = max(precison[Recalls.min():])

        area.append(pmax)

    ap = sum(area) / 11 # len(recallRange)
    
    return ap
def test_step(model, Data_loader, image_size=(300,300), device="cpu"):
    
    d=default()
    detect=Detect()#상위 n개에 대한 detection
    tensor_d = d.forward()
    tensor_d = tensor_d.to(device)  
    model.to(device)
    
    w, h = image_size #임시
    
    ap = 0
    total_Recall = 0
    total_Precison = 0
    N=0
    for idx, data in enumerate(Data_loader):
        images = data[0].to(device)
        labels = [label.cpu() for label in data[1]] # batch, 객체개수, 5
        
        total_labels=0
        for label in labels:
            total_labels+=len(label)

        with torch.no_grad():
            cls, loc = model(images)
            
            ''' 
            #For Debug
            cls = (batch,ddobx,3)
            value, i= torch.max(cls, dim=-1)
            s1 = i == 1
            s2 = i == 2
            bg = i == 0
            상위 200개에 대한 detection
            size=(batch, numclass ,200, 5)
            '''
            

            output = detect.forward(loc, cls, tensor_d, num_classes = cls.size(-1),  bkg_label=0, top_k=200, conf_thresh=0.4, nms_thresh=0.5)
        all_boxes = [[[] for _ in range(output.size(0))]
                 for _ in range(output.size(1))]  #all_boxes[cls][image]

        for i in range(output.size(0)):
            for j in range(1, output.size(1)):#class 종류
                dets = output[i, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()#confidence < 0
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0: #감지된게 없으면
                    continue
                boxes = dets[:, 1:]
                scores = dets[:, 0].cpu().numpy()

                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i] = cls_dets

        TPFN,TPFP = get_correct_ration(all_boxes, labels, batch = output.size(0), n_class = output.size(1), iou_threshold = 0.5)
        
        total_TP_1 = 0
        total_TP_2 = 0 #검수용
        total_FN = 0
        total_FP = 0
        
        for batch_TPFN in TPFN:
            for real_det in batch_TPFN:
                con, state = real_det[:2]
                if state=="FN":
                    total_FN+=1
                    
                if state=="TP":
                    total_TP_1+=1
                    
        TPFP_cls_filter=[[] for _ in range(output.size(1))] #n_class AP 계산용
        TPFP_filter=[] # total AP 계산용
        for cls_idx, cls_TPFP in enumerate(TPFP):
            if cls_idx == 0:
                continue
            TPFP_cls_list=[]
            for batch_TPFP in cls_TPFP:
                for pred_det in batch_TPFP:
                    if len(pred_det) == 0:
                        continue
                    con, state = pred_det[:2]
                    
                    if state=="FP":
                        total_FP+=1
                    
                    if state=="TP":
                        total_TP_2+=1
                        
                    TPFP_cls_list.append(pred_det[:2])
                    TPFP_filter.append(pred_det[:2])
                    
            TPFP_cls_filter[cls_idx] = TPFP_cls_list

        #print(TPFP)
        #print(TPFN)
        #print(total_TP_1,total_TP_2)
        #print(total_FP,total_FN)
        
        Precison = total_TP_1 / (total_TP_1 + total_FP) 
        Recall = total_TP_1 / (total_TP_1 + total_FN) #total_labels
        total_Recall += Recall
        total_Precison += Precison

        ap +=  get_ap(total_TP_1, total_FP, total_FN, TPFP_filter)
        N = idx+1
        
    total_Recall /= N
    total_Precison /= N
    mAP = ap / N
    return total_Recall, total_Precison, mAP

