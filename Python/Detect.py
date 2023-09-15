#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from utils import *

class Detect:
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
        
    def forward(self, loc_data, conf_data, prior_data, num_classes, bkg_label=0, top_k=30, conf_thresh=0.4, nms_thresh=0.5):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = [0.1, 0.2]
        
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1) # batch, num_class, dbox_num 클래스별 confidence 취득
        # Decode predictions into bboxes.
        for i in range(num): #배치별 반복
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)#xmin,xmax,ymin,ymax 변환
            # For each class, perform nms
            conf_scores = conf_preds[i].clone() # (num_class, dbox)

            for cl in range(1, self.num_classes):#클래스별 confidence계산
                c_mask = conf_scores[cl].gt(self.conf_thresh) #해당값 이하면 false 이상이면 True , size= (dbox)
                scores = conf_scores[cl][c_mask] #인식된 default box의 점수만 남음

                if scores.size(0) == 0: #인식된 개체 없으면 넘어감
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes) # dbox,1 -> dbox,4
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


# In[ ]:




