#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch import nn
import torch.nn.functional as F
from utils import *

class MultiBoxLoss(nn.Module):
    """SSD의 손실함수 클래스 """

    def __init__(self, thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = thresh  # 0.5 match 함수의 jaccard 계수의 임계치
        self.negpos_ratio = neg_pos  # 3:1 Hard Negative Mining의 음과 양 비율
        self.device = device  # 계산 device(CPU | GPU)

    def forward(self, predictions, targets, dboxs):
        """
        파라미터 설명
        ----------
        predictions :모델의 예측값 cls와 loc
        cls는 batch dbox의 개수, 클래스 개수로 이루어짐
        loc은 batch dbox의 개수, 4
        
        targets : [num_batch, 객체개수, 5]
            5는 라벨 정보[xmin, ymin, xmax, ymax, label_ind]

        """

        conf_data, loc_data = predictions
        dbox_list = dboxs
        
        num_batch = loc_data.size(0)  # 배치 크기
        num_dbox = loc_data.size(1)  # DBox의 수 
        num_classes = conf_data.size(2)  # 클래스 수

        # 손실 계산에 사용할 것을 저장하는 변수 작성
        # conf_t_label：각 DBox에 가장 가까운 정답 BBox의 라벨을 저장 
        # loc_t: 각 DBox에 가장 가까운 정답 BBox의 위치 정보 저장 
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        
        #conf_t_label.fill_(0)#테스트용도
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        # loc_t와 conf_t_label에 
        # DBox와 정답 어노테이션 targets를 amtch한 결과 덮어쓰기
        for idx in range(num_batch):  # 미니 배치 루프

            truths = targets[idx][:, :-1].to(self.device)
            labels = targets[idx][:, -1].to(self.device)
            dbox = dbox_list.to(self.device)

            # match 함수를 실행하여 loc_t와 conf_t_label 내용 갱신
            # loc_t: 각 DBox에 가장 가까운 정답 BBox 위치 정보가 덮어써짐.
            # conf_t_label：각 DBox에 가장 가까운 정답 BBox 라벨이 덮어써짐.
            # 단, 가장 가까운 BBox와 iou가 0.5보다 작은 경우,
            # 정답 BBox의 라벨 conf_t_label은 배경 클래스 0으로 한다.
            variance = [0.1, 0.2]
            
            # 라벨을 dbox에 대한 offset으로 변환
            match(self.jaccard_thresh, truths, dbox,
                  variance, labels, loc_t, conf_t_label, idx)


        #물체를 발견한 offset만 손실 계산
        pos_mask = conf_t_label > 0  # size: batch,dbox,1
        true_count = pos_mask[0].sum().item()

        # pos_mask를 loc_data 크기로 변형
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)
        
        # Positive DBox의 loc_data와 offset loc_t 취득
        loc_p = loc_data[pos_idx].view(-1, 4).to(self.device)
        loc_t = loc_t[pos_idx].view(-1, 4)

        # 물체를 발견한 Positive DBox의 오프셋 정보 loc_t의 손실(오차)를 계산
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        # ----------
        # 클래스 예측의 손실 : loss_c를 계산
        # 교차 엔트로피 오차 함수로 손실 계산. 단 배경 클래스가 정답인 DBox가 압도적으로 많으므로,
        # Hard Negative Mining을 실시하여 물체 발견 DBox 및 배경 클래스 DBox의 비율이 1:3이 되도록 한다.
        # 배경 클래스 DBox로 예상한 것 중 손실이 적은 것은 클래스 예측 손실에서 제외
        # ----------
        batch_conf = conf_data.view(-1, num_classes)

        # 클래스 예측의 손실함수 계산(reduction='none'으로 하여 합을 취하지 않고 차원 보존)
        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none')#batch * dbox_n

        
        #------------이 아래 부분은 이해 못함 

        
        
        # -----------------
        # Negative DBox중 Hard Negative Mining으로 
        # 추출하는 것을 구하는 마스크 작성
        # -----------------

        # 물체를 발견한 Positive DBox의 손실을 0으로 한다.
        # (주의) 물체는 label이 1 이상, 라벨 0은 배경을 의미
        num_pos = pos_mask.long().sum(1, keepdim=True)  # 미니 배치별 물체 클래스 예측 수
        loss_c = loss_c.view(num_batch, -1)  # torch.Size([num_batch, dbox])
        loss_c[pos_mask] = 0  # 물체를 발견한 DBox는 손실 0으로 한다.

        # Hard Negative Mining
        # 각 DBox 손실의 크기 loss_c 순위 idx_rank를 구함
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        #  (주의) 구현된 코드는 특수하며 직관적이지 않음.
        # 위 두 줄의 요점은 각 DBox에 대해 손실 크기가 몇 번째인지의 정보를 
        # idx_rank 변수로 빠르게 얻는 코드이다.
        
        # DBox의 손실 값이 큰 쪽부터 내림차순으로 정렬하여, 
        # DBox의 내림차순의 index를 loss_idx에 저장한다.
        # 손실 크기 loss_c의 순위 idx_rank를 구한다.
        # 내림차순이 된 배열 index인 loss_idx를 0부터 8732까지 오름차순으로 다시 정렬하기 위하여
        # 몇 번째 loss_idx의 인덱스를 취할 지 나타내는 것이 idx_rank이다.
        # 예를 들면 idx_rank 요소의 0번째 = idx_rank[0]을 구하는 것은 loss_idx의 값이 0인 요소,
        # 즉 loss_idx[?] =0은 원래 loss_c의 요소 0번째는 내림차순으로 정렬된 loss_idx의 
        # 몇 번째입니까? 를구하는 것이 되어 결과적으로, 
        # ? = idx_rank[0]은 loss_c의 요소 0번째가 내림차순으로 몇 번째인지 나타냄

        # 배경 DBox의 수 num_neg를 구한다. HardNegative Mining으로 
        # 물체 발견 DBox의 수 num_pos의 세 배 (self.negpos_ratio 배)로 한다.
        # DBox의 수를 초과한 경우에는 DBox의 수를 상한으로 한다.
        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)

        # idx_rank에 각 DBox의 손실 크기가 위에서부터 몇 번째인지 저장되었다.
        # 배경 DBox의 수 num_neg보다 순위가 낮은(손실이 큰) DBox를 취하는 마스크 작성
        # torch.Size([num_batch, 8732])
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # -----------------
        # (종료) 지금부터 Negative DBox 중 Hard Negative Mining으로 추출할 것을 구하는 마스크를 작성
        # -----------------

        # 마스크 모양을 고쳐 conf_data에 맞춘다
        # pos_idx_mask는 Positive DBox의 conf를 꺼내는 마스크이다.
        # neg_idx_mask는 Hard Negative Mining으로 추출한 Negative DBox의 conf를 꺼내는 마스크이다.
        # pos_mask：torch.Size([num_batch, 8732])
        # --> pos_idx_mask：torch.Size([num_batch, 8732, 21])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # conf_data에서 pos와 neg만 꺼내서 conf_hnm으로 한다. 
        # 형태는 torch.Size([num_pos+num_neg, 21])
        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)
                             ].view(-1, num_classes)
        # gt는 greater than (>)의 약칭. mask가 1인 index를 꺼낸다
        # pos_idx_mask+neg_idx_mask는 덧셈이지만 index로 mask를 정리할 뿐임.
        # pos이든 neg이든 마스크가 1인 것을 더해 하나의 리스트로 만들어 이를 gt로 췯그한다.

        # 마찬가지로 지도 데이터인 conf_t_label에서 pos와 neg만 꺼내, conf_t_label_hnm 으로 
        # torch.Size([pos+neg]) 형태가 된다
        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

        # confidence의 손실함수 계산(요소의 합계=sum을 구함)
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        # 물체를 발견한 BBox의 수 N (전체 미니 배치의 합계) 으로 손실을 나눈다.
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N
        
        #print("-"*100)
        
        return loss_l, loss_c


# In[ ]:




