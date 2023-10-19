
import torch
from torch import nn
import torch.nn.functional as F
from utils import *

class MultiBoxLoss(nn.Module):
    def __init__(self, thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = thresh  # 0.5 match 함수의 jaccard 계수의 임계치
        self.negpos_ratio = neg_pos  # 3:1 Hard Negative Mining의 음과 양 비율
        self.device = device 

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
        batch_conf = conf_data.view(-1, num_classes)

        # 클래스 예측의 손실함수 계산(reduction='none'으로 하여 합을 취하지 않고 차원 보존)
        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none')#batch * dbox_n

        num_pos = pos_mask.long().sum(1, keepdim=True)  # 미니 배치별 물체 클래스 예측 수
        loss_c = loss_c.view(num_batch, -1)  # torch.Size([num_batch, dbox])
        loss_c[pos_mask] = 0  # 물체를 발견한 DBox는 손실 0으로 한다.

        # Hard Negative Mining
        # 각 DBox 손실의 크기 loss_c 순위 idx_rank를 구함
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)
                             ].view(-1, num_classes)
        
        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

        # confidence의 손실함수 계산(요소의 합계=sum을 구함)
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        # 물체를 발견한 BBox의 수 N (전체 미니 배치의 합계) 으로 손실을 나눈다.
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N
        
        #print("-"*100)
        
        return loss_l, loss_c




