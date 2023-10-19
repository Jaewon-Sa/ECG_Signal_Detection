

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=2400, img_size=300):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, (50,patch_size), stride=(50, patch_size), padding=(0,2)),
            Rearrange('b e (h) (w) -> b (h w) e')
        )

        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn(114+1, emb_size))

    def forward(self, x):
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding to prejected patches
        x += self.positions
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.num_heads) # b, 197, 728 -> b, 8, 197, 91
        keys = rearrange(self.keys(x), 'b n (h d) -> b h n d', h=self.num_heads)
        values = rearrange(self.values(x), 'b n (h d) -> b h n d', h=self.num_heads)
        # sum up over the last axis, b,h,197,197
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_head, query_len, key_len
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values) # 197x91
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x



class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, drop_p=0., forward_expansion=4, forward_drop_p=0., **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class CustomHeadCls(nn.Module):
    def __init__(self, emb_size=768, box_n = 4, n_classes = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            #Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, box_n * n_classes),
            nn.ReLU(),
            nn.Linear(box_n * n_classes, 256),
            nn.ReLU(),
            nn.Linear(256, box_n * n_classes),
            Rearrange('b e (an out) -> b e an out', an = box_n),
            Rearrange('b e an out -> b (e an) out')
        )
        
    def forward(self, x, mask=None):
        x=x[:,1:,:]
        x = self.mlp(x)
        return x
    
class CustomHeadLoc(nn.Module):
    def __init__(self, emb_size=768, box_n = 4, n_classes = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            #Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, box_n * n_classes),
            nn.ReLU(),
            nn.Linear(box_n * n_classes, 256),
            nn.ReLU(),
            nn.Linear(256, box_n * n_classes),
            Rearrange('b e (an out) -> b e an out', an = box_n),
            Rearrange('b e an out -> b (e an) out')
        )
        
    def forward(self, x, mask=None):
        x=x[:,1:,:]
        x = self.mlp(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, 
                 emb_size=2400, img_size=300, depth=12, 
                 box_n=4, n_classes=7, **kwargs):

        super().__init__()
        self.PatchEmbedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.TransformerEncoder = TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        #self.CustomHead = CustomHead(emb_size, box_n, n_classes)
        self.CustomHeadCls = CustomHeadCls(emb_size, box_n, n_classes-4)
        self.CustomHeadLoc = CustomHeadLoc(emb_size, box_n, 4)
        
    def forward(self, x):
        x = self.PatchEmbedding(x)
        x = self.TransformerEncoder(x)
        cls = self.CustomHeadCls(x)
        loc = self.CustomHeadLoc(x)
        return cls, loc

class VIT_Det(nn.Module):
    def __init__(self, in_channels=3, emb_size=800, n_class=3, default_box_n=4, depth=6, state = "Test"):
        super().__init__()
        emb_size = emb_size*in_channels
        self.state = state
        self.softmax = nn.Softmax(dim=-1)
        self.n_class = n_class
        self.vit= ViT(in_channels=in_channels, patch_size=16, emb_size=emb_size, img_size=300, 
                      depth=depth, box_n=default_box_n, n_classes = n_class+4)

    def forward(self, x):
        cls, loc = self.vit(x)

        if self.state == "Test":
            cls = self.softmax(cls.view(cls.size(0), -1, self.n_class))
        return cls, loc
    







