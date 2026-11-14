"""Paper-informed Orthogonal Motion Conformer (OMC) model."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,d_model,max_length=8192):
        super().__init__(); pos=torch.arange(max_length).float().unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.)/d_model))
        pe=torch.zeros(max_length,d_model); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div[:pe[:,1::2].shape[1]])
        self.register_buffer('pe',pe.unsqueeze(0),persistent=False)
    def forward(self,x):
        if x.size(1)>self.pe.size(1): raise ValueError('Sequence too long')
        return x+self.pe[:,:x.size(1)]

class FeedForward(nn.Module):
    def __init__(self,d,mult=2,dropout=.1):
        super().__init__(); self.net=nn.Sequential(nn.Linear(d,d*mult),nn.SiLU(),nn.Dropout(dropout),nn.Linear(d*mult,d),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)

class ConformerConv(nn.Module):
    def __init__(self,d,kernel=15,dropout=.1):
        super().__init__()
        if kernel%2==0: raise ValueError('Conformer kernel must be odd')
        self.pw1=nn.Conv1d(d,2*d,1); self.dw=nn.Conv1d(d,d,kernel,padding=kernel//2,groups=d)
        self.bn=nn.BatchNorm1d(d); self.pw2=nn.Conv1d(d,d,1); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        x=x.transpose(1,2); x=F.glu(self.pw1(x),dim=1); x=self.dw(x); x=F.silu(self.bn(x)); x=self.pw2(x)
        return self.drop(x.transpose(1,2))

class ConformerBlock(nn.Module):
    def __init__(self,d,heads=4,ff_mult=2,kernel=15,dropout=.1):
        super().__init__(); self.n1=nn.LayerNorm(d); self.ff1=FeedForward(d,ff_mult,dropout)
        self.na=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True); self.ad=nn.Dropout(dropout)
        self.nc=nn.LayerNorm(d); self.conv=ConformerConv(d,kernel,dropout)
        self.n2=nn.LayerNorm(d); self.ff2=FeedForward(d,ff_mult,dropout); self.out=nn.LayerNorm(d)
    def forward(self,x):
        x=x+.5*self.ff1(self.n1(x)); q=self.na(x); x=x+self.ad(self.attn(q,q,q,need_weights=False)[0])
        x=x+self.conv(self.nc(x)); x=x+.5*self.ff2(self.n2(x)); return self.out(x)

class ArcFaceClassifier(nn.Module):
    def __init__(self,dim,nclass,scale=30.,margin=.35):
        super().__init__(); self.weight=nn.Parameter(torch.empty(nclass,dim)); nn.init.xavier_uniform_(self.weight)
        self.scale=float(scale); self.margin=float(margin)
    def logits(self,z,labels=None,with_margin=False):
        w=F.normalize(self.weight,dim=1); cosine=F.linear(F.normalize(z,dim=-1),w).clamp(-1+1e-6,1-1e-6)
        if not with_margin or labels is None: return self.scale*cosine
        target=torch.cos(torch.acos(cosine)+self.margin); mask=F.one_hot(labels,num_classes=self.weight.shape[0]).bool()
        return self.scale*torch.where(mask,target,cosine)
    def orthogonality_loss(self):
        w=F.normalize(self.weight,dim=1); gram=w@w.T; n=gram.shape[0]
        if n<2: return gram.sum()*0
        mask=~torch.eye(n,dtype=torch.bool,device=gram.device); return gram[mask].square().mean()

class OMC(nn.Module):
    def __init__(self,seq_dim,global_dim,num_classes,dim=64,depth=6,heads=4,embed_dim=128,ff_mult=2,conv_kernel=15,dropout=.1,use_emf=True,use_vat=True,input_mode='accgyro6'):
        super().__init__(); self.use_emf=use_emf; self.use_vat=use_vat
        if use_emf: self.channel_indices=list(range(seq_dim)); extra=global_dim
        else: self.channel_indices=list(range(3)) if input_mode=='acc3' else [0,1,2,12,13,14]; extra=0
        self.proj=nn.Linear(len(self.channel_indices),dim); self.pos=SinusoidalPositionalEncoding(dim)
        self.blocks=nn.ModuleList([ConformerBlock(dim,heads,ff_mult,conv_kernel,dropout) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim); self.embedding=nn.Sequential(nn.Linear(dim+extra,embed_dim),nn.LayerNorm(embed_dim))
        self.velocity_head=nn.Linear(embed_dim,3) if use_vat else None; self.arcface=ArcFaceClassifier(embed_dim,num_classes); self.extra_global_dim=extra
    def forward(self,seq,global_motion):
        x=self.pos(self.proj(seq[...,self.channel_indices]))
        for block in self.blocks: x=block(x)
        pooled=self.norm(x).mean(1)
        if self.extra_global_dim: pooled=torch.cat([pooled,global_motion],dim=-1)
        emb=self.embedding(pooled); z=F.normalize(emb,dim=-1)
        return z,self.velocity_head(emb) if self.velocity_head is not None else None

class BoundedUncertaintyWeighting(nn.Module):
    def __init__(self): super().__init__(); self.params=nn.Parameter(torch.zeros(2))
    def forward(self,a,b):
        w=torch.tanh(self.params); return .5*torch.exp(-w[0])*a+.5*torch.exp(-w[1])*b+.5*w.sum()

def compute_loss(model,awl,z,vlogits,identity,velocity,ort_lambda=.1,use_arcface=True):
    logits=model.arcface.logits(z,identity,with_margin=use_arcface); cls=F.cross_entropy(logits,identity)
    ort=model.arcface.orthogonality_loss() if ort_lambda else cls.detach()*0
    lid=cls+ort_lambda*ort
    if vlogits is not None:
        lv=F.cross_entropy(vlogits,velocity); total=awl(lid,lv)
    else: lv=cls.detach()*0; total=lid
    return total,{'classification':float(cls.detach()),'orthogonal':float(ort.detach()),'velocity':float(lv.detach()),'total':float(total.detach())}
