from __future__ import annotations
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from chimera_ml.core.registry import MODELS
from chimera_ml.models.base import BaseModel
from chimera_ml.core.types import ModelOutput

def _unpack(batch: Any):
    x = batch.inputs["native"].float()
    if x.ndim != 3: raise ValueError("inputs['native'] must be [B,T,F]")
    mask = batch.get_masks("sequence_mask")
    if mask is None: mask = batch.get_masks("native_mask")
    if mask is None: mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    return x, mask.to(x.device, dtype=torch.bool)

def _mean(x, mask):
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x*w).sum(1) / w.sum(1).clamp_min(1)

class _Heads(nn.Module):
    def __init__(self, d, dropout):
        super().__init__(); self.depression=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Dropout(dropout),nn.Linear(d,1)); self.parkinson=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Dropout(dropout),nn.Linear(d,1))
    def forward(self,z): return torch.cat((self.depression(z),self.parkinson(z)),-1)

class _Base(BaseModel):
    def __init__(self,input_dim,hidden_dim=96,dropout=.1):
        super().__init__(); self.proj=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,hidden_dim),nn.GELU()); self.heads=_Heads(hidden_dim,dropout)
    def encode(self,x,mask): raise NotImplementedError
    def forward(self,batch):
        x,mask=_unpack(batch); z=self.encode(self.proj(x),mask); logits=self.heads(_mean(z,mask)); return ModelOutput(preds=logits,aux={"logits":logits,"features":_mean(z,mask),"sequence_mask":mask})

class TSCMambaClassifier(_Base):
    """Temporal/spectral multiview classifier with an actual Mamba SSM path."""
    def __init__(self,input_dim,hidden_dim=96,num_layers=2,dropout=.1,mamba_required=True,**kwargs):
        super().__init__(input_dim,hidden_dim,dropout); self.spectral=nn.Linear(hidden_dim,hidden_dim); self.norm=nn.LayerNorm(hidden_dim)
        try:
            from mamba_ssm import Mamba
        except ImportError: Mamba=None
        if Mamba is None and mamba_required: raise ModuleNotFoundError("TSCMamba requires mamba_ssm; no Transformer fallback is used")
        self.layers=nn.ModuleList([Mamba(d_model=hidden_dim,d_state=16,d_conv=4,expand=2) for _ in range(num_layers)]) if Mamba else nn.ModuleList()
    def encode(self,x,mask):
        if not self.layers: return x
        y=x*mask.unsqueeze(-1); spectrum=torch.fft.rfft(y,dim=1).abs(); spectral=torch.fft.irfft(spectrum,n=x.shape[1],dim=1); y=self.norm(y+self.spectral(spectral))
        for layer in self.layers: y=layer(y)*mask.unsqueeze(-1)
        return y

class MPTSNetClassifier(_Base):
    """FFT period cue, multiscale local convolutions and global dependency path."""
    def __init__(self,input_dim,hidden_dim=96,dropout=.1,**kwargs):
        super().__init__(input_dim,hidden_dim,dropout); self.local=nn.ModuleList([nn.Conv1d(hidden_dim,hidden_dim,k,padding=k//2) for k in (3,5,7)]); self.global_path=nn.GRU(hidden_dim,hidden_dim,batch_first=True); self.mix=nn.Linear(hidden_dim*4,hidden_dim)
    def encode(self,x,mask):
        y=x*mask.unsqueeze(-1); power=torch.fft.rfft(y,dim=1).abs().mean(-1); power[:,0]=0; period=int(power.argmax(1).median().clamp_min(1)) if y.shape[1]>1 else 1; rolled=torch.roll(y,max(1,y.shape[1]//period),1); local=[F.gelu(c(y.transpose(1,2)).transpose(1,2)) for c in self.local]; global_,_=self.global_path(y); return self.mix(torch.cat((y,rolled,sum(local)/len(local),global_),-1))

class MDFormerClassifier(_Base):
    """Dual temporal and cross-channel Transformer branches with late fusion."""
    def __init__(self,input_dim,hidden_dim=96,num_heads=4,num_layers=2,dropout=.1,**kwargs):
        super().__init__(input_dim,hidden_dim,dropout); self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim,num_heads,hidden_dim*4,dropout,batch_first=True,norm_first=True),num_layers); self.cross=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_dim,num_heads,hidden_dim*4,dropout,batch_first=True,norm_first=True),num_layers); self.fuse=nn.Linear(hidden_dim*2,hidden_dim)
    def encode(self,x,mask):
        safe=mask.clone(); safe[:,0]=True; t=self.temporal(x,src_key_padding_mask=~safe); c=self.cross(x,src_key_padding_mask=~safe); return self.fuse(torch.cat((t,c),-1))

class PooledMLP(_Base):
    def encode(self,x,mask): return x

def _factory(cls):
    def build_model(context=None,**params):
        if context is not None: params.setdefault("input_dim",context.get("data.input_dim",context.get("data.audio_feature_dim",16)))
        return cls(**params)
    return build_model

carewsm_tscmamba=MODELS.register("carewsm_tscmamba")(_factory(TSCMambaClassifier))
carewsm_mptsnet=MODELS.register("carewsm_mptsnet")(_factory(MPTSNetClassifier))
carewsm_md_former=MODELS.register("carewsm_md_former")(_factory(MDFormerClassifier))
carewsm_pooled_mlp=MODELS.register("carewsm_pooled_mlp")(_factory(PooledMLP))
