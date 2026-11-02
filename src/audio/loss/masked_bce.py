import torch
from torch.nn import functional as F
def masked_multitask_bce(logits,targets,observed,pos_weight=None):
    losses=[]
    for i in range(2):
        m=observed[:,i].bool()
        if m.any(): losses.append(F.binary_cross_entropy_with_logits(logits[m,i],targets[m,i].float(),pos_weight=None if pos_weight is None else pos_weight[i]))
    return sum(losses)/len(losses) if losses else logits.sum()*0
