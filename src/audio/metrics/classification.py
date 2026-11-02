def binary_metrics(preds,targets,observed):
    out={}
    for i,name in enumerate(("depression","parkinson")):
        keep=observed[:,i].bool(); p=preds[keep,i].int(); y=targets[keep,i].int(); tp=((p==1)&(y==1)).sum().item(); tn=((p==0)&(y==0)).sum().item(); fp=((p==1)&(y==0)).sum().item(); fn=((p==0)&(y==1)).sum().item(); pr=tp/max(tp+fn,1); hr=tn/max(tn+fp,1); f1p=2*tp/max(2*tp+fp+fn,1); f1h=2*tn/max(2*tn+fp+fn,1); n=int(keep.sum()); out[name]={"count":n,"Macro F1":(f1p+f1h)/2,"Weighted F1":(f1p*(tp+fn)+f1h*(tn+fp))/max(n,1),"Macro Recall":(pr+hr)/2,"Weighted Recall":(pr*(tp+fn)+hr*(tn+fp))/max(n,1),"Patient Recall":pr,"Healthy Recall":hr}
    return out

def corpus_average(care, wsm):
    """Metric-wise arithmetic Average; inputs are separate corpus metric blocks."""
    return {task: {metric: (care[task][metric] + wsm[task][metric]) / 2 for metric in care[task] if metric != "count"} for task in care}
