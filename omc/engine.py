import csv,json,random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score,roc_auc_score,confusion_matrix

def fix_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def get_device(req='auto'):
    if req=='auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(req)

def train_epoch(model,awl,loader,opt,device,ort_lambda=.1,use_arcface=True):
    from .model import compute_loss
    model.train(); awl.train(); sums={}; n=0
    for b in loader:
        seq=b['seq'].to(device); glob=b['global'].to(device); y=b['identity'].to(device); v=b['velocity'].to(device)
        opt.zero_grad(set_to_none=True); z,vl=model(seq,glob)
        loss,parts=compute_loss(model,awl,z,vl,y,v,ort_lambda,use_arcface)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(awl.parameters()),1.)
        opt.step()
        for k,val in parts.items(): sums[k]=sums.get(k,0.)+val
        n+=1
    return {k:v/max(n,1) for k,v in sums.items()}

@torch.no_grad()
def inference(model,loader,device):
    model.eval(); ps=[]; ys=[]; rows=[]
    for b in loader:
        z,vl=model(b['seq'].to(device),b['global'].to(device)); probs=torch.softmax(model.arcface.logits(z),dim=1).cpu().numpy(); y=b['identity'].numpy()
        vp=None if vl is None else torch.softmax(vl,dim=1).cpu().numpy(); ps.append(probs); ys.append(y)
        for i in range(len(y)):
            r={'path':b['path'][i],'subject':b['subject'][i],'character':b['character'][i],
               'identity_label':int(y[i]),'predicted_label':int(probs[i].argmax()),'confidence':float(probs[i].max())}
            if vp is not None: r.update(velocity_label=int(b['velocity'][i]),velocity_prediction=int(vp[i].argmax()))
            rows.append(r)
    if not ps: raise ValueError('No evaluation samples')
    return np.concatenate(ys),np.concatenate(ps),rows

def metrics(y,p,nclass):
    pred=p.argmax(1); m={'n_samples':len(y),'accuracy':float(accuracy_score(y,pred)),
      'macro_f1':float(f1_score(y,pred,average='macro',labels=np.arange(nclass),zero_division=0)),
      'cohen_kappa':float(cohen_kappa_score(y,pred,labels=np.arange(nclass))),
      'class_counts':np.bincount(y,minlength=nclass).tolist()}
    if len(set(y.tolist()))==nclass:
        try: m['roc_auc_ovr_macro']=float(roc_auc_score(np.eye(nclass)[y],p,average='macro'))
        except ValueError: m['roc_auc_ovr_macro']=None
    else: m['roc_auc_ovr_macro']=None
    return m,confusion_matrix(y,pred,labels=np.arange(nclass))

def write_report(out,met,cm,rows,subjects):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(met,ensure_ascii=False,indent=2),encoding='utf8')
    np.savetxt(out/'confusion_matrix.csv',cm,fmt='%d',delimiter=',')
    ext=[]
    for row in rows:
        r=dict(row); r['true_subject']=subjects[r['identity_label']]; r['predicted_subject']=subjects[r['predicted_label']]; ext.append(r)
    with (out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ext[0].keys())); w.writeheader(); w.writerows(ext)
