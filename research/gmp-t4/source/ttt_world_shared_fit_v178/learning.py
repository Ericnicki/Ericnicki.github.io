import numpy as np,torch
from torch.nn import functional as F
from common import tensor
from model import reader_features
def batch(tasks,queries):
 ps=[tasks[tid]['pairs'][qi] for tid,qi in queries];B=len(ps);T=max(len(p['support_features']) for p in ps);sf=np.zeros((B,T,95),np.float32);sy=np.zeros((B,T,22),np.float32);valid=np.zeros((B,T),np.float32)
 for i,p in enumerate(ps):
  n=len(p['support_features']);sf[i,:n]=p['support_features'];sy[i,:n]=p['support_targets'];valid[i,:n]=1
 arrays=dict(support=tensor(sf),support_target=tensor(sy),support_valid=tensor(valid))
 for key in ['state','context','first_command','anchor_acc','controller','target','value','corrections','actual_first_commands']:arrays[key]=tensor(np.stack([p[key] for p in ps]))
 return arrays
def sample(tasks,train_ids,rng,n):return [(int(tid),int(rng.integers(len(tasks[int(tid)]['pairs'])))) for tid in rng.choice(train_ids,n,replace=False)]
def features(b,corrections=None):
 B=len(b['state']);corr=b['corrections'] if corrections is None else corrections;K=corr.shape[-1]
 expand=lambda x:x.unsqueeze(1).expand(-1,K,-1)
 return reader_features(expand(b['state']),expand(b['context']),expand(b['first_command']),corr,b['anchor_acc'].unsqueeze(1).expand(-1,K),expand(b['controller']))
def world_loss(w,b):
 W,support_loss=w.adapt(b['support'],b['support_target'],b['support_valid']);pred=w.predict(features(b),W);weight=w.coordinate_weight
 absolute=((pred-b['target']).square()*weight).mean();pred_slope=(pred[:,2]-pred[:,1])/.1;actual_slope=(b['target'][:,2]-b['target'][:,1])/.1;contrast=((pred_slope-actual_slope).square()*weight).mean()
 return absolute+.25*support_loss+.5*contrast,dict(absolute=absolute,support=support_loss,contrast=contrast)
def value_loss(v,b):
 c=b['context'].unsqueeze(1).expand(-1,3,-1);pred=v(b['target'],c);target=b['value'];absolute=F.smooth_l1_loss(pred[...,0],target[...,0])+.5*F.smooth_l1_loss(pred[...,1:],target[...,1:]);contrast=F.mse_loss((pred[:,2,1:]-pred[:,1,1:])/.1,(target[:,2,1:]-target[:,1,1:])/.1)
 return absolute+.1*contrast,dict(absolute=absolute,contrast=contrast)
