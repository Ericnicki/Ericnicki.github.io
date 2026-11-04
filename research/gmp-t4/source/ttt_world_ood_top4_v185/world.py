"""Physical TTT writer and policy-conditioned macro-action reader."""
import math
import torch
from torch import nn
from torch.nn import functional as F
def mlp(n,out,width=128):return nn.Sequential(nn.Linear(n,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,out))
class World(nn.Module):
 def __init__(self):
  super().__init__();self.observed_base=mlp(95,22);self.observed_key=nn.Sequential(nn.Linear(95,64),nn.SiLU(),nn.Linear(64,32),nn.Tanh());self.reader_base=mlp(60,22);self.reader_key=nn.Sequential(nn.Linear(60,64),nn.SiLU(),nn.Linear(64,32),nn.Tanh());self.initial=nn.Parameter(torch.zeros(32,22));self.log_eta=nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
  self.register_buffer('coordinate_weight',torch.tensor([1.]*12+[.1]*8+[0.,0.]))
  for head in (self.observed_base,self.reader_base):nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
 def fixed_coordinates(self,state,delta,features):
  return torch.cat([state[...,:20]+delta[...,:20],state[...,20:21]+.1*features[...,-1:],state[...,21:22]],-1)
 def predict(self,features,W,observed=False):
  base=self.observed_base(features) if observed else self.reader_base(features);key=self.observed_key(features) if observed else self.reader_key(features)
  fast=torch.matmul(key.unsqueeze(-2),W).squeeze(-2) if W.ndim==key.ndim+1 else torch.matmul(key,W)
  delta=base+fast;return self.fixed_coordinates(features[...,:22],delta,features)
 def write_from_components(self,W,features,base,key,target,valid=None):
  # One real executed block per task: exact d/dW weighted coordinate MSE.
  pred=self.fixed_coordinates(features[...,:22],base+torch.matmul(key.unsqueeze(-2),W).squeeze(-2),features);err=(pred-target)*self.coordinate_weight
  g=2*key.unsqueeze(-1)*err.unsqueeze(-2)/22
  if valid is not None:g=g*valid.unsqueeze(-1).unsqueeze(-1)
  norm=torch.linalg.vector_norm(g,dim=(-2,-1),keepdim=True);g=g*(1/(norm+1e-6)).clamp(max=1)
  return W-F.softplus(self.log_eta).clamp(max=.3)*g,pred
 def write(self,W,features,target):
  return self.write_from_components(W,features,self.observed_base(features),self.observed_key(features),target)[0]
 def adapt(self,features,targets,valid,meta=True):
  B,T,_=features.shape;W=self.initial.unsqueeze(0).expand(B,-1,-1);base=self.observed_base(features);key=self.observed_key(features);loss=features.sum()*0
  for j in range(T):
   W,pred=self.write_from_components(W,features[:,j],base[:,j],key[:,j],targets[:,j],valid[:,j]);loss=loss+((pred-targets[:,j]).square()*self.coordinate_weight*valid[:,j:j+1]).sum()
  return W if meta else W.detach(),loss/(valid.sum().clamp(min=1)*22)
class Value(nn.Module):
 def __init__(self):
  super().__init__();self.net=mlp(45,3)
 def forward(self,state,context):return self.net(torch.cat([state[...,:12],context,state[...,20:21]],-1))
 def score(self,state,context,kind='landing'):
  v=self(state,context)
  return 1-v[...,1:].abs().amax(-1) if kind=='landing' else 5*v[...,0]
class Correction(nn.Module):
 def __init__(self):
  super().__init__();self.net=mlp(76,1,width=64);nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,state,context,belief):return .05*torch.tanh(self.net(torch.cat([state,context,belief],-1))).squeeze(-1)
def projection(first_command,correction,anchor_acc,old,previous,base_delta):
 acc=(anchor_acc+correction).clamp(-.5,.5);new=(old+.1*acc).clamp(-.1,.1);y=(previous+base_delta+.1*(old+new)/2).clamp(-.85,.25);dv=(y-previous-base_delta)/.1
 return torch.stack([first_command[...,0],(y+.6)/.3,first_command[...,2],dv/.5],-1)
def reader_features(state,context,first_command,correction,anchor_acc,controller,horizon=8):
 actual=projection(first_command,correction,anchor_acc,controller[...,0],controller[...,1],controller[...,2]);h=torch.ones_like(correction)*float(horizon)/8
 return torch.cat([state,context,actual,correction.unsqueeze(-1)/.5,h.unsqueeze(-1)],-1)
def observed_features(state,context,commands,horizon):
 assert 1<=horizon<=8
 shape=state.shape[:-1];pad=torch.zeros(*shape,8,4,device=state.device,dtype=state.dtype);pad[...,:horizon,:]=commands;mask=torch.zeros(*shape,8,device=state.device,dtype=state.dtype);mask[...,:horizon]=1;h=torch.full((*shape,1),horizon/8,device=state.device,dtype=state.dtype)
 return torch.cat([state,context,pad.flatten(-2),mask,h],-1)
class DynamicsMemory:
 def __init__(self,world,enabled=True):self.world=world;self.W=world.initial.detach().clone();self.enabled=enabled;self.updates=0
 @torch.no_grad()
 def observe(self,features,target):
  if self.enabled:self.W=self.world.write(self.W,features,target).detach();self.updates+=int(self.enabled)
