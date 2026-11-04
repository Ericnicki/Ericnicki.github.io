"""Obs/action-only TTT-Linear fast memory in eight residual TTT/MLP blocks.

One GD write per16-frame Transformer window (observed obs/action events). Gradient is exactly
that of .5*sum||kW-v||^2/valid_count, not a GRU or a Hebbian replacement. Slow gradients
differentiate through the fast writes. Current candidate action enters Q only.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

STATE_DIM=22
EVENT_DIM=23
WIDTH=32
FAST_DIM=16
CHUNK=16

class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.groups=nn.ModuleList([nn.Sequential(nn.Linear(n,16),nn.SiLU(),nn.Linear(16,16)) for n in [6,6,10]])
        self.obs=nn.Sequential(nn.Linear(48,WIDTH),nn.LayerNorm(WIDTH))
        self.action=nn.Sequential(nn.Linear(1,16),nn.SiLU(),nn.Linear(16,WIDTH))
    def encode_obs(self,x):
        return self.obs(torch.cat([m(g) for m,g in zip(self.groups,torch.split(x,[6,6,10],dim=-1))],dim=-1))
    def encode_events(self,x):return self.encode_obs(x[...,:STATE_DIM])+self.action(x[...,STATE_DIM:])

class TTTResidual(nn.Module):
    def __init__(self):
        super().__init__();self.norm=nn.LayerNorm(WIDTH)
        self.position=nn.Parameter(torch.randn(CHUNK,WIDTH)*.01)
        layer=nn.TransformerEncoderLayer(WIDTH,4,64,dropout=0.,batch_first=True,norm_first=True,activation='gelu')
        self.window_transformer=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.k=nn.Linear(WIDTH,FAST_DIM);self.v=nn.Linear(WIDTH,FAST_DIM);self.q=nn.Linear(WIDTH,FAST_DIM)
        self.initial=nn.Parameter(torch.zeros(FAST_DIM,FAST_DIM))
        self.log_eta=nn.Parameter(torch.tensor(math.log(math.expm1(.2))))
        self.read=nn.Linear(FAST_DIM,WIDTH)
        self.mlp=nn.Sequential(nn.Linear(WIDTH,64),nn.SiLU(),nn.Linear(64,WIDTH))
    def forward(self,hist,query,mask):
        B,T,_=hist.shape;W=self.initial.unsqueeze(0).expand(B,-1,-1)
        eta=F.softplus(self.log_eta).clamp(max=1.)
        if T:
            assert T%CHUNK==0
            chunks=T//CHUNK
            windows=(self.norm(hist)*mask.unsqueeze(-1)).reshape(B*chunks,CHUNK,WIDTH)
            valid=mask.reshape(B*chunks,CHUNK).bool();safe=valid.clone()
            safe[~safe.any(dim=1),0]=True
            # Bidirectional only over the past/observed window. Hidden/future
            # padding is excluded; a write commits after window completion.
            contextual=self.window_transformer(windows+self.position,
                src_key_padding_mask=~safe).reshape(B,T,WIDTH)
            keys=F.normalize(self.k(contextual),dim=-1,eps=1e-6)
            vals=torch.tanh(self.v(contextual));qs=F.normalize(self.q(contextual),dim=-1,eps=1e-6)
            reads=[]
            for start in range(0,T,CHUNK):
                k=keys[:,start:start+CHUNK];v=vals[:,start:start+CHUNK];q=qs[:,start:start+CHUNK]
                m=mask[:,start:start+CHUNK].unsqueeze(-1);count=m.sum(dim=1,keepdim=True).clamp(min=1.)
                error=(torch.bmm(k,W)-v)*m
                W=W-eta*torch.bmm(k.transpose(1,2),error)/count
                reads.append(torch.bmm(q,W))
            hist=hist+self.mlp(self.read(torch.cat(reads,dim=1)))*mask.unsqueeze(-1)
        q=F.normalize(self.q(self.norm(query)),dim=-1,eps=1e-6)
        read_query=torch.bmm(q.unsqueeze(1),W).squeeze(1)
        query=query+self.mlp(self.read(read_query))
        return hist,query

class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=FeatureEncoder()
        # (TTT -> MLP -> add -> TTT -> MLP -> add) repeated four times.
        self.blocks=nn.ModuleList([TTTResidual() for _ in range(8)])
        self.wm_head=nn.Sequential(nn.Linear(95,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,12))
        self.value_head=nn.Sequential(nn.Linear(54,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        for head in (self.wm_head,self.value_head):
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        self.head=nn.Sequential(nn.Linear(WIDTH+STATE_DIM+1,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    def context(self,hist,mask,state):
        h=self.encoder.encode_events(hist);q=self.encoder.encode_obs(state)
        for block in self.blocks:h,q=block(h,q,mask)
        return q
    def forward(self,hist,mask,state,action,writer_grad=True):
        if writer_grad:q=self.context(hist,mask,state)
        else:
            with torch.no_grad():q=self.context(hist,mask,state)
        return self.head(torch.cat([q,state,action],dim=-1)).squeeze(-1)
    def auxiliary_parameters(self):
        yield from self.wm_head.parameters();yield from self.value_head.parameters()
    def load_checkpoint(self, weights):
        if any(k.startswith("wm_head.") for k in weights):
            self.load_state_dict(weights,strict=True)
        else:
            result=self.load_state_dict(weights,strict=False)
            assert not result.unexpected_keys
            assert set(result.missing_keys)=={k for k in self.state_dict() if k.startswith(("wm_head.","value_head."))}
    def writer_parameters(self):
        yield from self.encoder.parameters();yield from self.blocks.parameters()

class ResidualActor(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(WIDTH+STATE_DIM,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,context,state,previous_residual_velocity,acc_bound):
        # Choose applied average d_V via bounded acceleration increment. Gradient
        # through tanh/controller has the same semantics as offline action input.
        acc=acc_bound*torch.tanh(self.net(torch.cat([context,state],dim=-1)))
        return previous_residual_velocity+.05*acc,acc
