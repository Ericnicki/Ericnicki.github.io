"""Exact TTT prefix cache while collection weights remain fixed. Partial windows are speculative, never double-written."""
import numpy as np
import torch
from torch.nn import functional as F
class Memory:
    def __init__(self,model):
        self.model=model;self.pending=[];self.completed=[];self.current=[]
        self.W=[b.initial.detach().unsqueeze(0).clone() for b in model.blocks]
        self.committed_windows=0
    @torch.no_grad()
    def _process(self,state=None,commit=False):
        model=self.model;device=next(model.parameters()).device
        query=model.encoder.encode_obs(torch.as_tensor(state,device=device).reshape(1,22)) if state is not None else None
        hist=None;valid=None
        if self.pending:
            n=len(self.pending);x=np.zeros((1,16,23),np.float32);x[0,:n]=self.pending
            hist=model.encoder.encode_events(torch.as_tensor(x,device=device))
            valid=torch.arange(16,device=device).reshape(1,16)<n
        for bi,block in enumerate(model.blocks):
            W=self.W[bi]
            if hist is not None:
                m=valid.unsqueeze(-1).to(hist.dtype)
                contextual=block.window_transformer(block.norm(hist)*m+block.position,src_key_padding_mask=~valid)
                k=F.normalize(block.k(contextual),dim=-1,eps=1e-6)
                v=torch.tanh(block.v(contextual));q=F.normalize(block.q(contextual),dim=-1,eps=1e-6)
                error=(torch.bmm(k,W)-v)*m
                W=W-F.softplus(block.log_eta).clamp(max=1.)*torch.bmm(k.transpose(1,2),error)/m.sum(dim=1,keepdim=True)
                hist=hist+block.mlp(block.read(torch.bmm(q,W)))*m
            if query is not None:
                q=F.normalize(block.q(block.norm(query)),dim=-1,eps=1e-6)
                query=query+block.mlp(block.read(torch.bmm(q.unsqueeze(1),W).squeeze(1)))
            if commit:self.W[bi]=W
        if commit:
            if self.pending:self.committed_windows+=1
            self.pending=[]
        return query
    def context(self,state):return self._process(state).detach()
    def observe(self,event):
        event=np.asarray(event,np.float32);assert event.shape==(23,)
        self.pending.append(event);self.current.append(event)
        if len(self.pending)==16:self._process(commit=True)
    def end_trial(self):
        self._process(commit=True);self.completed.append(np.asarray(self.current,np.float32).reshape(-1,23));self.current=[]
    @torch.no_grad()
    def reference(self,state):
        parts=self.completed+[np.asarray(self.current,np.float32).reshape(-1,23)]
        h=[];m=[]
        for part in parts:
            if not len(part):continue
            n=((len(part)+15)//16)*16;x=np.zeros((n,23),np.float32);x[:len(part)]=part
            mask=np.zeros(n,np.float32);mask[:len(part)]=1;h.append(x);m.append(mask)
        x=np.concatenate(h) if h else np.zeros((0,23),np.float32)
        mask=np.concatenate(m) if m else np.zeros(0,np.float32)
        device=next(self.model.parameters()).device
        return self.model.context(torch.as_tensor(x,device=device).unsqueeze(0),torch.as_tensor(mask,device=device).unsqueeze(0),torch.as_tensor(state,device=device).reshape(1,22))
