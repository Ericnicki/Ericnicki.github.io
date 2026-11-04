"""Causal executed-block writer; candidates do not commit memory writes."""
from common import *
from world import World,Value,Correction,observed_features,reader_features
class Online:
 def __init__(self,checkpoint):
  from legacy import Critic,ResidualActor
  from memory import Memory
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['schema']=='joint-world-gradient-v183'
  self.anchor_model=freeze(Critic().cuda());self.anchor_model.load_checkpoint(ck['anchor_model']);self.anchor_actor=freeze(ResidualActor().cuda());self.anchor_actor.load_state_dict(ck['anchor_actor']);self.memory=Memory(self.anchor_model)
  self.world=freeze(World().cuda());self.world.load_state_dict(ck['world']);self.value=freeze(Value().cuda());self.value.load_state_dict(ck['value']);self.actor=freeze(Correction().cuda());self.actor.load_state_dict(ck['correction']);self.scale=tensor(ck['belief_scale'].numpy());self.mode='joint';self.W=self.world.initial.detach().clone();self.logs=[];self.pending=[];self.next_decision=None;self.held=0.;self.decisions=0
 def start_trial(self):assert not self.pending;self.next_decision=None;self.held=0.
 @torch.no_grad()
 def commit(self,state):
  if not self.pending:return
  rr=self.pending;f=observed_features(tensor(rr[0]['state']),tensor(rr[0]['context']),tensor(np.stack([command(r) for r in rr])),len(rr));target=tensor(state);before=self.W.clone()
  if self.mode in ('prediction_only','joint'):self.W=self.world.write(self.W,f,target).detach()
  self.logs.append(dict(features=f.cpu(),target=target.cpu(),W_before=before.cpu(),W_after=self.W.cpu(),commands=len(rr),enabled=self.mode in ('prediction_only','joint')));self.pending=[]
 def arrival(self,state):
  if len(self.pending)==8:self.commit(state)
 @torch.no_grad()
 def choose(self,r,state,context,acc,control):
  if not r['critical']:return 0.,None
  if self.next_decision is not None and control<self.next_decision:return self.held,None
  state=tensor(state).reshape(1,22);first=np.r_[(np.asarray(r['nominal_command'][0]['tcp_xyz_wxyz'][:3])-[.1,-.6,.365])/[.2,.3,.2],0.].astype(np.float32);first=tensor(first).reshape(1,4);ctr=tensor([[r['residual_velocity_before_m_s'],r['previous_executed_y_m'],r['base_delta_y_m']]]);aa=tensor([acc]);zero=tensor([0.]);f=reader_features(state,context,first,zero,aa,ctr);W=self.W.unsqueeze(0);before=self.W.clone();belief=self.world.predict(f,W)-self.world.predict(f,self.world.initial.detach().unsqueeze(0));feature=(belief/self.scale).clamp(-5,5);delta=self.actor(state,context,feature);nx=self.world.predict(reader_features(state,context,first,delta,aa,ctr),W);score=self.value.score(nx,context);assert torch.equal(before,self.W)
  self.held=float(delta.item());self.next_decision=control+8;self.decisions+=1
  return self.held,dict(first_command=first.cpu().numpy()[0],controller=ctr.cpu().numpy()[0],belief=belief.cpu().numpy()[0],world_goal_score=float(score.item()),world_writes=len(self.logs),world_W_sha256=__import__('hashlib').sha256(self.W.cpu().numpy().tobytes()).hexdigest())
 def record(self,r):self.pending.append(r)
 def end_trial(self,last_state):
  # Last command ended at judgment/reset: it never supplies a supervised target.
  if self.pending:self.pending.pop();self.commit(last_state)
  self.next_decision=None;self.held=0.
