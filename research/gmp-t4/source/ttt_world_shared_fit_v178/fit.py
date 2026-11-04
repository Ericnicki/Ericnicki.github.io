from common import *
from model import World,Value,Correction,reader_features
from learning import batch,sample,features,world_loss
from torch.nn import functional as fn
WORLD_STEPS=1000
ACTOR_STEPS=1000
ARMS=['prediction_only','joint']
def equal(a,b):
 if torch.is_tensor(a):assert torch.equal(a.cpu(),b.cpu())
 elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b,(a,b)
def tasks_load():
 m=json.loads((F/'cache_manifest.json').read_text());out={}
 for row in m['tasks']:
  p=F/'cache'/row['path'];assert sha(p)==row['sha256'];out[row['task']]=torch.load(p,map_location='cpu',weights_only=False)
 return out
def networks():
 ck=torch.load(F/'model_final.pt',map_location='cpu',weights_only=False);w=World().cuda();v=Value().cuda();w.load_state_dict(ck['world']);v.load_state_dict(ck['value']);opt=torch.optim.AdamW(w.parameters(),lr=1e-4,weight_decay=1e-4);opt.load_state_dict(ck['world_optimizer'])
 for group in opt.param_groups:group['lr']=5e-5
 return w,freeze(v),opt
def objective(w,v,b,joint):
 physics,_=world_loss(w,b);W,_=w.adapt(b['support'],b['support_target'],b['support_valid']);pred=w.predict(features(b),W);vv=v(pred,b['context'].unsqueeze(1).expand(-1,3,-1));goal=.1*fn.smooth_l1_loss(vv[...,0],b['value'][...,0])+.05*fn.smooth_l1_loss(vv[...,1:],b['value'][...,1:]);return physics+(goal if joint else 0),dict(physics=float(physics.detach()),goal_outer=float(goal.detach()))
def private(w,tasks,ids):
 cache={};raw=[]
 for tid in ids:
  for qi,p in enumerate(tasks[tid]['pairs']):
   b=batch(tasks,[(tid,qi)]);W,_=w.adapt(b['support'],b['support_target'],b['support_valid'],meta=False);f=reader_features(b['state'],b['context'],b['first_command'],tensor([0.]),b['anchor_acc'],b['controller'])
   with torch.no_grad():belief=(w.predict(f,W)-w.predict(f,w.initial.detach().unsqueeze(0))).detach()
   cache[(tid,qi)]=dict(W=W[0].cpu(),belief=belief[0].cpu());raw.append(belief[0].cpu())
 return cache,torch.stack(raw).std(0,unbiased=False).clamp(min=.001)
def actor_loss(actor,w,v,tasks,queries,cache,scale):
 b=batch(tasks,queries);W=tensor(torch.stack([cache[q]['W'] for q in queries]).numpy());belief=(tensor(torch.stack([cache[q]['belief'] for q in queries]).numpy())/tensor(scale.numpy())).clamp(-5,5);delta=actor(b['state'],b['context'],belief);nx=w.predict(reader_features(b['state'],b['context'],b['first_command'],delta,b['anchor_acc'],b['controller']),W);return -v.score(nx,b['context']).mean()+2*delta.square().mean()
def main():
 setup(2178000);assert not O.exists();assert json.loads((B/'audit.json').read_text())['status']=='passed';result=json.loads((B/'result.json').read_text());go=result['paired']['adaptive-frozen']['CI95'][0]>0 and result['paired']['adaptive-swap']['task_paired_delta_G']>=0;O.mkdir();import shutil;shutil.copytree(S,O/'source',ignore=shutil.ignore_patterns('__pycache__','._*'));write(O/'manifest.json',dict(sources=sources(),baseline_result_sha256=sha(B/'result.json'),parent_fit_sha256=sha(F/'model_final.pt'),conditional_gate_passed=bool(go),exploratory_benchmark_reuse=True))
 if not go:write(O/'result.json',dict(status='skipped',reason='No supported isolated adaptation benefit under declared condition',sources=sources()));return
 tasks=tasks_load();ids=sorted(tid for tid,t in tasks.items() if t['spec']['partition']=='train')
 for arm in ARMS:
  setup(2178000);w,v,opt=networks();vh=state_hash(v);rng=np.random.default_rng(2178100);logs=[]
  for step in range(WORLD_STEPS):
   qs=sample(tasks,ids,rng,12);b=batch(tasks,qs);loss,parts=objective(w,v,b,arm=='joint');opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(w.parameters(),1));opt.step();assert all(p.grad is None for p in v.parameters());logs.append(dict(queries=qs,loss=float(loss.detach()),parts=parts,norm=norm))
  save(O/f'{arm}_world_samples.pt',logs);save(O/f'{arm}_world.pt',dict(world=w.state_dict(),value=v.state_dict(),optimizer=opt.state_dict(),sampler_rng=rng.bit_generator.state,steps=WORLD_STEPS));w.zero_grad(set_to_none=True);freeze(w);cache,scale=private(w,tasks,ids);save(O/f'{arm}_private.pt',dict(cache=cache,scale=scale));setup(2178200);actor=Correction().cuda();opt=torch.optim.Adam(actor.parameters(),lr=1e-4);rng=np.random.default_rng(2178300);logs=[];wh=state_hash(w);save(O/f'{arm}_actor_initial.pt',actor.state_dict())
  for step in range(ACTOR_STEPS):
   qs=sample(tasks,ids,rng,24);loss=actor_loss(actor,w,v,tasks,qs,cache,scale);opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1));opt.step();assert all(p.grad is None for p in list(w.parameters())+list(v.parameters()));logs.append(dict(queries=qs,loss=float(loss.detach()),norm=norm))
  assert wh==state_hash(w) and vh==state_hash(v);save(O/f'{arm}_actor_samples.pt',logs);save(O/f'{arm}_actor.pt',dict(correction=actor.state_dict(),optimizer=opt.state_dict(),sampler_rng=rng.bit_generator.state,belief_scale=scale,steps=ACTOR_STEPS));print('SHARED_OUTER_ARM_FIT',arm,flush=True)
 write(O/'fit_complete.json',dict(status='completed',world_steps_each=WORLD_STEPS,actor_steps_each=ACTOR_STEPS,sources=sources(),value_frozen=True,local_writer_supervised_only=True))
if __name__=='__main__':main()
