"""Rebuild native features, controller and causal memory from immutable raw data."""
import time
from common import *
from memory import Memory
from controller import Controller
from dataset import features
from official import configurations,agent_for,reward
from collect import ARMS,specs
from online import Online
def equal(a,b):
 if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b,(a,b)
def audit_trial(tr,m,a,mem,acfg,initial,reference=None,query=None,online=None):
 rr=tr['records'];st,ev,phases=features(rr);first=copy.deepcopy(tr['initial_observation']);prior=copy.deepcopy(initial)
 for o in (first,prior):
  for robot in o['robots_obs']:robot.pop('arm_qacc',None)
 equal(first,prior);ctr=Controller();
 if online:online.start_trial()
 agent=agent_for(acfg,[tr['speed']]*6);previous=[];errors=[]
 for j,r in enumerate(rr):
  np.testing.assert_array_equal(r['state'],st[j]);np.testing.assert_array_equal(r['event'],ev[j]);assert r['proxy_phase']==phases[j]
  assert r['current_observed_events']==j and r['prior_trial_count']==len(mem.completed)
  ctx=mem.context(r['state']);err=float(np.max(abs(ctx.cpu().numpy()[0]-r['context'])));errors.append(err);assert err<2e-6
  with torch.no_grad():aa=float((.5*torch.tanh(a.net(torch.cat([ctx,tensor(r['state']).reshape(1,22)],-1)))).item()) if r['critical'] else 0.
  assert aa==r['actor_acc_m_s2']
  if online:online.arrival(r['state'])
  expected_delta,decision=online.choose(r,r['state'],ctx,aa,j) if online else (0.,None)
  equal(decision,r['world_decision']);assert expected_delta==r['held_correction_m_s2'] and r['noise_m_s2']==0 and r['noise_rng_draws']==0
  cc=ctr.step(r['base_y_m'],r['base_previous_y_m'],r['previous_executed_y_m'],aa+expected_delta,r['critical'],r['active'])
  for k,v in cc.items():assert v==r[k],(j,k)
  nominal=agent.predict_actions([r['before']['robots_obs']],[r['before']['env_objs_obs']],[previous])[0];equal(nominal,r['nominal_command']);previous=r['executed_command'];assert abs(float(previous[0]['tcp_xyz_wxyz'][1])-r['command_y_m'])<1e-10
  for k in (0,2):assert previous[0]['tcp_xyz_wxyz'][k]==nominal[0]['tcp_xyz_wxyz'][k]
  assert r['terminal']==(r['judge_after']>r['judge_before']) and abs(r['time_after']-r['time_before']-.1)<1e-9
  if j:equal(r['before'],rr[j-1]['after'])
  if reference is not None and j<=query:
   np.testing.assert_array_equal(r['state'],reference[j]['state']);np.testing.assert_array_equal(r['context'],reference[j]['context'])
   if j<query:equal(r['executed_command'],reference[j]['executed_command'])
  mem.observe(r['event'])
  if online:online.record(r)
 if online:online.end_trial(rr[-1]['state'])
 mem.end_trial()
 if tr['status']=='completed':
  xy=rr[-1]['after']['env_objs_obs'][1]['object_pose_xyz_wxyz'][:2].tolist();assert xy==tr['stop_xy'] and reward(xy)==tr['terminal_return'] and tr['official_success']==(reward(xy)>0) and rr[-1]['terminal']
 else:assert 'terminal_return' not in tr and not rr[-1]['terminal']
 return dict(controls=len(rr),max_context_error=max(errors),judged=tr['status']=='completed')
def main():
 setup();begin=time.time();complete=json.loads((O/'collection_complete.json').read_text());assert complete['sources']==sources();rows=json.loads((O/'summary.json').read_text());assert [r['spec'] for r in rows]==specs();assert all(r['spec']['friction']<.005 or r['spec']['friction']>.015 for r in rows);m,a=anchor();mh=state_hash(m);ah=state_hash(a);cfg,acfg=configurations();stats=[];bindings={};writes=0
 for row in rows:
  folder=O/f"task{row['spec']['task']:04d}";assert sha(folder/'initial.pt')==row['initial_sha256'];initial=None;firststate=None
  for arm in ARMS:
   raw=torch.load(folder/f'{arm}.pt',map_location='cpu',weights_only=False);assert sha(folder/f'{arm}.pt')==row['arms'][arm]['sha256'];mem=Memory(m);online=None if arm=='swap' else Online(arm)
   for ti,tr in enumerate(raw['trials']):
    if initial is None:initial=tr['initial_observation'];firststate=tr['records'][0]['state']
    np.testing.assert_array_equal(firststate,tr['records'][0]['state']);equal(tr,torch.load(folder/f'{arm}_t{ti+1:02d}.pt',map_location='cpu',weights_only=False));assert tr['restored_friction_sha256']==hashlib.sha256(torch.load(folder/'initial.pt',weights_only=False)['geom_friction'].tobytes()).hexdigest();stats.append(audit_trial(tr,m,a,mem,acfg,initial,online=online))
   if online:
    assert len(online.logs)==len(raw['world_logs']);torch.testing.assert_close(online.W.cpu(),raw['final_W'],rtol=0,atol=0)
    for log,recorded in zip(online.logs,raw['world_logs']):
     assert log.keys()==recorded.keys()
     for k in log:
      if torch.is_tensor(log[k]):torch.testing.assert_close(log[k],recorded[k],rtol=0,atol=0)
      else:assert log[k]==recorded[k]
     # Differentiate the supervised loss independently, including all clipping.
     W=log['W_before'].cuda().requires_grad_(True);f=log['features'].cuda();target=log['target'].cuda();pred=online.world.predict(f,W,observed=True);loss=((pred-target).square()*online.world.coordinate_weight).mean();g=torch.autograd.grad(loss,W)[0];g=g*(1/(torch.linalg.vector_norm(g)+1e-6)).clamp(max=1);want=W-torch.nn.functional.softplus(online.world.log_eta).clamp(max=.3)*g if arm in ARMS else W;torch.testing.assert_close(want.detach().cpu(),log['W_after'],rtol=1e-5,atol=2e-8);writes+=int(arm in ARMS)
    assert raw['global_hashes']==[state_hash(x) for x in online.models()];assert all(not p.requires_grad for model in online.models() for p in model.parameters())
  for path in folder.rglob('*.pt'):bindings[str(path.relative_to(O))]=sha(path)
  print('WORLD_CONTROL_AUDIT_TASK',row['spec']['task'],len(stats),flush=True)
 assert len(stats)==1920 and all(s['judged'] for s in stats);assert state_hash(m)==mh and state_hash(a)==ah
 write(O/'audit.json',dict(status='passed',trials=1920,controls=sum(s['controls'] for s in stats),independently_differentiated_world_writes=writes,all_native_features_context_actor_projection_and_private_W_replayed=True,matched_initials=True,global_weights_unchanged=True,censors=0,raw_bindings=bindings,seconds=time.time()-begin,sources=sources()))
if __name__=='__main__':main()
