from common import *
from online import Online
from world import World,Value,Correction
setup(2182000);assert not O.exists();O.mkdir();import shutil;shutil.copytree(S,O/'source',ignore=shutil.ignore_patterns('__pycache__','._*'));native=R/'ttt_world_shared_control_v179_output';review=R/'ttt_world_shared_review_v180_output';assert json.loads((native/'audit.json').read_text())['status']=='passed' and json.loads((review/'audit.json').read_text())['status']=='passed';assert sha(P)==P_SHA
parent=torch.load(P,map_location='cpu',weights_only=False);m,a=anchor();canonical_anchor=copy.deepcopy(m.state_dict());
for key in canonical_anchor:
 if key.startswith(('wm_head.','value_head.')):canonical_anchor[key].zero_()
 else:assert torch.equal(canonical_anchor[key].cpu(),parent['model'][key].cpu())
m.load_state_dict(canonical_anchor);wc=torch.load(JF/'joint_world.pt',map_location='cpu',weights_only=False);ac=torch.load(JF/'joint_actor.pt',map_location='cpu',weights_only=False);ck=dict(schema='joint-world-gradient-v183',anchor_model=canonical_anchor,anchor_actor=parent['actor'],world=wc['world'],value=wc['value'],correction=ac['correction'],belief_scale=ac['belief_scale'],controller=dict(dt=.1,acc_max=.5,velocity_max=.1,y_min=-.85,y_max=.25,correction_max=.05,held_controls=8),contract='Observation-only local physical writes; original causal context; physicalreset eachtrial, taskmemory persists; no candidate self-writes',sources=sources(),parents={str(p.relative_to(R)):sha(p) for p in [P,JF/'joint_world.pt',JF/'joint_actor.pt',JF/'audit.json',native/'audit.json',review/'audit.json']},status='experimental, no automatic deployment');save(O/'joint_inference.pt',ck);policy=Online(O/'joint_inference.pt');assert state_hash(m)==state_hash(policy.anchor_model) and state_hash(a)==state_hash(policy.anchor_actor);count=0;maxerr=0.
for tid in range(30):
 raw=torch.load(native/f'task{tid:04d}/joint.pt',map_location='cpu',weights_only=False);snapshots=[policy.world.initial.detach().cpu()]+[x['W_after'] for x in raw['world_logs']]
 for tr in raw['trials']:
  for r in tr['records']:
   recorded=r.get('world_decision')
   if recorded is None:continue
   policy.W=snapshots[recorded['world_writes']].cuda().clone();policy.next_decision=None;policy.logs=[None]*recorded['world_writes'];before=policy.W.clone();state=tensor(r['state']).reshape(1,22);ctx=tensor(r['context']).reshape(1,32)
   with torch.no_grad():aa=float((.5*torch.tanh(policy.anchor_actor.net(torch.cat([ctx,state],-1)))).item())
   assert aa==r['actor_acc_m_s2'];delta,decision=policy.choose(r,r['state'],ctx,aa,r['control']);err=max(abs(delta-r['held_correction_m_s2']),abs(decision['world_goal_score']-recorded['world_goal_score']),float(np.max(abs(decision['belief']-recorded['belief']))));maxerr=max(maxerr,err);assert err==0,(tid,r['control'],err);assert torch.equal(before,policy.W);assert decision['world_W_sha256']==recorded['world_W_sha256'];count+=1
 assert state_hash(m)==state_hash(policy.anchor_model) and state_hash(a)==state_hash(policy.anchor_actor)
assert count>0;receipt=dict(status='passed',tasks=30,trials=360,macro_decisions_checked=count,max_inference_error=maxerr,candidate_W_unchanged=True,anchor_unchanged=True,checkpoint_sha256=sha(O/'joint_inference.pt'),parents=ck['parents'],sources=sources(),physics_starts=0,optimizer_updates=0);write(O/'audit.json',receipt);write(O/'file_manifest.json',{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='file_manifest.json'});print(json.dumps(receipt,indent=2))
