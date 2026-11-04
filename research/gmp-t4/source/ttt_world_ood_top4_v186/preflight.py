"""OOD-only fixture, checkpoint provenance and live/offline policy parity."""
from common import *
from collect import environment,specs,parents,ARMS,fresh_start
from trial import trial
from memory import Memory
from online import Online
from audit import audit_trial
import common,trial as trial_module
setup(2185000);assert not O.exists();plans=specs();assert len(plans)==32 and all(s['friction']<.005 or s['friction']>.015 for s in plans);assert plans[0]['friction']==.0025 and plans[16]['friction']==.0225
prior=json.loads((PRIOR/'result.json').read_text());rank_success=sorted(prior['overall'],key=lambda k:prior['overall'][k]['successes'],reverse=True)[:4];rank_return=sorted(prior['overall'],key=lambda k:prior['overall'][k]['mean_G'],reverse=True)[:4];assert rank_success==ARMS[:4]==rank_return
for p in [F/'audit.json',F/'actor_audit.json',JF/'audit.json',QF/'audit.json']:assert json.loads(p.read_text())['status']=='passed'
common.O=R/'ttt_world_ood_top4_v186_fixture_output';trial_module.O=common.O;assert not common.O.exists();common.O.mkdir();env,acfg,cfg=environment();m,a=anchor();budget=dict(trial_starts=0,entries=[]);fixtures=[]
try:
 for i,arm in enumerate(ARMS):
  spec=plans[0 if i%2==0 else 16];start=fresh_start(env,spec);speed=float(np.random.default_rng(spec['speed_seed']).uniform(.247,.435));mem=Memory(m);online=None if arm=='swap' else Online(arm);hashes=[state_hash(x) for x in online.models()] if online else [];initial_W=online.W.clone() if online else None;tr=trial(env,acfg,start,speed,m,a,mem,budget,dict(kind='OOD_fixture',arm=arm,friction=spec['friction']),online=online,parity=True)
  assert tr['status']=='completed';audit_trial(tr,m,a,Memory(m),acfg,tr['initial_observation'],online=None if arm=='swap' else Online(arm));assert hashes==([state_hash(x) for x in online.models()] if online else [])
  rr=tr['records']
  if online:
   from world import observed_features
   W=initial_W
   for j in range(0,len(rr)-1,8):
    end=min(j+8,len(rr)-1);f=observed_features(tensor(rr[j]['state']),tensor(rr[j]['context']),tensor(np.stack([command(r) for r in rr[j:end]])),end-j);W=online.world.write(W,f,tensor(rr[end]['state'])).detach()
   torch.testing.assert_close(W,online.W,rtol=0,atol=0);new=Online(arm);torch.testing.assert_close(new.W,initial_W,rtol=0,atol=0);online.start_trial();torch.testing.assert_close(online.W,W,rtol=0,atol=0)
  save(common.O/f'{arm}.pt',dict(spec=spec,initial=start,trial=tr,world_logs=online.logs if online else []));fixtures.append(dict(arm=arm,friction=spec['friction'],controls=len(rr)))
 write(R/'ttt_world_ood_top4_v186_preflight.json',dict(status='passed',trial_starts=5,fixtures=fixtures,ID_tasks=0,optimizer_updates=0,live_offline_block_parity=True,independent_native_actor_replay=True,task_trial_memory_reset_pass=True,selection_success=rank_success,selection_return=rank_return,parents=parents(),sources=sources()))
finally:env.physics.free()
