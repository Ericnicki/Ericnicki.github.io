"""Fresh OOD-only matched native evaluation; no calibration or slow optimization."""
import time,shutil,math
import hydra
from common import *
from official import configurations,snap
from memory import Memory
from trial import trial
from online import Online
ARMS=['joint','prediction_only','adaptive','aligned','swap']
LABELS=dict(swap='Original swap',joint='Joint shared-memory',prediction_only='Prediction-only shared-memory',adaptive='Adaptive direct-world',aligned='Gradient-aligned Q')
def environment():
 cfg,acfg=configurations();env=hydra.utils.instantiate(cfg.task.env);env.init_simulator();env.cube_reset_waiting_time=math.inf;env.judge.time_limit_s=math.inf;original=env.objects[0].set_pose_xyz_wxyz
 def tracked(p):original(p);env._teleported=True;env._near_teleported=True
 env.objects[0].set_pose_xyz_wxyz=tracked
 return env,acfg,cfg
def specs():
 rng=np.random.default_rng(2185110);out=[]
 for region,lo,hi,boundary in [('low',.0025,.005,.0025),('high',.015,.0225,.0225)]:
  for j in range(16):
   tid=len(out);friction=boundary if j==0 else float(rng.uniform(lo,hi));assert friction<.005 or friction>.015
   out.append(dict(task=tid,namespace='world_ood_top4_v185',region=region,cohort='broad',friction=friction,seed=2185200+tid,speed_seed=2185300+tid,outer_boundary=j==0))
 return out
def parents():
 paths=[P,F/'model_final.pt',F/'actor_final.pt',F/'audit.json',F/'actor_audit.json',JF/'joint_world.pt',JF/'prediction_only_world.pt',JF/'joint_actor.pt',JF/'prediction_only_actor.pt',JF/'audit.json',QF/'aligned_actor.pt',QF/'aligned_Q.pt',QF/'audit.json',PRIOR/'result.json']
 return {str(p.relative_to(R)):sha(p) for p in paths}
def fresh_start(env,spec):
 env.reset(dict(seed=spec['seed'],episode_idx=spec['seed'],sliding_friction=spec['friction']));env.judge.time_limit_s=math.inf
 assert math.isinf(env.cube_reset_waiting_time)
 start=snap(env);start['geom_friction']=env.physics.model.geom_friction.copy()
 assert np.any(start['geom_friction'][:,0]==spec['friction']),('native_friction_not_applied',spec)
 return start
def main():
 import sys
 worker=int(sys.argv[1]);assert worker in range(4);setup(2185100);pf=json.loads((R/'ttt_world_ood_top4_v185_preflight.json').read_text());assert pf['parents']==parents();manifest=json.loads((O/'manifest.json').read_text());assert manifest['sources']==sources();plans=manifest['specs'];env,acfg,cfg=environment();rows=[];begin=time.time();m,a=anchor();mh=state_hash(m);ah=state_hash(a);budget=dict(trial_starts=0,near_starts=0,entries=[])
 import trial as trial_module
 trial_module.O=O/f'worker{worker}';trial_module.O.mkdir()
 try:
  for spec in plans:
   tid=spec['task']
   if tid%4!=worker or tid in manifest['retained_task_ids']:continue
   folder=O/f'task{tid:04d}';folder.mkdir();start=fresh_start(env,spec);save(folder/'initial.pt',start);speed=float(np.random.default_rng(spec['speed_seed']).uniform(.247,.435));row=dict(spec=spec,speed=speed,initial_sha256=sha(folder/'initial.pt'),arms={})
   for arm in ARMS:
    mem=Memory(m);online=None if arm=='swap' else Online(arm);hashes=[state_hash(x) for x in online.models()] if online else [];trials=[]
    for ti in range(12):
     tr=trial(env,acfg,start,speed,m,a,mem,budget,dict(task=tid,trial=ti+1,arm=arm,worker=worker),online=online,parity=tid==16);assert tr['status']=='completed';trials.append(tr);save(folder/f'{arm}_t{ti+1:02d}.pt',tr)
    assert hashes==([state_hash(x) for x in online.models()] if online else [])
    save(folder/f'{arm}.pt',dict(spec=spec,speed=speed,trials=trials,world_logs=online.logs if online else [],final_W=online.W.cpu() if online else None,global_hashes=hashes));row['arms'][arm]=dict(returns=[t['terminal_return'] for t in trials],sha256=sha(folder/f'{arm}.pt'),world_writes=len(online.logs) if online else 0);write(folder/'status.json',dict(completed_arms=list(row['arms']),policy_starts=budget['trial_starts']))
    print('OOD_ARM',worker,tid,arm,budget['trial_starts'],flush=True)
   assert state_hash(m)==mh and state_hash(a)==ah;rows.append(row);write(O/f'worker{worker}_summary.json',rows);write(O/f'worker{worker}_status.json',dict(status='running',tasks=len(rows),policy_starts=budget['trial_starts'],seconds=time.time()-begin));print('OOD_TASK',worker,tid,spec['region'],spec['friction'],len(rows),flush=True)
  expected=60*sum(s['task']%4==worker and s['task'] not in manifest['retained_task_ids'] for s in plans);assert budget['trial_starts']==expected and parents()==pf['parents'];write(O/f'worker{worker}_complete.json',dict(status='completed',tasks=len(rows),policy_starts=expected,fixture_starts=0,ID_tasks=0,optimizer_updates=0,global_weights_unchanged=True,seconds=time.time()-begin,sources=sources()))
 except BaseException as e:write(O/f'worker{worker}_failed.json',dict(error=repr(e),completed_tasks=len(rows),budget=budget));raise
 finally:env.physics.free()
if __name__=='__main__':main()
