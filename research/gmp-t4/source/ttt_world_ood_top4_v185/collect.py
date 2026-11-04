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
 setup(2185100);assert not O.exists();pf=json.loads((R/'ttt_world_ood_top4_v185_preflight.json').read_text());assert pf['status']=='passed' and pf['sources']==sources() and pf['parents']==parents()
 O.mkdir();shutil.copytree(S,O/'source',ignore=shutil.ignore_patterns('__pycache__','._*'))
 write(O/'manifest.json',dict(specs=specs(),arms=ARMS,labels=LABELS,selection='top four v181 primary successes and mean_G rankings agree',policy_slots=1920,fixture_starts=5,independent_replay_cap=20,calibration_starts=0,training_range=[.005,.015],bands={'low':[.0025,.005],'high':[.015,.0225]},ID_tasks=0,explicit_outer_boundary_tasks=[0,16],sources=sources(),parents=parents(),primary='T2–12',optimizer_updates=0,global_weights_frozen=True,standard_real_observation_task_private_adaptation=True,base_speed_uniform=[.247,.435],native_reset_timeout_cutoffs='disabled; every trial reaches judgment',new_task_memory_reset=True,trials_per_method_task=12))
 budget=dict(trial_starts=5,near_starts=0,entries=[dict(kind='OOD_fixture',count=5)]);env,acfg,cfg=environment();rows=[];begin=time.time();m,a=anchor();mh=state_hash(m);ah=state_hash(a)
 try:
  for spec in specs():
   tid=spec['task'];folder=O/f'task{tid:04d}';folder.mkdir();start=fresh_start(env,spec);save(folder/'initial.pt',start);speed=float(np.random.default_rng(spec['speed_seed']).uniform(.247,.435));row=dict(spec=spec,speed=speed,initial_sha256=sha(folder/'initial.pt'),arms={})
   for arm in ARMS:
    mem=Memory(m);online=None if arm=='swap' else Online(arm);hashes=[state_hash(x) for x in online.models()] if online else [];trials=[]
    for ti in range(12):
     tr=trial(env,acfg,start,speed,m,a,mem,budget,dict(task=tid,trial=ti+1,arm=arm),online=online,parity=tid in [0,16]);assert tr['status']=='completed';trials.append(tr);save(folder/f'{arm}_t{ti+1:02d}.pt',tr)
    assert hashes==([state_hash(x) for x in online.models()] if online else [])
    save(folder/f'{arm}.pt',dict(spec=spec,speed=speed,trials=trials,world_logs=online.logs if online else [],final_W=online.W.cpu() if online else None,global_hashes=hashes));row['arms'][arm]=dict(returns=[t['terminal_return'] for t in trials],sha256=sha(folder/f'{arm}.pt'),world_writes=len(online.logs) if online else 0);write(folder/'status.json',dict(completed_arms=list(row['arms']),policy_starts=budget['trial_starts']))
    print('OOD_ARM',tid,arm,len(trials),budget['trial_starts'],flush=True)
   assert state_hash(m)==mh and state_hash(a)==ah;rows.append(row);write(O/'summary.json',rows);write(O/'status.json',dict(status='running',stage='native_collection',tasks=len(rows),policy_starts=budget['trial_starts']-5,seconds=time.time()-begin));print('OOD_TASK',tid,spec['region'],spec['friction'],len(rows),flush=True)
  assert budget['trial_starts']==1925 and parents()==pf['parents'];write(O/'collection_complete.json',dict(status='completed',tasks=32,policy_starts=1920,fixture_starts=5,calibration_starts=0,ID_tasks=0,optimizer_updates=0,global_weights_unchanged=True,seconds=time.time()-begin,sources=sources()))
 except BaseException as e:write(O/'failed.json',dict(error=repr(e),completed_tasks=len(rows),budget=budget));raise
 finally:env.physics.free()
if __name__=='__main__':main()
