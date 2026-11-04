from common import *
from trial import environment
from probe import restore
def phys(obs):return np.r_[obs['robots_obs'][0]['tcp_xyz_wxyz'],obs['env_objs_obs'][1]['object_pose_xyz_wxyz'],obs['env_objs_obs'][1]['object_vel_xyz_xyz']]
setup();assert json.loads((O/'audit.json').read_text())['status']=='passed';env=environment();count=controls=0;max_error=0.
try:
 for tid in [0,16]:
  folder=O/f'task{tid:04d}';initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
  for ti in [1,2,6,12]:
   count+=1;assert count<=8;tr=torch.load(folder/f'gmp_t{ti:02d}.pt',map_location='cpu',weights_only=False);env.physics.model.geom_friction[:]=initial['geom_friction'];restore(env,initial);obs=obs_now(env,True)
   for r in tr['records']:
    err=float(np.max(abs(phys(obs)-phys(r['before']))));max_error=max(max_error,err);assert err<1e-9;assert env.episode_current_timestamp==r['time_before'] and env.judge.current_trial_num==r['judge_before'];obs,_,done,info=env.step(r['requested_command'],render_image=r['token']==7);err=float(np.max(abs(phys(obs)-phys(r['after']))));max_error=max(max_error,err);assert err<1e-9;assert env.episode_current_timestamp==r['time_after'] and env.judge.current_trial_num==r['judge_after'];equal(info['executed_action'],r['executed_command']);assert bool(done)==r['done'];controls+=1
   if tr['status']=='completed':assert reward(obs['env_objs_obs'][1]['object_pose_xyz_wxyz'][:2])==tr['terminal_return']
   print('GMP_PHYSICAL_REPLAY',tid,ti,count,flush=True)
finally:env.physics.free()
write(O/'independent_review.json',dict(status='passed',replay_starts=count,replayed_controls=controls,max_physical_error=max_error,optimizer_updates=0,ID_tasks=0,sources=sources()))
