"""Independent raw-block reconstruction plus OOD physical command replay."""
import time
from common import *
from collect import environment,ARMS
from official import restore,reward
from world import observed_features
from online import Online
def phys(obs):return np.r_[obs['robots_obs'][0]['tcp_xyz_wxyz'],obs['env_objs_obs'][1]['object_pose_xyz_wxyz'],obs['env_objs_obs'][1]['object_vel_xyz_xyz']]
def main():
 setup(2185400);begin=time.time();assert json.loads((O/'audit.json').read_text())['status']=='passed';rows=json.loads((O/'summary.json').read_text());blocks=0
 for row in rows:
  for arm in ARMS:
   if arm=='swap':continue
   raw=torch.load(O/f"task{row['spec']['task']:04d}/{arm}.pt",map_location='cpu',weights_only=False);online=Online(arm);W=online.W;logs=raw['world_logs'];li=0;snapshots=[W.clone()]
   for tr in raw['trials']:
    rr=tr['records']
    for j in range(0,len(rr)-1,8):
     end=min(j+8,len(rr)-1);assert all(not r['terminal'] and not r['teleported'] for r in rr[j:end]);f=observed_features(tensor(rr[j]['state']),tensor(rr[j]['context']),tensor(np.stack([command(r) for r in rr[j:end]])),end-j);target=tensor(rr[end]['state']);log=logs[li]
     for got,want in [(f.cpu(),log['features']),(target.cpu(),log['target']),(W.cpu(),log['W_before'])]:torch.testing.assert_close(got,want,atol=0,rtol=0)
     W=online.world.write(W,f,target).detach();torch.testing.assert_close(W.cpu(),log['W_after'],atol=0,rtol=0);snapshots.append(W.clone());li+=1;blocks+=1
   assert li==len(logs);torch.testing.assert_close(W.cpu(),raw['final_W'],atol=0,rtol=0)
   for tr in raw['trials']:
    for r in tr['records']:
     d=r.get('world_decision')
     if d:assert hashlib.sha256(snapshots[d['world_writes']].cpu().numpy().tobytes()).hexdigest()==d['world_W_sha256']
  print('OOD_INDEPENDENT_BLOCK_REVIEW',row['spec']['task'],blocks,flush=True)
 env,acfg,cfg=environment();count=controls=0;maxerr=0.
 try:
  for tid in [0,16]:
   folder=O/f'task{tid:04d}';start=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
   for arm in ARMS:
    raw=torch.load(folder/f'{arm}.pt',map_location='cpu',weights_only=False)
    for ti in [1,11]:
     count+=1;assert count<=20;env.physics.model.geom_friction[:]=start['geom_friction'];restore(env,start);obs=obs_now(env);tr=raw['trials'][ti]
     for r in tr['records']:
      err=float(np.max(abs(phys(obs)-phys(r['before']))));maxerr=max(maxerr,err);assert err<1e-9;assert env.episode_current_timestamp==r['time_before'] and env.judge.current_trial_num==r['judge_before'];obs,_,done,info=env.step(r['executed_command'],render_image=False);err=float(np.max(abs(phys(obs)-phys(r['after']))));maxerr=max(maxerr,err);assert err<1e-9;assert env.episode_current_timestamp==r['time_after'] and env.judge.current_trial_num==r['judge_after'];np.testing.assert_array_equal(info['executed_action'][0]['tcp_xyz_wxyz'],r['executed_command'][0]['tcp_xyz_wxyz']);assert bool(done)==r['done'];controls+=1
     xy=obs['env_objs_obs'][1]['object_pose_xyz_wxyz'][:2];assert reward(xy)==tr['terminal_return'] and bool(env.judge.trials_successful[-1])==tr['official_success'];print('OOD_NATIVE_REPLAY',tid,arm,ti+1,count,flush=True)
 finally:env.physics.free()
 assert count==20;write(O/'independent_review.json',dict(status='passed',independent_raw_observed_blocks=blocks,all_private_memory_snapshots_checked=True,native_replay_starts=count,replayed_controls=controls,max_physical_error=maxerr,optimizer_updates=0,ID_tasks=0,seconds=time.time()-begin,sources=sources()))
if __name__=='__main__':main()
