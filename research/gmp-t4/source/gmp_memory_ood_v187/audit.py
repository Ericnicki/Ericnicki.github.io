import sys,time
from common import *
from policy import Official
from robot_utils.pose_utils import quat_wxyz_to_rot_6d_batch

def main():
 worker=int(sys.argv[1]);setup();start_time=time.time();p=Official();mh=state_hash(p.model);nh=state_hash(p.norm);rows=json.loads((O/'summary.json').read_text());controls=decisions=judged=0;bindings={}
 for row in rows:
  tid=row['spec']['task']
  if tid%4!=worker:continue
  folder=O/f'task{tid:04d}';assert sha(folder/'initial.pt')==row['initial_sha256']==sha(B/f'task{tid:04d}/initial.pt');raw=torch.load(folder/'gmp.pt',map_location='cpu',weights_only=False);assert sha(folder/'gmp.pt')==row['raw_sha256'];p.new_task(tid);expected_initial=None
  for ti,tr in enumerate(raw['trials']):
   equal(tr,torch.load(folder/f'gmp_t{ti+1:02d}.pt',map_location='cpu',weights_only=False));initial=copy.deepcopy(tr['initial_observation'])
   if expected_initial is None:expected_initial=initial
   else:
    for robot in initial['robots_obs']:robot.pop('arm_qacc',None)
    ref=copy.deepcopy(expected_initial)
    for robot in ref['robots_obs']:robot.pop('arm_qacc',None)
    equal(initial,ref)
   reference=torch.load(B/f'task{tid:04d}/swap_t{ti+1:02d}.pt',map_location='cpu',weights_only=False)['initial_observation'];actual=copy.deepcopy(tr['initial_observation'])
   for x in [reference,actual]:
    for robot in x['robots_obs']:robot.pop('arm_qacc',None)
   equal(reference,actual);equal(p.history(),tr['history_start']);rr=tr['records'];di=0
   for j,r in enumerate(rr):
    if j:equal(r['before'],rr[j-1]['after'])
    assert abs(r['time_after']-r['time_before']-.1)<1e-9 and not r['teleported']
    if r['token']==0:
     d=tr['decisions'][di];assert d['control']==j;equal(p.history(),d['history_before']);assert torch.equal(p.model.torch_rng.get_state().cpu(),d['rng_before']);torch.set_rng_state(d['global_rng_before']['cpu']);torch.cuda.set_rng_state_all(d['global_rng_before']['cuda']);p.model.torch_rng.set_state(d['rng_before']);raw_input=d['raw_input'];assert set(raw_input)=={'third_person_camera','robot0_10d','episode_idx'} and raw_input['episode_idx']==[tid]
     robot=r['before']['robots_obs'][0];pose=robot['tcp_xyz_wxyz'];want=np.r_[pose[:3],quat_wxyz_to_rot_6d_batch(pose[None,3:])[0],robot['gripper_width']];np.testing.assert_array_equal(raw_input['robot0_10d'][0,0],want)
     inferred=p.predict_raw(raw_input)
     for k in inferred:np.testing.assert_array_equal(inferred[k],d['physical_action'][k])
     equal(p.last['normalized_action'],d['normalized_action']);equal(p.history(),d['history_after']);assert p.last['gate']==d['gate'] and p.last['history_attention_skipped']==d['history_attention_skipped'];assert len(d['decoded_actions'])==16
     # Independent official Y smoothing; disabled axes/gripper must stay at original defaults.
     raw_y=d['physical_action']['action0_10d'][0,:,1].astype(float);prev=float(d['previous'][0]['tcp_xyz_wxyz'][1]);expected=[]
     for token in range(8):
      y=.6*raw_y[token]+.2*prev+.2*raw_y[token+1];expected.append(y);prev=y
     for token in range(8):assert abs(float(d['decoded_actions'][token][0]['tcp_xyz_wxyz'][1])-expected[token])<2e-7
     decisions+=1;di+=1
    d=tr['decisions'][r['decision']];equal(r['requested_command'],d['decoded_actions'][r['token']]);assert abs(r['requested_command'][0]['tcp_xyz_wxyz'][1]-r['executed_command'][0]['tcp_xyz_wxyz'][1])<1e-9
    np.testing.assert_array_equal(r['requested_command'][0]['tcp_xyz_wxyz'][[0,2,3,4,5,6]],np.asarray([.1,.365,.707,-.707,0.,0.],dtype=r['requested_command'][0]['tcp_xyz_wxyz'].dtype));np.testing.assert_array_equal(r['requested_command'][0]['gripper_width'],[.08]);controls+=1
   assert di==len(tr['decisions']);equal(p.history(),tr['history_end'])
   if tr['status']=='completed':
    xy=rr[-1]['after']['env_objs_obs'][1]['object_pose_xyz_wxyz'][:2].tolist();assert xy==tr['stop_xy'] and reward(xy)==tr['terminal_return'] and tr['official_success']==(reward(xy)>0) and rr[-1]['terminal'];judged+=1
   else:assert len(rr)==1200 and 'terminal_return' not in tr and not rr[-1]['terminal']
  equal(p.history(),raw['final_history']);print('OFFICIAL_GMP_AUDIT_TASK',worker,tid,controls,decisions,flush=True)
  for path in folder.glob('*.pt'):bindings[str(path.relative_to(O))]=sha(path)
 assert state_hash(p.model)==mh and state_hash(p.norm)==nh;write(O/f'audit_worker{worker}.json',dict(status='passed',trials=96,controls=controls,decisions=decisions,judged=judged,censors=96-judged,all_model_inferences_rng_history_and_action_transforms_replayed=True,matched_initials=True,frozen_parameters=True,ID_tasks=0,optimizer_updates=0,model_sha256=mh,normalizer_sha256=nh,sources=sources(),raw_bindings=bindings,seconds=time.time()-start_time))
if __name__=='__main__':main()
