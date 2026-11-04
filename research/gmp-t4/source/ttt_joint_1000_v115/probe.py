"""Locked same-state one-control action probes from exact physical replay."""
import copy,hashlib,json,socket,time
from pathlib import Path
from collections import deque
import hydra,numpy as np,torch
assert socket.gethostname()=='c1d3019784aa'
ROOT=Path('/public/home/2023141520023');R=ROOT/'return_selector_20261006';D=R/'recurrent_mage_random_official_v2';O=R/'recurrent_mage_contact_probe_v2'
CONFIG=ROOT/'gmp/gated-memory-policy/mujoco-env/config'
OFFSET,SCALE=-.6150000095367432,.18500010669231415
ENV_KEYS=['episode_current_timestamp','cube_pass_line_time','cube_last_movement_time','cube_last_pos_y']
JUDGE_KEYS=['current_trial_num','cube_status','stop_in_target_zone','time_limit_reached','last_object_pose','time_since_finished_s','finished_timestamp','last_trial_timestamp','trials_successful','total_trial_num']
ARRAY_KEYS=['ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat','userdata']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def state(obs,env,history,last):
    ys=list(history);ys=[ys[0]]*(4-len(ys))+ys;c=obs['env_objs_obs'][1]['object_pose_xyz_wxyz']
    return np.asarray([(c[0]-.1)/.1]+[(y+.7)/.1 for y in ys]+[(obs['robots_obs'][0]['tcp_xyz_wxyz'][1]+.73)/.1,max(0,1-(env.episode_current_timestamp-env.judge.last_trial_timestamp)/20),(last-OFFSET)/SCALE],np.float32)
def command(obs,y):return [dict(name=obs['robots_obs'][0]['name'],tcp_xyz_wxyz=np.array([.1,y,.365,.707,-.707,0.,0.]),gripper_width=np.array([.08]))]
def snap(env):
    return dict(physics=env.physics.get_state().copy(),arrays={k:np.array(getattr(env.data,k),copy=True) for k in ARRAY_KEYS},env={k:copy.deepcopy(getattr(env,k)) for k in ENV_KEYS},judge={k:copy.deepcopy(getattr(env.judge,k)) for k in JUDGE_KEYS},robot=[copy.deepcopy(r.latest_action_tcp_xyz_wxyz) for r in env.robots],rng=copy.deepcopy(env.rng.bit_generator.state) if hasattr(env,'rng') else None)
def restore(env,s):
    env.physics.set_state(s['physics']);env.physics.forward()
    for k,v in s['arrays'].items():getattr(env.data,k)[:]=v
    for k,v in s['env'].items():setattr(env,k,copy.deepcopy(v))
    for k,v in s['judge'].items():setattr(env.judge,k,copy.deepcopy(v))
    for r,v in zip(env.robots,s['robot']):r.latest_action_tcp_xyz_wxyz=copy.deepcopy(v)
    if s['rng'] is not None:env.rng.bit_generator.state=copy.deepcopy(s['rng'])
    np.testing.assert_array_equal(env.physics.get_state(),s['physics'])
def main():
    torch.set_num_threads(4);O.mkdir(exist_ok=False);manifest=json.loads((D/'manifest.json').read_text());cap=json.loads((R/'recurrent_mage_capacity20k_v1/manifest.json').read_text());train=cap['train_task_ids'][:2];val=manifest['partitions']['validation'][:2];selected=[];tasks={}
    for task_id in train+val:
        d=torch.load(D/f'task{task_id:04d}/task.pt',map_location='cpu',weights_only=False);tasks[task_id]=d
        for trial in [1,2]:
            rr=[r for r in d['records'] if r['trial']==trial and r['phase']=='pushing' and r['contact']]
            assert len(rr)>=4
            # Fixed early/middle positions in previously recorded contact phase;
            # retrospective diagnostic state selection, no endpoint filtering.
            for label,k in [('early',1),('middle',len(rr)//2)]:
                z=rr[k];selected.append(dict(task=task_id,trial=trial,step=z['step'],event_index=z['event_index'],stage=label,partition='train' if task_id in train else 'development_validation'))
    variants=[0.,0.,-.00025,.00025,-.001,.001,-.002,.002]
    budget=dict(prefix_trial_replays=0,branches_started=0,branch_cap=len(selected)*len(variants),prefix_trial_cap=8,combined_start_cap=136)
    write(O/'manifest.json',dict(states=selected,offsets_m=variants,one_control_duration_s=.1,scope='Physical one-control secants, not value/endpoint gradients or complete trial outcomes.',task_sha256={str(i):sha(D/f'task{i:04d}/task.pt') for i in tasks},source_sha256=sha(Path(__file__)),budget=budget));write(O/'budget.json',budget)
    with hydra.initialize_config_dir(config_dir=str(CONFIG),version_base=None):cfg=hydra.compose(config_name='task/push_cube')
    env=hydra.utils.instantiate(cfg.task.env);env.init_simulator();teleported=[False];reset0=env.objects[0].set_pose_xyz_wxyz
    def reset(p):reset0(p);teleported[0]=True
    env.objects[0].set_pose_xyz_wxyz=reset
    output=[];start=time.time()
    try:
        for tid,d in tasks.items():
            cfg=d['config'];obs,_=env.reset(dict(seed=cfg['seed'],episode_idx=cfg['seed'],sliding_friction=cfg['friction']));history=deque([float(obs['env_objs_obs'][1]['object_pose_xyz_wxyz'][1])],maxlen=4);last=float(obs['robots_obs'][0]['tcp_xyz_wxyz'][1]);target={s['step']:s for s in selected if s['task']==tid};seen=set();replayed=set()
            for rec in d['records']:
                idx=rec['event_index'];x=state(obs,env,history,last);np.testing.assert_array_equal(x,d['events'][idx,:8]);assert abs(obs['env_objs_obs'][1]['object_pose_xyz_wxyz'][1]-rec['before_xy'][1])<1e-10
                if rec['trial'] in [1,2] and rec['trial'] not in replayed:
                    replayed.add(rec['trial']);budget['prefix_trial_replays']+=1;assert budget['prefix_trial_replays']<=8;write(O/'budget.json',budget)
                if rec['step'] in target:
                    st=target[rec['step']];ss=snap(env);answers=[]
                    torch.save(dict(snapshot=ss,obs=copy.deepcopy(obs),history=list(history),last=last,event_prefix=d['events'][:idx],record=rec,state=st),O/f'state_t{tid:04d}_s{rec["step"]:04d}.pt')
                    for variant,offset in enumerate(variants):
                        restore(env,ss);teleported[0]=False;budget['branches_started']+=1;assert budget['branches_started']<=budget['branch_cap'];write(O/'budget.json',budget)
                        after,_,done,info=env.step(command(obs,rec['command_y']+offset),render_image=False);assert not teleported[0] and not done and env.judge.current_trial_num==ss['judge']['current_trial_num']
                        actual=float(info['executed_action'][0]['tcp_xyz_wxyz'][1]);assert abs(actual-rec['command_y']-offset)<1e-10
                        hh=deque(history,maxlen=4);hh.append(float(after['env_objs_obs'][1]['object_pose_xyz_wxyz'][1]));nx=state(after,env,hh,actual);answers.append(dict(variant=variant,offset_m=offset,next_state=nx.tolist(),after_xy=after['env_objs_obs'][1]['object_pose_xyz_wxyz'][:2].tolist(),arm_y=float(after['robots_obs'][0]['tcp_xyz_wxyz'][1]),cube_velocity=after['env_objs_obs'][1]['object_vel_xyz_xyz'].tolist()))
                    np.testing.assert_array_equal(answers[0]['next_state'],answers[1]['next_state']);np.testing.assert_array_equal(answers[0]['cube_velocity'],answers[1]['cube_velocity']);restore(env,ss)
                    np.testing.assert_array_equal(np.asarray(answers[0]['next_state'],np.float32),d['events'][idx,9:17]);output.append(dict(state=st,x=x.tolist(),action_normalized=float(d['events'][idx,8]),answers=answers,repeat_exact=True));seen.add(rec['step']);print('STATE_PROBED',st,flush=True)
                teleported[0]=False;obs,_,done,info=env.step(command(obs,rec['command_y']),render_image=False);last=float(info['executed_action'][0]['tcp_xyz_wxyz'][1]);history.append(float(obs['env_objs_obs'][1]['object_pose_xyz_wxyz'][1]));np.testing.assert_array_equal(state(obs,env,history,last),d['events'][idx,9:17]);np.testing.assert_allclose(obs['env_objs_obs'][1]['object_vel_xyz_xyz'],rec['cube_velocity'],atol=1e-10,rtol=0)
                assert teleported[0]==rec['teleported']
                if teleported[0]:env.physics.forward();obs=dict(robots_obs=env._get_robots_obs(False),env_objs_obs=env._get_env_objs_obs(False));history.clear();history.append(float(obs['env_objs_obs'][1]['object_pose_xyz_wxyz'][1]))
                if seen==set(target):break
            assert seen==set(target)
        write(O/'labels.json',output);write(O/'completion.json',dict(status='completed',states=len(output),budget=budget,replay_exact=True,repeat_exact=True,seconds=time.time()-start));print('PROBES_COMPLETE',len(output))
    finally:env.physics.free()
if __name__=='__main__':main()
