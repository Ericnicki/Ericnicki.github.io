"""Six repeated clean base programs, causal memory, explicit native command evidence."""
import copy,itertools,math
import numpy as np
import torch
from common import O,write,save,clean,obs_now
from controller import Controller,noise,A_MAX
from trial_budget import claim_start
from features_online import state_phase
from official import agent_for,restore,reward
def trial(env,acfg,start,speed,model,actor,memory,budget,identity,online=None,parity=False):
    claim_start(budget,identity);write(O/'budget.json',budget)
    env.physics.model.geom_friction[:]=start['geom_friction'];restore(env,start)
    obs=obs_now(env);agent=agent_for(acfg,[speed]*6);previous=[]
    last=base_previous=float(obs['robots_obs'][0]['tcp_xyz_wxyz'][1]);controller=Controller()
    started=False;records=[];prior=None;t0=float(env.episode_current_timestamp)
    result=dict(status='not_started',speed=speed,identity=identity,override=None)
    initial=clean(obs);friction_sha=__import__('hashlib').sha256(env.physics.model.geom_friction.tobytes()).hexdigest()
    assert math.isinf(env.cube_reset_waiting_time) and math.isinf(env.judge.time_limit_s)
    if online:online.start_trial()
    for control in itertools.count():
        assert np.isfinite(env.physics.get_state()).all()
        phase=agent.phase;nominal=copy.deepcopy(agent.predict_actions([obs['robots_obs']],[obs['env_objs_obs']],[previous])[0])
        critical=bool(nominal[0]['is_critical'][0]);base_y=float(nominal[0]['tcp_xyz_wxyz'][1])
        if critical and not started:
            started=True;result['status']='started'
        before=clean(obs);tb=float(env.episode_current_timestamp);jb=int(env.judge.current_trial_num)
        r=dict(control=control,phase=phase,critical=critical,base_y_m=base_y,base_previous_y_m=base_previous,
            previous_executed_y_m=last,before=before,time_before=tb,judge_before=jb,
            nominal_command=nominal,base_delta_y_m=base_y-base_previous,offset_before_m=controller.offset,
            residual_velocity_before_m_s=controller.residual_velocity,acc_bound_m_s2=A_MAX)
        state,proxy=state_phase(r,prior,t0);context=memory.context(state)
        if online:online.arrival(state)
        err=0.
        if parity and control in [0,1,15,16,31,32]:
            err=float((context-memory.reference(state)).abs().max());assert err<3e-5,('online_memory_parity',identity,control,err)
        with torch.no_grad():
            st=torch.as_tensor(state,device='cuda').reshape(1,22)
            actor_acc=float((A_MAX*torch.tanh(actor.net(torch.cat([context,st],dim=-1)))).item()) if critical else 0.
        delta,decision=online.choose(r,state,context,actor_acc,control) if online else (0.,None)
        c=controller.step(base_y,base_previous,last,actor_acc+delta,critical,started)
        cmd=copy.deepcopy(nominal);cmd[0]['tcp_xyz_wxyz'][1]=c['command_y_m']
        env._teleported=False;env._near_teleported=False
        after,native_reward,done,info=env.step(cmd,render_image=False)
        actual=float(info['executed_action'][0]['tcp_xyz_wxyz'][1]);assert abs(actual-c['command_y_m'])<1e-10
        terminal=int(env.judge.current_trial_num)>jb
        with torch.no_grad():
            a=torch.tensor([[c['effective_d_V_m_s']/.5]],device='cuda',dtype=torch.float32)
            q=float(model.head(torch.cat([context,st,a],dim=-1)).item()*5)
            value=float(model.value_head(torch.cat([context,st],dim=-1)).item()*5)
        r.update(after=clean(after),time_after=float(env.episode_current_timestamp),judge_after=int(env.judge.current_trial_num),
            terminal=terminal,teleported=bool(env._teleported),native_reward=copy.deepcopy(native_reward),
            world_decision=decision,noise_m_s2=0.,noise_rng_draws=0,held_correction_m_s2=delta,actor_acc_m_s2=actor_acc,done=bool(done),nominal_command=nominal,
            executed_command=copy.deepcopy(info['executed_action']),state=state,context=context.cpu().numpy()[0],
            proxy_phase=proxy,q_raw_return=q,v_raw_return=value,context_parity_error=err,prior_trial_count=len(memory.completed),
            current_observed_events=len(memory.current),**c)
        event=np.r_[state,c['effective_d_V_m_s']/.5].astype(np.float32);r['event']=event
        memory.observe(event);records.append(r)
        if online:online.record(r)
        obs=after;previous=cmd;last=actual;base_previous=base_y;prior=r
        if terminal:
            xy=after['env_objs_obs'][1]['object_pose_xyz_wxyz'][:2].tolist();g=reward(xy)
            success=bool(env.judge.trials_successful[-1]);assert success==(g>0)
            result.update(status='completed',terminal_return=g,official_success=success,stop_xy=xy);break
        if env._teleported or done:raise RuntimeError('Unexpected reset or timeout before judgment')
    if online:online.end_trial(records[-1]['state'])
    memory.end_trial()
    result.update(records=records,initial_observation=initial,restored_friction_sha256=friction_sha,
        peak_abs_offset_m=max(abs(r['offset_after_m']) for r in records),controls=len(records))
    assert records and all(abs(r['residual_velocity_after_m_s'])<=.100000000001 and abs(r['applied_d_acc_m_s2'])<=.50000000001 for r in records)
    return result
