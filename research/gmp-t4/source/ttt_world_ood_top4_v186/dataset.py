"""Causal obs/action features; all outcomes live outside the event tensors."""
import numpy as np
import torch
STATE_DIM=22
PHASES=['contact_early','contact_late','about_to_leave','sliding']

def features(records):
    states=[];events=[];phases=[];prior=None;t0=records[0]['time_before']
    for r in records:
        tcp=np.asarray(r['before']['robots_obs'][0]['tcp_xyz_wxyz'][:3],float)
        obj=r['before']['env_objs_obs'][1];cube=np.asarray(obj['object_pose_xyz_wxyz'][:3],float)
        cube_v=np.asarray(obj['object_vel_xyz_xyz'][:3],float)
        dt=r['time_before']-prior['time_before'] if prior else .1
        tcp_v=(tcp-np.asarray(prior['before']['robots_obs'][0]['tcp_xyz_wxyz'][:3]))/dt if prior else np.zeros(3)
        robot=np.r_[(tcp-[.1,-.6,.365])/[.2,.3,.2],tcp_v/.5]
        cube_features=np.r_[(cube-[.1,-.1,.345])/[.2,.5,.1],cube_v/.5]
        prev_v=prior['executed_delta_y_m']/.1 if prior else 0.
        ctrl=np.array([(r['previous_executed_y_m']+.6)/.3,prev_v/.5,
            (r['base_previous_y_m']+.6)/.3,r['base_delta_y_m']/.05,
            r['offset_before_m']/.2,r['residual_velocity_before_m_s']/.5,
            prior['effective_d_V_m_s']/.5 if prior else 0.,float(prior is not None),(r['time_before']-t0)/8,r['acc_bound_m_s2']/2.])
        x=np.r_[robot,cube_features,ctrl].astype(np.float32);assert x.shape==(STATE_DIM,) and np.isfinite(x).all()
        states.append(x);events.append(np.r_[x,r['effective_d_V_m_s']/.5].astype(np.float32))
        # Explicit geometric/motion proxies, independent of Q and outcome. These
        # are not measured force contacts; no future trial state is consulted.
        gap=cube[1]-tcp[1];moving=abs(cube_v[1])>.005
        contact=abs(cube[0]-tcp[0])<.03 and .010<=gap<=.045 and moving
        phase=None
        if contact:phase='contact_early' if tcp[1]<-.60 else 'contact_late'
        elif moving and .045<gap<=.10 and r['phase'] in ['pushing','slowing_down']:phase='about_to_leave'
        elif moving and gap>.045 and not r['critical']:phase='sliding'
        phases.append(phase);prior=r
    return np.array(states),np.array(events),phases

def batch(tasks,queries,all_history,device):
    histories=[];states=[];actions=[];targets=[]
    for task_id,ti,index in queries:
        trials=tasks[task_id]['trials'];current=trials[ti]
        previous=trials[:ti] if all_history else trials[max(0,ti-1):ti]
        parts=[t['events'] for t in previous]+[current['events'][:index]]
        padded=[];valid=[]
        for part in parts:
            L=len(part)
            if L:
                N=((L+15)//16)*16
                x=np.zeros((N,23),np.float32);x[:L]=part;padded.append(x)
                m=np.zeros(N,np.float32);m[:L]=1;valid.append(m)
        history=np.concatenate(padded) if padded else np.empty((0,23),np.float32)
        hm=np.concatenate(valid) if valid else np.empty(0,np.float32)
        histories.append((history,hm));states.append(current['states'][index]);actions.append(current['events'][index,-1:]);targets.append(current['return']/5.)
    T=max(len(h[0]) for h in histories);B=len(queries)
    h=np.zeros((B,T,23),np.float32);mask=np.zeros((B,T),np.float32)
    for i,(x,m) in enumerate(histories):h[i,:len(x)]=x;mask[i,:len(x)]=m
    return [torch.as_tensor(x,device=device) for x in [h,mask,np.asarray(states),np.asarray(actions),np.asarray(targets,np.float32)]]
