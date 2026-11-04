"""Streaming equivalent of the frozen 22-state/23-event feature contract."""
import numpy as np
def state_phase(r,prior,t0):
    tcp=np.asarray(r['before']['robots_obs'][0]['tcp_xyz_wxyz'][:3],float)
    obj=r['before']['env_objs_obs'][1];cube=np.asarray(obj['object_pose_xyz_wxyz'][:3],float)
    cube_v=np.asarray(obj['object_vel_xyz_xyz'][:3],float)
    dt=r['time_before']-prior['time_before'] if prior else .1
    tcp_v=(tcp-np.asarray(prior['before']['robots_obs'][0]['tcp_xyz_wxyz'][:3]))/dt if prior else np.zeros(3)
    robot=np.r_[(tcp-[.1,-.6,.365])/[.2,.3,.2],tcp_v/.5]
    cf=np.r_[(cube-[.1,-.1,.345])/[.2,.5,.1],cube_v/.5]
    prev_v=prior['executed_delta_y_m']/.1 if prior else 0.
    ctrl=np.array([(r['previous_executed_y_m']+.6)/.3,prev_v/.5,(r['base_previous_y_m']+.6)/.3,
        r['base_delta_y_m']/.05,r['offset_before_m']/.2,r['residual_velocity_before_m_s']/.5,
        prior['effective_d_V_m_s']/.5 if prior else 0.,float(prior is not None),(r['time_before']-t0)/8,r['acc_bound_m_s2']/2.])
    x=np.r_[robot,cf,ctrl].astype(np.float32);assert x.shape==(22,) and np.isfinite(x).all()
    gap=cube[1]-tcp[1];moving=abs(cube_v[1])>.005
    contact=abs(cube[0]-tcp[0])<.03 and .010<=gap<=.045 and moving
    phase=None
    if contact:phase='contact_early' if tcp[1]<-.60 else 'contact_late'
    elif moving and .045<gap<=.10 and r['phase'] in ['pushing','slowing_down']:phase='about_to_leave'
    elif moving and gap>.045 and not r['critical']:phase='sliding'
    return x,phase
