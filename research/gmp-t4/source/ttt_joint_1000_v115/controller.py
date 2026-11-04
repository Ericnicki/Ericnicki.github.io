"""Physical acceleration projection; Q action is actual average residual velocity."""
from dataclasses import dataclass
import numpy as np
import torch
DT=.1
A_MAX=.5
V_MAX=.1
Y_MIN,Y_MAX=-.85,.25
SIGMA=.05
NOISE_MAX=.15
def noise(rng):
    draws=0
    while True:
        x=float(rng.normal(0,SIGMA));draws+=1
        if abs(x)<=NOISE_MAX:return x,draws
def actor_action(actor,context,state,old_v,previous_y,base_delta):
    z=actor.net(torch.cat([context,state],dim=-1)).squeeze(-1)
    acc=A_MAX*torch.tanh(z)
    new_v=(old_v+DT*acc).clamp(-V_MAX,V_MAX)
    dv=(old_v+new_v)/2
    requested=previous_y+base_delta+DT*dv
    y=requested.clamp(Y_MIN,Y_MAX)
    effective=(y-previous_y-base_delta)/DT
    return effective,acc,new_v
@dataclass
class Controller:
    residual_velocity:float=0.
    offset:float=0.
    def step(self,base_y,base_previous,previous_y,proposal,critical,started):
        old=self.residual_velocity;offset=self.offset
        if not started:acc=0.;new=0.;dv=0.;requested=base_y
        else:
            acc=float(np.clip(proposal,-A_MAX,A_MAX)) if critical else float(np.clip(-old/DT,-A_MAX,A_MAX))
            new=float(np.clip(old+DT*acc,-V_MAX,V_MAX));dv=(old+new)/2
            requested=previous_y+(base_y-base_previous)+DT*dv
        y=float(np.clip(requested,Y_MIN,Y_MAX));effective=(y-previous_y-(base_y-base_previous))/DT if started else 0.
        self.residual_velocity=new;self.offset=y-base_y
        return dict(d_acc_proposal_m_s2=proposal,requested_d_acc_m_s2=acc,
            applied_d_acc_m_s2=(new-old)/DT,residual_velocity_before_m_s=old,
            residual_velocity_after_m_s=new,d_V_m_s=dv,effective_d_V_m_s=effective,
            offset_before_m=offset,offset_after_m=self.offset,requested_y_m=requested,command_y_m=y,
            base_delta_y_m=base_y-base_previous,executed_delta_y_m=y-previous_y,
            active=started,constraint_intervention=y!=requested,acc_bound_m_s2=A_MAX,
            acceleration_saturation=started and critical and abs(proposal)>A_MAX,
            velocity_saturation=started and abs(old+DT*acc)>V_MAX)
