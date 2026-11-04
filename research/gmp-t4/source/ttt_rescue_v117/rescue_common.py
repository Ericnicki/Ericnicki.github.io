import sys, copy, socket
from pathlib import Path
R=Path('/public/home/2023141520023/return_selector_20261006')
sys.path.insert(1,str(R/'ttt_joint_1000_v115'))
import numpy as np, torch
from common import write,save,sha
from model import Critic,ResidualActor
from memory import Memory
O=R/'ttt_rescue_v117_output'
PATHS={'old':R/'ttt_joint_v100_output/joint_final.pt','new':R/'ttt_joint_1000_v115_output/joint_final.pt'}
HASHES={'old':'d7d513ca8ed90830b92c5ed5d58aad059637f8c68bfa34e4a78095f9dc6cea75','new':'569d6925ed8e5960f8fc11f21c0f76b004dfbb4e78ec6b6dab43804fc71efacd'}
def setup():
    assert socket.gethostname()=='c1d3019784aa';torch.set_num_threads(1)
    torch.manual_seed(2030000);np.random.seed(2030000)
def load(which='new',actor_from=None):
    assert sha(PATHS[which])==HASHES[which]
    ck=torch.load(PATHS[which],map_location='cpu',weights_only=False)
    m=Critic().cuda().eval();a=ResidualActor().cuda().eval()
    m.load_state_dict(ck['model']);a.load_state_dict(ck['actor'] if actor_from is None else torch.load(PATHS[actor_from],map_location='cpu',weights_only=False)['actor'])
    for p in list(m.parameters())+list(a.parameters()):p.requires_grad_(False)
    return m,a
def claim_start(budget,key,entry):
    assert budget[key]<budget['limit'];budget[key]+=1;budget['entries'].append(entry)
def rebuild(model,trials):
    mem=Memory(model)
    for t in trials:
        for r in t['records']:mem.observe(r['event'])
        mem.end_trial()
    return mem
def env_setup():
    import hydra
    from official import configurations
    cfg,acfg=configurations();env=hydra.utils.instantiate(cfg.task.env);env.init_simulator()
    original=env.objects[0].set_pose_xyz_wxyz
    def tracked(p):original(p);env._teleported=True;env._near_teleported=True
    env.objects[0].set_pose_xyz_wxyz=tracked
    return env,acfg
def brief(trials):
    gs=[t.get('terminal_return') for t in trials];valid=[g for g in gs if g is not None]
    success=[g is not None and g>0 for g in gs]
    return dict(returns=gs,judged=len(valid),successes=sum(success),mean_return=float(np.mean(valid)) if valid else None,
      last_three_all=all(success[-3:]),failure_after_success=any(not success[i] and any(success[:i]) for i in range(len(success))),
      first_success=next((i+1 for i,s in enumerate(success) if s),None))
