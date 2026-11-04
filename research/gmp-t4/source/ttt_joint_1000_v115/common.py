"""Exclusive joint-training run bindings and atomic artifacts."""
import copy,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
R=Path('/public/home/2023141520023/return_selector_20261006')
O=R/'ttt_joint_1000_v115_output'
P=R/'ttt_joint_v100_output/joint_final.pt'
CRITIC_PARENT=P
TASKS=1000
GROUPS=125
INIT_Q=76300
INIT_ACTOR=9450
INIT_WRITER=12630
P_SHA='d7d513ca8ed90830b92c5ed5d58aad059637f8c68bfa34e4a78095f9dc6cea75'
sys.path.insert(1,str(Path(__file__).parent))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def save(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('wb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def obs_now(env):return dict(robots_obs=env._get_robots_obs(False),env_objs_obs=env._get_env_objs_obs(False))
def clean(obs):
    return dict(robots_obs=copy.deepcopy(obs['robots_obs']),env_objs_obs=[{k:copy.deepcopy(v) for k,v in o.items()
        if not ('camera' in k and isinstance(v,np.ndarray) and v.ndim==3)} for o in obs['env_objs_obs']])
def load_networks():
    from model import Critic,ResidualActor
    assert sha(P)==P_SHA
    ck=torch.load(P,map_location='cpu',weights_only=False)
    assert ck['Q_steps']==76300 and ck['writer_steps']==12630 and ck['actor_steps']==9450
    model=Critic().cuda().eval();model.load_state_dict(ck['model'],strict=True)
    actor=ResidualActor().cuda().eval();actor.load_state_dict(ck['actor'],strict=True)
    for p in model.parameters():p.requires_grad_(True)
    return model,actor,ck
def frozen_equal(model,ck):
    for k,v in model.state_dict().items():assert torch.equal(v.detach().cpu(),ck['model'][k]),('frozen_changed',k)
    assert all(p.grad is None for p in model.parameters())
def sources():
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'PROTOCOL.md',
        R/'endpoint_currentnode_runtime_v2/runtime.sh']
    return {str(p):sha(p) for p in files}
