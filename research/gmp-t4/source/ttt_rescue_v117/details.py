from rescue_common import *
import json
from controller import actor_action
setup();out={}
rows=json.loads((R/'ttt_joint_1000_v115_output/evaluation/summary.json').read_text())
for model_name,path in [('old',R/'ttt_joint_completion_v102_output'),('new',R/'ttt_joint_1000_v115_output')]:
 m,a=load(model_name);stats=[]
 for row in rows:
  tid=row['spec']['task'];d=torch.load(path/f'evaluation/task{tid:04d}/actor_deterministic.pt',map_location='cpu',weights_only=False)
  for ti,t in enumerate(d['trials']):
   rr=[r for r in t['records'] if r['critical']];ctx=torch.tensor(np.array([r['context'] for r in rr]),device='cuda');s=torch.tensor(np.array([r['state'] for r in rr]),device='cuda')
   with torch.no_grad():z=a.net(torch.cat([ctx,s],-1)).squeeze(-1)
   acts=torch.tensor([[r['effective_d_V_m_s']/.5] for r in rr],device='cuda',dtype=torch.float32,requires_grad=True);q=m.head(torch.cat([ctx,s,acts],-1));dq=torch.autograd.grad(q.sum(),acts)[0].flatten()
   g=t.get('terminal_return');xy=t.get('stop_xy');stats.append(dict(task=tid,trial=ti+1,region=row['spec']['region'],G=g,xy=xy,
    overshoot=xy[1]>.2 if xy else None,x_dominant=abs(xy[0]-.1)>abs(xy[1]-.15) if xy else None,
    active=len(rr),mean_acc=float(np.mean([r['actor_acc_m_s2'] for r in rr])),max_abs_offset=t['peak_abs_offset_m'],
    tanh_saturated=float((z.tanh().abs()>.95).float().mean()),velocity_saturated=float(np.mean([r['velocity_saturation'] for r in rr])),
    envelope=float(np.mean([r['constraint_intervention'] for r in rr])),mean_abs_dQ_da=float(dq.abs().mean()),positive_Q_slope=float((dq>0).float().mean())))
 out[model_name]={}
 for region in ['all','low','middle','high']:
  rs=[r for r in stats if region=='all' or r['region']==region];out[model_name][region]={k:float(np.mean([r[k] for r in rs])) for k in ['G','overshoot','x_dominant','mean_acc','max_abs_offset','tanh_saturated','velocity_saturated','envelope','mean_abs_dQ_da','positive_Q_slope']}
 write(O/f'{model_name}_trajectory_details.json',stats)
write(O/'trajectory_details.json',out);print(json.dumps(out,indent=2))
