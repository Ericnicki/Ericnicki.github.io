from rescue_common import *
import json,collections
setup();rows=json.loads((R/'ttt_joint_1000_v115_output/evaluation/summary.json').read_text());out={}
for name,path in [('old',R/'ttt_joint_completion_v102_output'),('new',R/'ttt_joint_1000_v115_output')]:
 stats=collections.defaultdict(list)
 for row in rows:
  tid=row['spec']['task'];d=torch.load(path/f'evaluation/task{tid:04d}/actor_deterministic.pt',weights_only=False)
  for t in d['trials']:
   critical=[r for r in t['records'] if r['critical']]
   for k,part in [('first_half',critical[:len(critical)//2]),('second_half',critical[len(critical)//2:])]:
    stats[(row['spec']['region'],k)].append(dict(acc=float(np.mean([r['actor_acc_m_s2'] for r in part])),residual_velocity=float(np.mean([r['effective_d_V_m_s'] for r in part])),peak_command_offset=max(r['offset_after_m'] for r in part),min_command_offset=min(r['offset_after_m'] for r in part)))
 out[name]={f'{r}/{p}':{k:float(np.mean([x[k] for x in values])) for k in values[0]} for (r,p),values in stats.items()}
write(O/'phase_details.json',out);print(json.dumps(out,indent=2))
