from rescue_common import *
import json
setup();rows=json.loads((O/'probes.json').read_text());out={}
for region in ['all','low','middle','high']:
 rs=[r for r in rows if region=='all' or r['region']==region];items=[]
 for r in rs:
  aa=r['answers'][1:];g=np.array([a['return_'] for a in aa]);eq=np.array([a['early_Q'] for a in aa]);lq=np.array([a['late_Q'] for a in aa]);orig=r['answers'][0]['return_']
  items.append(dict(task=r['task'],true_range=float(np.ptp(g)),early_range=float(np.ptp(eq)),late_range=float(np.ptp(lq)),oracle_gain=float(max(g.max(),orig)-orig),early_gain=float(g[eq.argmax()]-orig),late_gain=float(g[lq.argmax()]-orig),random_gain=float(g.mean()-orig),zero_contrast_mae=float(np.mean(abs(g-g[2]))),teacher_regret_delta=float((g.max()-g[lq.argmax()])-(g.max()-g[eq.argmax()]))))
 stats={k:float(np.mean([x[k] for x in items])) for k in items[0] if k!='task'}
 tasks=sorted({x['task'] for x in items});v=np.array([np.mean([x['teacher_regret_delta'] for x in items if x['task']==tid]) for tid in tasks]);rng=np.random.default_rng(2038000);stats['teacher_minus_early_regret_CI95']=np.quantile(v[rng.integers(0,len(v),(10000,len(v)))].mean(1),[.025,.975]).tolist();out[region]=stats
write(O/'probe_baselines.json',out);print(json.dumps(out,indent=2))
