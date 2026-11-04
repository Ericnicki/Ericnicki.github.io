from audit_results import *
import itertools
setup();assert json.loads((FITS/'complete.json').read_text())['status']=='completed';queries=json.loads((O/'probes.json').read_text());out={};raw={}
for tid in sorted({q['task'] for q in queries}):raw[tid]=torch.load(R/f'ttt_joint_1000_v115_output/evaluation/task{tid:04d}/actor_deterministic.pt',map_location='cpu',weights_only=False)['trials']
for seed in [2033000,2033001]:
 for arm in ['mc','hindsight','grounded']:
  key=f'{arm}_{seed}';m,_=load();ck=torch.load(FITS/key/'step1500.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);rs=[]
  for query in queries:
   tid,ti,ix=query['task'],query['trial'],query['control'];trials=raw[tid];rr=trials[ti]['records'];mem=rebuild(m,trials[:ti])
   for r in rr[:ix]:mem.observe(r['event'])
   ctx=mem.context(rr[ix]['state']);s=torch.tensor(rr[ix]['state'],device='cuda').reshape(1,22);aa=query['answers'][1:];acts=torch.tensor([[a['dv']/.5] for a in aa],device='cuda',dtype=torch.float32)
   with torch.no_grad():q=m.head(torch.cat([ctx.expand(5,-1),s.expand(5,-1),acts],-1)).flatten().cpu().numpy()*5
   g=np.array([a['return_'] for a in aa]);teacher=np.array([a['late_Q'] for a in aa]);pairs=[(i,j) for i,j in itertools.combinations(range(5),2) if abs(g[i]-g[j])>.01 and abs(aa[i]['dv']-aa[j]['dv'])>1e-7]
   rs.append(dict(task=tid,region=query['region'],signs=int(sum((q[j]-q[i])*(g[j]-g[i])>0 for i,j in pairs)),pairs=len(pairs),regret=float(g.max()-g[q.argmax()]),contrast_mae=float(np.mean(abs((q-q[2])-(g-g[2])))),teacher_relative_mae=float(np.mean(abs((q-q[2])-(teacher-teacher[2])))),range=float(np.ptp(q))))
  out[key]={}
  for region in ['all','low','middle','high']:
   ss=[r for r in rs if region=='all' or r['region']==region];out[key][region]=dict(signs=sum(r['signs'] for r in ss),pairs=sum(r['pairs'] for r in ss),**{k:float(np.mean([r[k] for r in ss])) for k in ['regret','contrast_mae','teacher_relative_mae','range']})
  print('SCORED',key,flush=True)
write(FITS/'reference_policy_branch_scores.json',dict(scope='Frozen v115 query histories and branch continuation, not refitted actor current-policy branches',source_sha256=sha(Path(__file__)),scores=out))
