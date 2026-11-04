from audit_results import *
import itertools
setup();targets=torch.load(FITS/'branch_targets.pt',map_location='cpu',weights_only=False);m,_=load();raw={};rows=json.loads((D/'summary.json').read_text());out=[]
for b in targets:
 tid,ti,ix=b['task'],b['trial'],b['index']
 if tid not in raw:raw[tid]=torch.load(D/f'task{tid:04d}/task.pt',map_location='cpu',weights_only=False)
 d=raw[tid];r=d['trials'][ti]['records'][ix];ctx=torch.tensor(r['context'],device='cuda').reshape(1,-1);s=torch.tensor(r['state'],device='cuda').reshape(1,-1)
 with torch.no_grad():q=m.head(torch.cat([ctx.expand(5,-1),s.expand(5,-1),b['actions'].cuda()],-1)).flatten().cpu().numpy()*5
 late=b['teacher_q'].numpy()*5;g=b['returns'].numpy()*5;actions=b['actions'].numpy().flatten();pairs=[(i,j) for i,j in itertools.combinations(range(5),2) if abs(g[i]-g[j])>.01 and abs(actions[i]-actions[j])>2e-7]
 z=dict(task=tid,region=d['spec']['region'],trial=ti,index=ix,true_range=float(np.ptp(g)))
 for name,pred in [('early',q),('late',late)]:z[name]=dict(signs=int(sum((pred[j]-pred[i])*(g[j]-g[i])>0 for i,j in pairs)),pairs=len(pairs),regret=float(g.max()-g[pred.argmax()]),contrast_mae=float(np.mean(abs((pred-pred[2])-(g-g[2])))))
 out.append(z)
summary={}
for region in ['all','low','middle','high']:
 rs=[r for r in out if region=='all' or r['region']==region];summary[region]=dict(states=len(rs),true_range=float(np.mean([r['true_range'] for r in rs])))
 for name in ['early','late']:summary[region][name]=dict(signs=sum(r[name]['signs'] for r in rs),pairs=sum(r[name]['pairs'] for r in rs),regret=float(np.mean([r[name]['regret'] for r in rs])),contrast_mae=float(np.mean([r[name]['contrast_mae'] for r in rs])))
write(FITS/'teacher_quality.json',dict(source_sha256=sha(Path(__file__)),summary=summary,queries=out));print(json.dumps(summary,indent=2))
