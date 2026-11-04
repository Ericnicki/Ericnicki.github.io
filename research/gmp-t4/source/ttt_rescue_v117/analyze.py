import rescue_common as C
from rescue_common import *
import json,itertools
def ci(vals):
    x=np.asarray(vals,float);rng=np.random.default_rng(2031000)
    means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return [float(x.mean()),*np.quantile(means,[.025,.975]).tolist()]
def summarize():
    setup();rows=json.loads((O/'summary.json').read_text());probes=json.loads((O/'probes.json').read_text());out={}
    for region in ['all','low','middle','high']:
        rs=[r for r in rows if region=='all' or r['spec']['region']==region];out[region]={}
        if not rs:continue
        for arm in rs[0]['arms']:
            aa=[r['arms'][arm] for r in rs]
            out[region][arm]=dict(mean_return=float(np.mean([a['mean_return'] for a in aa])),successes=sum(a['successes'] for a in aa),judged=sum(a['judged'] for a in aa),
                gain_vs_new=ci([r['arms'][arm]['mean_return']-r['arms']['new']['mean_return'] for r in rs]),
                last3=sum(a.get('last_three_all',False) for a in aa),lost_success=sum(a.get('failure_after_success',False) for a in aa))
    pp=[]
    for p in probes:
        ans=p['answers'][1:];assert all(a['status']=='completed' for a in ans)
        g=np.array([a['return_'] for a in ans]);r=dict(task=p['task'],trial=p['trial'],control=p['control'],region=p['region'],range=float(np.ptp(g)))
        for name in ['early','late']:
            q=np.array([a[name+'_Q'] for a in ans]);pairs=[(i,j) for i,j in itertools.combinations(range(len(ans)),2) if abs(g[i]-g[j])>.01 and abs(ans[i]['dv']-ans[j]['dv'])>1e-7]
            r[name]=dict(signs=int(sum((q[j]-q[i])*(g[j]-g[i])>0 for i,j in pairs)),n=len(pairs),regret=float(g.max()-g[q.argmax()]),contrast_mae=float(np.mean(abs((q-q[2])-(g-g[2])))))
        pp.append(r)
    out['probes']={}
    for region in ['all','low','middle','high']:
        ps=[p for p in pp if region=='all' or p['region']==region]
        if not ps:continue
        out['probes'][region]=dict(states=len(ps),mean_range=float(np.mean([p['range'] for p in ps])))
        for name in ['early','late']:
            n=sum(p[name]['n'] for p in ps);sg=sum(p[name]['signs'] for p in ps)
            out['probes'][region][name]=dict(signs=sg,n=n,agreement=sg/n if n else None,regret=float(np.mean([p[name]['regret'] for p in ps])),contrast_mae=float(np.mean([p[name]['contrast_mae'] for p in ps])))
    write(O/'analysis.json',out);write(O/'probe_metrics.json',pp);print(json.dumps(out,indent=2))
if __name__=='__main__':summarize()
