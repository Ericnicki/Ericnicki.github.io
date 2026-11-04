"""OOD-only native statistics, task-paired intervals and audited classic plots."""
import csv,base64,html,itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from collect import ARMS,LABELS
from plot_style import plot_function
def bootstrap(d,rng,bands=None):
 d=np.asarray(d,float)
 if bands is None:b=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
 else:b=sum(d[ix][rng.integers(0,len(ix),(10000,len(ix)))].mean(1) for ix in bands)/len(bands)
 return dict(mean=float(d.mean()),CI95=np.quantile(b,[.025,.975]).tolist(),tasks=len(d))
def main():
 setup(2185500);audit=json.loads((O/'audit.json').read_text());review=json.loads((O/'independent_review.json').read_text());assert audit['status']==review['status']=='passed';rows=json.loads((O/'summary.json').read_text());assert len(rows)==32;M=O/'plots';M.mkdir();arrays={arm:np.asarray([r['arms'][arm]['returns'] for r in rows],float) for arm in ARMS};assert all(a.shape==(32,12) and np.isfinite(a).all() for a in arrays.values());ixs={r:[i for i,row in enumerate(rows) if row['spec']['region']==r] for r in ['low','high']};assert all(len(ix)==16 for ix in ixs.values());stats={};paired={};rng=np.random.default_rng(2185500)
 for name,ix in [('overall',list(range(32))),*ixs.items()]:
  stats[name]={};paired[name]={}
  for arm,a in arrays.items():
   x=a[ix];stats[name][arm]=dict(tasks=len(ix),judged_primary=len(ix)*11,successes=int((x[:,1:]>0).sum()),success_rate=float((x[:,1:]>0).mean()),mean_G=float(x[:,1:].mean()),trial1_success_rate=float((x[:,0]>0).mean()),trial1_mean_G=float(x[:,0].mean()),all12_success_rate=float((x>0).mean()),all12_mean_G=float(x.mean()),by_trial=dict(success_rate=(x>0).mean(0).tolist(),mean_G=x.mean(0).tolist()),history_segments={label:dict(success_rate=float((x[:,lo:hi]>0).mean()),mean_G=float(x[:,lo:hi].mean())) for label,lo,hi in [('T2-6',1,6),('T7-12',6,12)]})
  for a,b in itertools.combinations(ARMS,2):
   dg=(arrays[a][ix,1:]-arrays[b][ix,1:]).mean(1);ds=((arrays[a][ix,1:]>0).astype(float)-(arrays[b][ix,1:]>0).astype(float)).mean(1);bands=list(ixs.values()) if name=='overall' else None;paired[name][a+'-'+b]=dict(return_gain=bootstrap(dg,rng,bands),success_rate_gain=bootstrap(ds,rng,bands))
 with (O/'native_outcomes.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['task','band','friction','speed','method','trial','native_G','success']);writer.writeheader()
  for row in rows:
   for arm in ARMS:
    for ti,g in enumerate(row['arms'][arm]['returns']):writer.writerow(dict(task=row['spec']['task'],band=row['spec']['region'],friction=row['spec']['friction'],speed=row['speed'],method=arm,trial=ti+1,native_G=g,success=int(g>0)))
 plot,binding=plot_function(M);(O/'reused_plot_trace.py').write_text(binding.pop('adapted')+'\n');panels=[];bindings={};controls=0
 for arm in ARMS:
  for tid in [0,1,16,17]:
   row=rows[tid];path=O/f'task{tid:04d}/{arm}.pt';raw=torch.load(path,map_location='cpu',weights_only=False);bindings[str(path.relative_to(O))]=sha(path);spec=dict(row['spec'],region=f'{arm}_task{tid:04d}',method_label=LABELS[arm],region_label=row['spec']['region']+' OOD');plot(spec,raw);stem=f'{spec["region"]}_value_Y_time';export=list(csv.DictReader((M/f'{stem}.csv').open()));j=0;cursor=0.
   for ti,tr in enumerate(raw['trials']):
    rr=tr['records'];start=rr[0]['time_before']
    for k,r in enumerate(rr):
     e=export[j];j+=1;assert int(e['trial'])==ti+1 and int(e['control'])==k;assert float(e['concatenated_time_s'])==float(r['time_before']-start+cursor);assert float(e['ee_y_m'])==float(r['before']['robots_obs'][0]['tcp_xyz_wxyz'][1]);assert float(e['cube_y_m'])==float(r['before']['env_objs_obs'][1]['object_pose_xyz_wxyz'][1]);assert float(e['q_raw_return'])==r['q_raw_return'];assert float(e['actual_final_return'])==tr['terminal_return']
    cursor+=rr[-1]['time_after']-start
   assert j==len(export);controls+=j;panels.append(dict(arm=arm,task=tid,region=row['spec']['region'],stem=stem))
 f,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
 for col,band in enumerate(['low','high']):
  for arm in ARMS:
   d=stats[band][arm]['by_trial'];axes[0,col].plot(range(1,13),100*np.asarray(d['success_rate']),marker='o',label=LABELS[arm]);axes[1,col].plot(range(1,13),d['mean_G'],marker='o',label=LABELS[arm])
  axes[0,col].set(title=f'{band.capitalize()} OOD: '+('[0.0025, 0.005)' if band=='low' else '(0.015, 0.0225]'),ylabel='Native success (%)',ylim=(-3,103));axes[1,col].set(xlabel='Physical trial',ylabel='Mean native G');axes[0,col].legend(fontsize=8)
 for ax in axes.flat:ax.axvspan(.6,1.4,color='grey',alpha=.15);ax.axvline(6.5,color='grey',linestyle=':');ax.set_xticks(range(1,13));ax.grid(alpha=.2)
 f.suptitle('OOD-only: 32 fresh matched tasks / 12 trials / five frozen policies');f.tight_layout()
 for ext in ['png','pdf','svg']:f.savefig(M/f'ood_native_by_trial.{ext}',dpi=160)
 plt.close(f)
 result=dict(status='completed',primary='T2–12',tasks=32,methods=ARMS,unique_native_slots=1920,censors=0,ID_tasks=0,optimizer_updates=0,global_weights_frozen=True,stats=stats,paired=paired,bootstrap=dict(resamples=10000,seed=2185500,unit='matched task with all trial histories',overall='stratified resampling inside low/high bands, fixed equal weights',multiple_comparison_correction=False),source_binding=binding,audit_sha256=sha(O/'audit.json'),review_sha256=sha(O/'independent_review.json'),scope='Fresh OOD-only evaluation of fixed top four endpoints plus original swap; no training or checkpoint selection');write(O/'result.json',result)
 page='<!doctype html><html><head><meta charset="utf-8"><title>OOD top four plus swap</title><style>body{font-family:system-ui;max-width:1450px;margin:auto;padding:24px;background:#fafafa}img{max-width:100%}td,th{padding:8px 18px;border-bottom:1px solid #ddd}select{padding:8px;margin:8px}article{background:white;padding:12px;margin:12px 0}pre{white-space:pre-wrap}</style></head><body><h1>OOD-only: top four methods plus original swap</h1><p>Low [0.0025,0.005), high (0.015,0.0225]. 32 fresh matched broad tasks,12 trials. No training or in-distribution tasks. Fixed global weights; normal real-observation task-private memories persist within task. Primary metrics exclude trial1. Orange unchanged anchor Q is diagnostic for the correction methods.</p>'
 for band in ['low','high','overall']:
  page+='<h2>'+band.capitalize()+'</h2><table><tr><th>Method</th><th>Success / judged</th><th>Mean G</th></tr>'
  for arm in ARMS:
   d=stats[band][arm];page+=f'<tr><td>{LABELS[arm]}</td><td>{d["successes"]}/{d["judged_primary"]} ({100*d["success_rate"]:.1f}%)</td><td>{d["mean_G"]:+.3f}</td></tr>'
  page+='</table>'
 page+='<h2>Native outcomes by trial</h2><img src="data:image/png;base64,'+base64.b64encode((M/'ood_native_by_trial.png').read_bytes()).decode()+'"><h2>Classic trajectories</h2><p>First two tasks per OOD band, selected before outcomes. All12native final returns marked.</p><select id="arm"><option value="all">All methods</option>'+''.join(f'<option value="{a}">{LABELS[a]}</option>' for a in ARMS)+'</select><select id="region"><option value="all">Both OOD bands</option><option>low</option><option>high</option></select>'
 for panel in panels:page+=f'<article data-arm="{panel["arm"]}" data-region="{panel["region"]}"><img alt="{html.escape(LABELS[panel["arm"]])} OOD task {panel["task"]}" src="data:image/svg+xml;base64,'+base64.b64encode((M/(panel['stem']+'.svg')).read_bytes()).decode()+'"></article>'
 page+='<details><summary>Task-paired intervals and full metrics</summary><pre>'+html.escape(json.dumps(result,indent=2))+'</pre></details><script>const selects=["arm","region"].map(id=>document.getElementById(id));function filter(){for(const el of document.querySelectorAll("article")){el.hidden=selects.some(s=>s.value!=="all"&&el.dataset[s.id]!==s.value)}}for(const s of selects)s.addEventListener("change",filter);</script></body></html>';assert page.count('<article ')==20;(O/'index.html').write_text(page)
 write(O/'plot_audit.json',dict(status='passed',graphs=20,exported_controls_checked=controls,all_native_G_and_trial_boundaries_checked=True,raw_bindings=bindings,selection=[0,1,16,17],parent_source=binding,sources=sources(),physics_starts=0,optimizer_updates=0));write(O/'status.json',dict(status='completed',stage='native_outcomes_and_exports_audited'));write(O/'file_manifest.json',{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='file_manifest.json'})
if __name__=='__main__':main()
