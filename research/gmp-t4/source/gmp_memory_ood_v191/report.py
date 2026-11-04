import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"gmp_memory_ood_v187"))
"""Matched OOD statistics and scientific plots; execute only on compute node."""
import csv,html,base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
METHODS=['gmp','swap','joint','prediction_only','adaptive','aligned']
LABELS=dict(gmp='Official GMP + memory',swap='Swap',joint='Joint',prediction_only='Prediction only',adaptive='Adaptive',aligned='Aligned')
def ci(x,rng):
 x=np.asarray(x,float)
 if not len(x):return None
 return dict(mean=float(x.mean()),CI95=np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975]).tolist(),tasks=len(x))
def metric(a):
 ok=np.isfinite(a);n=int(a.size);j=int(ok.sum());s=int((a>0).sum());return dict(attempted=n,judged=j,censors=n-j,successes=s,success_rate=s/n,judged_success_rate=s/j if j else None,mean_G=float(a[ok].mean()) if j else None)
def main():
 setup(2187500);assert json.loads((O/'audit.json').read_text())['status']==json.loads((O/'independent_review.json').read_text())['status']=='passed';rows=json.loads((O/'summary.json').read_text());parents=json.loads((B/'summary.json').read_text());manifest=json.loads((O/'manifest.json').read_text());assert sha(B/'summary.json')==manifest['parent_summary_sha256'];P=O/'plots';P.mkdir();arrays={'gmp':np.asarray([[np.nan if g is None else g for g in r['returns']] for r in rows])};arrays.update({m:np.asarray([r['arms'][m]['returns'] for r in parents],float) for m in METHODS[1:]});assert all(a.shape==(32,12) for a in arrays.values());groups={'low':np.arange(16),'high':np.arange(16,32),'overall':np.arange(32)};stats={};paired={};rng=np.random.default_rng(2187500)
 for band,ix in groups.items():
  stats[band]={};paired[band]={}
  for method,a in arrays.items():
   x=a[ix];stats[band][method]=dict(primary=metric(x[:,1:]),all12=metric(x),by_trial=[metric(x[:,t]) for t in range(12)],segments={label:metric(x[:,lo:hi]) for label,lo,hi in [('T2-6',1,6),('T7-12',6,12)]})
  valid=np.isfinite(arrays['gmp'][ix,1:]).all(1)
  for method in METHODS[1:]:
   g=arrays['gmp'][ix,1:];b=arrays[method][ix,1:];dg=(g[valid]-b[valid]).mean(1)
   # Attempted success treats censored slots as no observed success, preserving explicit censor counts.
   ds=((g>0).astype(float)-(b>0).astype(float)).mean(1)
   paired[band]['gmp-'+method]=dict(return_gain_complete_tasks=ci(dg,rng),excluded_tasks=int((~valid).sum()),attempted_success_gain=ci(ds,rng),interval='Task cluster percentile bootstrap, 10000 draws; unadjusted')
 with (O/'native_outcomes.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['task','band','friction','method','trial','native_G','observed_success','judged'])
  for i,r in enumerate(rows):
   for m,a in arrays.items():
    for t,g in enumerate(a[i]):w.writerow([i,r['spec']['region'],r['spec']['friction'],m,t+1,float(g) if np.isfinite(g) else '',int(g>0),int(np.isfinite(g))])
 memory=[];panels=[];bindings={};plotted=0
 with (O/'memory_decisions.csv').open('w',newline='') as memory_file:
  memory_writer=csv.writer(memory_file);memory_writer.writerow(['task','band','trial','decision','control','history_before','history_after','gate','history_attention_skipped'])
  for i,r in enumerate(rows):
   path=O/f'task{i:04d}/gmp.pt';raw=torch.load(path,map_location='cpu',weights_only=False);bindings[str(path.relative_to(O))]=sha(path)
   for t,tr in enumerate(raw['trials']):
    for d in tr['decisions']:
     memory_writer.writerow([i,r['spec']['region'],t+1,d['control']//8,d['control'],d['history_before']['action_chunks'],d['history_after']['action_chunks'],d['gate'],int(d['history_attention_skipped'])]);memory.append(dict(task=i,trial=t+1,band=r['spec']['region'],gate=d['gate'],skip=d['history_attention_skipped'],chunks=d['history_before']['action_chunks']))
   if i not in [0,1,16,17]:continue
   fig,axes=plt.subplots(3,4,figsize=(17,10),sharey=True);stem=f'gmp_task{i:04d}_trajectories';export=[]
   for t,(tr,ax) in enumerate(zip(raw['trials'],axes.flat)):
    rr=tr['records'];tt=np.asarray([x['time_before']-rr[0]['time_before'] for x in rr]);ee=np.asarray([x['before']['robots_obs'][0]['tcp_xyz_wxyz'][1] for x in rr]);cube=np.asarray([x['before']['env_objs_obs'][1]['object_pose_xyz_wxyz'][1] for x in rr]);g=tr.get('terminal_return');ax.plot(tt,ee,color='#2467bb',label='End effector Y');ax.plot(tt,cube,color='#23964c',label='Cube Y');ax.axhspan(.10,.20,color='#23964c',alpha=.08);ax.axhline(.15,color='gray',ls=':',lw=.7);ax.set_title(f'Trial {t+1}: G={g:+.3f}' if g is not None else f'Trial {t+1}: censored',fontsize=10);ax.grid(alpha=.18);ax.set_xlabel('Time (s)');ax.set_ylabel('Y (m)')
    for j,x in enumerate(rr):export.append([t+1,j,float(tt[j]),float(ee[j]),float(cube[j]),g if g is not None else '',tr['status']]);plotted+=1
   axes[0,0].legend(fontsize=8);fig.suptitle(f'Official GMP + memory | {r["spec"]["region"]} OOD task {i} | friction={r["spec"]["friction"]:.6f}\nSame physical initial state each trial; policy history retained. Green band: target Y tolerance.');fig.tight_layout()
   for ext in ['png','pdf','svg']:fig.savefig(P/f'{stem}.{ext}',dpi=155)
   plt.close(fig)
   with (P/f'{stem}.csv').open('w',newline='') as curve_file:
    w=csv.writer(curve_file);w.writerow(['trial','control','time_s','ee_y_m','cube_y_m','native_G','status']);w.writerows(export)
   checked=list(csv.DictReader((P/f'{stem}.csv').open()));assert len(checked)==len(export)
   for a,b in zip(export,checked):assert int(b['trial'])==a[0] and int(b['control'])==a[1] and all(float(b[k])==a[j] for j,k in [(2,'time_s'),(3,'ee_y_m'),(4,'cube_y_m')]) and (b['native_G']=='' if a[5]=='' else float(b['native_G'])==a[5])
   panels.append(stem)
 fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True)
 for col,band in enumerate(['low','high']):
  for m in METHODS:
   d=stats[band][m]['by_trial'];kw=dict(marker='o',lw=3 if m=='gmp' else 1.4,alpha=1 if m=='gmp' else .7,label=LABELS[m]);axes[0,col].plot(range(1,13),[100*x['success_rate'] for x in d],**kw);axes[1,col].plot(range(1,13),[np.nan if x['mean_G'] is None else x['mean_G'] for x in d],**kw)
  axes[0,col].set(title=f'{band.capitalize()} friction OOD',ylabel='Observed success / attempted (%)',ylim=(-3,103));axes[1,col].set(xlabel='Trial',ylabel='Mean native G of judged slots');axes[0,col].legend(fontsize=8)
 for ax in axes.flat:ax.set_xticks(range(1,13));ax.axvspan(.6,1.4,color='gray',alpha=.12);ax.grid(alpha=.2)
 fig.suptitle('Matched OOD: official GMP with memory versus retained five-method results');fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(P/f'ood_comparison.{ext}',dpi=160)
 plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True);memstats={}
 for col,band in enumerate(['low','high']):
  selected=[x for x in memory if x['band']==band];memstats[band]=dict(decisions=len(selected),history_attention_enabled=sum(not x['skip'] for x in selected),history_nonempty=sum(x['chunks']>0 for x in selected),history_used_with_nonempty=sum(x['chunks']>0 and not x['skip'] for x in selected))
  for t in range(1,13):
   z=[x for x in selected if x['trial']==t];axes[0,col].plot(t,100*np.mean([not x['skip'] for x in z]),'o',color='#8047ae');axes[1,col].plot(t,np.mean([x['chunks'] for x in z]),'o',color='#2467bb')
  axes[0,col].set(title=f'{band.capitalize()} OOD',ylabel='Decisions with history attention (%)',ylim=(-3,103));axes[1,col].set(ylabel='Mean retained history chunks',xlabel='Trial',ylim=(-1,31))
 for ax in axes.flat:ax.set_xticks(range(1,13));ax.grid(alpha=.2)
 fig.suptitle('Official memory diagnostics: retention and learned gate decisions');fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(P/f'memory_diagnostics.{ext}',dpi=160)
 plt.close(fig)
 result=dict(status='completed',primary='T2–12',tasks=32,new_policy_slots=384,ID_tasks=0,optimizer_updates=0,checkpoint_sha256=CK_SHA,stats=stats,paired=paired,memory=memstats,censors=int((~np.isfinite(arrays['gmp'])).sum()),scope='Official pretrained gated diffusion policy; learned memory retained over repeated matched physical resets; previous five methods reused without simulation',limitations=['One diffusion seed per task; this comparison does not isolate memory causality','GMP uses original absolute action decoder; previous methods use residual controllers','Censored G is missing; complete-task return intervals exclude tasks with any primary censor','Unadjusted task bootstrap; overall pooling uses task resampling rather than stratification']);write(O/'result.json',result)
 page='<!doctype html><html><head><meta charset="utf-8"><title>Official GMP memory OOD</title><style>body{font-family:system-ui;max-width:1500px;margin:auto;padding:24px;background:#fafafa}img{width:100%}td,th{padding:9px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style></head><body><h1>Official GMP + memory: OOD evaluation</h1><p>32 matched tasks. Low [0.0025,0.005), high (0.015,0.0225]. Twelve trials, frozen weights, no training or ID evaluation. Primary: trials2–12. Policy history persists across physical resets; learned gate retains its original behavior.</p>'
 for band in ['low','high','overall']:
  page+=f'<h2>{band.capitalize()}</h2><table><tr><th>Method</th><th>Success / attempted</th><th>Censored</th><th>Mean native G (judged)</th></tr>'
  for m in METHODS:
   d=stats[band][m]['primary'];g='Missing' if d['mean_G'] is None else f'{d["mean_G"]:+.3f}';page+=f'<tr><td>{LABELS[m]}</td><td>{d["successes"]}/{d["attempted"]} ({100*d["success_rate"]:.1f}%)</td><td>{d["censors"]}</td><td>{g}</td></tr>'
  page+='</table>'
 for stem in ['ood_comparison','memory_diagnostics',*panels]:page+='<h2>'+html.escape(stem.replace('_',' '))+'</h2><img src="data:image/png;base64,'+base64.b64encode((P/f'{stem}.png').read_bytes()).decode()+'">'
 page+='<details><summary>Full metrics, paired intervals, and limitations</summary><pre>'+html.escape(json.dumps(result,indent=2))+'</pre></details></body></html>'; (O/'index.html').write_text(page)
 write(O/'plot_audit.json',dict(status='passed',trajectory_graphs=4,aggregate_graphs=2,trajectory_controls_checked=plotted,native_labels_and_CSV_exact=True,raw_bindings=bindings,selection=[0,1,16,17],sources=sources(),supplement_sources={q.name:sha(q) for q in Path(__file__).resolve().parent.iterdir() if q.suffix in ('.py','.md','.sh')}));write(O/'status.json',dict(status='completed',stage='all_audits_and_plots_complete'))
if __name__=='__main__':main()
