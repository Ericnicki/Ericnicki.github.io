import sys
from pathlib import Path

"""Independent evaluation checks, scientific metrics/plots, annotated media and artifact bundle."""
import copy,csv,json,socket,subprocess,tarfile,time
from pathlib import Path
import cv2,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import R,O,sha,write,load_networks,frozen_equal
from controller import Controller,A_MAX
from official import configurations,agent_for,reward
from dataset import features,PHASES
from memory import Memory
C=O
E=C/'evaluation';M=C/'media'
FF=Path('/public/home/2023141520023/miniconda3/envs/qplanning/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
def auc(labels,scores):
    labels=np.asarray(labels,bool);scores=np.asarray(scores,float);n1=int(labels.sum());n0=len(labels)-n1
    if not n1 or not n0:return None
    order=np.argsort(scores,kind='stable');ranks=np.empty(len(scores));i=0
    while i<len(scores):
        j=i+1
        while j<len(scores) and scores[order[j]]==scores[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2.;i=j
    return float((ranks[labels].sum()-n1*(n1+1)/2)/(n1*n0))
def phase_metrics(rows):
    out={}
    for p in ['all']+PHASES:
        a=[r for r in rows if p=='all' or r['phase']==p];y=np.array([r['return'] for r in a]);q=np.array([r['q'] for r in a])
        out[p]=dict(n=len(a),positives=int((y>0).sum()),auroc=auc(y>0,q) if len(a) else None,return_mae=float(abs(y-q).mean()) if len(a) else None)
    return out
def plot_trace(spec,d):
    region=spec['region'];tid=spec['task'];fig,ax=plt.subplots(figsize=(13.8,6.2));aq=ax.twinx();cursor=0.;csvrows=[];frame_rows=[]
    for ti,t in enumerate(d['trials']):
        rr=t['records'];g=t.get('terminal_return');xx=np.array([r['time_before']-rr[0]['time_before'] for r in rr])+cursor
        end=cursor+rr[-1]['time_after']-rr[0]['time_before'];q=np.array([r['q_raw_return'] for r in rr])
        ax.plot(xx,[r['before']['robots_obs'][0]['tcp_xyz_wxyz'][1] for r in rr],color='#1768c4',lw=1.8,label='Observed EE Y' if ti==0 else None)
        ax.plot(xx,[r['before']['env_objs_obs'][1]['object_pose_xyz_wxyz'][1] for r in rr],color='#149277',lw=1.8,label='Observed cube Y' if ti==0 else None)
        aq.plot(xx,q,color='#e67818',lw=1.8,label='Final TTT Q(s, executed residual)' if ti==0 else None)
        if g is not None:aq.hlines(g,cursor,end,color='#666666',linestyles='--',lw=1.2,label='Actual final return G' if ti==0 else None)
        if ti:ax.axvline(cursor,color='#a9adb4',ls=':',lw=1.2)
        text=f'T{ti+1}: G={g:+.3f}' if g is not None else f'T{ti+1}: censored'
        ax.text((cursor+end)/2,1.025,text,ha='center',va='bottom',transform=ax.get_xaxis_transform(),fontsize=10)
        for i,r in enumerate(rr):
            out=dict(task=tid,trial=ti+1,control=i,concatenated_time_s=float(xx[i]),trial_time_s=float(xx[i]-cursor),
                ee_y_m=float(r['before']['robots_obs'][0]['tcp_xyz_wxyz'][1]),cube_y_m=float(r['before']['env_objs_obs'][1]['object_pose_xyz_wxyz'][1]),
                q_raw_return=float(q[i]),effective_d_V_m_s=r['effective_d_V_m_s'],actual_final_return=g,phase=r['proxy_phase'] or '',prior_trial_count=ti)
            csvrows.append(out);frame_rows.append((ti,r,g))
        cursor=end
    ax.set_xlabel('Concatenated trial time (s); dotted lines mark physical resets');ax.set_ylabel('Observed Y (m)',color='#1768c4')
    aq.set_ylabel('Predicted / actual final return (raw units)',color='#e67818');ax.grid(alpha=.18);ax.set_xlim(0,cursor)
    fig.suptitle(f'{region.capitalize()} friction | development task {tid} | friction={spec["friction"]:.5f}',y=.98)
    hh,ll=ax.get_legend_handles_labels();h2,l2=aq.get_legend_handles_labels();fig.legend(hh+h2,ll+l2,loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,frameon=False)
    fig.text(.5,.012,'Continuation: +12500 Q / +6250 actor / +1250 writer updates; fixed base across six trials. No returns enter TTT.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.085,right=.91,bottom=.20,top=.83)
    for ext in ['png','pdf','svg']:fig.savefig(M/f'{region}_value_Y_time.{ext}',dpi=180)
    plt.close(fig)
    with (M/f'{region}_value_Y_time.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    return dict(region=region,task=tid,controls=len(frame_rows),trial_returns=[t.get('terminal_return') for t in d['trials']])
def main():
    assert socket.gethostname()=='c1d3019784aa';torch.set_num_threads(1);t0=time.monotonic();M.mkdir(exist_ok=False)
    audit=json.loads((C/'audit.json').read_text());complete=json.loads((E/'evaluation_complete.json').read_text());assert audit['status']=='passed' and complete['status']=='completed'
    rows=json.loads((E/'summary.json').read_text());model,actor,parent=load_networks()
    final=torch.load(O/'joint_final.pt',map_location='cpu',weights_only=False);model.load_state_dict(final['model']);actor.load_state_dict(final['actor']);parent=final
    for p in model.parameters():p.requires_grad_(False)
    _,acfg=configurations();metric={'base':[],'actor_deterministic':[]};paired=[];byarm={'base':[],'actor_deterministic':[]};interventions={};media=[];errors=[]
    for row in rows:
        spec=row['spec'];tid=spec['task'];armdata={}
        for mode in ['base','actor_deterministic']:
            path=E/f'task{tid:04d}/{mode}.pt';assert sha(path)==row['arms'][mode]['sha256']
            d=torch.load(path,map_location='cpu',weights_only=False);assert d['speed']==row['speed'];armdata[mode]=d
            mem=Memory(model);bound_counts={'controls':0,'acc_saturation':0,'velocity_saturation':0,'envelope_interventions':0}
            for ti,t in enumerate(d['trials']):
                rr=t['records'];states,events,phases=features(rr);c=Controller();agent=agent_for(acfg,[row['speed']]*6);previous=[]
                for i,r in enumerate(rr):
                    np.testing.assert_array_equal(states[i],r['state']);np.testing.assert_array_equal(events[i],r['event']);assert r['noise_m_s2']==0.
                    if i in {0,len(rr)-1}:
                        ctx=mem.reference(r['state']);err=float(np.max(abs(ctx.cpu().numpy()[0]-r['context'])));errors.append(err);assert err<3e-5
                    mem.observe(r['event']);st=torch.as_tensor(r['state'],device='cuda').reshape(1,22);ctx=torch.as_tensor(r['context'],device='cuda').reshape(1,32)
                    with torch.no_grad():
                        acc=float((A_MAX*torch.tanh(actor.net(torch.cat([ctx,st],dim=-1)))).item()) if r['critical'] and mode!='base' else 0.
                        pred=float(model.head(torch.cat([ctx,st,torch.tensor([[r['effective_d_V_m_s']/.5]],device='cuda')],dim=-1)).item()*5)
                    assert acc==r['actor_acc_m_s2'] and abs(pred-r['q_raw_return'])<1e-6
                    calc=c.step(r['base_y_m'],r['base_previous_y_m'],r['previous_executed_y_m'],acc,r['critical'],r['active'])
                    for k,v in calc.items():assert abs(v-r[k])<1e-12
                    nominal=agent.predict_actions([r['before']['robots_obs']],[r['before']['env_objs_obs']],[previous])[0]
                    for a,b in zip(nominal,r['nominal_command']):np.testing.assert_array_equal(a['tcp_xyz_wxyz'],b['tcp_xyz_wxyz'])
                    previous=copy.deepcopy(r['executed_command']);assert abs(previous[0]['tcp_xyz_wxyz'][1]-r['command_y_m'])<1e-10
                    if mode=='base':assert r['effective_d_V_m_s']==0.
                    for key,rkey in [('acc_saturation','acceleration_saturation'),('velocity_saturation','velocity_saturation'),('envelope_interventions','constraint_intervention')]:bound_counts[key]+=bool(r[rkey])
                    bound_counts['controls']+=1
                mem.end_trial()
                if t['status']=='completed':
                    g=reward(t['stop_xy']);assert g==t['terminal_return'] and t['official_success']==(g>0);byarm[mode].append(g)
                    for phase in PHASES:
                        pool=[i for i,p in enumerate(phases) if p==phase]
                        for ix in sorted({pool[0],pool[len(pool)//2]} if pool else set()):metric[mode].append(dict(task=tid,trial=ti+1,phase=phase,**{'return':g},q=rr[ix]['q_raw_return']))
                else:assert 'terminal_return' not in t
            interventions[f'{tid}_{mode}']=bound_counts
        pairs=[(a['terminal_return'],b['terminal_return']) for a,b in zip(armdata['base']['trials'],armdata['actor_deterministic']['trials']) if a['status']==b['status']=='completed']
        if pairs:
            a,b=np.array(pairs).T;paired.append(dict(task=tid,region=spec['region'],cohort=spec['evaluation_cohort'],matched_trials=len(pairs),base_mean=float(a.mean()),actor_mean=float(b.mean()),gain=float((b-a).mean())))
        if complete['selected_plot_tasks'][spec['region']]==tid:media.append(plot_trace(spec,armdata['actor_deterministic']))
    frozen_equal(model,parent);gains=np.array([r['gain'] for r in paired]);rng=np.random.default_rng(2027000)
    boot=np.array([rng.choice(gains,len(gains),replace=True).mean() for _ in range(10000)])
    metrics=dict(status='passed',evaluation_tasks=12,trial_slots=144,paired_tasks=paired,
        task_mean_return_gain=float(gains.mean()),task_bootstrap_95CI=np.quantile(boot,[.025,.975]).tolist(),
        arms={k:dict(judged=len(v),censored=72-len(v),successes=sum(x>0 for x in v),mean_return=float(np.mean(v)),phase_metrics=phase_metrics(metric[k])) for k,v in byarm.items()},
        control_interventions=interventions,media=media,max_cache_error=max(errors),Q_updates=0,slow_writer_updates=0,
        limitations='phase labels are geometric/motion proxies; descriptive Q metrics; MC mixed-behavior continuation forecasts; action gradients not certified',seconds=time.monotonic()-t0)
    # Paired comparison with the previous audited joint actor on matched tasks.
    prior_values=[];previous_comparison=[]
    for row in rows:
        path=Path(row['prior_joint_path']);assert sha(path)==row['prior_joint_arm']['sha256']
        prior=torch.load(path,map_location='cpu',weights_only=False)
        current=torch.load(E/f"task{row['spec']['task']:04d}/actor_deterministic.pt",map_location='cpu',weights_only=False)
        assert prior['spec']==current['spec'] and prior['speed']==current['speed']
        pairs=[(a['terminal_return'],b['terminal_return']) for a,b in zip(prior['trials'],current['trials']) if a['status']==b['status']=='completed']
        prior_values.extend(t['terminal_return'] for t in prior['trials'] if t['status']=='completed')
        if pairs:
            a,b=np.array(pairs).T
            previous_comparison.append(dict(task=row['spec']['task'],region=row['spec']['region'],cohort=row['spec']['evaluation_cohort'],matched_trials=len(pairs),prior_mean=float(a.mean()),new_mean=float(b.mean()),gain=float((b-a).mean()),success_gain=float(((b>0)-(a>0).astype(int)).mean())))
    gains_previous=np.array([r['gain'] for r in previous_comparison]);rng=np.random.default_rng(2027001)
    previous_boot=np.array([rng.choice(gains_previous,len(gains_previous),replace=True).mean() for _ in range(10000)])
    metrics['previous_joint']=dict(judged=len(prior_values),successes=sum(g>0 for g in prior_values),mean_return=float(np.mean(prior_values)),
        task_comparisons=previous_comparison,task_mean_gain=float(gains_previous.mean()),task_bootstrap_95CI=np.quantile(previous_boot,[.025,.975]).tolist(),data_reused=True)
    training=json.loads((O/'summary.json').read_text());training_returns=[];training_task_rows=[]
    for row in training:
        d=torch.load(O/f"task{row['task']:04d}/task.pt",map_location='cpu',weights_only=False)
        vals=[t['terminal_return'] for t in d['trials'] if t['status']=='completed'];training_returns.extend(vals)
        training_task_rows.append(dict(task=row['task'],region=row['region'],cohort=row['cohort'],judged=len(vals),successes=sum(g>0 for g in vals),mean_return=float(np.mean(vals)) if vals else None))
    metrics['training']=dict(task_count=1000,trial_slots=6000,judged=len(training_returns),censored=6000-len(training_returns),successes=sum(g>0 for g in training_returns),
        success_rate=sum(g>0 for g in training_returns)/len(training_returns),mean_return=float(np.mean(training_returns)),tasks=training_task_rows,
        limitation='exploratory changing behavior throughout fitting; not deterministic final-policy evaluation')
    for value in metrics['arms'].values():value['success_rate']=value['successes']/value['judged'] if value['judged'] else None
    metrics['cohorts']={cohort:dict(tasks=[p for p in paired if p['cohort']==cohort],mean_task_gain=float(np.mean([p['gain'] for p in paired if p['cohort']==cohort]))) for cohort in ['near','broad']}
    write(C/'metrics.json',metrics)
    report=f'# Joint continuation v115, recovered from v109\n\nCompleted1000additional fresh trainingtasks/6000trialslots and12500Q/6250actor/1250slow-writer updates. Independent full optimizer replay passed.\n\nMatched reused-development task mean return gain: {metrics["task_mean_return_gain"]:+.6f}; task bootstrap95%CI {metrics["task_bootstrap_95CI"]}.\n\n'
    for k,v in metrics['arms'].items():report+=f'{k}: {v["successes"]}/{v["judged"]} successes, {v["censored"]} censored, mean return {v["mean_return"]:+.6f}.\n\n'
    report+=f'Prior joint actor: {metrics["previous_joint"]["successes"]}/{metrics["previous_joint"]["judged"]} successes, meanG {metrics["previous_joint"]["mean_return"]:+.6f}; new-minus-prior taskgain {metrics["previous_joint"]["task_mean_gain"]:+.6f},95%CI {metrics["previous_joint"]["task_bootstrap_95CI"]}.\n\n'
    report+='Q uses current state/base velocity and applied residual velocity. TTT uses prior observations/executed actions only. Increasing predicted Q does not by itself demonstrate control improvement. Phase AUROC/MAE and individual trial returns are in metrics.json; three plots are in media/.\n'
    (C/'REPORT.md').write_text(report)
    files=[R/'ttt_joint_1000_v115_fixture.json',O/'manifest.json',O/'training_complete.json',C/'audit.json',C/'metrics.json',O/'joint_final.pt',C/'REPORT.md',O/'updates.jsonl',E/'evaluation_complete.json',E/'summary.json']
    files+=list((O/'source').glob('*'))+list(M.glob('*'))+list(O.glob('group*_networks.pt'))
    files+=[p for p in (O/'recovery').rglob('*') if p.is_file()]
    files+=list(O.glob('task*/task.pt'))+list(O.glob('task*/initial.pt'))+list(E.glob('task*/*.pt'))
    def arcname(p):return str(p.relative_to(O)) if p.is_relative_to(O) else str(p.relative_to(C)) if p.is_relative_to(C) else 'preflight.json'
    archive_index={arcname(p):sha(p) for p in files};write(C/'archive_manifest.json',dict(files=archive_index))
    with tarfile.open(R/'ttt_joint_1000_v115_archive.tar.gz','w:gz') as tar:
        for p in files+[C/'archive_manifest.json']:tar.add(p,arcname=arcname(p),recursive=False)
    write(C/'report_complete.json',dict(status='completed',archive_sha256=sha(R/'ttt_joint_1000_v115_archive.tar.gz'),files=len(archive_index),seconds=time.monotonic()-t0))
    print('JOINT_REPORT_COMPLETE',json.dumps(dict(gain=metrics['task_mean_return_gain'],CI=metrics['task_bootstrap_95CI'],media=media,seconds=time.monotonic()-t0)),flush=True)
if __name__=='__main__':main()
