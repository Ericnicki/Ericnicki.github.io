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
E=O/'evaluation';M=O/'media'
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
def plot_video(spec,d,raw_path):
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
    fig.text(.5,.012,'Joint phase: +6300 Q / +3150 actor / +630 writer updates; fixed base across six trials. No returns enter TTT.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.085,right=.91,bottom=.20,top=.83)
    for ext in ['png','pdf','svg']:fig.savefig(M/f'{region}_value_Y_time.{ext}',dpi=180)
    plt.close(fig)
    with (M/f'{region}_value_Y_time.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    cap=cv2.VideoCapture(str(raw_path));tmp=M/f'{region}_annotated.mp4';w=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,300));assert w.isOpened()
    count=0
    try:
        for ti,r,g in frame_rows:
            ok,frame=cap.read();assert ok
            panel=np.full((300,640,3),245,np.uint8);panel[22:278,:256]=frame
            lines=[f'{region.upper()} friction {spec["friction"]:.5f}',f'Dev task {tid} | trial {ti+1}/6',
                'Final deterministic residual actor',f'Fixed base {d["speed"]:.4f} m/s',
                f't={r["time_after"]-d["trials"][ti]["records"][0]["time_before"]:.1f}s | dV={r["effective_d_V_m_s"]:+.3f} m/s',
                f'Pre-control Q={r["q_raw_return"]:+.3f}',f'Final G={g:+.3f} (retrospective)' if g is not None else 'Final G: censored']
            for j,line in enumerate(lines):cv2.putText(panel,line,(270,34+j*35),cv2.FONT_HERSHEY_SIMPLEX,.49,(35,35,35),1,cv2.LINE_AA)
            w.write(panel);count+=1
        assert not cap.read()[0]
    finally:cap.release();w.release()
    subprocess.run([str(FF),'-y','-i',str(tmp),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(M/f'{region}.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    tmp.unlink();return dict(region=region,task=tid,frames=count,trial_returns=[t.get('terminal_return') for t in d['trials']])
def main():
    assert socket.gethostname()=='92c316f35f5f';torch.set_num_threads(1);t0=time.monotonic();M.mkdir(exist_ok=False)
    audit=json.loads((O/'audit.json').read_text());complete=json.loads((E/'evaluation_complete.json').read_text());assert audit['status']=='passed' and complete['status']=='completed'
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
        if complete['selected_video_tasks'][spec['region']]==tid:media.append(plot_video(spec,armdata['actor_deterministic'],E/f'task{tid:04d}/raw.mp4'))
    frozen_equal(model,parent);gains=np.array([r['gain'] for r in paired]);rng=np.random.default_rng(2007000)
    boot=np.array([rng.choice(gains,len(gains),replace=True).mean() for _ in range(10000)])
    metrics=dict(status='passed',evaluation_tasks=12,trial_slots=144,paired_tasks=paired,
        task_mean_return_gain=float(gains.mean()),task_bootstrap_95CI=np.quantile(boot,[.025,.975]).tolist(),
        arms={k:dict(judged=len(v),censored=72-len(v),successes=sum(x>0 for x in v),mean_return=float(np.mean(v)),phase_metrics=phase_metrics(metric[k])) for k,v in byarm.items()},
        control_interventions=interventions,media=media,max_cache_error=max(errors),Q_updates=0,slow_writer_updates=0,
        limitations='phase labels are geometric/motion proxies; descriptive Q metrics; MC mixed-behavior continuation forecasts; action gradients not certified',seconds=time.monotonic()-t0)
    write(O/'metrics.json',metrics)
    report=f'# Joint online phase v100\n\nCompleted500additional fresh trainingtasks/3000trialslots and6300Q/3150actor/630slow-writer updates. Independent full optimizer replay passed.\n\nMatched held-out task mean return gain: {metrics["task_mean_return_gain"]:+.6f}; task bootstrap95%CI {metrics["task_bootstrap_95CI"]}.\n\n'
    for k,v in metrics['arms'].items():report+=f'{k}: {v["successes"]}/{v["judged"]} successes, {v["censored"]} censored, mean return {v["mean_return"]:+.6f}.\n\n'
    report+='Q uses current state/base velocity and applied residual velocity. TTT uses prior observations/executed actions only. Increasing predicted Q does not by itself demonstrate control improvement. Phase AUROC/MAE and individual trial returns are in metrics.json; three videos/plots are in media/.\n'
    (O/'REPORT.md').write_text(report)
    files=[R/'ttt_joint_v100_fixture.json',O/'manifest.json',O/'training_complete.json',O/'audit.json',O/'metrics.json',O/'joint_final.pt',O/'REPORT.md',O/'updates.jsonl',E/'evaluation_complete.json',E/'summary.json']
    files+=list((O/'source').glob('*'))+list(M.glob('*'))+list(O.glob('group*_networks.pt'))
    files+=list(O.glob('task*/task.pt'))+list(O.glob('task*/initial.pt'))+list(E.glob('task*/*.pt'))
    def arcname(p):return str(p.relative_to(O)) if p.is_relative_to(O) else 'preflight.json'
    archive_index={arcname(p):sha(p) for p in files};write(O/'archive_manifest.json',dict(files=archive_index))
    with tarfile.open(R/'ttt_joint_v100_archive.tar.gz','w:gz') as tar:
        for p in files+[O/'archive_manifest.json']:tar.add(p,arcname=arcname(p),recursive=False)
    write(O/'report_complete.json',dict(status='completed',archive_sha256=sha(R/'ttt_joint_v100_archive.tar.gz'),files=len(archive_index),seconds=time.monotonic()-t0))
    print('ACTOR_ONLY_REPORT_COMPLETE',json.dumps(dict(gain=metrics['task_mean_return_gain'],CI=metrics['task_bootstrap_95CI'],media=media,seconds=time.monotonic()-t0)),flush=True)
if __name__=='__main__':main()
