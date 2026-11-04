import sys
from pathlib import Path
R=Path('/public/home/2023141520023/return_selector_20261006');sys.path.insert(1,str(R/'ttt_rescue_v117'))
from rescue_common import *
import json,collections,itertools,time
from dataset import features
D=R/'ttt_rescue_train_v118_output';FITS=R/'ttt_rescue_fit_v119_output';CONF=R/'ttt_rescue_confirm_v120_output';OUT=R/'ttt_rescue_review_v121_output'
def check_trial(t):
    rr=t['records'];states,events,ph=features(rr)
    np.testing.assert_allclose(states,np.array([r['state'] for r in rr]),atol=2e-6,rtol=0)
    np.testing.assert_array_equal(events,np.array([r['event'] for r in rr]))
    for r in rr:
        c=r['executed_command'][0]['tcp_xyz_wxyz'];assert abs(float(c[1])-r['command_y_m'])<1e-10
        if r['active']:
            dv=(r['command_y_m']-r['previous_executed_y_m']-r['base_delta_y_m'])/.1
            assert abs(dv-r['effective_d_V_m_s'])<1e-10
        assert abs(r['residual_velocity_after_m_s'])<=.1000000001
        assert abs(r['applied_d_acc_m_s2'])<=.500000001
        assert r['noise_m_s2']==0.
    if t['status']=='completed':
        r=rr[-1];assert r['terminal'] and r['judge_after']>r['judge_before'];xy=np.asarray(r['after']['env_objs_obs'][1]['object_pose_xyz_wxyz'][:2]);np.testing.assert_array_equal(xy,t['stop_xy'])
        g=1-float(np.max(abs(xy-[.1,.15])))/.05;assert abs(g-t['terminal_return'])<1e-12 and (g>0)==t['official_success']
    return len(rr)
def bootstrap(vals,rows):
    x=np.asarray(vals,float);rng=np.random.default_rng(2037000);draw=[]
    for region in ['low','middle','high']:
      for cohort in ['near','broad']:
        ids=np.array([i for i,r in enumerate(rows) if r['spec']['region']==region and r['spec']['cohort']==cohort])
        if len(ids):draw.append(ids[rng.integers(0,len(ids),(10000,len(ids)))])
    means=x[np.concatenate(draw,1)].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist())
def summarize(rows):
    out={};arms=list(rows[0]['arms'])
    for group in ['all','low','middle','high','near','broad']:
        rs=[r for r in rows if group=='all' or r['spec']['region']==group or r['spec']['cohort']==group];out[group]={}
        for arm in arms:
            aa=[r['arms'][arm] for r in rs];gs=np.array([a['mean_return'] for a in aa]);base=np.array([r['arms']['base']['mean_return'] for r in rs]);new=np.array([r['arms']['new']['mean_return'] for r in rs]);old=np.array([r['arms']['old']['mean_return'] for r in rs])
            out[group][arm]=dict(mean_return=float(gs.mean()),successes=sum(a['successes'] for a in aa),judged=sum(a['judged'] for a in aa),slots=6*len(rs),last3_all=sum(a['last_three_all'] for a in aa),lost_success=sum(a['failure_after_success'] for a in aa),first_trial=float(np.mean([a['returns'][0] for a in aa])),last3_return=float(np.mean([g for a in aa for g in a['returns'][-3:] if g is not None])),
                gain_new=bootstrap(gs-new,rs),gain_old=bootstrap(gs-old,rs),gain_base=bootstrap(gs-base,rs),damage_tasks=int(sum(gs<base)),worst_task_gain_base=float((gs-base).min()),p10_gain_base=float(np.quantile(gs-base,.1)))
            if arm.startswith(('hindsight','grounded')):
                seed=arm.split('_')[-1];mc=np.array([r['arms']['mc_'+seed]['mean_return'] for r in rs]);out[group][arm]['gain_matched_mc']=bootstrap(gs-mc,rs)
    return out

def main():
    setup();assert json.loads((CONF/'complete.json').read_text())['status']=='completed';OUT.mkdir(exist_ok=False);counts=collections.Counter();t0=time.monotonic();hashes={}
    for root in [O,D,CONF]:
      for p in sorted(root.glob('task*/*.pt')):
        obj=torch.load(p,map_location='cpu',weights_only=False);hashes[str(p)]=sha(p)
        if p.name=='initial.pt':continue
        if p.name=='queries.pt':continue
        trials=obj['trials'] if isinstance(obj,dict) and 'trials' in obj else [obj]
        for t in trials:
            counts['trials']+=1;counts['controls']+=check_trial(t);counts[t['status']]+=1
    # Exact branch prefix/state and factual continuation outcome checks, independently loaded.
    dp=json.loads((O/'probes.json').read_text())
    for q in dp:
      tid,ti,ix=q['task'],q['trial'],q['control'];original=torch.load(R/f'ttt_joint_1000_v115_output/evaluation/task{tid:04d}/actor_deterministic.pt',weights_only=False)['trials'][ti]
      for ci,answer in enumerate(q['answers']):
        tr=torch.load(O/f'task{tid:04d}/branch_t{ti}_i{ix}_c{ci}.pt',weights_only=False)
        for j in range(ix):
          np.testing.assert_array_equal(tr['records'][j]['event'],original['records'][j]['event'])
        np.testing.assert_array_equal(tr['records'][ix]['state'],original['records'][ix]['state'])
        assert tr.get('terminal_return')==answer['return_']
        if ci==0:assert tr['terminal_return']==original['terminal_return']
    trainrows=json.loads((D/'summary.json').read_text());trainids={r['spec']['task'] for r in trainrows};assert len(trainids)==48
    for row in trainrows:
      tid=row['spec']['task'];assert row['spec']['partition']=='train';d=torch.load(D/f'task{tid:04d}/task.pt',weights_only=False)
      for q in row['queries']:
       ti,ix=q['trial'],q['control'];orig=d['trials'][ti]
       for ci,answer in enumerate(q['answers']):
        tr=torch.load(D/f'task{tid:04d}/branch_t{ti}_c{ci}.pt',weights_only=False)
        for j in range(ix):np.testing.assert_array_equal(tr['records'][j]['event'],orig['records'][j]['event'])
        np.testing.assert_array_equal(tr['records'][ix]['state'],orig['records'][ix]['state']);assert tr.get('terminal_return')==answer['return_']
        if ci==0:assert tr.get('terminal_return')==orig.get('terminal_return')
    fitchecks=[]
    for seed in [2033000,2033001]:
      schedules=[]
      for arm in ['mc','hindsight','grounded']:
        folder=FITS/f'{arm}_{seed}';p=folder/'step1500.pt';ck=torch.load(p,map_location='cpu',weights_only=False);hashes[str(p)]=sha(p)
        logs=[json.loads(s) for s in (folder/'updates.jsonl').read_text().splitlines()];assert len(logs)==1500
        for i,l in enumerate(logs,1):
          assert l['step']==i and l['writer']==(i%10==0);assert len({q[0] for q in l['queries']})==8
          assert all(q[0] in trainids for q in l['queries']+l['aux_queries']);assert np.isfinite(l['mc']) and np.isfinite(l['aux'])
        schedules.append([(l['queries'],l['aux_queries']) for l in logs])
        for key,expected in [('head_optimizer',90300),('writer_optimizer',14030),('actor_optimizer',16450)]:assert all(int(v['step'])==expected for v in ck[key]['state'].values())
        assert ck['parent']==HASHES['new'];fitchecks.append(dict(arm=arm,seed=seed,sha256=sha(p),Q=1500,writer=150,actor=750,first100_mc=float(np.mean([l['mc'] for l in logs[:100]])),last100_mc=float(np.mean([l['mc'] for l in logs[-100:]])),first100_aux=float(np.mean([l['aux'] for l in logs[:100]])),last100_aux=float(np.mean([l['aux'] for l in logs[-100:]]))))
      assert schedules[0]==schedules[1]==schedules[2]
    rows=json.loads((CONF/'summary.json').read_text());assert len(rows)==60
    for row in rows:
      for arm,reported in row['arms'].items():
        tr=torch.load(CONF/f'task{row["spec"]["task"]:04d}/{arm}.pt',weights_only=False)['trials'];assert brief(tr)==reported
    results=summarize(rows);write(OUT/'metrics.json',results);write(OUT/'fit_checks.json',fitchecks);write(OUT/'data_hashes.json',hashes)
    write(OUT/'audit.json',dict(status='passed',counts=dict(counts),same_fit_query_schedules=True,optimizer_counts_verified=True,exact_branch_prefixes=True,exact_factual_returns=True,native_outcomes=True,features_and_executed_actions=True,fit_full_optimizer_replay=False,seconds=time.monotonic()-t0,source_sha256=sha(Path(__file__))))
    print(json.dumps(results['all'],indent=2))
if __name__=='__main__':main()
