import rescue_common as C
from rescue_common import *
import argparse,time,json
import trial_rescue as T
from controller import Controller
def main():
    setup();O.mkdir(exist_ok=False);T.O=O
    write(O/'manifest.json',dict(stage='diagnosis',checkpoints=HASHES,sources={p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()}))
    rows=json.loads((R/'ttt_joint_1000_v115_output/evaluation/summary.json').read_text())
    models={name:load(w,a) for name,w,a in [('new','new',None),('old','old',None),('old_actor','new','old'),('old_writer','old','new')]}
    env,acfg=env_setup();budget=dict(evaluation_starts=0,training_starts=0,entries=[],limit=2000);t0=time.monotonic();out=[];probes=[]
    try:
      for row in rows:
        tid=row['spec']['task'];start=torch.load(R/f'acc_noise_full_v86_output/task{tid:04d}/initial.pt',weights_only=False)
        folder=O/f'task{tid:04d}';folder.mkdir();res=dict(spec=row['spec'],speed=row['speed'],arms={})
        originals={}
        for key,path in [('new',R/'ttt_joint_1000_v115_output'),('old',R/'ttt_joint_completion_v102_output')]:
            d=torch.load(path/f'evaluation/task{tid:04d}/actor_deterministic.pt',weights_only=False);originals[key]=d['trials'];res['arms'][key]=brief(d['trials'])
        res['arms']['base']=row['arms']['base']
        for arm in ['old_actor','old_writer','reset','half']:
            m,a=models[arm if arm in models else 'new'];mem=Memory(m);trials=[]
            for ti in range(6):
                if arm=='reset':mem=Memory(m)
                tr=T.trial(env,acfg,start,row['speed'],m,a,mem,np.random.default_rng(2030000),budget,dict(task=tid,trial=ti+1,partition='development'),mode='actor_deterministic',scale=.5 if arm=='half' else 1.)
                trials.append(tr)
            save(folder/f'{arm}.pt',dict(trials=trials));res['arms'][arm]=brief(trials)
        m,a=models['new'];original=originals['new']
        for ti in [0,3]:
            rr=original[ti]['records'];contact=[i for i,r in enumerate(rr) if r['critical'] and r['proxy_phase'] in ['contact_early','contact_late']]
            assert contact
            # Additional other-trial evidence excludes the query trial entirely.
            early=rebuild(m,original[:ti]);teacher=rebuild(m,original[:ti]+original[ti+1:])
            for ix in sorted(set([contact[0],contact[-1]])):
                r=rr[ix];st=torch.as_tensor(r['state'],device='cuda').reshape(1,22)
                em=rebuild(m,original[:ti]);tm=rebuild(m,original[:ti]+original[ti+1:])
                for prior in rr[:ix]:em.observe(prior['event']);tm.observe(prior['event'])
                contexts=[em.context(r['state']),tm.context(r['state'])];answers=[]
                candidates=[r['actor_acc_m_s2'],-.5,-.25,0.,.25,.5]
                for ci,acc in enumerate(candidates):
                    mem=rebuild(m,original[:ti])
                    tr=T.trial(env,acfg,start,row['speed'],m,a,mem,np.random.default_rng(2030000),budget,dict(task=tid,trial=ti+1,partition='development',probe=ix,candidate=ci),mode='actor_deterministic',override=(ix,acc))
                    assert len(tr['records'])>ix
                    for j in range(ix+1):np.testing.assert_allclose(tr['records'][j]['state'],rr[j]['state'],atol=2e-6,rtol=0)
                    if ci==0:
                        assert tr['status']==original[ti]['status'];assert abs(tr['terminal_return']-original[ti]['terminal_return'])<1e-8
                    dv=tr['records'][ix]['effective_d_V_m_s'];act=torch.tensor([[dv/.5]],device='cuda',dtype=torch.float32)
                    with torch.no_grad():qs=[float(m.head(torch.cat([c,st,act],-1)).item()*5) for c in contexts]
                    answers.append(dict(acc=acc,dv=dv,early_Q=qs[0],late_Q=qs[1],return_=tr.get('terminal_return'),status=tr['status'],xy=tr.get('stop_xy')))
                    save(folder/f'branch_t{ti}_i{ix}_c{ci}.pt',tr)
                probes.append(dict(task=tid,trial=ti,control=ix,region=row['spec']['region'],phase=r['proxy_phase'],answers=answers))
                write(O/'probes.json',probes)
        out.append(res);write(O/'summary.json',out);print('DIAG_TASK',tid,json.dumps(res['arms']),time.monotonic()-t0,flush=True)
      write(O/'complete.json',dict(status='completed',seconds=time.monotonic()-t0,tasks=len(out),probes=len(probes),starts=budget['evaluation_starts']))
    finally:env.physics.free()
if __name__=='__main__':main()
