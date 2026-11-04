import sys
from pathlib import Path
R=Path('/public/home/2023141520023/return_selector_20261006')
sys.path.insert(1,str(R/'ttt_rescue_v117'))
from rescue_common import *
import json,time
import trial_rescue as T
O=R/'ttt_rescue_train_v118_output'
def main():
    setup();O.mkdir(exist_ok=False);T.O=O
    allspec=json.loads((R/'ttt_joint_1000_v115_output/manifest.json').read_text())['all1000_specs'];rng=np.random.default_rng(2032000);specs=[]
    for region in ['low','middle','high']:
      for cohort in ['near','broad']:
        pool=[s for s in allspec if s['region']==region and s['cohort']==cohort]
        specs.extend(pool[int(i)] for i in rng.choice(len(pool),8,replace=False))
    write(O/'manifest.json',dict(specs=specs,checkpoint=HASHES['new'],source_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),training_tasks=48,selection='8 per region/cohort, seed2032000; original TRAIN only',query='median critical contact, trials1 and4',candidates=[-.5,-.25,0,.25,.5]))
    m,a=load();env,acfg=env_setup();budget=dict(evaluation_starts=0,training_starts=0,entries=[],limit=1500);rows=[];t0=time.monotonic()
    try:
      for spec in specs:
        tid=spec['task'];folder=O/f'task{tid:04d}';folder.mkdir()
        source=R/f'ttt_joint_1000_v115_output/task{tid:04d}';start=torch.load(source/'initial.pt',map_location='cpu',weights_only=False);speed=json.loads((source/'program.json').read_text())['speed']
        mem=Memory(m);trials=[]
        for ti in range(6):trials.append(T.trial(env,acfg,start,speed,m,a,mem,rng,budget,dict(task=tid,trial=ti+1,partition='train'),mode='actor_deterministic'))
        save(folder/'task.pt',dict(spec=spec,speed=speed,trials=trials));queries=[]
        for ti in [0,3]:
            rr=trials[ti]['records'];ids=[i for i,r in enumerate(rr) if r['critical'] and r['proxy_phase'] in ['contact_early','contact_late']]
            if not ids:ids=[i for i,r in enumerate(rr) if r['critical']]
            assert ids;ix=ids[len(ids)//2];answers=[]
            for ci,acc in enumerate([rr[ix]['actor_acc_m_s2'],-.5,-.25,0.,.25,.5]):
                mem=rebuild(m,trials[:ti]);tr=T.trial(env,acfg,start,speed,m,a,mem,rng,budget,dict(task=tid,trial=ti+1,partition='train',probe=ix,candidate=ci),mode='actor_deterministic',override=(ix,acc))
                for j in range(ix+1):np.testing.assert_allclose(tr['records'][j]['state'],rr[j]['state'],atol=2e-6,rtol=0)
                if ci==0:
                    assert tr['status']==trials[ti]['status']
                    if tr['status']=='completed':assert abs(tr['terminal_return']-trials[ti]['terminal_return'])<1e-8
                answers.append(dict(acc=acc,dv=tr['records'][ix]['effective_d_V_m_s'],return_=tr.get('terminal_return'),status=tr['status']))
                save(folder/f'branch_t{ti}_c{ci}.pt',tr)
            queries.append(dict(task=tid,trial=ti,control=ix,answers=answers))
        save(folder/'queries.pt',queries);rows.append(dict(spec=spec,speed=speed,queries=queries,baseline=brief(trials)));write(O/'summary.json',rows)
        print('TRAIN_BRANCH_TASK',tid,len(rows),time.monotonic()-t0,flush=True)
      write(O/'complete.json',dict(status='completed',tasks=len(rows),starts=budget['training_starts'],seconds=time.monotonic()-t0))
    finally:env.physics.free()
if __name__=='__main__':main()
