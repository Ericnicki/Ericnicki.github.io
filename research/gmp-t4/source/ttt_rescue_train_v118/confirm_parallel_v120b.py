import sys
from pathlib import Path
R=Path('/public/home/2023141520023/return_selector_20261006');sys.path.insert(1,str(R/'ttt_rescue_v117'))
from rescue_common import *
import json,time,hydra
import trial_rescue as T
import near
from official import configurations,snap
from train import tensor_hash
FITS=R/'ttt_rescue_fit_v119_output';O=R/'ttt_rescue_confirm_v120_output'
def worker_main(worker):
    setup();assert json.loads((FITS/'complete.json').read_text())['status']=='completed';WO=O/f'worker{worker}';WO.mkdir(exist_ok=False);T.O=WO;near.O=WO
    rng=np.random.default_rng(2035000);specs=[]
    for ri,region in enumerate(['low','middle','high']):
      for cohort in ['near','broad']:
       for j in range(10):
        tid=len(specs);specs.append(dict(task=tid,namespace='rescue_confirm_v120',partition='confirmation',region=region,cohort=cohort,friction=float(rng.uniform(.005+ri*.01/3,.005+(ri+1)*.01/3)),seed=2035100+tid,speed_seed=2035200+tid))
    specs=[s for s in specs if s['task']%4==worker]
    models={'old':load('old'),'new':load('new'),'old_actor':load('new','old'),'half':load('new')}
    hashes={k:HASHES[k] for k in ['old','new']};hashes.update(old_actor={'writer':HASHES['new'],'actor':HASHES['old']},half=HASHES['new'])
    for seed in [2033000,2033001]:
      for arm in ['mc','hindsight','grounded']:
        key=f'{arm}_{seed}';p=FITS/key/'step1500.pt';m,a=load();ck=torch.load(p,map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);a.load_state_dict(ck['actor']);models[key]=(m,a);hashes[key]=sha(p)
    write(WO/'manifest.json',dict(specs=specs,checkpoints=hashes,source_sha256=sha(Path(__file__)),arms=['base']+list(models),fit_manifest_sha256=sha(FITS/'manifest.json'),selection='all fixed endpoints; no model selection on fresh outcomes'))
    cfg,acfg=configurations();env=hydra.utils.instantiate(cfg.task.env);env.init_simulator();original=env.objects[0].set_pose_xyz_wxyz
    def tracked(p):original(p);env._teleported=True;env._near_teleported=True
    env.objects[0].set_pose_xyz_wxyz=tracked
    budget=dict(evaluation_starts=0,training_starts=0,near_starts=0,entries=[],limit=5000);rows=[];t0=time.monotonic()
    mh={k:(tensor_hash(m),tensor_hash(a)) for k,(m,a) in models.items()}
    try:
      for spec in specs:
        tid=spec['task'];folder=O/f'task{tid:04d}';folder.mkdir();env.reset(dict(seed=spec['seed'],episode_idx=spec['seed'],sliding_friction=spec['friction']));start=snap(env);start['geom_friction']=env.physics.model.geom_friction.copy();save(folder/'initial.pt',start)
        speed=float(near.calibrate(env,acfg,cfg,spec,start,budget,folder)['speed']) if spec['cohort']=='near' else float(np.random.default_rng(spec['speed_seed']).uniform(.247,.435))
        row=dict(spec=spec,speed=speed,arms={})
        for arm in ['base']+list(models):
            m,a=models['new' if arm=='base' else arm];mem=Memory(m);trials=[]
            for ti in range(6):trials.append(T.trial(env,acfg,start,speed,m,a,mem,rng,budget,dict(task=tid,trial=ti+1,partition='confirmation',arm=arm),mode='base' if arm=='base' else 'actor_deterministic',scale=.5 if arm=='half' else 1.))
            save(folder/f'{arm}.pt',dict(trials=trials));row['arms'][arm]=brief(trials)
            key='new' if arm=='base' else arm;assert (tensor_hash(m),tensor_hash(a))==mh[key]
        rows.append(row);write(WO/'summary.json',rows);print('CONFIRM_TASK',tid,len(rows),time.monotonic()-t0,flush=True)
      write(WO/'complete.json',dict(status='completed',tasks=len(rows),starts=budget['evaluation_starts'],calibrations=budget['near_starts'],seconds=time.monotonic()-t0,slow_weights_unchanged=True))
    finally:env.physics.free()
def main():
    import argparse,subprocess,os
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);args=parser.parse_args()
    if args.worker is not None:return worker_main(args.worker)
    # Model fitting runs as a separate owned process; no optimizer interruption.
    started=time.monotonic()
    while not (FITS/'complete.json').exists():
        assert time.monotonic()-started<3600,'fit wait exceeded one hour'
        try:os.kill(1355,0)
        except ProcessLookupError:raise RuntimeError('fit process absent before completion')
        time.sleep(10)
    O.mkdir(exist_ok=False);jobs=[]
    write(O/'parallel_owner.json',dict(pid=os.getpid(),workers=4,source_sha256=sha(Path(__file__)),preserved_sequential_source_sha256=sha(Path(__file__).with_name('confirm.py'))))
    for i in range(4):
        log=(O/f'worker{i}.log').open('w');proc=subprocess.Popen([sys.executable,str(Path(__file__)), '--worker',str(i)],stdout=log,stderr=subprocess.STDOUT);jobs.append((proc,log))
    codes=[]
    for proc,log in jobs:codes.append(proc.wait());log.close()
    write(O/'worker_exit_codes.json',codes);assert codes==[0]*4,codes
    rows=[];specs=[];receipts=[]
    for i in range(4):
        wd=O/f'worker{i}';rows+=json.loads((wd/'summary.json').read_text());man=json.loads((wd/'manifest.json').read_text());specs+=man['specs'];receipts.append(json.loads((wd/'complete.json').read_text()))
    rows.sort(key=lambda r:r['spec']['task']);specs.sort(key=lambda s:s['task']);assert [s['task'] for s in specs]==list(range(60))
    write(O/'summary.json',rows);write(O/'manifest.json',dict(specs=specs,checkpoints=man['checkpoints'],source_sha256=sha(Path(__file__)),arms=man['arms'],fit_manifest_sha256=sha(FITS/'manifest.json'),workers=4,partition='task ID modulo4, independent task histories',selection='all fixed endpoints; no selection on confirmation'))
    write(O/'complete.json',dict(status='completed',tasks=len(rows),starts=sum(r['starts'] for r in receipts),calibrations=sum(r['calibrations'] for r in receipts),seconds=time.monotonic()-started,slow_weights_unchanged=all(r['slow_weights_unchanged'] for r in receipts),worker_exit_codes=codes))
if __name__=='__main__':main()
