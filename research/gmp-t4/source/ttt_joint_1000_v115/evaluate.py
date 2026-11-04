import sys
from pathlib import Path

"""Matched held-out BASE/final deterministic actor; record trajectories for three plots."""
import copy,json,socket,time
from pathlib import Path
import cv2,hydra,numpy as np,torch
from common import R,O,write,save,sha,load_networks,frozen_equal
from official import configurations
from memory import Memory
import trial as trial_module
from trial import trial
C=O
E=C/'evaluation'
def main():
    assert socket.gethostname()=='c1d3019784aa';torch.set_num_threads(1);t0=time.monotonic()
    audit=json.loads((C/'audit.json').read_text());assert audit['status']=='passed'
    model,actor,parent=load_networks();final=torch.load(O/'joint_final.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(final['model']);actor.load_state_dict(final['actor']);parent=final;
    for p in model.parameters():p.requires_grad_(False)
    E.mkdir(exist_ok=False)
    prior_rows=json.loads((R/'ttt_joint_completion_v102_output/evaluation/summary.json').read_text())
    old_manifest=json.loads((R/'acc_noise_full_v86_output/manifest.json').read_text())
    ids=json.loads((O/'manifest.json').read_text())['eval_ids'];specs=[s for s in old_manifest['all500_prospective_specs'] if s['task'] in ids]
    rng=np.random.default_rng(2006000);selected={}
    for region in ['low','middle','high']:
        ss=[s for s in specs if s['region']==region];assert len(ss)==4
        cohorts=rng.permutation(['near']*2+['broad']*2)
        for s,c in zip(ss,cohorts):s['evaluation_cohort']=str(c)
        selected[region]=min(s['task'] for s in ss)
    cfg,acfg=configurations();env=hydra.utils.instantiate(cfg.task.env);env.init_simulator()
    original=env.objects[0].set_pose_xyz_wxyz
    def tracked(p):original(p);env._teleported=True;env._near_teleported=True
    env.objects[0].set_pose_xyz_wxyz=tracked
    budget=json.loads((O/'budget.json').read_text());budget['evaluation_starts']=0;rows=[];trial_module.O=C
    try:
        for s in specs:
            tid=s['task'];old=R/f'acc_noise_full_v86_output/task{tid:04d}'
            raw=torch.load(old/'task.pt',map_location='cpu',weights_only=False)
            start=torch.load(old/'initial.pt',map_location='cpu',weights_only=False)
            assert s['partition']=='development' and raw['config']['friction']==s['friction']
            if s['evaluation_cohort']=='near':speed=float(next(p['speed'] for p in raw['programs'] if p['cohort']=='near'))
            else:speed=float(np.random.default_rng(2006100+tid).uniform(.247,.435))
            folder=E/f'task{tid:04d}';folder.mkdir()
            row=dict(spec=s,speed=speed,initial_sha256=sha(old/'initial.pt'),source_task_sha256=sha(old/'task.pt'),arms={})
            prior=next(r for r in prior_rows if r['spec']['task']==tid)
            assert row['spec']==prior['spec'] and row['speed']==prior['speed']
            assert row['initial_sha256']==prior['initial_sha256'] and row['source_task_sha256']==prior['source_task_sha256']
            row['prior_joint_arm']=prior['arms']['actor_deterministic']
            old_arm=R/f'ttt_joint_completion_v102_output/evaluation/task{tid:04d}/actor_deterministic.pt'
            assert sha(old_arm)==row['prior_joint_arm']['sha256']
            row['prior_joint_path']=str(old_arm)

            for mode in ['base','actor_deterministic']:
                mem=Memory(model);trials=[];writer=None
                try:
                    for ti in range(6):
                        identity=dict(task=tid,trial=ti+1,partition='development',actor_steps=15700 if mode!='base' else 0)
                        tr=trial(env,acfg,start,speed,model,actor,mem,np.random.default_rng(2006200+tid),budget,identity,mode=mode,parity=ti==5,writer=writer)
                        trials.append(tr)
                finally:
                    if writer is not None:writer.release()
                result=dict(spec=s,speed=speed,mode=mode,trials=trials,actor_checkpoint_sha256=sha(O/'joint_final.pt'))
                save(folder/f'{mode}.pt',result)
                rr=[t['terminal_return'] for t in trials if t['status']=='completed']
                row['arms'][mode]=dict(judged=len(rr),censored=6-len(rr),successes=sum(g>0 for g in rr),mean_return=float(np.mean(rr)) if rr else None,
                    returns=[t.get('terminal_return') for t in trials],sha256=sha(folder/f'{mode}.pt'))
            rows.append(row);write(E/'summary.json',rows);print('EVAL_TASK',tid,s['region'],s['evaluation_cohort'],json.dumps(row['arms']),flush=True)
        frozen_equal(model,parent);assert budget['evaluation_starts']==144
        write(E/'evaluation_complete.json',dict(status='completed',tasks=12,trial_slots=144,selected_plot_tasks=selected,
            checkpoint_sha256=sha(O/'joint_final.pt'),seconds=time.monotonic()-t0,Q_updates=0,slow_writer_updates=0,actor_updates=0))
    finally:env.physics.free()
if __name__=='__main__':main()
