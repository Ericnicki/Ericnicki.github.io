"""Reconstruct the unsaved optimizer tail; preserve every completed v109 task."""
import copy,json,random,shutil,socket,time
from pathlib import Path
import numpy as np
import torch
from common import R,O,P_SHA,sha,write,save,sources,load_networks
from train import specs,replay_task,sample,tensor_hash,optimizers,update_q,update_actor

OLD=R/'ttt_joint_1000_v109_output'


def assert_networks(model,actor,path,q):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['Q_updates']==q and ck['actor_updates']==q//2 and ck['writer_updates']==q//10
    for module,key in [(model,'model'),(actor,'actor')]:
        assert all(torch.equal(v.detach().cpu(),ck[key][k]) for k,v in module.state_dict().items()),(q,key)


def main():
    assert socket.gethostname()=='c1d3019784aa';torch.set_num_threads(1);t0=time.monotonic()
    assert not O.exists(),'Recovery output already exists; inspect it instead of overwriting'
    source=sources();fixture=json.loads((R/'ttt_joint_1000_v115_fixture.json').read_text())
    assert fixture['status']=='passed' and fixture['source_sha256']==source
    assert len(fixture['trial_budget_boundary_checks'])==5
    manifest=json.loads((OLD/'manifest.json').read_text())
    assert manifest['all1000_specs']==specs() and manifest['parent_checkpoint_sha256']==P_SHA
    for p,h in manifest['source_sha256'].items():assert sha(p)==h,('old_source_changed',p)
    failed=json.loads((OLD/'failed.json').read_text())
    assert failed['stage']=='collecting' and failed['task']==specs()[500]
    assert (failed['completed_tasks'],failed['Q_updates'],failed['actor_updates'],failed['writer_updates'])==(500,6200,3100,620)
    assert failed['error']=='AssertionError()' and not (OLD/'training_complete.json').exists()
    budget=json.loads((OLD/'budget.json').read_text())
    assert budget['training_starts']==3000 and budget['evaluation_starts']==0
    assert len([e for e in budget['entries'] if e.get('partition')=='train'])==3000
    rows=json.loads((OLD/'summary.json').read_text());assert len(rows)==500
    assert [r['task'] for r in rows]==list(range(500))
    lines=(OLD/'updates.jsonl').read_text().splitlines();log=[json.loads(line) for line in lines]
    assert len(log)==6200 and [r['Q_updates'] for r in log]==list(range(1,6201))
    receipt=json.loads((OLD/'checkpoint_receipt.json').read_text())
    assert receipt['Q_updates']==6000 and receipt['tasks']==480 and sha(OLD/'rolling.pt')==receipt['sha256']
    rolling=torch.load(OLD/'rolling.pt',map_location='cpu',weights_only=False)
    assert rolling['Q_updates']==6000 and rolling['completed_groups']==60 and rolling['rows']==rows[:480]
    assert rolling['source_sha256']==manifest['source_sha256']
    model,actor,parent=load_networks();head,writer,aopt=optimizers(model,actor,parent)
    model.load_state_dict(rolling['model']);actor.load_state_dict(rolling['actor'])
    for opt,key,count in [(head,'head_optimizer',82300),(writer,'writer_optimizer',13230),(aopt,'actor_optimizer',12450)]:
        opt.load_state_dict(rolling[key]);assert all(int(v['step'])==count for v in opt.state.values())
    assert_networks(model,actor,OLD/'group60_networks.pt',6000)
    rng=np.random.default_rng();rng.bit_generator.state=rolling['numpy_rng']
    random.setstate(rolling['python_rng']);torch.set_rng_state(rolling['torch_rng']);torch.cuda.set_rng_state_all(rolling['cuda_rng'])
    replay={};raw_paths={};group_hash={}
    for gi in range(63):
        ck=torch.load(OLD/f'group{gi:02d}_networks.pt',map_location='cpu',weights_only=False)
        assert ck['Q_updates']==100*gi
        group_hash[gi]=(ck['model_tensor_sha256'],ck['actor_tensor_sha256'])
    for row in rows:
        tid=row['task'];path=OLD/f'task{tid:04d}/task.pt';assert sha(path)==row['task_sha256']
        d=torch.load(path,map_location='cpu',weights_only=False)
        assert d['spec']==specs()[tid] and len(d['trials'])==6
        assert (d['model_tensor_sha256'],d['actor_tensor_sha256'])==group_hash[tid//8]
        assert d['Q_updates']==(tid//8)*100 and d['actor_updates']==(tid//8)*50 and d['writer_updates']==(tid//8)*10
        if tid<496:replay[tid]=replay_task(d)
        raw_paths[str(tid)]=dict(task_sha256=sha(path),initial_sha256=sha(path.with_name('initial.pt')))
    # Only tasks available at each original update enter the sampler/causal prefix.
    # This is optimizer-state reconstruction, not additional production training.
    for row in log[6000:]:
        q=row['Q_updates'];gi=(q-1)//100;available=(gi+1)*8
        assert row['group']==gi and row['available_tasks']==available
        if q%100==1:assert_networks(model,actor,OLD/f'group{gi:02d}_networks.pt',q-1)
        live={tid:d for tid,d in replay.items() if tid<available}
        queries=sample(live,rng);assert [list(x) for x in queries]==row['queries']
        slow=q%10==0;assert row['writer_updated']==slow
        metrics=update_q(model,actor,live,queries,head,writer,slow)
        assert abs(metrics['Q_loss']-row['Q_loss'])<1e-7,(q,'Q_loss')
        if q%2==0:
            queries=sample(live,rng,True);assert [list(x) for x in queries]==row['actor_queries']
            metrics=update_actor(model,actor,live,queries,aopt)
            assert abs(metrics['actor_loss']-row['actor_loss'])<1e-7,(q,'actor_loss')
        else:assert row['actor_queries'] is None
        assert row['actor_updates']==q//2 and row['writer_updates']==q//10
        if q%100==0:assert_networks(model,actor,OLD/f'group{gi+1:02d}_networks.pt',q)
    assert_networks(model,actor,OLD/'group62_networks.pt',6200)
    for opt,count in [(head,82500),(writer,13250),(aopt,12550)]:
        assert all(int(v['step'])==count for v in opt.state.values())
    # Do not create a resumable output until all reconstruction checks have passed.
    O.mkdir();(O/'source').mkdir();backups=O/'recovery';backups.mkdir()
    for i,p in enumerate(source):shutil.copy2(p,O/'source'/f'{i:02d}_{Path(p).name}')
    for name in ['manifest.json','failed.json','pipeline_failed.json','budget.json','progress.json','summary.json','checkpoint_receipt.json']:
        shutil.copy2(OLD/name,backups/f'v109_{name}')
    shutil.copytree(OLD/'source',backups/'v109_source')
    shutil.copy2(OLD/'rolling.pt',backups/'v109_rolling_6000.pt')
    shutil.copy2(R/'ttt_joint_1000_v109.log',backups/'v109_failed.log')
    for row in rows:
        name=f'task{row["task"]:04d}';shutil.copytree(OLD/name,O/name)
        assert sha(O/name/'task.pt')==row['task_sha256']
    for gi in range(63):shutil.copy2(OLD/f'group{gi:02d}_networks.pt',O/f'group{gi:02d}_networks.pt')
    prelude=OLD/'task0500';assert (prelude/'initial.pt').exists() and (prelude/'program.json').exists()
    assert not any(prelude.glob('trial*.pt')) and not (prelude/'task.pt').exists()
    shutil.copytree(prelude,backups/'task0500_rejected_attempt')
    new_manifest=copy.deepcopy(manifest);new_manifest.update(source_sha256=source,
        inherited_source_sha256=manifest['source_sha256'],inherited_manifest_sha256=sha(OLD/'manifest.json'),
        recovery_from=str(OLD),recovered_completed_tasks=500,reconstructed_updates=200,
        training_trial_limit=6000,original_rejected_critical_requests=1)
    write(O/'manifest.json',new_manifest)
    budget['rejected_training_critical_requests']=1
    write(O/'budget.json',budget);write(O/'summary.json',rows)
    (O/'updates.jsonl').write_text('\n'.join(lines)+'\n')
    payload=dict(model=model.state_dict(),actor=actor.state_dict(),head_optimizer=head.state_dict(),writer_optimizer=writer.state_dict(),actor_optimizer=aopt.state_dict(),
        Q_steps=82500,writer_steps=13250,actor_steps=12550,Q_updates=6200,actor_updates=3100,slow_writer_updates=620,completed_groups=62,rows=rows[:496],
        replay_index=[dict(task=r['task'],path=f'task{r["task"]:04d}/task.pt',sha256=r['task_sha256']) for r in rows[:496]],
        numpy_rng=rng.bit_generator.state,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
        parent_checkpoint_sha256=P_SHA,source_sha256=source,action='effective residual velocity / .5',gamma=1.,target_scale=5.,actor_to_writer_gradient=False)
    save(O/'rolling.pt',payload);shutil.copy2(O/'rolling.pt',backups/'recovered_start_6200.pt')
    write(O/'checkpoint_receipt.json',dict(Q_updates=6200,actor_updates=3100,writer_updates=620,tasks=496,sha256=sha(O/'rolling.pt')))
    write(backups/'receipt.json',dict(status='passed',source_sha256=source,old_output=str(OLD),old_checkpoint_sha256=receipt['sha256'],
        old_update_log_sha256=sha(OLD/'updates.jsonl'),old_manifest_sha256=sha(OLD/'manifest.json'),
        recovered_checkpoint_sha256=sha(O/'rolling.pt'),replayed_Q_updates=200,replayed_actor_updates=100,replayed_writer_updates=20,
        exact_group60_61_62_tensors=True,recorded_losses_and_samples_match=True,optimizer_steps_verified=True,
        preserved_tasks=500,optimizer_replay_tasks=496,partial_group_tasks=4,raw_hashes=raw_paths,
        discarded_task_data=0,new_physics_starts=0,new_production_updates=0,
        rejected_critical_requests=1,rejected_request_executed_critical_commands=0,seconds=time.monotonic()-t0))
    print('RECOVERY_PASSED',json.dumps(dict(tasks_preserved=500,Q_updates=6200,remaining_tasks=500)),flush=True)


if __name__=='__main__':main()
