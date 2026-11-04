"""Fresh-task joint MC fitting: 10 Q / 5 actor / 1 slow writer; causal prefix replay."""
import argparse,copy,hashlib,json,os,random,shutil,signal,socket,time
from pathlib import Path
import hydra,numpy as np,torch
from torch.nn import functional as F
from common import R,O,P,P_SHA,CRITIC_PARENT,write,save,sha,sources,load_networks
from official import configurations,snap
from near import calibrate
from memory import Memory
from trial import trial
from controller import actor_action
from dataset import PHASES,batch as history_batch

def tensor_hash(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def specs():
    rng=np.random.default_rng(2021000);cohorts=[]
    for _ in range(125):cohorts.extend(rng.permutation(['near']*4+['broad']*4).tolist())
    out=[]
    for i,(f,c) in enumerate(zip(rng.uniform(.005,.015,1000),cohorts)):
        region='low' if f<.005+.01/3 else 'middle' if f<.005+2*.01/3 else 'high'
        out.append(dict(task=i,namespace='joint_1000_v109',partition='train',friction=float(f),region=region,cohort=c,
            seed=2022000+i,speed_seed=2023000+i,noise_seed=2024000+i))
    assert len(out)==1000 and sum(s['cohort']=='near' for s in out)==500
    return out
def replay_task(d):
    trials=[]
    for t in d['trials']:
        rr=t['records'];judged=t['status']=='completed'
        active=[i for i,r in enumerate(rr) if judged and r['critical'] and r['active']]
        eligible=list(range(len(rr))) if judged else []
        pools={p:[i for i in eligible if rr[i]['proxy_phase']==p] for p in PHASES}
        actor_pools={p:[i for i in active if rr[i]['proxy_phase']==p] for p in PHASES}
        trials.append(dict(states=np.asarray([r['state'] for r in rr],np.float32),events=np.asarray([r['event'] for r in rr],np.float32),
            old=np.asarray([r['residual_velocity_before_m_s'] for r in rr],np.float32),
            previous=np.asarray([r['previous_executed_y_m'] for r in rr],np.float32),delta=np.asarray([r['base_delta_y_m'] for r in rr],np.float32),
            eligible=eligible,active=active,pools=pools,actor_pools=actor_pools,status=t['status'],**{'return':t.get('terminal_return')}))
    assert any(t['active'] for t in trials),('no_judged_active_queries',d['spec'])
    return dict(cohort=d['spec']['cohort'],trials=trials)
def sample(replay,rng,actor=False):
    ids=[]
    for cohort in ['near','broad']:
        pool=[tid for tid,d in replay.items() if d['cohort']==cohort];assert len(pool)>=4
        ids.extend(int(v) for v in rng.choice(pool,4,replace=False))
    rng.shuffle(ids);assert len(set(ids))==8
    result=[]
    for tid in ids:
        trials=replay[tid]['trials'];field='active' if actor else 'eligible';pf='actor_pools' if actor else 'pools'
        ti=int(rng.choice([i for i,t in enumerate(trials) if t[field]]));t=trials[ti]
        phases=[p for p in PHASES if t[pf][p]];phase=str(rng.choice(phases)) if phases else 'unclassified'
        ix=int(rng.choice(t[pf][phase] if phases else t[field]));result.append((tid,ti,ix,phase))
    return result
def batch(replay,queries):
    return history_batch(replay,[q[:3] for q in queries],True,'cuda')
def controls(replay,queries):
    return [torch.as_tensor(np.asarray([replay[tid]['trials'][ti][field][ix] for tid,ti,ix,_ in queries]),device='cuda',dtype=torch.float32)
        for field in ['old','previous','delta']]
def optimizers(model,actor,parent):
    assert sha(P)==P_SHA
    head=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=1e-4)
    writer=torch.optim.AdamW(model.writer_parameters(),lr=1e-5,weight_decay=1e-4)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)
    for opt,key,count in [(head,'head_optimizer',76300),(writer,'writer_optimizer',12630),(actor_opt,'actor_optimizer',9450)]:
        opt.load_state_dict(parent[key]);assert all(int(v['step'])==count for v in opt.state.values())
    head.param_groups[0]['lr']=1e-4;writer.param_groups[0]['lr']=1e-5;actor_opt.param_groups[0]['lr']=1e-5
    return head,writer,actor_opt

def update_q(model,actor,replay,queries,head,writer,slow):
    head.zero_grad(set_to_none=True);writer.zero_grad(set_to_none=True);actor.zero_grad(set_to_none=True)
    h,m,s,a,y=batch(replay,queries)
    # Stored collection contexts are deliberately ignored; regenerate with current weights.
    pred=model(h,m,s,a,writer_grad=slow);loss=F.smooth_l1_loss(pred,y)
    assert torch.isfinite(loss);loss.backward()
    hn=float(torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.,error_if_nonfinite=True));wn=None
    assert all(p.grad is None for p in actor.parameters())
    if slow:
        wn=float(torch.nn.utils.clip_grad_norm_(model.writer_parameters(),1.,error_if_nonfinite=True));writer.step()
    else:assert all(p.grad is None for p in model.writer_parameters())
    head.step();model.zero_grad(set_to_none=True)
    return dict(Q_loss=float(loss.detach()),Q_raw_return_MAE=float((pred.detach()-y).abs().mean()*5),head_grad_norm=hn,writer_grad_norm=wn,context_events=int(m.sum()))
def update_actor(model,actor,replay,queries,opt):
    model.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
    for p in model.parameters():p.requires_grad_(False)
    try:
        h,m,s,_,_=batch(replay,queries)
        with torch.no_grad():context=model.context(h,m,s)
        old,prev,delta=controls(replay,queries)
        dv,acc,new=actor_action(actor,context.detach(),s,old,prev,delta)
        pred=model.head(torch.cat([context.detach(),s,(dv/.5).unsqueeze(-1)],dim=-1)).squeeze(-1)
        loss=-pred.mean();assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None for p in model.parameters())
        norm=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True));opt.step()
        return dict(actor_loss=float(loss.detach()),actor_candidate_Q_raw=float(pred.detach().mean()*5),actor_grad_norm=norm,actor_to_writer_gradient=False)
    finally:
        for p in model.parameters():p.requires_grad_(True)
        actor.zero_grad(set_to_none=True)

def collect_task(env,cfg,acfg,spec,model,actor,budget,qsteps,asteps,wsteps):
    folder=O/f'task{spec["task"]:04d}';folder.mkdir(exist_ok=False);write(folder/'spec.json',spec)
    env.reset(dict(seed=spec['seed'],episode_idx=spec['seed'],sliding_friction=spec['friction']))
    prelude=O/'recovery'/'task0500_rejected_attempt'
    if spec['task']==500 and prelude.exists():
        assert json.loads((prelude/'spec.json').read_text())==spec and spec['cohort']=='broad'
        start=torch.load(prelude/'initial.pt',map_location='cpu',weights_only=False)
        shutil.copy2(prelude/'initial.pt',folder/'initial.pt')
    else:
        start=snap(env);start['geom_friction']=env.physics.model.geom_friction.copy();save(folder/'initial.pt',start)
    speed=float(calibrate(env,acfg,cfg,spec,start,budget,folder)['speed']) if spec['cohort']=='near' else float(np.random.default_rng(spec['speed_seed']).uniform(.247,.435))
    if spec['task']==500 and prelude.exists():assert json.loads((prelude/'program.json').read_text())['speed']==speed
    write(folder/'program.json',dict(speed=speed,repeated_speeds=[speed]*6,clean_official_base=True))
    memory=Memory(model);rng=np.random.default_rng(spec['noise_seed']);trials=[];mh=tensor_hash(model);ah=tensor_hash(actor)
    for ti in range(6):
        identity=dict(task=spec['task'],trial=ti+1,partition='train',actor_steps=9450+asteps,Q_steps=76300+qsteps,writer_steps=12630+wsteps)
        result=trial(env,acfg,start,speed,model,actor,memory,rng,budget,identity,parity=spec['task']<2)
        assert tensor_hash(model)==mh and tensor_hash(actor)==ah
        save(folder/f'trial{ti+1:02d}.pt',result);trials.append(result)
        write(folder/'progress.json',dict(completed_trials=len(trials),speed=speed,Q_updates=qsteps,actor_updates=asteps,writer_updates=wsteps))
    d=dict(spec=spec,speed=speed,trials=trials,Q_updates=qsteps,actor_updates=asteps,writer_updates=wsteps,
        model_tensor_sha256=mh,actor_tensor_sha256=ah,parent_checkpoint_sha256=P_SHA,rng_final=copy.deepcopy(rng.bit_generator.state),memory_windows=memory.committed_windows)
    save(folder/'task.pt',d);returns=[t['terminal_return'] for t in trials if t['status']=='completed']
    row=dict(**spec,speed=speed,Q_updates=qsteps,actor_updates=asteps,writer_updates=wsteps,judged=len(returns),censored=6-len(returns),
        successes=sum(g>0 for g in returns),mean_return=float(np.mean(returns)) if returns else None,controls=sum(len(t['records']) for t in trials),
        actor_tensor_sha256=ah,model_tensor_sha256=mh,task_sha256=sha(folder/'task.pt'))
    write(folder/'summary.json',row);return d,row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert args.resume,'This recovery package must use the reconstructed checkpoint (--resume)'
    assert socket.gethostname()=='c1d3019784aa';torch.set_num_threads(1)
    fixture=json.loads((R/'ttt_joint_1000_v115_fixture.json').read_text());source=sources();assert fixture['status']=='passed' and fixture['source_sha256']==source
    model,actor,parent=load_networks();head,writer,aopt=optimizers(model,actor,parent)
    rng=np.random.default_rng(2025000);random.seed(2025000);torch.manual_seed(2025000)
    plans=specs();replay={};rows=[];qs=ast=ws=groups=0;started=time.monotonic();last_save=started;stopped=False
    def stop(signum,frame):
        nonlocal stopped;stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.resume:
        assert O.is_dir() and not (O/'training_complete.json').exists()
        assert json.loads((O/'manifest.json').read_text())['source_sha256']==source
        ck=torch.load(O/'rolling.pt',map_location='cpu',weights_only=False);assert ck['parent_checkpoint_sha256']==P_SHA
        model.load_state_dict(ck['model']);actor.load_state_dict(ck['actor'])
        head.load_state_dict(ck['head_optimizer']);writer.load_state_dict(ck['writer_optimizer']);aopt.load_state_dict(ck['actor_optimizer'])
        qs=ck['Q_updates'];ast=ck['actor_updates'];ws=ck['slow_writer_updates'];groups=ck['completed_groups'];rows=ck['rows']
        assert qs==100*groups and ast==qs//2 and ws==qs//10
        rng.bit_generator.state=ck['numpy_rng'];random.setstate(ck['python_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        budget=json.loads((O/'budget.json').read_text())
        for row in rows:
            path=O/f'task{row["task"]:04d}/task.pt';assert sha(path)==row['task_sha256'];replay[row['task']]=replay_task(torch.load(path,map_location='cpu',weights_only=False))
    else:
        O.mkdir(exist_ok=False);(O/'source').mkdir()
        for i,p in enumerate(source):shutil.copy2(p,O/'source'/f'{i:02d}_{Path(p).name}')
        write(O/'manifest.json',dict(source_sha256=source,parent_checkpoint_sha256=P_SHA,critic_parent_sha256=sha(CRITIC_PARENT),all1000_specs=plans,
            tasks=1000,trials=6000,Q_updates=12500,actor_updates=6250,slow_writer_updates=1250,initial_Q_steps=76300,initial_actor_steps=9450,initial_writer_steps=12630,
            group_tasks=8,updates_per_group=100,critic_batch_tasks=8,actor_batch_tasks=8,cohorts={'near':500,'broad':500},
            target='MC recorded terminal return /5, gamma=1; mixed historical behavior continuations, not exact current-policy Q',
            replay='all tasks collected in this new phase only; uniformly 4near/4broad distinct tasks, one query/task',
            head_lr=1e-4,writer_lr=1e-5,actor_lr=1e-5,clip=1.,optimizer_moments_resumed=True,
            writer_every=10,actor_every=2,actor_to_writer_gradient=False,writer_loss='critic MC Huber only',
            TTT_inputs='observations/executed residual velocity only; no reward/return tokens',full_causal_BPTT=True,
            context_cache='none in replay; all sampled contexts reconstructed under current writer',collection_model_fixed=True,
            eval_ids=[400,401,402,403,404,405,406,407,408,412,413,414],eval_task_count=12,
            exploration={'acc_sigma':.05,'acc_noise_bound':.15,'acc_bound':.5,'residual_velocity_bound':.1},
            rolling_every_Q_updates=1000,rolling_seconds=600,calibration_failure='stop, no substitution'))
        budget=dict(near_starts=0,training_starts=0,evaluation_starts=0,entries=[]);write(O/'budget.json',budget)
    resumed_groups=groups
    def checkpoint(force=False):
        nonlocal last_save
        if not(force or qs%1000==0 or time.monotonic()-last_save>=600):return
        assert qs==groups*100 and ast==qs//2 and ws==qs//10
        payload=dict(model=model.state_dict(),actor=actor.state_dict(),head_optimizer=head.state_dict(),writer_optimizer=writer.state_dict(),actor_optimizer=aopt.state_dict(),
            Q_steps=76300+qs,writer_steps=12630+ws,actor_steps=9450+ast,Q_updates=qs,actor_updates=ast,slow_writer_updates=ws,completed_groups=groups,rows=rows,
            replay_index=[dict(task=r['task'],path=f'task{r["task"]:04d}/task.pt',sha256=r['task_sha256']) for r in rows],
            numpy_rng=rng.bit_generator.state,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            parent_checkpoint_sha256=P_SHA,source_sha256=source,action='effective residual velocity / .5',gamma=1.,target_scale=5.,actor_to_writer_gradient=False)
        save(O/'rolling.pt',payload);write(O/'checkpoint_receipt.json',dict(Q_updates=qs,actor_updates=ast,writer_updates=ws,tasks=len(rows),sha256=sha(O/'rolling.pt')));last_save=time.monotonic()
    write(O/'owner.json',dict(pid=os.getpid(),hostname=socket.gethostname(),started_unix=time.time()));checkpoint(force=True)
    cfg,acfg=configurations();env=hydra.utils.instantiate(cfg.task.env);env.init_simulator();original=env.objects[0].set_pose_xyz_wxyz
    def tracked(p):original(p);env._teleported=True;env._near_teleported=True
    env.objects[0].set_pose_xyz_wxyz=tracked
    log=(O/'updates.jsonl').open('a' if args.resume else 'x',buffering=1);stage='initializing';spec=None
    try:
        for gi in range(groups,125):
            block=plans[gi*8:min((gi+1)*8,1000)];start=time.monotonic();mh=tensor_hash(model);ah=tensor_hash(actor)
            path=O/f'group{gi:02d}_networks.pt'
            if not path.exists():save(path,dict(model=model.state_dict(),actor=actor.state_dict(),Q_updates=qs,actor_updates=ast,writer_updates=ws,model_tensor_sha256=mh,actor_tensor_sha256=ah))
            for spec in block:
                stage='collecting';write(O/'progress.json',dict(stage=stage,task=spec['task'],completed_tasks=len(rows),Q_updates=qs,actor_updates=ast,writer_updates=ws,elapsed_s=time.monotonic()-started))
                path=O/f'task{spec["task"]:04d}/task.pt'
                if path.exists():
                    d=torch.load(path,map_location='cpu',weights_only=False);row=json.loads(path.with_name('summary.json').read_text())
                    assert d['model_tensor_sha256']==mh and d['actor_tensor_sha256']==ah and d['spec']==spec and sha(path)==row['task_sha256']
                else:d,row=collect_task(env,cfg,acfg,spec,model,actor,budget,qs,ast,ws)
                replay[spec['task']]=replay_task(d);rows.append(row);write(O/'summary.json',rows)
                assert tensor_hash(model)==mh and tensor_hash(actor)==ah
                print('JOINT_TASK',spec['task'],row['cohort'],row['mean_return'],'completed',len(rows),flush=True)
            stage='joint_updates';optstart=time.monotonic()
            for _ in range(100):
                queries=sample(replay,rng);slow=(qs+1)%10==0
                metrics=update_q(model,actor,replay,queries,head,writer,slow);qs+=1;ws+=int(slow)
                aq=None
                if qs%2==0:
                    aq=sample(replay,rng,actor=True);metrics.update(update_actor(model,actor,replay,aq,aopt));ast+=1
                log.write(json.dumps(dict(Q_updates=qs,actor_updates=ast,writer_updates=ws,writer_updated=slow,group=gi,available_tasks=len(replay),
                    queries=queries,actor_queries=aq,seconds=time.monotonic()-started,**metrics))+'\n')
            groups=gi+1;checkpoint(force=stopped or groups==125)
            progress=dict(stage='group_complete',completed_tasks=len(rows),completed_groups=groups,Q_updates=qs,actor_updates=ast,writer_updates=ws,
                elapsed_s=time.monotonic()-started,group_seconds=time.monotonic()-start,optimization_seconds=time.monotonic()-optstart,
                completed_groups_this_process=groups-resumed_groups,
                estimated_training_remaining_s=(time.monotonic()-started)/(groups-resumed_groups)*(125-groups))
            write(O/'progress.json',progress);print('JOINT_GROUP',json.dumps(progress),flush=True)
            if stopped:write(O/'stopped.json',progress);break
        else:
            assert len(rows)==1000 and qs==12500 and ast==6250 and ws==1250
            for opt,count in [(head,88800),(writer,13880),(aopt,15700)]:assert all(int(v['step'])==count for v in opt.state.values())
            shutil.copy2(O/'rolling.pt',O/'joint_final.pt')
            write(O/'training_complete.json',dict(status='completed',tasks=1000,trial_slots=6000,Q_updates=qs,actor_updates=ast,slow_writer_updates=ws,
                Q_steps=88800,writer_steps=13880,actor_steps=15700,checkpoint_sha256=sha(O/'joint_final.pt'),parent_checkpoint_sha256=P_SHA,
                judged=sum(r['judged'] for r in rows),censored=sum(r['censored'] for r in rows),successes=sum(r['successes'] for r in rows),
                seconds=json.loads((O/'recovery/v109_failed.json').read_text())['elapsed_s']+time.monotonic()-started,
                resumed_training_seconds=time.monotonic()-started,budget={k:v for k,v in budget.items() if k!='entries'},actor_to_writer_gradient=False,recovery_receipt_sha256=sha(O/'recovery/receipt.json')))
    except BaseException as e:
        write(O/'failed.json',dict(error=repr(e),stage=stage,task=spec,completed_tasks=len(rows),Q_updates=qs,actor_updates=ast,writer_updates=ws,elapsed_s=time.monotonic()-started));raise
    finally:log.close();env.physics.free()
if __name__=='__main__':main()
