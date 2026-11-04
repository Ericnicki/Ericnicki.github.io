"""Independent native/controller/cache and all joint optimizers replay audit."""
import copy,json,socket,time
import numpy as np,torch
from common import R,O,P_SHA,sha,write,load_networks,frozen_equal
from controller import Controller,noise,actor_action,A_MAX
from official import configurations,agent_for,reward
from dataset import features
from memory import Memory
from train import replay_task,batch,tensor_hash,specs,optimizers,controls,sample
from torch.nn import functional as F
def main():
    assert socket.gethostname()=='92c316f35f5f';torch.set_num_threads(1);t0=time.monotonic()
    complete=json.loads((O/'training_complete.json').read_text());manifest=json.loads((O/'manifest.json').read_text())
    assert complete['status']=='completed' and complete['tasks']==500 and complete['actor_updates']==3150 and complete['Q_updates']==6300 and complete['slow_writer_updates']==630
    for p,h in manifest['source_sha256'].items():assert sha(p)==h
    final=torch.load(O/'joint_final.pt',map_location='cpu',weights_only=False)
    assert sha(O/'joint_final.pt')==complete['checkpoint_sha256'] and final['actor_steps']==9450
    model,actor,parent=load_networks()
    assert all(int(v['step'])==9450 for v in final['actor_optimizer']['state'].values())
    assert manifest['all500_specs']==specs();_,acfg=configurations()
    replay={};judged=success=controls=0;errors=[];group_hash={}
    for gi in range(63):
        checkpoint=torch.load(O/f'group{gi:02d}_networks.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']);actor.load_state_dict(checkpoint['actor']);assert checkpoint['Q_updates']==gi*100
        h=tensor_hash(actor);mh=tensor_hash(model);assert h==checkpoint['actor_tensor_sha256'] and mh==checkpoint['model_tensor_sha256'];group_hash[gi]=(h,mh)
        for spec in manifest['all500_specs'][gi*8:min((gi+1)*8,500)]:
            tid=spec['task'];folder=O/f'task{tid:04d}';d=torch.load(folder/'task.pt',map_location='cpu',weights_only=False)
            assert d['spec']==spec and d['Q_updates']==gi*100 and d['actor_updates']==gi*50 and d['writer_updates']==gi*10 and d['model_tensor_sha256']==mh and d['actor_tensor_sha256']==h
            assert d['parent_checkpoint_sha256']==P_SHA and len(d['trials'])==6
            row=final['rows'][tid];assert sha(folder/'task.pt')==row['task_sha256']
            assert all(t['speed']==d['speed'] for t in d['trials'])
            rng=np.random.default_rng(spec['noise_seed']);initial=None;mem=Memory(model)
            if spec['cohort']=='near':
                cal=json.loads((folder/'calibration/completion.json').read_text());assert cal['qualified'] and cal['selected']['speed']==d['speed']
            else:assert d['speed']==float(np.random.default_rng(spec['speed_seed']).uniform(.247,.435))
            for ti,t in enumerate(d['trials']):
                rr=t['records'];states,events,phases=features(rr)
                first=rr[0]['before']['robots_obs'][0]['tcp_xyz_wxyz']
                if initial is None:initial=first
                else:np.testing.assert_array_equal(first,initial)
                c=Controller();agent=agent_for(acfg,[d['speed']]*6);previous=[];prior=None
                for i,r in enumerate(rr):
                    np.testing.assert_array_equal(states[i],r['state']);np.testing.assert_array_equal(events[i],r['event']);assert phases[i]==r['proxy_phase']
                    assert r['prior_trial_count']==ti and r['current_observed_events']==i
                    eps,draws=noise(rng) if r['critical'] else (0.,0)
                    assert r['noise_m_s2']==eps and r['noise_rng_draws']==draws
                    if r['critical']:
                        st=torch.as_tensor(r['state'],device='cuda').reshape(1,22);ctx=torch.as_tensor(r['context'],device='cuda').reshape(1,32)
                        with torch.no_grad():acc=float((A_MAX*torch.tanh(actor.net(torch.cat([ctx,st],dim=-1)))).item())
                    else:acc=0.
                    assert acc==r['actor_acc_m_s2']
                    calc=c.step(r['base_y_m'],r['base_previous_y_m'],r['previous_executed_y_m'],acc+eps,r['critical'],r['active'])
                    for k,v in calc.items():assert abs(v-r[k])<1e-12,(tid,ti,i,k)
                    assert r['terminal']==(r['judge_after']>r['judge_before'])
                    assert abs(r['time_after']-r['time_before']-.1)<1e-10
                    assert abs(r['executed_command'][0]['tcp_xyz_wxyz'][1]-r['command_y_m'])<1e-10
                    nominal=agent.predict_actions([r['before']['robots_obs']],[r['before']['env_objs_obs']],[previous])[0]
                    for a,b in zip(nominal,r['nominal_command']):
                        for k in ['name','tcp_xyz_wxyz','gripper_width','is_critical']:
                            if k in a:np.testing.assert_array_equal(a[k],b[k])
                    if prior:
                        assert r['previous_executed_y_m']==prior['command_y_m']
                        np.testing.assert_array_equal(r['before']['robots_obs'][0]['tcp_xyz_wxyz'],prior['after']['robots_obs'][0]['tcp_xyz_wxyz'])
                    if tid%25==0:
                        if i in {0,16,len(rr)-1}:
                            ctx=mem.reference(r['state']);err=float(np.max(abs(ctx.cpu().numpy()[0]-r['context'])));errors.append(err)
                            assert err<3e-5,(tid,ti,i,err)
                        mem.observe(r['event'])
                    prior=r;previous=copy.deepcopy(r['executed_command']);controls+=1
                if tid%25==0:mem.end_trial()
                if t['status']=='completed':
                    assert rr[-1]['terminal'];g=reward(t['stop_xy']);assert g==t['terminal_return'] and t['official_success']==(g>0)
                    judged+=1;success+=g>0
                else:assert 'terminal_return' not in t
            assert rng.bit_generator.state==d['rng_final'];replay[tid]=replay_task(d)
    assert len(replay)==500 and judged==complete['judged'] and success==complete['successes']
    # Replay logged samples and optimizer operations, independently of training update functions.
    model.load_state_dict(parent['model']);actor.load_state_dict(parent['actor']);head,writer,aopt=optimizers(model,actor,parent)
    updates=actor_updates=writer_updates=0;task_exposure={tid:0 for tid in replay};rng=np.random.default_rng(2015000)
    def check_queries(queries,available,active):
        ids=[q[0] for q in queries];assert len(ids)==len(set(ids))==8 and sum(replay[tid]['cohort']=='near' for tid in ids)==4
        assert all(0<=tid<available for tid in ids)
        for tid,ti,ix,phase in queries:
            tr=replay[tid]['trials'][ti];assert tr['return'] is not None and ix in tr['active' if active else 'eligible']
            assert phase=='unclassified' or ix in tr['actor_pools' if active else 'pools'][phase];task_exposure[tid]+=1
    for line in (O/'updates.jsonl').open():
        row=json.loads(line);updates+=1;gi=(updates-1)//100;available=min((gi+1)*8,500)
        assert row['Q_updates']==updates and row['group']==gi and row['available_tasks']==available
        if updates%100==1:assert (tensor_hash(actor),tensor_hash(model))==group_hash[gi]
        live={tid:d for tid,d in replay.items() if tid<available}
        expected=[list(q) for q in sample(live,rng)];assert expected==row['queries'];queries=row['queries'];check_queries(queries,available,False)
        slow=updates%10==0;assert row['writer_updated']==slow
        head.zero_grad(set_to_none=True);writer.zero_grad(set_to_none=True);actor.zero_grad(set_to_none=True)
        h,m,st,a,y=batch(replay,queries);pred=model(h,m,st,a,writer_grad=slow);loss=F.smooth_l1_loss(pred,y)
        assert abs(float(loss.detach())-row['Q_loss'])<1e-7
        loss.backward();torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.,error_if_nonfinite=True)
        assert all(p.grad is None for p in actor.parameters())
        if slow:
            torch.nn.utils.clip_grad_norm_(model.writer_parameters(),1.,error_if_nonfinite=True);writer.step();writer_updates+=1
        else:assert all(p.grad is None for p in model.writer_parameters())
        head.step();model.zero_grad(set_to_none=True)
        if updates%2==0:
            queries=row['actor_queries'];expected=[list(q) for q in sample(live,rng,True)];assert expected==queries;check_queries(queries,available,True)
            model_snapshot=tensor_hash(model);aopt.zero_grad(set_to_none=True)
            for p in model.parameters():p.requires_grad_(False)
            h,m,st,_,_=batch(replay,queries)
            with torch.no_grad():ctx=model.context(h,m,st)
            old,prev,delta=controls(replay,queries);dv,acc,new=actor_action(actor,ctx.detach(),st,old,prev,delta)
            pred=model.head(torch.cat([ctx.detach(),st,(dv/.5).unsqueeze(-1)],dim=-1)).squeeze(-1);loss=-pred.mean()
            assert abs(float(loss.detach())-row['actor_loss'])<1e-7
            loss.backward();assert all(p.grad is None for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True);aopt.step();actor.zero_grad(set_to_none=True);actor_updates+=1
            for p in model.parameters():p.requires_grad_(True)
            assert tensor_hash(model)==model_snapshot
        else:assert row['actor_queries'] is None
        assert row['actor_updates']==actor_updates and row['writer_updates']==writer_updates
    assert updates==6300 and actor_updates==3150 and writer_updates==630
    for module,key in [(model,'model'),(actor,'actor')]:
        assert all(torch.equal(v.detach().cpu(),final[key][k]) for k,v in module.state_dict().items()),key
    for opt,key,count in [(head,'head_optimizer',76300),(writer,'writer_optimizer',12630),(aopt,'actor_optimizer',9450)]:
        actual=opt.state_dict();expected=final[key];assert actual['param_groups']==expected['param_groups']
        assert set(actual['state'])==set(expected['state'])
        for pi,values in actual['state'].items():
            for k,v in values.items():
                assert torch.equal(v.detach().cpu(),expected['state'][pi][k]) if torch.is_tensor(v) else v==expected['state'][pi][k]
            assert int(values['step'])==count
    assert rng.bit_generator.state==final['numpy_rng']
    budget=json.loads((O/'budget.json').read_text());assert budget['training_starts']<=3000 and budget['evaluation_starts']==0
    receipt=dict(status='passed',tasks=500,trial_slots=3000,judged=judged,censored=3000-judged,successes=success,controls=controls,
        Q_updates=updates,actor_updates=actor_updates,slow_writer_updates=writer_updates,all_optimizer_replay_exact=True,
        all_optimizer_moments_exact=True,actor_to_writer_gradient=False,max_cache_reference_error=max(errors),cache_queries=len(errors),
        eight_distinct_task_batches=True,task_gradient_exposures=task_exposure,checkpoint_sha256=sha(O/'joint_final.pt'),seconds=time.monotonic()-t0,
        physics_starts=0,production_optimizer_updates=0)
    write(O/'audit.json',receipt);print('JOINT_AUDIT_PASSED',json.dumps({k:v for k,v in receipt.items() if k!='task_gradient_exposures'}),flush=True)
if __name__=='__main__':main()
