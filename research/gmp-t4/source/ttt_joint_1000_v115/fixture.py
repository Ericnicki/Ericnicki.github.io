"""Compute-only real-history gradient-routing and causal/controller preflight."""
import copy,json,socket,time
import numpy as np,torch
from common import R,P_SHA,load_networks,write,sources
from train import replay_task,batch,sample,specs,optimizers,update_q,update_actor,tensor_hash
from memory import Memory
from trial_budget import boundary_check
from controller import Controller,actor_action,noise,A_MAX

def main():
    assert socket.gethostname()=='c1d3019784aa';torch.set_num_threads(1);torch.manual_seed(2020000);start=time.monotonic()
    budget_checks=boundary_check()
    model,actor,parent=load_networks();raw=torch.load(R/'ttt_joint_v100_output/task0000/task.pt',map_location='cpu',weights_only=False)
    replay={i:replay_task(raw) for i in range(8)}
    for i,d in replay.items():d['cohort']='near' if i<4 else 'broad'
    queries=[(i,5,20,'fixture') for i in range(8)]
    h,m,s,a,y=batch(replay,queries);assert h.shape[-1]==23 and s.shape[-1]==22 and len(raw['trials'])==6
    # Full-prefix gradient reaches the earliest observed event, not only the last window.
    leaf=h.clone().requires_grad_(True);ctx=model.context(leaf,m,s)
    grad=torch.autograd.grad(ctx.square().mean(),leaf)[0]
    earliest=float(grad[:,0].abs().sum());assert earliest>0
    # Future observations and prior-trial labels never enter this query context.
    changed=copy.deepcopy(replay)
    for d in changed.values():
        d['trials'][5]['events'][20:]+=123.
        for t in d['trials'][:5]:t['return']=12345.
    hh,mm,ss,aa,yy=batch(changed,queries)
    assert torch.equal(h,hh) and torch.equal(m,mm) and torch.equal(s,ss) and torch.equal(y,yy)
    errors=[]
    def cache_check():
        memory=Memory(model)
        for ti,t in enumerate(raw['trials']):
            for i,r in enumerate(t['records']):
                if i in {0,15,16,len(t['records'])-1}:
                    err=float((memory.context(r['state'])-memory.reference(r['state'])).abs().max());assert err<3e-5;errors.append(err)
                memory.observe(r['event'])
            memory.end_trial()
        assert len(memory.completed)==6
        fresh=Memory(model);assert not fresh.pending and not fresh.completed
    cache_check()
    head,writer,aopt=optimizers(model,actor,parent);initial_model=tensor_hash(model);initial_actor=tensor_hash(actor)
    writer_initial={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('head.')}
    rng=np.random.default_rng(2020001);qseconds=0.;aseconds=0.;slowseconds=0.;writer_changes=0
    for step in range(1,11):
        aq=sample(replay,rng);asnap=tensor_hash(actor);t0=time.monotonic()
        row=update_q(model,actor,replay,aq,head,writer,slow=step==10);qseconds+=time.monotonic()-t0
        assert tensor_hash(actor)==asnap
        current={k:v for k,v in model.state_dict().items() if not k.startswith('head.')}
        if step<10:assert all(torch.equal(v,writer_initial[k]) for k,v in current.items())
        else:
            assert any(not torch.equal(v,writer_initial[k]) for k,v in current.items());writer_changes+=1;slowseconds=time.monotonic()-t0
        if step%2==0:
            frozen=tensor_hash(model);t0=time.monotonic();update_actor(model,actor,replay,sample(replay,rng,True),aopt);aseconds+=time.monotonic()-t0
            assert tensor_hash(model)==frozen and all(p.grad is None for p in model.parameters())
    assert writer_changes==1 and tensor_hash(model)!=initial_model and tensor_hash(actor)!=initial_actor
    assert all(int(v['step'])==76310 for v in head.state.values())
    assert all(int(v['step'])==12631 for v in writer.state.values())
    assert all(int(v['step'])==9455 for v in aopt.state.values())
    cache_check()
    old=torch.tensor([0,.09,-.09,.1,-.1,0,0,.02],device='cuda');prev=torch.tensor([-.7,-.7,-.7,-.7,-.7,.249,-.849,-.7],device='cuda')
    delta=torch.tensor([.01,.01,.01,.01,.01,.02,-.02,.01],device='cuda')
    c=torch.zeros(8,32,device='cuda');st=torch.zeros(8,22,device='cuda');dv,acc,new=actor_action(actor,c,st,old,prev,delta)
    for i in range(8):
        out=Controller(float(old[i]),0).step(float(prev[i]+delta[i]),float(prev[i]),float(prev[i]),float(acc[i]),True,True)
        assert abs(out['effective_d_V_m_s']-float(dv[i]))<7e-7
    draws=np.array([noise(rng)[0] for _ in range(10000)]);assert np.max(abs(draws))<=.15 and .47<(draws>0).mean()<.53
    for size in [8,496,1000]:
        fake={s['task']:dict(replay[s['task']%8],cohort=s['cohort']) for s in specs()[:size]}
        for actor_batch in [False,True]:
            qs=sample(fake,rng,actor_batch);ids=[q[0] for q in qs]
            assert len(ids)==len(set(ids))==8 and sum(fake[i]['cohort']=='near' for i in ids)==4
    fresh=specs();assert len(fresh)==1000 and sum(d['cohort']=='near' for d in fresh)==500
    previous=[]
    for path,key in [(R/'acc_noise_full_v86_output/manifest.json','all500_prospective_specs'),(R/'ttt_actor_only_v96_output/manifest.json','all500_specs'),(R/'ttt_joint_v100_output/manifest.json','all500_specs')]:
        previous.extend(json.loads(path.read_text())[key])
    oldseeds={d['seed'] for d in previous}
    assert not ({d['seed'] for d in fresh}&oldseeds)
    assert len({d['seed'] for d in fresh})==1000
    receipt=dict(status='passed',trial_budget_boundary_checks=budget_checks,source_sha256=sources(),parent_checkpoint_sha256=P_SHA,physics_starts=0,production_optimizer_updates=0,
        max_cache_error=max(errors),cache_queries=len(errors),earliest_history_gradient_L1=earliest,
        actor_to_writer_gradient=False,actor_updates_preserve_entire_critic=True,writer_updates_only_from_MC=True,
        tested_Q_updates=10,tested_actor_updates=5,tested_writer_updates=1,optimizer_moments_resume=True,
        future_and_return_input_exclusion=True,eight_distinct_balanced_tasks=True,controller_parity=True,
        sampled_batch_mean_Q_seconds=qseconds/10,mean_actor_seconds=aseconds/5,writer_step_seconds=slowseconds,seconds=time.monotonic()-start)
    write(R/'ttt_joint_1000_v115_fixture.json',receipt);print('JOINT_PREFLIGHT_PASSED',json.dumps(receipt),flush=True)
if __name__=='__main__':main()
