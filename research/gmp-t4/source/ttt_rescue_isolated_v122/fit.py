import sys
from pathlib import Path
R=Path('/public/home/2023141520023/return_selector_20261006');sys.path.insert(1,str(R/'ttt_rescue_v117'))
from rescue_common import *
import json,time,copy
from torch.nn import functional as F
from dataset import batch as hb
from train import replay_task,sample,controls
from controller import actor_action
D=R/'ttt_rescue_train_v118_output';FITS=R/'ttt_rescue_fit_v122_output'
def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float32)
def main():
    setup();assert json.loads((D/'complete.json').read_text())['status']=='completed';FITS.mkdir(exist_ok=False)
    parent=torch.load(PATHS['new'],map_location='cpu',weights_only=False);rows=json.loads((D/'summary.json').read_text());replay={};raw={};branch=[]
    teacher,_=load()
    for row in rows:
        tid=row['spec']['task'];d=torch.load(D/f'task{tid:04d}/task.pt',map_location='cpu',weights_only=False);raw[tid]=d;replay[tid]=replay_task(d)
        for query in row['queries']:
            if not all(a['status']=='completed' for a in query['answers']):continue
            ti,ix=query['trial'],query['control'];rr=d['trials'][ti]['records'];tm=rebuild(teacher,d['trials'][:ti]+d['trials'][ti+1:])
            for r in rr[:ix]:tm.observe(r['event'])
            state=tensor(rr[ix]['state']).reshape(1,22);context=tm.context(rr[ix]['state']);actions=tensor([[a['dv']/.5] for a in query['answers'][1:]])
            with torch.no_grad():q=teacher.head(torch.cat([context.expand(5,-1),state.expand(5,-1),actions],-1)).flatten().cpu()
            branch.append(dict(task=tid,trial=ti,index=ix,actions=actions.cpu(),returns=torch.tensor([a['return_']/5 for a in query['answers'][1:]]),teacher_q=q))
    save(FITS/'branch_targets.pt',branch)
    write(FITS/'manifest.json',dict(parent=HASHES['new'],data_manifest_sha256=sha(D/'manifest.json'),data_complete_sha256=sha(D/'complete.json'),source_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).with_name('FIT_PROTOCOL.md')),arms=['mc','hindsight','grounded'],seeds=[2033000,2033001],updates=1500,actor_every=2,writer_every=10,branch_queries=len(branch),coefficient=.1,contrast_scale=.1))
    t0=time.monotonic()
    for seed in [2033000,2033001]:
      for arm in ['mc','hindsight','grounded']:
        folder=FITS/f'{arm}_{seed}';folder.mkdir();m,a=load();
        for p in list(m.parameters())+list(a.parameters()):p.requires_grad_(True)
        ho=torch.optim.AdamW(m.head.parameters(),lr=1e-4,weight_decay=1e-4);wo=torch.optim.AdamW(m.writer_parameters(),lr=1e-5,weight_decay=1e-4);ao=torch.optim.Adam(a.parameters(),lr=1e-5)
        for opt,key in [(ho,'head_optimizer'),(wo,'writer_optimizer'),(ao,'actor_optimizer')]:opt.load_state_dict(copy.deepcopy(parent[key]))
        assert all(int(v['step'])==n for key,n in [('head_optimizer',88800),('writer_optimizer',13880),('actor_optimizer',15700)] for v in parent[key]['state'].values())
        rng=np.random.default_rng(seed);torch.manual_seed(seed);log=(folder/'updates.jsonl').open('w',buffering=1)
        for step in range(1,1501):
            qs=sample(replay,rng);h,mask,s,ac,y=hb(replay,[q[:3] for q in qs],True,'cuda');slow=step%10==0
            m.zero_grad(set_to_none=True);a.zero_grad(set_to_none=True)
            mc=F.smooth_l1_loss(m(h,mask,s,ac,writer_grad=slow),y)
            # Draw identical task-balanced auxiliary queries in all three arms.
            selected=[]
            for cohort in ['near','broad']:
                ids=[tid for tid,d in replay.items() if d['cohort']==cohort]
                for tid in rng.choice(ids,4,replace=False):
                    eligible=[b for b in branch if b['task']==tid];selected.append(eligible[int(rng.integers(len(eligible)))])
            aux=torch.zeros((),device='cuda');
            if arm!='mc':
                hh,mm,ss,_,_=hb(replay,[(b['task'],b['trial'],b['index']) for b in selected],True,'cuda')
                if slow:ctx=m.context(hh,mm,ss)
                else:
                    with torch.no_grad():ctx=m.context(hh,mm,ss)
                acts=torch.stack([b['actions'] for b in selected]).cuda();pred=m.head(torch.cat([ctx[:,None,:].expand(-1,5,-1),ss[:,None,:].expand(-1,5,-1),acts],-1)).squeeze(-1)
                target=torch.stack([b['teacher_q'] if arm=='hindsight' else b['returns'] for b in selected]).cuda()
                aux=F.smooth_l1_loss(((pred-pred[:,2:3])-(target-target[:,2:3]))/.1,torch.zeros_like(pred))
            loss=mc+.1*aux;assert torch.isfinite(loss);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.head.parameters(),1.,error_if_nonfinite=True);ho.step()
            if slow:torch.nn.utils.clip_grad_norm_(m.writer_parameters(),1.,error_if_nonfinite=True);wo.step()
            else:assert all(p.grad is None for p in m.writer_parameters())
            m.zero_grad(set_to_none=True);al=None
            if step%2==0:
                aq=sample(replay,rng,actor=True);h,mask,s,_,_=hb(replay,[q[:3] for q in aq],True,'cuda')
                with torch.no_grad():ctx=m.context(h,mask,s)
                old,prev,delta=controls(replay,aq)
                for p in m.parameters():p.requires_grad_(False)
                ao.zero_grad(set_to_none=True);dv,acc,_=actor_action(a,ctx,s,old,prev,delta)
                al=-m.head(torch.cat([ctx,s,(dv/.5).unsqueeze(-1)],-1)).mean();al.backward()
                assert all(p.grad is None for p in m.parameters());torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True);ao.step()
                for p in m.parameters():p.requires_grad_(True)
            log.write(json.dumps(dict(step=step,mc=float(mc.detach()),aux=float(aux.detach()),actor=float(al.detach()) if al is not None else None,writer=slow,queries=qs,aux_queries=[(b['task'],b['trial'],b['index']) for b in selected]))+'\n')
            if step%500==0:
                save(folder/f'step{step}.pt',dict(model=m.state_dict(),actor=a.state_dict(),head_optimizer=ho.state_dict(),writer_optimizer=wo.state_dict(),actor_optimizer=ao.state_dict(),step=step,seed=seed,arm=arm,parent=HASHES['new']))
                print('FIT',seed,arm,step,time.monotonic()-t0,flush=True)
        assert all(int(v['step'])==n for key,n in [('head_optimizer',88800),('writer_optimizer',13880),('actor_optimizer',15700)] for v in parent[key]['state'].values())
        assert all(int(v['step'])==n for opt,n in [(ho,90300),(wo,14030),(ao,16450)] for v in opt.state.values())
        log.close();write(folder/'complete.json',dict(Q_updates=1500,actor_updates=750,writer_updates=150,sha256=sha(folder/'step1500.pt')))
    write(FITS/'complete.json',dict(status='completed',arms=3,seeds=2,total_Q_updates=9000,total_actor_updates=4500,total_writer_updates=900,seconds=time.monotonic()-t0))
if __name__=='__main__':main()
