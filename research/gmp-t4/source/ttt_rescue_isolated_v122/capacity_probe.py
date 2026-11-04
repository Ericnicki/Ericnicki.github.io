from review import A
from audit_results import *
import copy,time
setup();P=R/'ttt_rescue_capacity_v125_output';P.mkdir(exist_ok=False);targets=torch.load(FITS/'branch_targets.pt',map_location='cpu',weights_only=False);m,_=load();rows=[];raw={}
for b in targets:
 tid,ti,ix=b['task'],b['trial'],b['index']
 if tid not in raw:raw[tid]=torch.load(D/f'task{tid:04d}/task.pt',weights_only=False)
 r=raw[tid]['trials'][ti]['records'][ix];cs=np.r_[r['context'],r['state']];rows.append(np.column_stack([np.repeat(cs[None,:],5,axis=0),b['actions'].numpy()]))
x=torch.tensor(np.array(rows),device='cuda',dtype=torch.float32);g=torch.stack([b['returns'] for b in targets]).cuda();t0=time.monotonic();out={}
write(P/'manifest.json',dict(source_sha256=sha(Path(__file__)),parent=HASHES['new'],target_sha256=sha(FITS/'branch_targets.pt'),variants=[1.,10.],head_only=True,writer_frozen=True,full_batch_queries=len(targets),steps_per_variant=10000,lr=.0003,clip=5.,fresh_head_optimizer=True,objective='MSE normalized relative Q contrasts/.1 + .1*MSE absolute branch Q',scope='TRAIN fit capacity/conditioning diagnostic; no actor updates and no physical-control claim'))
for scale in [1.,10.]:
 head=copy.deepcopy(m.head)
 for p in head.parameters():p.requires_grad_(True)
 with torch.no_grad():head[0].weight[:,-1]/=scale
 xx=x.clone();xx[:,:,-1]*=scale
 with torch.no_grad():assert float((head(xx)-m.head(x)).abs().max())<2e-6
 opt=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.0001);trace=[]
 def metrics():
  with torch.no_grad():
   p=head(xx).squeeze(-1);err=(p-p[:,2:3])-(g-g[:,2:3])
   return dict(contrast_mae=float(err.abs().mean()*5),value_mae=float((p-g).abs().mean()*5),contrast_rmse=float(err.square().mean().sqrt()*5))
 trace.append(dict(step=0,**metrics()))
 for step in range(1,10001):
  opt.zero_grad(set_to_none=True);p=head(xx).squeeze(-1);err=(p-p[:,2:3])-(g-g[:,2:3]);loss=(err/.1).square().mean()+.1*(p-g).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.,error_if_nonfinite=True);opt.step()
  if step%500==0:trace.append(dict(step=step,**metrics()))
 save(P/f'head_scale{scale:g}.pt',dict(head=head.state_dict(),action_input_multiplier=scale,parent=HASHES['new'],head_updates=10000,actor_updates=0,writer_updates=0))
 # Existing diagnostic states, always frozen v115 continuation labels.
 allq=json.loads((O/'probes.json').read_text());stats=[];dev={}
 for q in allq:
  tid,ti,ix=q['task'],q['trial'],q['control']
  if tid not in dev:dev[tid]=torch.load(R/f'ttt_joint_1000_v115_output/evaluation/task{tid:04d}/actor_deterministic.pt',weights_only=False)['trials']
  r=dev[tid][ti]['records'][ix];cs=np.r_[r['context'],r['state']];aa=q['answers'][1:];xx2=np.column_stack([np.repeat(cs[None,:],5,axis=0),np.array([a['dv']/.5*scale for a in aa])]);gg=np.array([a['return_'] for a in aa])
  with torch.no_grad():qq=head(torch.tensor(xx2,device='cuda',dtype=torch.float32)).flatten().cpu().numpy()*5
  stats.append(dict(task=tid,region=q['region'],contrast_mae=float(np.mean(abs((qq-qq[2])-(gg-gg[2])))),regret=float(gg.max()-gg[qq.argmax()])))
 out[str(scale)]=dict(trace=trace,development=dict(contrast_mae=float(np.mean([s['contrast_mae'] for s in stats])),regret=float(np.mean([s['regret'] for s in stats]))));write(P/'results.json',out);print('CAPACITY',scale,trace[0],trace[-1],out[str(scale)]['development'],flush=True)
write(P/'complete.json',dict(status='completed',head_updates=20000,actor_updates=0,writer_updates=0,new_physics=0,seconds=time.monotonic()-t0))
