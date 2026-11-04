from review import A
from audit_results import *
from dataset import batch as hb
from train import replay_task
setup();targets=torch.load(FITS/'branch_targets.pt',map_location='cpu',weights_only=False);replay={}
for tid in sorted({b['task'] for b in targets}):replay[tid]=replay_task(torch.load(D/f'task{tid:04d}/task.pt',map_location='cpu',weights_only=False))
results={}
for seed in [2033000,2033001]:
 for arm in ['mc','hindsight','grounded']:
  key=f'{arm}_{seed}';results[key]={}
  for step in [0,500,1000,1500]:
   m,_=load()
   if step:m.load_state_dict(torch.load(FITS/key/f'step{step}.pt',map_location='cpu',weights_only=False)['model'])
   pred=[]
   for i in range(0,len(targets),8):
    bs=targets[i:i+8];h,mask,s,_,_=hb(replay,[(b['task'],b['trial'],b['index']) for b in bs],True,'cuda');acts=torch.stack([b['actions'] for b in bs]).cuda()
    with torch.no_grad():
     c=m.context(h,mask,s);p=m.head(torch.cat([c[:,None,:].expand(-1,5,-1),s[:,None,:].expand(-1,5,-1),acts],-1)).squeeze(-1);pred.append(p.cpu())
   pred=torch.cat(pred);g=torch.stack([b['returns'] for b in targets]);t=torch.stack([b['teacher_q'] for b in targets]);delta=pred-pred[:,2:3];dg=g-g[:,2:3];dt=t-t[:,2:3]
   results[key][str(step)]=dict(branch_value_mae=float((pred-g).abs().mean()*5),true_contrast_mae=float((delta-dg).abs().mean()*5),teacher_contrast_mae=float((delta-dt).abs().mean()*5),grounded_aux=float(torch.nn.functional.smooth_l1_loss((delta-dg)/.1,torch.zeros_like(pred))),hindsight_aux=float(torch.nn.functional.smooth_l1_loss((delta-dt)/.1,torch.zeros_like(pred))))
  print('TRAIN_SCORED',key,results[key]['0'],results[key]['1500'],flush=True)
write(FITS/'fixed_train_scores.json',dict(source_sha256=sha(Path(__file__)),scope='Fixed95 TRAIN query bank; curves descriptive, no checkpoint selection',scores=results))
