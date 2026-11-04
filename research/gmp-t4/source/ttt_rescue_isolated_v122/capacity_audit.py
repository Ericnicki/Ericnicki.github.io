from review import A
from audit_results import *
setup();P=R/'ttt_rescue_capacity_v125_output';data=json.loads((P/'results.json').read_text());targets=torch.load(FITS/'branch_targets.pt',map_location='cpu',weights_only=False);m,_=load();raw={};rows=[]
for b in targets:
 tid,ti,ix=b['task'],b['trial'],b['index']
 if tid not in raw:raw[tid]=torch.load(D/f'task{tid:04d}/task.pt',weights_only=False)
 r=raw[tid]['trials'][ti]['records'][ix];rows.append(np.column_stack([np.repeat(np.r_[r['context'],r['state']][None,:],5,0),b['actions'].numpy()]))
x=torch.tensor(np.array(rows),device='cuda');g=torch.stack([b['returns'] for b in targets]).cuda();checks=[]
for scale in [1.,10.]:
 ck=torch.load(P/f'head_scale{scale:g}.pt',map_location='cpu',weights_only=False);m.head.load_state_dict(ck['head']);xx=x.clone();xx[:,:,-1]*=scale
 with torch.no_grad():pred=m.head(xx).squeeze(-1);err=(pred-pred[:,2:3])-(g-g[:,2:3]);mae=float(err.abs().mean()*5)
 assert abs(mae-data[str(scale)]['trace'][-1]['contrast_mae'])<1e-6;assert all(torch.isfinite(v).all() for v in ck['head'].values());assert ck['head_updates']==10000 and ck['actor_updates']==ck['writer_updates']==0
 checks.append(dict(scale=scale,recomputed_train_contrast_mae=mae,head_sha256=sha(P/f'head_scale{scale:g}.pt')))
write(P/'audit.json',dict(status='passed',checks=checks,source_sha256=sha(Path(__file__)),optimizer_replay=False,new_physics=0));print('CAPACITY_AUDIT_PASSED')
