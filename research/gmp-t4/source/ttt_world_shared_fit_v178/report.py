from fit import *
setup(2178000)
if json.loads((O/'audit.json').read_text())['status']=='skipped':exit()
tasks=tasks_load();metrics={};rows=[]
for arm in ARMS:
 ck=torch.load(O/f'{arm}_world.pt',map_location='cpu',weights_only=False);w=World().cuda();v=Value().cuda();w.load_state_dict(ck['world']);v.load_state_dict(ck['value']);freeze(w);freeze(v);ids=[tid for tid,t in tasks.items() if t['spec']['partition']=='heldout'];cache,_=private(w,tasks,ids);local=[]
 for tid in ids:
  for qi,p in enumerate(tasks[tid]['pairs']):
   b=batch(tasks,[(tid,qi)]);W=tensor(cache[(tid,qi)]['W'].numpy()).unsqueeze(0);delta=tensor([0.]).requires_grad_(True);nx=w.predict(reader_features(b['state'],b['context'],b['first_command'],delta,b['anchor_acc'],b['controller']),W);score=v.score(nx,b['context']);g=float(torch.autograd.grad(score.sum(),delta)[0]);native=float((b['value'][0,2,0]-b['value'][0,1,0])*5/.1);pred=w.predict(features(b),W);goal=v.score(pred,b['context'].unsqueeze(1).expand(-1,3,-1));local.append(dict(task=tid,phase=p['phase'],cube_Y_center_error_mm=float(abs(pred[0,0,7]-b['target'][0,0,7])*.5*1000),G_MAE=float((goal-5*b['value'][...,0]).abs().mean()),goal_gradient=g,native_gradient=native))
 eligible=[r for r in local if abs(r['native_gradient'])>1e-3];metrics[arm]=dict(tasks=24,pairs=len(local),cube_Y_center_MAE_mm=float(np.mean([np.mean([r['cube_Y_center_error_mm'] for r in local if r['task']==tid]) for tid in ids])),G_MAE=float(np.mean([r['G_MAE'] for r in local])),eligible=len(eligible),goal_gradient_sign=float(np.mean([r['goal_gradient']*r['native_gradient']>0 for r in eligible])),goal_gradient_MAE=float(np.mean([abs(r['goal_gradient']-r['native_gradient']) for r in eligible])))
 rows.extend(dict(arm=arm,**r) for r in local)
write(O/'diagnostics.json',dict(metrics=metrics,rows=rows));write(O/'result.json',dict(status='completed',metrics=metrics,world_updates_each=1000,actor_updates_each=1000,shared_correction_and_dynamics_private_W=True,joint_prediction_value_outer_test=True,native_evaluation='pending',exploratory_reused_benchmark=True,audit_sha256=sha(O/'audit.json')));write(O/'file_manifest.json',{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='file_manifest.json'})
