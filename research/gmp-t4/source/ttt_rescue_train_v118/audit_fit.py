from audit_results import *
setup();assert json.loads((FITS/'complete.json').read_text())['status']=='completed';parent=torch.load(PATHS['new'],map_location='cpu',weights_only=False);rows=[]
for seed in [2033000,2033001]:
 for arm in ['mc','hindsight','grounded']:
  folder=FITS/f'{arm}_{seed}';ck=torch.load(folder/'step1500.pt',map_location='cpu',weights_only=False);row=dict(arm=arm,seed=seed)
  for group in ['model','actor']:
   assert set(ck[group])==set(parent[group]);assert all(torch.isfinite(v).all() for v in ck[group].values());row[group+'_changed_tensors']=sum(not torch.equal(v,parent[group][k]) for k,v in ck[group].items())
  for key,n in [('head_optimizer',1500),('writer_optimizer',150),('actor_optimizer',750)]:
   old=parent[key];new=ck[key];assert new['param_groups']==old['param_groups'];assert set(old['state'])==set(new['state'])
   for idx,v in new['state'].items():
    assert int(v['step'])-int(old['state'][idx]['step'])==n
    assert all(torch.isfinite(t).all() for t in v.values() if isinstance(t,torch.Tensor))
   row[key+'_operations']=n
  logs=[json.loads(s) for s in (folder/'updates.jsonl').read_text().splitlines()]
  row['mc_curve']=[float(np.mean([l['mc'] for l in logs[i:i+100]])) for i in range(0,1500,100)]
  row['aux_curve']=[float(np.mean([l['aux'] for l in logs[i:i+100]])) for i in range(0,1500,100)]
  rows.append(row)
write(FITS/'optimizer_audit.json',dict(status='passed',source_sha256=sha(Path(__file__)),exact_full_optimizer_replay=False,finite_parameters_and_moments=True,parent_optimizer_settings_preserved=True,rows=rows));print(json.dumps([dict(arm=r['arm'],seed=r['seed'],mc_first=r['mc_curve'][0],mc_last=r['mc_curve'][-1],aux_first=r['aux_curve'][0],aux_last=r['aux_curve'][-1]) for r in rows],indent=2))
