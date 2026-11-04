from audit_results import *
setup();out={}
for seed in [2033000,2033001]:
 for arm in ['mc','hindsight','grounded']:
  k=f'{arm}_{seed}';ck=torch.load(FITS/k/'step1500.pt',map_location='cpu',weights_only=False);out[k]={key:sorted({int(v['step']) for v in ck[key]['state'].values()}) for key in ['head_optimizer','writer_optimizer','actor_optimizer']}
write(FITS/'step_counter_diagnosis.json',out);print(json.dumps(out,indent=2))
