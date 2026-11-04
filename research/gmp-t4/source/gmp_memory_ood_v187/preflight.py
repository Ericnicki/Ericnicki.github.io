import shutil
from common import *
from policy import Official
from trial import environment,run_trial
setup();assert not O.exists();fixture=R/'gmp_memory_ood_v187_fixture_output';assert not fixture.exists();fixture.mkdir();env=environment();p=Official();mh=state_hash(p.model);nh=state_hash(p.norm);rows=json.loads((B/'summary.json').read_text());checks=[]
try:
 for tid in [0,16]:
  row=rows[tid];assert row['spec']['friction']<.005 or row['spec']['friction']>.015;start=torch.load(B/f'task{tid:04d}/initial.pt',map_location='cpu',weights_only=False);p.new_task(tid);tr=run_trial(env,start,p,dict(task=tid,trial=0,kind='OOD_fixture'),fixture/f'task{tid:04d}',parity=True);before=p.history();assert before['action_chunks']>0
  reference=torch.load(B/f'task{tid:04d}/swap_t01.pt',map_location='cpu',weights_only=False)['initial_observation'];actual=copy.deepcopy(tr['initial_observation'])
  for x in [reference,actual]:
   for robot in x['robots_obs']:robot.pop('arm_qacc',None)
  equal(reference,actual)
  from probe import restore
  env.physics.model.geom_friction[:]=start['geom_friction'];restore(env,start);equal(before,p.history());checks.append(dict(task=tid,friction=row['spec']['friction'],controls=tr['controls'],native_status=tr['status'],matched_initial_observation=True,memory_retained_across_physical_reset=True,server_first_two_decisions_bit_exact=True));save(fixture/f'task{tid:04d}.pt',tr)
 assert state_hash(p.model)==mh and state_hash(p.norm)==nh;p.new_task(31);assert p.history()['action_chunks']==p.history()['image_chunks']==0
 write(R/'gmp_memory_ood_v187_preflight.json',dict(status='passed',fixtures=checks,fixture_starts=2,ID_tasks=0,optimizer_updates=0,strict_official_checkpoint_loading=True,memory_gate_enabled=True,original_action_decoder_and_smoothing=True,new_task_clear=True,global_weights_unchanged=True,checkpoint_sha256=CK_SHA,model_sha256=mh,normalizer_sha256=nh,sources=sources(),upstream_sources=upstream_files(),parent_OOD_manifest_sha256=sha(B/'manifest.json')))
finally:env.physics.free()
