"""Bind unchanged OOD design and byte-exact completed prefix; no new fixtures."""
import shutil
from common import *
from collect import specs,parents,ARMS,LABELS
setup(2185100);assert not O.exists();stop=json.loads((R/'ttt_world_ood_top4_v185_boundary_stop.json').read_text());assert stop['status']=='clean_task_boundary_stop' and stop['interrupted_policy_attempts']==0
pf=json.loads((R/'ttt_world_ood_top4_v185_preflight.json').read_text());assert pf['status']=='passed' and pf['parents']==parents();oldmanifest=json.loads((OLD/'manifest.json').read_text());assert oldmanifest['specs']==specs() and oldmanifest['arms']==ARMS
rows=json.loads((OLD/'summary.json').read_text());assert len(rows)==stop['completed_tasks'] and json.loads((OLD/'budget.json').read_text())['trial_starts']==5+60*len(rows)
O.mkdir();shutil.copytree(S,O/'source',ignore=shutil.ignore_patterns('__pycache__','._*'));bindings={}
for row in rows:
 tid=row['spec']['task'];assert row['spec']==specs()[tid];assert all(len(row['arms'][arm]['returns'])==12 for arm in ARMS);shutil.copytree(OLD/f'task{tid:04d}',O/f'task{tid:04d}')
 for arm in ARMS:assert sha(O/f'task{tid:04d}/{arm}.pt')==row['arms'][arm]['sha256']
 for path in (OLD/f'task{tid:04d}').rglob('*.pt'):bindings[str(path.relative_to(R))]=sha(path)
write(O/'retained_summary.json',rows);write(O/'manifest.json',dict(specs=specs(),arms=ARMS,labels=LABELS,policy_slots=1920,retained_task_ids=[r['spec']['task'] for r in rows],retained_policy_slots=60*len(rows),new_policy_slots=1920-60*len(rows),fixture_starts=5,new_fixture_starts=0,independent_replay_cap=20,ID_tasks=0,optimizer_updates=0,primary='T2–12',workers=4,training_range=[.005,.015],bands={'low':[.0025,.005],'high':[.015,.0225]},base_speed_uniform=[.247,.435],sources=sources(),parents=parents(),retained_raw_bindings=bindings,old_manifest_sha256=sha(OLD/'manifest.json'),old_preflight_sha256=sha(R/'ttt_world_ood_top4_v185_preflight.json'),boundary_stop_sha256=sha(R/'ttt_world_ood_top4_v185_boundary_stop.json'),global_weights_frozen=True,standard_real_observation_task_private_adaptation=True,sampling_and_policy_unchanged=True,interrupted_policy_attempts=0));write(O/'status.json',dict(status='running',stage='parallel_native_collection',retained_tasks=len(rows),retained_policy_slots=60*len(rows)));print('PARALLEL_OOD_PREPARED',len(rows),flush=True)
