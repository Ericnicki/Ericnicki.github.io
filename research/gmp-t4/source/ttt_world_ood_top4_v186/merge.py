"""Combine disjoint complete workers; enforce the unchanged native budget."""
from common import *
from collect import specs,ARMS,parents
import sys
setup();stage=sys.argv[1];manifest=json.loads((O/'manifest.json').read_text());assert sources()==manifest['sources']
if stage=='collect':
 rows=json.loads((O/'retained_summary.json').read_text());budget=json.loads((OLD/'budget.json').read_text());new_starts=0
 for worker in range(4):
  complete=json.loads((O/f'worker{worker}_complete.json').read_text());assert complete['status']=='completed' and complete['sources']==sources();wr=json.loads((O/f'worker{worker}_summary.json').read_text());assert len(wr)==complete['tasks'];rows.extend(wr);new_starts+=complete['policy_starts'];wb=json.loads((O/f'worker{worker}/budget.json').read_text());assert wb['trial_starts']==complete['policy_starts'];budget['trial_starts']+=wb['trial_starts'];budget['entries'].extend(wb['entries'])
 rows.sort(key=lambda row:row['spec']['task']);assert [r['spec'] for r in rows]==specs();assert len({r['spec']['task'] for r in rows})==32;assert budget['trial_starts']==1925 and new_starts==manifest['new_policy_slots'];assert parents()==manifest['parents'];write(O/'summary.json',rows);write(O/'budget.json',budget);write(O/'collection_complete.json',dict(status='completed',tasks=32,policy_starts=1920,retained_policy_starts=manifest['retained_policy_slots'],new_policy_starts=new_starts,fixture_starts=5,ID_tasks=0,optimizer_updates=0,interrupted_policy_attempts=0,sources=sources()));write(O/'status.json',dict(status='running',stage='parallel_native_audit',tasks=32,policy_starts=1920))
elif stage=='audit':
 audits=[json.loads((O/f'audit_worker{worker}.json').read_text()) for worker in range(4)];assert all(a['status']=='passed' and a['sources']==sources() and a['global_weights_unchanged'] for a in audits);bindings={}
 for a in audits:
  assert not set(bindings).intersection(a['raw_bindings']);bindings.update(a['raw_bindings'])
 assert sum(a['trials'] for a in audits)==1920
 for rel,digest in manifest['retained_raw_bindings'].items():assert sha(R/rel)==digest;local=O/Path(rel).relative_to(OLD.relative_to(R));assert sha(local)==digest
 write(O/'audit.json',dict(status='passed',trials=1920,controls=sum(a['controls'] for a in audits),independently_differentiated_world_writes=sum(a['independently_differentiated_world_writes'] for a in audits),all_native_features_context_actor_projection_and_private_W_replayed=True,matched_initials=True,global_weights_unchanged=True,retained_prefix_byte_exact=True,interrupted_policy_attempts=0,ID_tasks=0,optimizer_updates=0,censors=0,raw_bindings=bindings,worker_audit_sha256=[sha(O/f'audit_worker{w}.json') for w in range(4)],sources=sources()));write(O/'status.json',dict(status='running',stage='independent_review'))
else:raise ValueError(stage)
print('PARALLEL_OOD_MERGED',stage,flush=True)
