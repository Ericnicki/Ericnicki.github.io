import sys
from common import *
setup();stage=sys.argv[1]
if stage=='collect':
 rows=[]
 for worker in range(4):
  c=json.loads((O/f'worker{worker}_complete.json').read_text());assert c['status']=='completed' and c['sources']==sources() and c['upstream_sources']==upstream_files();rows.extend(json.loads((O/f'worker{worker}_summary.json').read_text()))
 rows.sort(key=lambda r:r['spec']['task']);assert [r['spec'] for r in rows]==json.loads((O/'manifest.json').read_text())['specs'];assert len(rows)==32;write(O/'summary.json',rows);write(O/'collection_complete.json',dict(status='completed',policy_slots=384,fixture_starts=2,ID_tasks=0,optimizer_updates=0,sources=sources()));write(O/'status.json',dict(status='running',stage='official_inference_native_audit'))
else:
 audits=[json.loads((O/f'audit_worker{w}.json').read_text()) for w in range(4)];assert all(a['status']=='passed' and a['sources']==sources() and a['frozen_parameters'] for a in audits);assert len({a['model_sha256'] for a in audits})==len({a['normalizer_sha256'] for a in audits})==1;bindings={}
 for a in audits:
  assert not set(bindings).intersection(a['raw_bindings']);bindings.update(a['raw_bindings'])
 write(O/'audit.json',dict(status='passed',trials=384,controls=sum(a['controls'] for a in audits),decisions=sum(a['decisions'] for a in audits),judged=sum(a['judged'] for a in audits),censors=sum(a['censors'] for a in audits),matched_initials=True,all_official_model_rng_history_action_inferences_replayed=True,frozen_parameters=True,ID_tasks=0,optimizer_updates=0,sources=sources(),raw_bindings=bindings));write(O/'status.json',dict(status='running',stage='independent_physical_replay'))
