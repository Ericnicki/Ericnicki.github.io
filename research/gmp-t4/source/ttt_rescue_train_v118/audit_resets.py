from audit_results import *
import hashlib
setup();assert json.loads((CONF/'complete.json').read_text())['status']=='completed';rows=json.loads((CONF/'summary.json').read_text());checked=0
for row in rows:
 folder=CONF/f'task{row["spec"]["task"]:04d}';initial=torch.load(folder/'initial.pt',weights_only=False);fsha=hashlib.sha256(initial['geom_friction'].tobytes()).hexdigest();reference=None
 for arm in row['arms']:
  ts=torch.load(folder/f'{arm}.pt',weights_only=False)['trials']
  for t in ts:
   assert t['restored_friction_sha256']==fsha
   state=t['records'][0]['state']
   if reference is None:reference=state
   else:np.testing.assert_array_equal(state,reference)
   assert t['records'][0]['current_observed_events']==0
   assert t['records'][0]['residual_velocity_before_m_s']==0 and t['records'][0]['offset_before_m']==0
   checked+=1
  assert [t['records'][0]['prior_trial_count'] for t in ts]==list(range(6))
 assert row['spec']['partition']=='confirmation' and row['spec']['namespace']=='rescue_confirm_v120'
write(CONF/'reset_audit.json',dict(status='passed',trial_resets=checked,same_initial_features=True,same_friction_per_task=True,controller_reset=True,persistent_memory_trial_counts=True,source_sha256=sha(Path(__file__))));print('RESET_AUDIT',checked)
