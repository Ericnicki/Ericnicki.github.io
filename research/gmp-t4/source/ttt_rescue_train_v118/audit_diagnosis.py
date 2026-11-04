from audit_results import *
setup();counts=collections.Counter()
for p in sorted(O.glob('task*/*.pt')):
 obj=torch.load(p,map_location='cpu',weights_only=False);trials=obj['trials'] if 'trials' in obj else [obj]
 for t in trials:counts['controls']+=check_trial(t);counts['trials']+=1;counts[t['status']]+=1
write(O/'native_audit.json',dict(status='passed',counts=dict(counts),source_sha256=sha(Path(__file__)),validator_sha256=sha(Path(__file__).with_name('audit_results.py'))));print(dict(counts))
