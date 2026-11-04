from review import A
from audit_results import *
setup();d=json.loads((OUT/'metrics.json').read_text());rows=json.loads((CONF/'summary.json').read_text());lines=[]
labels={'base':'BASE','old':'v100','new':'v115','old_actor':'Old actor / new writer','half':'Half acceleration','mc_2033000':'MC seed 0','mc_2033001':'MC seed 1','hindsight_2033000':'Hindsight seed 0','hindsight_2033001':'Hindsight seed 1','grounded_2033000':'Grounded seed 0','grounded_2033001':'Grounded seed 1'}
lines+=['| Arm | Successes / 360 attempts | Judged | Native mean G | Last-three successful tasks / 60 | Gain vs v115 [95% CI] |','|---|---:|---:|---:|---:|---:|']
for key,label in labels.items():
 z=d['all'][key];g=z['gain_new'];lines.append(f"| {label} | {z['successes']} | {z['judged']} | {z['mean_return']:.3f} | {z['last3_all']} | {g['mean']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")
lines+=['','| Fit arm | Gain over matching MC [95% CI] | Gain with censored=-10 [95% CI] |','|---|---:|---:|']
for key in labels:
 if key.startswith(('hindsight','grounded')):
  g=d['all'][key]['gain_matched_mc'];p=d['all'][key]['censor_penalty_sensitivity']['-10.0']['gain_matched_mc'];lines.append(f"| {labels[key]} | {g['mean']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] | {p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] |")
lines+=['','| Arm | Low success /120 | Low mean G | Middle success /120 | High success /120 | Failure after earlier success (tasks) |','|---|---:|---:|---:|---:|---:|']
for key,label in labels.items():lines.append(f"| {label} | {d['low'][key]['successes']} | {d['low'][key]['mean_return']:.3f} | {d['middle'][key]['successes']} | {d['high'][key]['successes']} | {d['all'][key]['lost_success']} |")
(OUT/'tables.md').write_text('\n'.join(lines)+'\n')
write(OUT/'statistics_binding.json',dict(metrics_sha256=sha(OUT/'metrics.json'),confirmation_summary_sha256=sha(CONF/'summary.json'),statistics_source_sha256=sha(Path(__file__).with_name('statistics_v124.py')),wrapper_source_sha256=sha(Path(__file__).with_name('review_complete.py')),tables_source_sha256=sha(Path(__file__)),bootstrap='10000 whole-task replicates, within friction/cohort strata; pointwise intervals, no multiple-comparison adjustment',censoring='all attempts in success; observed-native conditional means plus complete-task and -5/-10 hypothetical-penalty sensitivity'))
print((OUT/'tables.md').read_text())
