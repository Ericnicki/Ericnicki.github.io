# Fixed comparison v119

All fits start from v115 including Adam moments. Three arms x two sampler seeds
2033000/2033001, each1500 Q/750actor/150writer updates. LR/head1e-4,
writer1e-5,actor1e-5; clip1; inherited optimizers/decay. No tuning on development.
Real MC anchor uses all48 fresh deterministic TRAIN sequences, original balanced
8-task/phase sampler. Actor samples active queries separately, detached context.
Writer receives full causal BPTT only every10th Q update.

Auxiliary batch:8 distinct TRAIN tasks,4near4broad, uniform of each task's
trial1/4 branch queries. Five feasible actions. Teacher frozen at v115, additional
other-trial evidence excludes query trial; append the query's observed prefix.
Teacher outputs detached. Relative differences subtract zero-request candidate
(index2); normalization is /0.1 in normalized-Q units, coefficient0.1, SmoothL1.
MC arm coefficient0. Hindsight target uses later-memory relative Q. Grounded
uses native branch G/5 differences, same physical query and frozen-v115 causal
continuation. This controls schedule but grounded has true counterfactual labels.
Both learned auxiliary arms update Q and writer; actor updates use current Q.
Changing actor after fitting makes old targets policy-mismatched again; report it.

Snapshots500/1000/1500; only fixed1500 endpoints used in confirmation. Development
12 tasks can diagnose and compare all endpoints but never train/select a step.
Fresh confirmation:60 tasks,10 per friction-region x near/broad cell, seeds
2035000 onward; no recycling original development. Arms BASE,v100,v115, and all
six fitted endpoints, old-actor/new-writer swap and half-acceleration (11arms). Repeated6trials with frozen slow weights and causal
persistent memory. Near speed uses unchanged native calibration; broad uses IID
speed independent of friction. Native task-level return differences and last3
success, retention and tail damage reported, with whole-task bootstrap intervals.
This is a bounded rescue pilot (48 TRAIN tasks,2seeds), not a universal failure
claim or established TTT advantage over a trained recurrent baseline.
