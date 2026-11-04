# Completed and audited: 50% friction OOD, top four methods plus swap

2026-10-13. Evaluation only: **32 fresh OOD tasks ×12 trials ×five methods =1,920 judged outcomes**, zero censors, zero in-distribution tasks and zero training/optimizer updates. Original swap was added before evaluation launch, as requested. The four fixed correction endpoints were selected by the prior completed study's success and mean-return rankings, before inspecting OOD outcomes.

**Original swap has the highest low-friction success rate (13.64%). Joint and swap tie at high friction (37.50%). Joint has the highest mean native G in both bands, but the low-band return advantage is uncertain. All four corrections improve high-band mean G relative to swap under the reported unadjusted task-paired intervals; none establishes a success-rate improvement. OOD success remains weak.**

[Plan](../plans/RUN_PLAN_TTT_world_ood_top4_v185.md) · [Frozen original protocol](../../return-selector/ttt_world_ood_top4_v185/PROTOCOL.md) · [Frozen parallel supplement](../../return-selector/ttt_world_ood_top4_v186/PROTOCOL.md) · [Offline graphs](../../outputs/ttt_world_ood50_20261013/verified/ttt_world_ood_top4_v186_output/index.html) · [Native outcome CSV](../../outputs/ttt_world_ood50_20261013/verified/ttt_world_ood_top4_v186_output/native_outcomes.csv).

## OOD task definition and fixed policies

Low sliding friction **[0.0025,0.005)**; high **(0.015,0.0225]**, where0.0225=0.015+0.0075. Sixteen tasks per band. Task0 is exactly0.0025 and task16 exactly0.0225; the other fifteen frictions per band are uniform draws from the declared range. All exclude the trained friction interval[0.005,0.015]. Friction is stratification metadata, never an actor/world input. The native geometric friction arrays are checked against each requested coefficient and restored exactly for every trial.

All tasks are broad, with base speed drawn independently in[0.247,0.435]m/s; identical speed, physical initial snapshot and friction across all five methods and twelve trials of a task. No near-success calibration, BASE evaluation, action noise, outcome-based task selection, task replacement, new checkpoint selection or feature-scale tuning. Seeds: friction2185110, task2185200+taskID, speed2185300+taskID. The first two tasks of each band were selected for display before outcomes.

Global actor/critic/world/value weights and TRAIN feature scales remain fixed. Each method retains its ordinary test-time memory adaptation from real observations, including the original causal Transformer-TTT context; correction methods also update task-private physical W from actual executed-command blocks and observed successors. Memories persist through twelve physical trial resets and start fresh at a new task. These task-private writes are part of evaluation, with no slow model fitting or optimizer steps. Predicted transitions never become observation evidence.

Original controller unchanged: acceleration±0.5m/s², residual velocity±0.1m/s, dt0.1s, EE absoluteY[-0.85,0.25]; correction±0.05m/s² held eight controls. Native early resets, timeouts and control cutoffs disabled so every trial reaches native judgment. Original deployed checkpoint remains unchanged.

## Primary native outcomes: trials2–12

Each band has16×11=176slots per method; pooled352. Native G=`1-max(abs(stopXY-[.1,.15]))/.05`, successG>0; larger G is better. No in-distribution result is included here.

| Method | Low successes | Low rate | Low mean G | High successes | High rate | High mean G |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original swap | 24/176 | 13.64% | -4.518699 | 66/176 | 37.50% | -1.562406 |
| Joint shared-memory | 19/176 | 10.80% | -4.405777 | 66/176 | 37.50% | -1.337207 |
| Prediction-only shared-memory | 18/176 | 10.23% | -4.585557 | 64/176 | 36.36% | -1.378110 |
| Adaptive direct-world | 16/176 | 9.09% | -4.757187 | 63/176 | 35.80% | -1.392777 |
| Gradient-aligned Q | 15/176 | 8.52% | -4.724816 | 64/176 | 36.36% | -1.397244 |

## Task-paired differences versus original swap

Bootstrap10,000resamples, seed2185500, selecting whole matched tasks with all trial histories. Sixteen task pairs per band. Pooled sampling is stratified within the two bands with fixed equal weights. These are unadjusted95%intervals across ten method pairs; a barely positive pooled interval is not strong evidence of a uniform OOD advantage. Detailed success-rate intervals and all pairwise comparisons are in the [native result JSON](../../outputs/ttt_world_ood50_20261013/verified/ttt_world_ood_top4_v186_output/result.json).

| Method − swap | Low ΔG | Low95%CI | High ΔG | High95%CI |
| --- | ---: | --- | ---: | --- |
| Joint shared-memory | +0.112923 | [-0.199187,+0.439919] | +0.225199 | [+0.120039,+0.337416] |
| Prediction-only shared-memory | -0.066858 | [-0.380554,+0.238374] | +0.184296 | [+0.053768,+0.297376] |
| Adaptive direct-world | -0.238488 | [-0.675403,+0.170541] | +0.169629 | [+0.034420,+0.288222] |
| Gradient-aligned Q | -0.206117 | [-0.507411,+0.093915] | +0.165162 | [+0.035742,+0.281285] |

Joint−swap: low success difference−2.84percentage points, CI[−10.23,+2.84]; high difference0, CI[−2.27,+2.84]. Joint's high mean-return gain+.225199, CI[+.120039,+.337416], accompanies the same success count,66/176. The mean-return benefit is distinct from reaching the success threshold more often. Low return change+.112923 has CI[−.199187,+.439919], so a positive low-friction effect is unestablished.

## Pooled OOD summary and repeated trials

Both bands weighted equally; this summary contains only OOD tasks.

| Method | Pooled successes | Pooled rate | Mean G |
| --- | ---: | ---: | ---: |
| Original swap | 90/352 | 25.57% | -3.040553 |
| Joint shared-memory | 85/352 | 24.15% | -2.871492 |
| Prediction-only shared-memory | 82/352 | 23.30% | -2.981833 |
| Adaptive direct-world | 79/352 | 22.44% | -3.074982 |
| Gradient-aligned Q | 79/352 | 22.44% | -3.061030 |

Pooled joint−swap meanGgain+.169061, CI[+.001473,+.340626]; lower bound is close to zero and intervals are unadjusted. Pooled success difference−1.42percentage points, CI[−5.11,+1.70]. The other three pooled return intervals span zero.

Late trials7–12: low joint9.38%/G−4.491564, swap10.42%/−4.434419; high joint43.75%/−1.007874, swap42.71%/−1.231540. All five methods have zero trial1successes in both bands. High-friction performance improves across the early trial history and then plateaus; low-friction success stays unstable and low. This evaluation lacks a frozen-memory ablation, so these within-task changes do not isolate which memory mechanism causes improvement.

## Execution and audit evidence

Serialv185 completed six tasks/360policy outcomes plus five OOD fixtures. It was stopped at a complete task boundary for parallel execution, with budget365 and **zero interrupted policy attempts**. All accepted files were preserved and copied byte-exact. Supplementalv186 ran the26remaining tasks and full native audits in four disjoint processes, taskID modulo4. Task specs, checkpoints, controller, speeds and trial counts were unchanged. Source/output snapshots and the boundary-stop receipt remain archived; the old running status is a historical snapshot, not an unfinished accepted phase.

The full native audit passes **1,920trials/137,863controls/14,228independently differentiated physical-memory writes**, including exact retained-prefix hashes. It checks observed features, original causal contexts, actor choices, projected commands, global parameter invariance, matched physical resets, private W and native judged G. Independent review reconstructs all14,228actual supervised blocks/private-memory snapshots, then replays20native traces: boundary tasks0/16 ×five methods ×trials2/12, **1,642controls and maximum physical error0**. These replay checks validate recorded simulator trajectories; they do not claim learned physical forecasts are exact.

Twenty classic PNG/PDF/SVG/CSV graphs show two predetermined tasks per band/method, all twelve trials and final G annotations, using the established plotting function. Aggregate outcome graphs show low/high separately. Export audit checks18,105control points, every native G and trial boundary. All20embedded SVGs parse. Aggregate and joint/swap boundary PNGs were visually reviewed; annotations are readable. Viewer JavaScript syntax and all/joint-low/swap-high/reset filter cases pass. Interactive browser inspection was unavailable because this environment has no enabled browser surface.

[Local verification receipt](../../outputs/ttt_world_ood50_20261013/local_verification.json): **2,798files plus archive manifest** hash-verified, including both frozen source versions, raw data, plots, fixed parent checkpoints and receipts. ArchiveSHA256 `5246573ac936559f6a9e46f896ad60d3079dbd8bd8be20d8a1163535b47c7ddc`; manifestSHA256 `7ec33a202d258ba2b4f5e2a23136708e96ab7a2059aa77f53dbb192cfa91d865`.

All simulation, numerical analysis and scientific plotting ran on c1d3019784aa/173.0.75.3 through compute.sh. Owned serial45377/45390 and parallel45748 pipelines exited; no OOD phase remains queued. Zero in-distribution tasks, zero training updates, no deployment change.
