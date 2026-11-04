# Completed official GMP with memory: 50% friction OOD

2026-10-13. **32 matched OOD tasks ×12 trials =384 judged GMP outcomes; zero censors, zero training and zero in-distribution evaluation.** Official GMP with memory achieved **zero successes in either friction band** and lower mean native return than every previously evaluated method on these task snapshots. This is the original pretrained gated-diffusion policy, with frozen weights and learned memory gating retained.

[All six figures](../../outputs/official_gmp_memory_ood_20261013/verified/gmp_memory_ood_v187_output/index.html) · [Native outcomes](../../outputs/official_gmp_memory_ood_20261013/verified/gmp_memory_ood_v187_output/native_outcomes.csv) · [Full metrics and paired intervals](../../outputs/official_gmp_memory_ood_20261013/verified/gmp_memory_ood_v187_output/result.json) · [Plan](../plans/RUN_PLAN_official_GMP_memory_OOD_v187.md) · [Frozen production protocol](../../return-selector/gmp_memory_ood_v187/PROTOCOL.md).

## Native outcomes, trials2–12

Sixteen tasks per band,176primary slots per method. Native G=`1-max(abs(stopXY-[.1,.15]))/.05`; success requires G>0. All returns below are judged, with no missing slots. Previous five-method outcomes are reused byte-exactly from v186; no comparator simulation was rerun.

| Method | Low successes | Low success | Low mean G | High successes | High success | High mean G |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official GMP + memory | 0/176 | 0.00% | -7.749957 | 0/176 | 0.00% | -4.780563 |
| Original swap | 24/176 | 13.64% | -4.518699 | 66/176 | 37.50% | -1.562406 |
| Joint shared memory | 19/176 | 10.80% | -4.405777 | 66/176 | 37.50% | -1.337207 |
| Prediction only | 18/176 | 10.23% | -4.585557 | 64/176 | 36.36% | -1.378110 |
| Adaptive direct world | 16/176 | 9.09% | -4.757187 | 63/176 | 35.80% | -1.392777 |
| Gradient-aligned Q | 15/176 | 8.52% | -4.724816 | 64/176 | 36.36% | -1.397244 |

GMP pooled primary:0/352successes, mean G−6.265260. Including trial1:0/384successes, mean G−6.382909. Low/high all12means are−7.902880/−4.862938. Mean G improves relative to the first trial but does not improve further between T2–6 and T7–12: low−7.664651→−7.821045; high−4.659638→−4.881334. This is descriptive repeated-trial behavior, without a no-memory ablation.

## Paired differences

Task-cluster percentile bootstrap,10,000resamples, seed2187500; every task retains all eleven primary trials. These are unadjusted95%confidence intervals, not95%probabilities that a particular fixed effect lies inside the interval. All16tasks in each band are complete. Pooled resampling uses whole tasks without the previous report's fixed-band stratification; low/high comparisons are the main conclusions.

| GMP minus comparator | Low ΔG | Low95%CI | High ΔG | High95%CI |
| --- | ---: | --- | ---: | --- |
| Original swap | -3.231257 | [-5.284583, -1.112959] | -3.218157 | [-4.067458, -2.349166] |
| Joint shared memory | -3.344180 | [-5.402838, -1.165815] | -3.443356 | [-4.236507, -2.637282] |

Both band-specific return differences versus swap are negative throughout these intervals. Retain swap as the stronger observed success baseline on this OOD design; official GMP memory is not an outcome improvement here.

## Official policy and matched protocol

The cached [official GMP repository](https://github.com/real-stanford/gated-memory-policy) supplies `memmimic/push_cube_diffusion_gated.ckpt`, SHA256 `ff358946f65e559a8f8b677e0df2606f1ae1bf78221ad3b4d0c7ba505e3ebcc1`. Upstream commit `c23bb40566a9eec2f90a570deb67ec71868e0820`; HuggingFace revision `29be4b9b364d293ba399394fe72745ac5af78c6a`. Strict main-checkpoint loading includes the trained memory-gate parameters. The source is unchanged in the upstream inference/decoder files checked against Git. The275,025,693parameter policy uses original eight diffusion steps,16predicted/eight executed tokens and30history chunks. Normalization, image/proprio preprocessing, learned binary gate, Y-only action masking, fixed gripper and original decoder/smoothing remain in force.

Inputs are one third-person image and robot proprioception, with a task episode key for history; friction, terminal reward and direct object state are not policy inputs. There is no residual controller, speed program, actor/world/Q correction or invented Q curve. Per-task diffusion seed2187200+taskID continues across trials. Policy history clears at a new task and persists across all twelve physical resets; model/normalizer parameter hashes remain unchanged.

Same v186physical initial snapshots:16low[0.0025,0.005),16high(0.015,0.0225], including0.0025 and0.0225boundary tasks. This is a matched physical benchmark extension of official GMP's original six natural-reset trial benchmark. Native early teleport and20-second deadline are disabled for compatibility with the previous OOD comparison. An external120-second/1200-control failsafe was declared; all384trials judged before it, so no censor adjustments are needed. GMP chooses original absolute actions whereas the previous methods use residual controllers. A single diffusion seed is tested per task; this comparison does not isolate memory's causal contribution or establish general official-benchmark performance.

## Memory observations

History accumulates to its original30chunk limit and remains present across resets. Low band:2241decisions,2225with nonempty history,960with history attention enabled,944with nonempty history actually attended. High:1164decisions,1148with nonempty history,769with history attention enabled,753with nonempty history attended. These measurements confirm retained and used memory; they do not imply successful adaptation. The learned gate retains its original choice to suppress attention on other decisions.

[Performance curves](../../outputs/official_gmp_memory_ood_20261013/verified/gmp_memory_ood_v187_output/plots/ood_comparison.png) · [Memory diagnostics](../../outputs/official_gmp_memory_ood_20261013/verified/gmp_memory_ood_v187_output/plots/memory_diagnostics.png) · [Decision-level memory CSV](../../outputs/official_gmp_memory_ood_20261013/verified/gmp_memory_ood_v187_output/memory_decisions.csv).

## Audit and delivery

Two boundary OOD fixtures passed strict loading, bit-exact upstream serving inference parity for the first two decisions, matched physical initial observations, history retention across reset, fresh-task clearing and frozen parameters. Full accepted audit replays **3405model decisions and25858controls** across all384slots, checking recorded image/proprio inputs, RNG states, memory hashes, normalized/physical model outputs, all16decoded actions and requested/executed commands. All native outcomes and initial snapshot bindings pass. Eight independent physical command replays total500controls, **maximum physical error0**. Production, fixtures and physical replays use394starts in total. Inference audits and offline decoder checks add no physical starts.

Six scientific figures have PNG/SVG/PDF exports: aggregate native outcomes, memory diagnostics and four twelve-trial EE-Y/cube-Y trajectory figures. Tasks0,1,16,17were selected before outcomes; every trial has its true final G label. CSV trajectory exports are checked exactly against audited records. Representative low/high trajectories and both aggregate figures are visually inspected during local delivery verification. Standalone HTML embeds all six images without external dependencies. All computation, simulation and scientific plotting ran on c1d3019784aa through compute.sh.

Preserved failures: prototype `inspect.py` shadowed a standard-library import (zero inference/physics); v187reviewer used previous smoothed token instead of previous raw token; v188assumed disabled pose stayed constant after smoothing instead of incorporating the previous executed quaternion; v189reviewer shadowed a raw-task dictionary with an action array; v187report reused a nested CSV writer after closure. Production always invoked the unchanged upstream agent correctly. Frozen v190corrects the independent audit and v191corrects reporting; no policy trial was rerun or replaced. Failed sources/logs and partial report outputs remain in the archive.

Archive: [v191.tar.gz](../../outputs/official_gmp_memory_ood_20261013/v191.tar.gz),619members including the manifest. SHA256 `1f31b6db4b16fe3df639f51f9110255e5ec52ac806751ef3386587450420014e`; manifest SHA256 `6e75c048e60f19ec015fe508714cf9efb44fe9a97cf448c6e390c0062f2f742c`. The1.10GBofficial checkpoint binary remains cached on compute and is externally SHA/revision bound; the archive contains original inference sources, audited raw records and all export artifacts. Local hash verification and owned-process exit checks are recorded in [verification](../../outputs/official_gmp_memory_ood_20261013/local_verification.json).
