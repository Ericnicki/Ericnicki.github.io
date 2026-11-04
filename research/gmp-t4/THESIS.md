# Completed TTT world-gradient research — v169–v184

Completed on 2026-10-13: separate supervised physical TTT, native macro-action Jacobians, grounded successor value, conservative direct actor, MC versus physical-gradient-aligned critic/actor, and the conditional shared-memory joint outer objective. All fits, native comparisons, independent replays and delivered artifacts passed their required audits. Information-gain/probing remains the explicitly later extension in the supplied proposal.

The strongest experimental result is the joint shared-memory correction: **293/330 successes (88.79%) versus 238/330 (72.12%)** for the original swap, with mean native-return gain **+.174513, 95% CI [+.048967,+.309907]**. It also improves over the equal-budget prediction-only continuation. This is evidence on a reused 30-task benchmark, not an independent blind replication. Its physical prediction errors worsen, so the result does not establish uniformly accurate physical gradients. The original deployed swap remains unchanged; a separately audited experimental inference bundle is available.

[Completion plan](../plans/RESEARCH_COMPLETION_TTT_world_gradient_v169.md) · [Requirement audit](../notes/NOTE_TTT_world_gradient_completion_audit_v181.md) · [Implementation note](../notes/NOTE_TTT_world_gradient_implementation_v170_v174.md) · [Seven-method trajectory viewer](../../outputs/ttt_world_macro_20261013/verified/ttt_world_gradient_final_v181_output/index.html).

## Actual native outcomes

Thirty matched tasks, five per low/middle/high friction × near/broad cell, twelve trials per method. The initial direct-world experiment used fresh prospective tasks; later Q and shared-memory comparisons reused those declared tasks and initial snapshots. All seven methods have their own causal history and task-private memory. Physical state resets each trial; memories persist within each task. All global weights are frozen during evaluation. The original baseline is retained byte-exact and counted once: **2,520 unique judged outcomes, zero censors**.

Primary outcome excludes trial 1: 30 tasks ×11 trials =330 slots per arm. Native G is `1 − max(abs(stopXY − [.1,.15]))/.05`; success means G>0. Higher is better.

| Method | Successes | Success rate | Mean G |
| --- | ---: | ---: | ---: |
| Original swap | 238/330 | 72.12% | +.238516 |
| Direct actor, frozen physical memory | 211/330 | 63.94% | +.065139 |
| Direct actor, adaptive physical memory | 252/330 | 76.36% | +.290528 |
| MC-only macro-Q actor | 168/330 | 50.91% | −.071397 |
| Physical-gradient-aligned macro-Q actor | 248/330 | 75.15% | +.249880 |
| Shared memory, prediction-only continuation | 266/330 | 80.61% | +.305940 |
| Shared memory, joint prediction/value outer loss | 293/330 | 88.79% | +.413030 |

Consolidated v181 task-paired bootstrap intervals use 10,000 resamples, seed2176000. Each resample selects entire matched tasks with their trial histories, preserving within-task dependence. These exploratory intervals are not adjusted for multiple comparisons or benchmark reuse. Earlier reports retain their original bootstrap seeds and slightly different interval endpoints.

| Comparison | Mean ΔG | 95% CI | Interpretation on these tasks |
| --- | ---: | --- | --- |
| Frozen − swap | −.173377 | [−.274792,−.085431] | Regression |
| Adaptive − frozen | +.225389 | [+.069040,+.403881] | Useful physical-memory adaptation |
| Adaptive − swap | +.052012 | [−.072463,+.179997] | Anchor gain uncertain |
| MC − swap | −.309914 | [−.416945,−.190855] | Regression |
| Aligned − MC | +.321277 | [+.154402,+.483732] | Alignment rescues the MC correction |
| Aligned − swap | +.011363 | [−.127854,+.168882] | Anchor gain uncertain |
| Aligned − adaptive | −.040648 | [−.122197,+.028457] | No established difference |
| Prediction-only − swap | +.067424 | [−.051822,+.183965] | Return gain uncertain |
| Joint − prediction-only | +.107089 | [+.038466,+.179775] | Joint objective improves equal-budget control |
| Joint − adaptive | +.122502 | [+.038568,+.204918] | Better than the initial direct correction |
| Joint − swap | +.174513 | [+.048967,+.309907] | Supported exploratory benchmark gain |

Joint − swap success gain is **+16.67 percentage points**, CI[+8.79,+25.15]; joint − prediction-only is +8.18 points, CI[+3.63,+13.33]. Late trials7–12: joint89.44% success/G+.398670 versus swap65.56%/+.178783. Trial12: joint80% versus swap36.67%. These are observed closed-loop outcomes, separate from command diversity or fitted gradient agreement. Each friction/cohort cell contains only five tasks; aggregate improvement is not a guarantee for every task or friction.

A **95% confidence interval (CI)** describes uncertainty in the estimated average effect. Under repeated comparable sampling, this interval procedure aims to cover the underlying effect about95% of the time. It is not a95% success rate, nor a95% guarantee for a particular future task. An interval spanning zero leaves the effect's sign unresolved by this analysis.

## Implemented gradient and memory path

The experimental writer consumes current observed state22, original causal context32, eight actually executed commands×4, command masks8 and horizon1. A direct macro reader consumes state22/context32, exact first projected command4, held correction1 and horizon1. Separate encoders read the same task-private W32×22. Fast writes are supervised by actual next observations, once an executed block completes; valid partial blocks flush at trial end, excluding the terminal command whose successor is unavailable. A fresh task restores initial W. Predicted transitions never become observation evidence.

The macro action is a scalar acceleration correction `.05*tanh`, held for eight controls around the reactive frozen original actor. The unchanged native controller enforces acceleration±.5m/s², residual velocity±.1m/s, dt.1s and absoluteY[−.85,.25]. Candidate first-command projection is differentiated explicitly. Native early resets/time/control cutoffs are disabled for the policy comparisons, so every evaluated trial reaches native judgment.

The successor value takes physical successor12, causal context32 and known time1; it predicts G/5 and signed terminal XY/.05. Actor learning maximizes the geometric goal score derived from signed landing errors, with2δ² anchor penalty. World, value, causal history and private W are fixed during candidate autograd; the actor's memory feature is detached. Physical TTT still adapts from actual observations during deployment without slow actor/critic retraining.

The MC critic uses78 inputs: context32, state22, effective first dV1, held correction1, detached memory belief22. Aligned training adds `.05 MSE(∂Q/∂δ, detached ∂Goal(V(F_W))/∂δ /5)` to the MC loss. This is MAGE-inspired gradient distillation, not the exact MAGE TD objective. The teacher does not receive critic gradients.

The shared experiment was declared after inspecting the complete direct-control outcomes, conditional on audited useful adaptive-versus-frozen performance. Both arms restore identical v170 world parameters/AdamW moments and receive1,000 extra world updates plus1,000 fresh actor updates. Joint adds `.1 Huber(G/5) + .05 Huber(signedXY/.05)` through the frozen value head on predicted successors. Prediction-only controls the added training budget. Both correction actors and dynamics read the shared physical W; the original anchor's Transformer-TTT architecture is preserved. The local writer remains supervised by actual state transitions.

## Physical evidence and limitations

[v169 native data](RESULTS_TTT_world_macro_data_v169.md):72tasks,48TRAIN/24task-heldout,576matched±.05 macro-action pairs spanning precontact, first/middle contact and release windows;1,584production starts plus3fixtures,101,033controls. Three production censors are preserved. Four heldout fit pairs are excluded because task469/trial2 lacks the retained center outcome; valid physical ±branches remain in native derivative evidence. The fit uses572pairs. This is heldout from the new fits, not globally unseen historical tasks.

[v170 fit](RESULTS_TTT_world_macro_fit_v170.md):2,000world/5,000value/1,000direct-actor updates. Adaptive cube-Y MAE falls20.169→10.743mm; paired task-cluster CI for the change[−13.098,−5.814]mm. Goal-gradient sign rises48.18→65.69% on137nondegenerate heldout pairs, change CI[+6.06,+29.41]percentage points. Cube-direction sign is135/137(98.54%) for both. Free-G calibration does not improve. All47valid heldout release queries have native zero cube/G action authority; false predicted slopes remain explicit.

[v174 critic fits](RESULTS_TTT_world_gradient_q_v172_v174.md):3,000critic/1,000actor updates per MC/aligned arm. Aligned sign60.58→64.96% does not have a supported task-cluster improvement; physical teacher65.69%. Slope MAE6.331→8.084 and G MAE.559→.615 worsen, despite native rescue relative to MC.

[v178 shared fit and v180 review](RESULTS_TTT_world_shared_v178_v180.md):prediction-only versus joint heldout cube-Y MAE9.321/14.841mm, goal-value MAE.895/.547, goal-gradient sign64.96/75.91%. On their own native trajectories, cube-Y MAE6.916/17.520mm and continuation-G error.911/.475. These on-policy errors compare different visited states and continuation policies; they are not a matched-state physics estimate.

[Final model Jacobian figure v184](../../outputs/ttt_world_macro_20261013/verified/ttt_world_shared_jacobian_v184_output/final_shared_jacobian.png):both final models checked on all188heldout queries, reconstructed causal W, fixed candidate W, autograd versus smooth model finite differences±.001 and native secants±.05. All188model finite-difference checks pass per arm; maximum errors1.144e−4/9.868e−5. Cube-sign remains135/137 each, but cube-slope MAE worsens30.924→37.290mm/(m/s²). On51native-zero cube queries, false predicted absolute slopes worsen21.928→34.331mm/(m/s²). Goal-slope MAE improves8.750→7.826. Thus numerical derivative correctness and high nonzero cube-sign agreement do **not** establish physical gradient fidelity everywhere. The joint model is more useful for this control objective while distorting physical forecasts. Probing remains premature without further authority/coverage validation.

## Audits and deliverables

Every fixed fit endpoint has exact optimizer/sample/loss/tensor/moment/sampler replay. Fast-write gradients are independently differentiated; all native state, action projection, causal context, actor decisions, private W, task/trial resets and judged G are checked. Evaluation global weights remain unchanged. Independent v175 review reconstructs11,710observed blocks/8,790activewrites; v180 reconstructs5,913more blocks/writes. Independent native command replay covers60+24traces and3,616+1,455controls, with **zero physical error**. These review replays are separate from the2,520policy-outcome slots.

The final viewer includes42classic graphs, one method per graph, all12trials, using the established native plotting function. BlueEEY/greenCubeY/orangeoriginal-anchorQ/graynativeG; every final G is annotated. OrangeQ is the unchanged anchor diagnostic, whereas correction decisions use world/value or macro-Q. PNG/PDF/SVG/CSV exports audit30,496control points and all native G/trial boundaries. Representative figures were visually reviewed; all42SVGs parse. Browser automation was unavailable in this environment; viewer JavaScript syntax and all/combined/reset filter logic were checked in the JavaScript runtime.

[Experimental single inference checkpoint](../../outputs/ttt_world_macro_20261013/verified/ttt_world_gradient_inference_v183_output/joint_inference.pt) · [Runtime API](../../return-selector/ttt_world_gradient_inference_v183/online.py) · [Inference protocol](../../return-selector/ttt_world_gradient_inference_v183/PROTOCOL.md). Export v183 reproduces all **732recorded joint macro decisions** over30tasks/360trials exactly: anchor acceleration, held correction, belief, score and W hash; maximum inference error0 and candidate W unchanged. This is export parity, not a new independent physical evaluation. CheckpointSHA256 `631bf08510325b2a1e62eefaa62bf43a025e119c8f5224739e62ace4a9e101c6`.

Preserved failures: v172 retained its completed MC critic but failed an actor assertion due to stale critic gradient buffers; v174 explicitly clears buffers before freezing and audits the corrected continuation. v182 export compared random unused legacy auxiliary layers absent from the original checkpoint; v183 gives those unused layers deterministic zeros, verifies every original saved context/Q/actor weight unchanged, and passes decision parity. Neither failed artifact is accepted as final; their sources/logs remain archived.

[Local verification receipt](../../outputs/ttt_world_macro_20261013/local_verification.json) binds the locally verified data/fit, final native/review, uncertainty, inference and final-Jacobian packages. The final archive contains4,314hashed files plus its manifest; raw v169data are in the separate verified data/fit archive. Current frozen local sources match145archived source files.

| Archive | SHA256 |
| --- | --- |
| v169/v170data+fit | `800172ce73fe12988be2e3dff20914623ed6e431b6629fb6b0e1d8abffb04ff7` |
| v181final native/review package | `d2261a88da5a006618f82a2396b625e5af1a9db725adb9f5ecdb79a3eab6ff41` |
| v183inference/failure-preservation package | `5ef557c381a49d24c15c803f14f1c0b8149c326215fc3e4a1b096df18b3d41b2` |

All scientific computation ran on c1d3019784aa/173.0.75.3 through `return-selector/compute.sh`. Local work was artifact verification, display checks and documentation. All owned world-gradient pipeline parents exited; no further phase is queued. Original deployed checkpointSHA256 `eb992367ed10149a93284cbfb7c1918d5f1970fba924bfed107db48ae10c5599` is preserved. Independent blind task/OOD validation and authority-aware physical-model repair are reasonable future research; they were not part of this completed controlled comparison.
