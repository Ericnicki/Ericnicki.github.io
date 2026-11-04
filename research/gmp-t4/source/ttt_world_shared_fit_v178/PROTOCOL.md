# v178 exploratory shared physical memory / joint outer objective

Conditional follow-up from the supplied proposal. The v171 outcome arrays have
been inspected while its full audit runs; this experiment is exploratory and
not blind confirmatory. Require passing complete native audit, positive lower
95% task-paired CI for adaptive-minus-frozen nativeG and nonnegative mean change
against originalswap. This establishes a useful isolated adaptation effect,
not a supported gain over the original swap. If unsupported, record skipped.
The original anchor and original TransformerTTT memory remain frozen.

Both new correction actors and dynamics access the same physical32×22W. This
preserves the deployed anchor's original context while testing the proposed
joint prediction/value-related outer objective for the shared experimental
memory. Fast writes remain real-observation supervised only. No reward/friction
inference inputs, hypothetical writes, or actor/critic slow inference updates.

Two arms initialized from exact v170world/value endpoint and world optimizer
moments: prediction_only and joint. Each1000extra world meta steps,12distinct
TRAINtasks/querybatch, original weightedphysical/supervised/contrast loss.
Joint adds .1 smoothL1(G/5 prediction,target) + .05 smoothL1(signedXY prediction,
nativeXY/.05) on predicted8-control successors under frozen v170Value. This
outer goal signal can change world encoders/initialW/step size; it does not enter
the local supervised writer. Native branch outcomes on48TRAINonly. Both restore
AdamWmoments and use continuationLR5e-5,wd1e-4,clip1. No value updates; fixed1000
endpoint, no heldout tuning. Seed2178000, sampler2178100 eacharm.

Eacharm then trains freshzeroCorrection,1000steps,24distinctTRAINtasks, Adam1e-4,
clip1, directworldgoal gradient with2δ²penalty. Own TRAIN-only belief std floor.001,
clip5, no centering. World/value/W frozen duringcandidate autograd. No frozen-W
mixture: subsequent controlled comparison uses adaptiveW only for both.
Actorinit2178200, sampler2178300; samequery schedule, ±.05heldcorrection.
Exactreplay ofboth1000world+1000actor optimizer trajectories/moments/samples and
all privateW/scale features, maskedcausality and independentteacher routing.
Heldout physical/valuecalibration/phasegradient fidelity measured withouttuning.

Subsequent nativev179 reuses30v171tasks/speeds/snapshots,12trials/newarm (720starts)
plus2fixture; originalswap retained360slots exactly. Prediction-only continued
fit controls the additionaltraining budget. Report nativejoint-minus-control,
bothminusoriginaladaptive/baseline, raw traces/authority andphysicsprediction
tradeoffs. No automaticpromotion or assumedpositive result. Sharedmemory result
is for the correction policy, not modification of originalTTTanchorfastweights.
Information-gain/probing remains explicitly later inoriginalproposal.
