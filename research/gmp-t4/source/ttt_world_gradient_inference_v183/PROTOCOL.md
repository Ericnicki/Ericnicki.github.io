# v182 audited experimental joint inference export

No new physics, fitting or checkpoint selection. Export the already-audited v178
joint world/value/correction plus original frozen v122swap model/actor and fixed
TRAINbeliefscale. Require passingv179native andv180independent audits. Original
deployed asset remains unchanged. Include schema, model/actor/source/audit hashes,
native-controller constants and causal-observed-writer contract.

Online API is copied unchanged from actually evaluatedv179; only constructor
loads the single bundled checkpoint and creates original causalMemory. WorldW
is taskprivate; actor/globalmodels frozen. Native loop: state/context from real
observation, arrival(state), originalanchoracceleration, choose(...), native
Controller.step, record(actualrecord), trial-endflush excludingterminalaction.
Memory observes state22+executedDV/.5 andend_trial. Instantiate new Online at
newtask, preserve it acrossphysicaltrialresets. No reward/friction inferenceinput.
The initial/context/privateweights from bundle must exactly equal all parent
models. Verify bundled inference at every recordedjoint macrodecision over30
matchedtasks/12trials with actualcausal privateW snapshot and unchangedcandidateW;
anchor acceleration, correction, belief and goal-score parity. No rerunphysics.

Experimental benchmark improvement is not independentblindreplication; export
is reviewable and usable, not an automaticdeployment. Originalswap remains
currentdeployment. Archive checkpoint/source/parityreceipt; verify localhashes.

Correction supplement: v182's full-object anchor hash compared independently
initialized legacy unused auxiliary hidden layers. They are absent from the
original checkpoint and their zero final layers yield identicallyzero output;
they are not used by the deployed context/Q/actor. Preservefailedv182source/bundle/
log. v183bundles deterministiczero unusedauxiliaryweights and verifies every
original saved context/Qweight andactorweight exactly unchanged, then checks
allactual native macrodecisioninference outputs. No physical experiment changed.
Schema differs, failedexport neveraccepted asverified. CandidateworldW unchanged.
