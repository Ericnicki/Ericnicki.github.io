# Recovery of the 1,000 fresh-task continuation

Prepared and migrated on 2026-10-10 to user-provided compute c1d3019784aa /173.0.75.3; fixture and recovery must pass before training.
The original v109 source, manifests, raw data, failed log and checkpoints remain
immutable. This package fixes the inherited 3,000-start cap and reconstructs
the unsaved optimizer tail before continuing the same 1,000 sampled tasks.

## Preserved experiment contract

Keep the original `joint_1000_v109` task namespace and 202xxxx seeds: 500 near,
500 broad, 125 groups of four each, six trials/task. No additional cohort is
silently generated. Initialize from audited v100 Q76,300 / actor9,450 /
writer12,630; total phase targets remain +12,500Q / +6,250actor / +1,250writer.
Eight distinct replay tasks per update, single-Q MC G/5 with gamma1; Q:writer
10:1, actor every two Q updates. Head/writer/actor LR1e-4/1e-5/1e-5,
gradient norm cap1. Full causal writer BPTT and detached actor context.
No return/reward TTT inputs, RUDDER, settled-tail sampling change, or
actor-to-writer gradient change.

Controller unchanged: dt0.1, acceleration bound0.5m/s², residual-velocity
bound0.1m/s, midpoint integration, acceleration Gaussian sigma0.05m/s²
truncated at0.15m/s², native Y envelope[-0.85,0.25], bounded braking,
uncapped accumulated offset. Q/TTT action is executed effective residual
velocity, with base velocity and controller state in observation features.

## Recovery and provenance

1. Run `fixture.py` on authorized compute. In addition to existing causal,
   controller, gradient and moment preflight, test the actual budget helper:
   TRAIN starts3001 and6000 accepted;6001 rejected before budget mutation.
   Evaluation starts144 accepted;145 rejected before mutation.
2. Run `recover.py`. Validate all original source bindings, 500 raw task hashes,
   failure identity, 6,200 sequential log rows and full rolling checkpoint at
   Q6,000/actor3,000/writer600, groups60/tasks480.
3. Reconstruct only the already executed200Q/100actor/20writer optimizer
   operations with logged samples and restored RNG. Restrict replay to tasks
   available at each historical group. Require logged losses and sampler draws
   to match, and exact tensors against group60/61/62 snapshots. Do not count
   reconstruction as additional production updates or new collection.
4. Only after checks pass, create exclusive v115 output. Copy all500 completed
   task folders and63 network snapshots; preserve original failure/provenance
   under `recovery/`. Full recovered optimizer checkpoint has62 completed
   groups and496 replay tasks. The four already collected partial-group tasks
   496–499 are retained and reused when that group resumes.
5. Task500 failed before its first critical command. Preserve its rejected
   approach attempt separately, then restart the trial from its saved initial
   state with its unchanged broad base program. It contributes no training
   record/history from the aborted approach. The persisted original budget is
   3,000 starts, not3,001: failure happened before its budget write. Record
   one rejected critical-entry request separately, without calling it an
   executed push or silently dropping the attempt.
6. `train.py --resume`: collect the remaining500 tasks and finish63 collection
   groups (including the half-collected group), adding6,300Q /3,150actor /
   630writer beyond the recovered6,200/3,100/620. Restore all optimizer moments
   and sampler/RNG state; never start a fresh optimizer or fit from parent.
7. Independent full12,500-update native/optimizer audit, then matched BASE/new
   deterministic actor evaluation on the same12 development regression tasks
   and three low404/middle400/high401 Q/Y time plots. High401 is broad.
   Report native return/success and task-paired comparisons with prior v100;
   reused development performance does not establish fresh generalization.

`pipeline.py` owns recover → resume → audit → evaluate → report. The final
checkpoint targets remain Q88,800 / actor15,700 / writer13,880. Save atomically
around1,000Q updates or600s at group boundaries and at completion. Archive
recovery evidence along with the final sources/raw data/checkpoint/plots.

All science runs through `return-selector/compute.sh` on the authorized compute
node. The user supplied replacement hostname c1d3019784aa at the same IP; SSH and idle RTX4090D verified.
A replacement requires explicit node/IP information and versioned runtime /
hostname bindings before preflight; no training or plotting on the login node.
Estimated remaining training50–65min, audit/evaluation/plots15–20min after
access and recovery verification. Inspect near expected finish, not frequently.

Migration addendum: v113 preflight stopped at import before physics/optimization, because the legacy probe had a92c316f35f5f import guard. v115 copies official.py/probe.py beside this source; only the import directory and probe hostname guard change. The original frozen dependencies remain unchanged and are verified through inherited source bindings. Runtime v2 guard matches c1d3019784aa.
