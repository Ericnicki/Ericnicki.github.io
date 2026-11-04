# Degradation diagnosis and rescue, v117

Authorized by 2026-10-10 request to inspect degradation and test hindsight
consistency or other remedies. All compute runs on c1d3019784aa via compute.sh.
Frozen v100/v115 inputs are read only; new outputs have a separate namespace.

Diagnosis precedes fitting. Reuse the 12 development tasks solely as a diagnostic
set. Compare v100/v115, actor/writer swaps, per-trial memory reset, and half
acceleration. Q head swaps cannot change an otherwise fixed executed actor.
Measure native terminal XY return, success, last-three success and retention.

On v115 trial 1 and 4, first and last available contact queries, replay the exact
physical/controller/agent prefix from the initial snapshot. Change only one
acceleration request to [-.5,-.25,0,.25,.5] and subsequently continue v115 with
branch-specific causal memory. Also replay the original request to verify the
native endpoint and prefix. Q input remains feasible effective residual velocity.
Compare early-memory versus additional-other-trial-memory rankings of the same
actions against native branch returns. Later-memory Q is an unvalidated teacher,
not a privileged true current-policy return. No development labels used to fit.

Subsequent rescue variants and train/validation/confirmation splits will be
specified before fitting, following diagnosis. Retain real MC anchors for
hindsight relative-action consistency. Separate consistency agreement from
grounded discrimination and final closed-loop improvement. Fixed compute budgets,
whole-task uncertainty and censored outcomes must be reported. Negative results
are valid completion of a bounded experiment, not proof no method can work.

References checked: Bootstrapped Meta-Learning (ICLR 2022),
https://openreview.net/pdf?id=b-ny3x071E5 ; MAGE (NeurIPS 2020),
https://proceedings.neurips.cc/paper/2020/file/03255088ed63354a54e0e5ed957e9008-Paper.pdf .
The proposed hindsight relative-Q target is not an implementation of MAGE's
model-based TD action-gradient objective.
