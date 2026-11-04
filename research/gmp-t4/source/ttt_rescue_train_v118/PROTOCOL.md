# Fixed rescue experiment following v117 diagnosis

Use 48 existing v115 TRAIN tasks, 8 per friction-region/cohort cell, selection
seed2032000, unrelated to diagnostic outcomes. Collect six deterministic v115
trials and one median-contact one-control branch query in trials1 and4; five
acceleration candidates and factual replay. Reuse simulator snapshots and
programs. Labels are native current-policy continuation returns. Query current
state is fixed; branch memories subsequently evolve causally. Censored outcomes
are retained and excluded from supervised terminal-return losses.

Before fits: compare MC continuation against detached later-memory relative-Q
consistency with the same real MC anchor and actor schedule. A grounded contrast
arm uses branch-return differences. Exact fitting coefficients, seeds, budgets
and fresh confirmation panel will be frozen in a fit protocol before launch.
No development/test task may supply optimizer labels. All initial parent
checkpoints, architecture, action projection and native reward remain unchanged.
