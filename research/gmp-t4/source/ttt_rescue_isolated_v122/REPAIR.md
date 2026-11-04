# Isolated optimizer repair v122

The v119 postfit audit rejected comparison integrity. PyTorch load_state_dict
retained CPU scalar step tensor aliases into the shared parent optimizer dictionary.
After each arm, the next arm inherited incremented step counters despite fresh
parent weights and moment values. Saved counter sets verify the drift. Preserve
all original sources,logs,weights and incomplete confirmation output.

Pause confirmation worker PIDs1718–1721 before using its results. Re-run the same
six fixed fits in a new namespace, deep-copying each parent optimizer state before
load. Assert every parent scalar remains88800Q/13880writer/15700actor before and
after each arm, and every endpoint is90300/14030/16450. Same seeds,data,losses,
learning rates and budgets; no result-dependent objective or model selection.

Compare intended-fit tensors to preserved v119 endpoints. Existing physical
results can only be reused for fitted arms if every model/actor tensor is bitwise
identical; otherwise run corrected endpoints on the same fresh task specifications
in a new output directory. Do not claim exact equivalence merely because errors
are small. All discarded/replayed computation and interrupted starts are reported.
