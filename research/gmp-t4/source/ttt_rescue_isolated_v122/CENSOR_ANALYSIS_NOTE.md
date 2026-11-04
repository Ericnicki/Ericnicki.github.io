# Censored evaluation outcomes

Before consolidated results, progress checks found native trials without a
terminal judgment, including first trials. Preserve those records and include
all360attempts/arm in success denominators. Native return means use observed
returns within each task and then equal weight among tasks with a judgment;
report judged counts and paired complete-task sensitivity. First-trial return
means exclude unavailable values and carry their observed sample count.

Add explicitly post-hoc stress analyses assigning censored trials G=-5 and G=-10.
These are hypothetical penalties, not native labels or mathematical worst-case
bounds. Report their effects alongside the observed-native comparison. They do
not enter fitting or model selection. Use statistics_v124.py with the preserved
native validator through review_complete.py. This prevents undefined arithmetic
on missing first-trial outcomes without changing any experimental arm.
