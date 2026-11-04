# Execution-only confirmation amendment v120b

2026-10-10, before confirmation starts. Preserve the original sequential
`confirm.py` and pipeline.sh. Stop only the waiting pipeline shell PID1034;
the active optimizer process PID1355 continues uninterrupted and retains its log.
The new `confirm_parallel_v120b.py` waits for the same fixed fit completion, then
assigns task IDs modulo4 to four independent simulator processes on the same
current authorized compute node. Each has an isolated native environment,
model instances, task memories and budget file. Seeds/spec generation,11arms,
60tasks,6trials,calibration,model endpoints and all statistical decisions stay
unchanged. No confirmation labels enter any fit. The parent combines sorted
whole-task summaries only after all workers exit successfully. This is process
parallelism, not delegated research agents or an altered experiment.
