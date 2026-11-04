# GMP T4: Iterative Pushing with Test-Time Training

Research source snapshot by Keqi Ni. The environment is GMP's T4 PushCube (iterative pushing), not Push-T. This snapshot contains the TTT residual controller and matched friction out-of-distribution evaluation against the official pretrained Gated Memory Policy.

## Evidence

The featured controller is the joint shared physical-memory/prediction-value variant, displayed as **TTT memory sparse return RL**.

- ID: 293/330 successes (88.79%) on trials 2–12 of a reused 30-task benchmark.
- Low-friction OOD: 19/176 successes (10.80%).
- High-friction OOD: 66/176 successes (37.50%).
- ID trial success: 16.7% at trial 1, 73.3% at trial 2, 93.3% at trial 4.

See THESIS.md and results/joint_id_result.json. The OOD results in results/ttt_result.json use the same joint label. Official GMP achieved zero primary OOD successes under this matched custom extension, but different action interfaces limit causal comparisons. The original GMP last-three-trial criterion does not demonstrate that its first three attempts always fail or establish a matched speed advantage.

The three displayed curves are low/middle/high **ID** illustrative tasks, not aggregate learning curves. OOD boundary videos show failed attempts for the joint policy; they are not a representative success sample. Historic source and report files retain original names for provenance.

## Source map

- `source/ttt_world_shared_fit_v178/`: joint prediction/value shared-memory fitting.
- `source/ttt_world_gradient_inference_v183/`: exported joint-policy inference API and protocol.



- `source/ttt_joint_v100/` and `source/ttt_joint_1000_v115/`: original joint actor/critic training and TTT model implementation.
- `source/ttt_rescue_v117/`: common parent-loading and causal memory reconstruction used by the rescue fits.

- `source/ttt_rescue_train_v118/`: original TTT training implementation and protocol.
- `source/ttt_rescue_isolated_v122/`: isolated optimizer-state repair and confirmation utilities.
- `source/ttt_world_ood_top4_v185/`: serial OOD evaluation and original protocol.
- `source/ttt_world_ood_top4_v186/`: parallel completion; `legacy.py` implements TTT blocks, `memory.py` causal fast-write memory, `controller.py` residual control, `trial.py` native trial collection, and `audit.py` verification.
- `source/gmp_memory_ood_v187/`: official GMP comparator collection.
- `source/gmp_memory_ood_v190/`: corrected independent audit.
- `source/gmp_memory_ood_v191/`: final reporting correction.
- `SOURCE_SHA256.json`: hashes of unchanged copied source files.

## Reproduction status

This is an archival research release, not a self-contained installation. The unchanged research sources reference the original Linux/CUDA filesystem and hostname, parent checkpoints, dataset snapshots, and GMP configuration/assets. Machine-specific shell launchers, credentials, large checkpoints and raw replay data are excluded. No simulation or training was launched to prepare this website.

The TTT source needs Python, NumPy, PyTorch and the original MuJoCo/Hydra environment. Consult each frozen protocol for exact dependency paths and checkpoint hashes. Obtain the official environment and its dependency instructions from https://github.com/real-stanford/gated-memory-policy (comparison used commit c23bb40566a9eec2f90a570deb67ec71868e0820). Before executing research scripts elsewhere, configure paths/hostname guards in a separate working copy and supply the bound checkpoints and raw inputs. Do not remove guards and treat a run as a reproduced experiment without those inputs.

Original reports preserve historical relative links into the full private workspace; some links are intentionally unavailable in this compact release. Use the included `results/` files and the website project page for the public evidence. Upstream GMP retains its own license; no blanket license is asserted over third-party code.
