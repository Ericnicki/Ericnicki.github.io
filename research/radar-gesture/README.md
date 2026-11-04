# Radar-IMU Gesture Recognition with Cross-Modal JEPA

Compact research code release by Keqi Ni. Includes the v101 JEPA model and frozen temporal classifier, plus the v103 radar-only supervised baseline. The original EgoHand/mmVR dataset and baseline are credited to https://github.com/WhisperYi/mmVR. The reported original-method baseline is a local reproduction, not the paper's headline result.

## Included

`new_arch/v101_framewise_bundled_ema/`: framewise sensor-landmark JEPA, EMA targets, regularization, and frozen gesture probe.
`new_arch/v103_radar_cnn_transformer_direct/`: end-to-end radar-only CNN/Transformer classifier.
`THESIS.md`: experiment report with methods, splits and evaluation results.

## Setup

Use a Python environment compatible with PyTorch and torchvision. Install `requirements.txt`, obtain the mmVR dataset from the upstream project, and preserve its episode-level split lists. Data and pretrained weights are not bundled. Read each version's README before running. Run `python new_arch/v101_framewise_bundled_ema/train.py --help` and `python new_arch/v101_framewise_bundled_ema/probe.py --help` for arguments. No training is launched by opening this repository.

Dependencies are listed without pretending that a new pinned environment has been validated. The copied research files are unchanged and hash-bound in SOURCE_SHA256.json. Upstream code/data retains its own license and attribution.
