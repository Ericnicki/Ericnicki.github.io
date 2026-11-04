# v101 Framewise Bundled EMA JEPA

v101 is the stabilized v019 EMA-asymmetric architecture with one controlled
change: synchronous radar and IMU are fused into each 128D sensor frame token.
Radar and IMU are mandatory and inseparable.

There is no Transformer, temporal attention, recurrence, pooling, context
token, context loss, or change loss in representation pretraining. The
objective is exactly framewise:

`sensor_to_landmark + 0.25 * landmark_to_sensor + 0.05 * SIGReg`.

Both targets are full EMA copies of their frame encoders, with momentum
scheduled from 0.996 toward 1. The representation optimizer always executes
all 200 epochs; a source-only, label-free best checkpoint is retained for the
final frozen probes. A separate two-layer temporal Transformer consumes the 30
frozen tokens only during gesture classification.

The server queue runs both the official non-OOD split and the original strict
persons 1-8 to 9-12 split, with an independent 200-epoch representation run
for each protocol. Server output:

`/root/autodl-fs/mmVR/experiments_new/v101_framewise_bundled_ema/official_non_ood/`

`/root/autodl-fs/mmVR/experiments_new/v101_framewise_bundled_ema/person_1_8_to_9_12/`
