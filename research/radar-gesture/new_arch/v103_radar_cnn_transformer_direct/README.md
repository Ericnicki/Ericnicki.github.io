# v103 Direct Radar CNN + Transformer Classification

This is a direct supervised control with no JEPA, landmark encoder, keypoint
target, IMU branch, coordinate decoder, or oracle candidate selection.

Each of the 30 radar frames is independently compressed by a shared
single-channel ResNet18 and projected to 128 dimensions. A two-layer,
eight-head temporal Transformer consumes the 30 tokens plus a learned class
token and predicts one of eight gestures directly.

The official 4,151 training episodes retain a deterministic 90/10 fit and
validation split. The best checkpoint is selected only by source validation
accuracy (validation loss breaks ties). The complete 1,055-episode official
test is evaluated once after all 80 epochs.

Server output:

`/root/autodl-fs/mmVR/experiments_new/v103_radar_cnn_transformer_direct/official_non_ood/`
