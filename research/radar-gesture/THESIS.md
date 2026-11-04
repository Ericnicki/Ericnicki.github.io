# Gesture Recognition Experiment Report

## 1. Dataset

The released mmVR dataset contains eight gesture classes and approximately **5,200 episodes**: about **4,150 training episodes** and **1,050 unseen evaluation episodes**. Each episode contains synchronized radar, IMU, and hand-landmark data. Thirty frames are evenly sampled from each 40-frame episode.

About 3,740 training episodes are used to fit each model, while the remaining 415 are used for checkpoint selection. The unseen evaluation set is used only for final evaluation.

## 2. Results

| Method | Training-set accuracy | Unseen-evaluation accuracy |
|---|---:|---:|
| Authors' original method, prediction-only | 97.32% | 62.27% |
| v101 | 100.00% | 86.73% |
| v103 | 100.00% | 71.28% |

## 3. Methods

### 3.1 Authors' original method

The original method is a two-stage radar–IMU system:

1. A 3D CNN encodes the radar sequence, while a two-layer LSTM encodes the six-axis IMU sequence. Separate six-layer, eight-head Transformer encoders process the two modalities, which are fused using fixed weights of 0.9 for radar and 0.1 for IMU.
2. Pose and temporal decoders predict hand confidence and 21 three-dimensional joints for each hand candidate. Prediction confidence and temporal tracking select and arrange at most two predicted hands into a fixed \([30,42,3]\) tensor without using target annotations.
3. A ResNet-50 treats the 30 temporal frames as input channels and classifies the tensor into one of eight gestures.

The pose-estimation stage is trained for 200 epochs. The gesture classifier is trained for 500 epochs.

### 3.2 v101

v101 first learns framewise radar–IMU representations without gesture labels:

1. A shared ResNet-18 compresses each radar frame to a 128-dimensional feature. An MLP encodes the corresponding IMU measurement to 128 dimensions.
2. Radar and IMU features are fused using their concatenation, elementwise product, and difference.
3. A cross-modal JEPA objective trains the fused sensor representation to predict a moving-average landmark representation. Landmark information is used only during representation training.
4. The sensor encoder is frozen. A separate two-layer, eight-head temporal Transformer processes its 30 radar–IMU tokens and performs eight-class classification. Landmarks are not used during classification.

Representation learning runs for 200 epochs, followed by 200 epochs of temporal-classifier training.

### 3.3 v103

v103 is a direct supervised radar-only classifier:

1. A shared ResNet-18 independently compresses each of the 30 radar frames to a 128-dimensional token.
2. A learned classification token and positional embeddings are added.
3. A two-layer, eight-head Transformer models interaction across the 30 frames.
4. A linear head predicts one of eight gesture classes.

The complete network is trained end to end for 80 epochs. IMU, landmarks, keypoints, and JEPA objectives are not used.
