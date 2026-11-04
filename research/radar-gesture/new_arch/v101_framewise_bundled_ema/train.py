import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import append_jsonl, make_loaders, seed_everything
from dataset import EpisodeDataset
from model import FramewiseBundledEMAJEPA, latent_statistics, normalized_regression, sigreg_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--output-dir",
        default="experiments_new/v101_framewise_bundled_ema/official_non_ood/pretrain",
    )
    parser.add_argument("--protocol", default="official_non_ood")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--reverse-weight", type=float, default=0.25)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--sigreg-slices", type=int, default=256)
    parser.add_argument("--sigreg-points", type=int, default=17)
    parser.add_argument("--sigreg-frame-samples", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--min-effective-rank", type=float, default=8.0)
    parser.add_argument("--min-mean-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def cosine_momentum(step, total_steps, start=0.996, end=1.0):
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def sample_tokens(tokens, count, seed):
    if count <= 0 or count >= tokens.shape[1]:
        return tokens
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(int(seed) + 31_415_927)
    indices = torch.randint(
        tokens.shape[1],
        (tokens.shape[0], count),
        generator=generator,
        device=tokens.device,
    )
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def compute_batch(model, batch, device, args, seed, ablation=None):
    if "label" in batch:
        raise RuntimeError("gesture labels must not enter JEPA pretraining")
    if "radar" not in batch or "imu" not in batch:
        raise RuntimeError("both radar and IMU are mandatory")
    radar = batch["radar"].to(device, non_blocking=True)
    imu = batch["imu"].to(device, non_blocking=True)
    keypoints = batch["keypoints"].to(device, non_blocking=True)
    if ablation == "zero_sensor":
        radar, imu = torch.zeros_like(radar), torch.zeros_like(imu)
    elif ablation == "shuffle_sensor":
        radar, imu = radar.roll(1, 0), imu.roll(1, 0)
    elif ablation == "zero_landmarks":
        keypoints = torch.zeros_like(keypoints)
    elif ablation == "shuffle_landmarks":
        keypoints = keypoints.roll(1, 0)
    output = model(radar, imu, keypoints)
    forward = normalized_regression(
        output["sensor_to_landmark"], output["target_landmark"]
    )
    reverse = normalized_regression(
        output["landmark_to_sensor"], output["target_sensor"]
    )
    sensor_sample = sample_tokens(output["sensor"], args.sigreg_frame_samples, seed)
    landmark_sample = sample_tokens(
        output["landmark"], args.sigreg_frame_samples, seed + 1_000_003
    )
    sigreg_sensor = sigreg_loss(
        sensor_sample, seed, args.sigreg_slices, args.sigreg_points
    )
    sigreg_landmark = sigreg_loss(
        landmark_sample, seed + 1_000_003, args.sigreg_slices, args.sigreg_points
    )
    sigreg = 0.5 * (sigreg_sensor + sigreg_landmark)
    loss = forward + args.reverse_weight * reverse + args.sigreg_weight * sigreg
    with torch.no_grad():
        sensor_std, sensor_rank = latent_statistics(sensor_sample)
        landmark_std, landmark_rank = latent_statistics(landmark_sample)
    return loss, {
        "loss": loss,
        "sensor_to_landmark": forward,
        "landmark_to_sensor": reverse,
        "sigreg": sigreg,
        "sigreg_sensor": sigreg_sensor,
        "sigreg_landmark": sigreg_landmark,
        "sensor_mean_std": sensor_std,
        "sensor_effective_rank": sensor_rank,
        "landmark_mean_std": landmark_std,
        "landmark_effective_rank": landmark_rank,
    }, radar.shape[0]


METRICS = (
    "loss",
    "sensor_to_landmark",
    "landmark_to_sensor",
    "sigreg",
    "sigreg_sensor",
    "sigreg_landmark",
    "sensor_mean_std",
    "sensor_effective_rank",
    "landmark_mean_std",
    "landmark_effective_rank",
)


def run_epoch(model, loader, device, args, optimizer, epoch, total_steps=None):
    training = optimizer is not None
    model.train(training)
    model.target_sensor.eval()
    model.target_landmark.eval()
    totals = {name: 0.0 for name in METRICS}
    count = 0
    maximum = args.max_train_batches if training else args.max_eval_batches
    for batch_index, batch in enumerate(loader):
        if maximum and batch_index >= maximum:
            break
        step = (epoch - 1) * len(loader) + batch_index if training else 10_000_000 + batch_index
        with torch.enable_grad() if training else torch.no_grad():
            loss, metrics, batch_size = compute_batch(model, batch, device, args, step)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.update_targets(cosine_momentum(step, total_steps, args.ema_start))
        for name, value in metrics.items():
            totals[name] += float(value.detach()) * batch_size
        count += batch_size
    result = {name: value / max(count, 1) for name, value in totals.items()}
    result["count"] = count
    return result


def module_grad_norm(module):
    return sum(
        float(parameter.grad.norm())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def gradient_audit(model, batch, device):
    radar = batch["radar"][:2].to(device)
    imu = batch["imu"][:2].to(device)
    keypoints = batch["keypoints"][:2].to(device)
    model.eval()

    def collect():
        return {
            "sensor_encoder": module_grad_norm(model.enc_sensor),
            "landmark_encoder": module_grad_norm(model.enc_landmark),
            "target_sensor": module_grad_norm(model.target_sensor),
            "target_landmark": module_grad_norm(model.target_landmark),
            "sensor_predictor": module_grad_norm(model.predict_sensor_to_landmark),
            "landmark_predictor": module_grad_norm(model.predict_landmark_to_sensor),
        }

    model.zero_grad(set_to_none=True)
    output = model(radar, imu, keypoints)
    normalized_regression(output["sensor_to_landmark"], output["target_landmark"]).backward()
    forward = collect()
    model.zero_grad(set_to_none=True)
    output = model(radar, imu, keypoints)
    normalized_regression(output["landmark_to_sensor"], output["target_sensor"]).backward()
    reverse = collect()
    model.zero_grad(set_to_none=True)
    model.train()
    if not (
        forward["sensor_encoder"] > 0
        and forward["sensor_predictor"] > 0
        and forward["landmark_encoder"] == 0
        and forward["target_sensor"] == 0
        and forward["target_landmark"] == 0
        and forward["landmark_predictor"] == 0
        and reverse["landmark_encoder"] > 0
        and reverse["landmark_predictor"] > 0
        and reverse["sensor_encoder"] == 0
        and reverse["target_sensor"] == 0
        and reverse["target_landmark"] == 0
        and reverse["sensor_predictor"] == 0
    ):
        raise RuntimeError(f"stop-gradient routing failed: {forward}, {reverse}")
    return {"sensor_to_landmark": forward, "landmark_to_sensor": reverse}


@torch.no_grad()
def evaluate_ablation(model, loader, device, args, mode):
    model.eval()
    totals = {"sensor_to_landmark": 0.0, "landmark_to_sensor": 0.0}
    count = 0
    for batch_index, batch in enumerate(loader):
        if args.max_eval_batches and batch_index >= args.max_eval_batches:
            break
        _, metrics, batch_size = compute_batch(
            model, batch, device, args, 20_000_000 + batch_index, mode
        )
        for name in totals:
            totals[name] += float(metrics[name]) * batch_size
        count += batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


def main():
    args = parse_args()
    if args.protocol not in {"official_non_ood", "person_1_8_to_9_12"}:
        raise ValueError(f"unknown protocol: {args.protocol}")
    if args.reverse_weight != 0.25:
        raise ValueError("v101 fixes the stabilized reverse weight at 0.25")
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = EpisodeDataset(args.data_root, "source", args.protocol, True, False)
    fit_loader, validation_loader, fit_indices, validation_indices = make_loaders(
        source, args.batch_size, args.num_workers, args.validation_fraction, args.seed
    )
    (output / "split.json").write_text(
        json.dumps({"fit": fit_indices, "validation": validation_indices}, indent=2)
    )
    model = FramewiseBundledEMAJEPA(args.model_dim).to(device)
    audit = gradient_audit(model, next(iter(fit_loader)), device)
    (output / "gradient_audit.json").write_text(json.dumps(audit, indent=2))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    total_steps = args.epochs * len(fit_loader)
    best_loss, best_epoch = float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, fit_loader, device, args, optimizer, epoch, total_steps)
        validation = run_epoch(model, validation_loader, device, args, None, epoch)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train,
            "validation": validation,
        }
        append_jsonl(output / "history.jsonl", row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "architecture": "v101_framewise_bundled_ema_jepa",
            "model": model.state_dict(),
            "epoch": epoch,
            "config": vars(args),
        }
        torch.save(checkpoint, output / "last.pt")
        safe = (
            validation["sensor_effective_rank"] >= args.min_effective_rank
            and validation["landmark_effective_rank"] >= args.min_effective_rank
            and validation["sensor_mean_std"] >= args.min_mean_std
            and validation["landmark_mean_std"] >= args.min_mean_std
        )
        if safe and validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            torch.save(checkpoint, output / "best.pt")
        scheduler.step()
    if best_epoch < 0:
        raise RuntimeError("no checkpoint met rank/std gates after all 200 epochs")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    target = EpisodeDataset(args.data_root, "target", args.protocol, True, False)
    target_loader = DataLoader(
        target,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    target_metrics = run_epoch(model, target_loader, device, args, None, 1)
    held_out_name = "official_test_once" if args.protocol == "official_non_ood" else "target_test_once"
    results = {
        "protocol": args.protocol,
        "architecture": "v019 framewise EMA-asymmetric variant with mandatory radar+IMU fusion",
        "temporal_context_in_encoder": False,
        "radar_and_imu_inseparable": True,
        "imu_ablation": False,
        "objective": "sensor_to_landmark + 0.25*landmark_to_sensor + 0.05*SIGReg",
        "configured_epochs": args.epochs,
        "completed_epochs": args.epochs,
        "early_stopping": False,
        "labels_used_in_pretraining": False,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "source_count": len(source),
        "target_count": len(target),
        held_out_name: target_metrics,
        "held_out_ablations": {
            mode: evaluate_ablation(model, target_loader, device, args, mode)
            for mode in ("zero_sensor", "shuffle_sensor", "zero_landmarks", "shuffle_landmarks")
        },
    }
    (output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
