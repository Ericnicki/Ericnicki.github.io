import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import seed_everything
from dataset import EpisodeDataset
from model import FramewiseBundledEMAJEPA, TemporalClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("sensor", "landmark"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "v101_framewise_bundled_ema_jepa":
        raise ValueError("checkpoint is not v101")
    config = checkpoint["config"]
    model = FramewiseBundledEMAJEPA(config["model_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config, checkpoint["epoch"]


@torch.no_grad()
def extract(model, encoder, dataset, device, batch_size, workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    features, labels = [], []
    for batch in loader:
        if encoder == "sensor":
            if "keypoints" in batch:
                raise RuntimeError("deployable sensor probe must not load landmarks")
            if "radar" not in batch or "imu" not in batch:
                raise RuntimeError("sensor probe requires bundled radar and IMU")
            output = model.enc_sensor(
                batch["radar"].to(device), batch["imu"].to(device)
            )
        else:
            output = model.enc_landmark(batch["keypoints"].to(device))
        features.append(output.cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def predictions(classifier, features, device):
    classifier.eval()
    return torch.cat(
        [
            classifier(features[start : start + 256].to(device)).argmax(-1).cpu()
            for start in range(0, len(features), 256)
        ]
    )


def metrics(predicted, labels, classes=8):
    confusion = torch.zeros(classes, classes, dtype=torch.long)
    for truth, prediction in zip(labels.tolist(), predicted.tolist()):
        confusion[truth, prediction] += 1
    matrix = confusion.numpy()
    diagonal = np.diag(matrix).astype(np.float64)
    precision = diagonal / np.maximum(matrix.sum(axis=0), 1)
    recall = diagonal / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(100 * diagonal.sum() / max(matrix.sum(), 1)),
        "macro_precision": float(100 * precision.mean()),
        "macro_recall": float(100 * recall.mean()),
        "macro_f1": float(100 * f1.mean()),
        "confusion": matrix.tolist(),
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, encoder_epoch = load_model(args.checkpoint, device)
    source = EpisodeDataset(
        args.data_root,
        "source",
        config["protocol"],
        include_keypoints=args.encoder == "landmark",
        include_labels=True,
    )
    source_features, source_labels = extract(
        model, args.encoder, source, device, args.extract_batch_size, args.num_workers
    )
    split = json.loads((Path(args.checkpoint).parent / "split.json").read_text())
    fit_indices = torch.tensor(split["fit"], dtype=torch.long)
    validation_indices = torch.tensor(split["validation"], dtype=torch.long)
    fit_loader = DataLoader(
        TensorDataset(source_features[fit_indices], source_labels[fit_indices]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    classifier = TemporalClassifier(config["model_dim"]).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_accuracy, best_epoch = -1.0, -1
    history = []
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        for features, labels in fit_loader:
            loss = F.cross_entropy(classifier(features.to(device)), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        validation = metrics(
            predictions(classifier, source_features[validation_indices], device),
            source_labels[validation_indices],
        )
        history.append({"epoch": epoch, "source_validation": validation})
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            torch.save(classifier.state_dict(), output / "best.pt")
        scheduler.step()
    classifier.load_state_dict(
        torch.load(output / "best.pt", map_location=device, weights_only=True)
    )
    source_validation = metrics(
        predictions(classifier, source_features[validation_indices], device),
        source_labels[validation_indices],
    )
    target = EpisodeDataset(
        args.data_root,
        "target",
        config["protocol"],
        include_keypoints=args.encoder == "landmark",
        include_labels=True,
    )
    source_names = {row["filename"] for row in source.rows}
    target_names = {row["filename"] for row in target.rows}
    if source_names & target_names:
        raise RuntimeError("source/target filename leakage detected")
    target_features, target_labels = extract(
        model, args.encoder, target, device, args.extract_batch_size, args.num_workers
    )
    target_metrics = metrics(predictions(classifier, target_features, device), target_labels)
    held_out_name = "official_test_once" if config["protocol"] == "official_non_ood" else "target_zero_shot"
    results = {
        "protocol": config["protocol"],
        "encoder": args.encoder,
        "role": "deployable bundled radar+IMU frame encoder" if args.encoder == "sensor" else "privileged framewise landmark diagnostic",
        "encoder_epoch": encoder_epoch,
        "encoder_frozen": True,
        "label_gradient_to_encoder": False,
        "temporal_context_in_encoder": False,
        "temporal_context_in_classifier": True,
        "radar_and_imu_inseparable": args.encoder == "sensor",
        "imu_ablation": False,
        "landmarks_loaded_by_probe": args.encoder == "landmark",
        "source_count": len(source),
        "source_fit_count": len(fit_indices),
        "source_validation_count": len(validation_indices),
        "target_count": len(target),
        "source_target_filename_overlap": 0,
        "best_classifier_epoch": best_epoch,
        "source_validation": source_validation,
        held_out_name: target_metrics,
    }
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
