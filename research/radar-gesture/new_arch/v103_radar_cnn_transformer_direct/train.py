import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import RadarClassificationDataset
from model import RadarCNNTransformerClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--output-dir",
        default="experiments_new/v103_radar_cnn_transformer_direct/official_non_ood",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(length, fraction, seed):
    generator = np.random.default_rng(seed)
    indices = generator.permutation(length)
    validation_count = max(1, round(length * fraction))
    return indices[validation_count:].tolist(), indices[:validation_count].tolist()


def confusion_metrics(confusion):
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


def run_epoch(model, loader, device, optimizer=None, maximum_batches=0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    count = 0
    confusion = torch.zeros(8, 8, dtype=torch.long)
    for batch_index, batch in enumerate(loader):
        if maximum_batches and batch_index >= maximum_batches:
            break
        radar = batch["radar"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.enable_grad() if training else torch.no_grad():
            logits = model(radar)
            loss = F.cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        predictions = logits.detach().argmax(-1).cpu()
        for truth, prediction in zip(labels.cpu().tolist(), predictions.tolist()):
            confusion[truth, prediction] += 1
        total_loss += float(loss.detach()) * len(labels)
        count += len(labels)
    result = confusion_metrics(confusion)
    result["loss"] = total_loss / max(count, 1)
    result["count"] = count
    return result


def plot_history(history, output):
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train"]["loss"] for row in history], label="fit")
    axes[0].plot(
        epochs, [row["validation"]["loss"] for row in history], label="validation"
    )
    axes[0].set(title="Cross-entropy", xlabel="epoch", ylabel="loss")
    axes[1].plot(
        epochs, [row["train"]["accuracy"] for row in history], label="fit"
    )
    axes[1].plot(
        epochs,
        [row["validation"]["accuracy"] for row in history],
        label="validation",
    )
    axes[1].set(title="Direct classification", xlabel="epoch", ylabel="accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.model_dim % args.attention_heads:
        raise ValueError("model_dim must be divisible by attention_heads")
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source = RadarClassificationDataset(args.data_root, "train")
    official_test = RadarClassificationDataset(args.data_root, "eval")
    source_names = {row["filename"] for row in source.rows}
    test_names = {row["filename"] for row in official_test.rows}
    if source_names & test_names:
        raise RuntimeError("official train/eval filename leakage detected")
    fit_indices, validation_indices = split_indices(
        len(source), args.validation_fraction, args.seed
    )
    (output / "split.json").write_text(
        json.dumps({"fit": fit_indices, "validation": validation_indices}, indent=2)
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": args.num_workers > 0,
    }
    fit_loader = DataLoader(
        Subset(source, fit_indices), shuffle=True, drop_last=True, **loader_options
    )
    validation_loader = DataLoader(
        Subset(source, validation_indices), shuffle=False, **loader_options
    )
    test_loader = DataLoader(official_test, shuffle=False, **loader_options)

    model = RadarCNNTransformerClassifier(
        args.model_dim,
        args.transformer_layers,
        args.attention_heads,
        args.dropout,
    ).to(device)
    contract_batch = next(iter(fit_loader))
    if set(contract_batch) != {"radar", "label", "filename"}:
        raise RuntimeError(f"unexpected direct-classification inputs: {set(contract_batch)}")
    model.eval()
    with torch.no_grad():
        contract_logits, frame_tokens, contextual = model(
            contract_batch["radar"][:2].to(device), return_tokens=True
        )
    model.train()
    contract = {
        "input_keys": sorted(contract_batch),
        "radar_shape": list(contract_batch["radar"][:2].shape),
        "frame_token_shape": list(frame_tokens.shape),
        "contextual_token_shape": list(contextual.shape),
        "logit_shape": list(contract_logits.shape),
        "attention_heads": args.attention_heads,
        "transformer_layers": args.transformer_layers,
        "official_filename_overlap": 0,
    }
    (output / "architecture_audit.json").write_text(json.dumps(contract, indent=2))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model, fit_loader, device, optimizer, args.max_train_batches
        )
        validation = run_epoch(
            model, validation_loader, device, None, args.max_eval_batches
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train,
            "validation": validation,
        }
        history.append(row)
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        checkpoint = {
            "architecture": "v103_radar_cnn_transformer_direct",
            "model": model.state_dict(),
            "epoch": epoch,
            "config": vars(args),
            "parameter_count": parameter_count,
        }
        torch.save(checkpoint, output / "last.pt")
        improved = validation["accuracy"] > best_accuracy or (
            validation["accuracy"] == best_accuracy and validation["loss"] < best_loss
        )
        if improved:
            best_accuracy = validation["accuracy"]
            best_loss = validation["loss"]
            best_epoch = epoch
            torch.save(checkpoint, output / "best.pt")
        scheduler.step()

    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    source_validation = run_epoch(
        model, validation_loader, device, None, args.max_eval_batches
    )
    official_test_once = run_epoch(
        model, test_loader, device, None, args.max_eval_batches
    )
    results = {
        "architecture": "shared per-frame ResNet18 CNN -> 30-token temporal Transformer -> direct classifier",
        "input": "radar only",
        "imu_used": False,
        "landmarks_or_keypoints_used": False,
        "oracle_selection_used": False,
        "frame_tokens": 30,
        "model_dim": args.model_dim,
        "transformer_layers": args.transformer_layers,
        "attention_heads": args.attention_heads,
        "parameter_count": parameter_count,
        "configured_epochs": args.epochs,
        "completed_epochs": args.epochs,
        "best_epoch": best_epoch,
        "selection": "fixed source-training validation accuracy, then validation loss",
        "source_count": len(source),
        "source_fit_count": len(fit_indices),
        "source_validation_count": len(validation_indices),
        "official_test_count": len(official_test),
        "source_validation": source_validation,
        "official_test_once": official_test_once,
    }
    (output / "results.json").write_text(json.dumps(results, indent=2))
    plot_history(history, output)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
