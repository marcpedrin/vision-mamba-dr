"""
Evaluate a checkpoint on the EyePACS test split and on Messidor-2.

    python -m eval.evaluate --config configs/default.yaml
    python -m eval.evaluate --checkpoint outputs/default/best.pt

    Rebuilds the model and the same split from the config saved in the checkpoint.
    Writes results.json + one predictions csv per dataset.
    Expect lower scores on Messidor-2 (different camera, country, quality = domain shift).

"""

import argparse
import csv
import json
import os

import torch
import torch.nn as nn

from data import build_dataloaders, build_messidor_loader
from eval.metrics import compute_all_metrics
from train.engine import validate
from train.utils import load_checkpoint, set_seed
from vim import build_model
from vim.config import config_from_dict, load_config

GRADE_NAMES = ["0 none", "1 mild", "2 moderate", "3 severe", "4 prolif."]


def parse_args(argv=None):
    """Command line options."""
    parser = argparse.ArgumentParser(description="Evaluate a Vision Mamba checkpoint")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="config with the dataset paths")
    parser.add_argument("--override", action="append", default=[],
                        help="change one config value, e.g. eval.checkpoint=x.pt")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint to evaluate (default: eval.checkpoint)")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' or 'cpu'")
    return parser.parse_args(argv)


def evaluate_loader(model, loader, device, num_classes):
    """Returns (metrics, labels, predictions) for one loader."""
    #unweighted loss for reporting
    criterion = nn.CrossEntropyLoss()
    out = validate(model, loader, criterion, device)
    metrics = compute_all_metrics(out["labels"], out["predictions"], num_classes)
    metrics["loss"] = out["loss"]
    metrics["num_images"] = int(len(out["labels"]))
    return metrics, out["labels"], out["predictions"]


def write_predictions_csv(path, dataset, labels, predictions):
    """path, true, pred per image (loaders arent shuffled so order matches)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true", "pred"])
        for (image_path, _), true, pred in zip(dataset.samples, labels, predictions):
            writer.writerow([image_path, int(true), int(pred)])


def print_table(results):
    """Metrics side by side, one column per dataset."""
    names = list(results.keys())

    header = f"{'metric':<22}"
    for name in names:
        header += f"{name:>16}"
    print(header)

    for key in ("num_images", "accuracy", "qwk", "macro_precision", "macro_recall",
                "macro_f1", "loss"):
        line = f"{key:<22}"
        for name in names:
            value = results[name][key]
            if isinstance(value, int):
                line += f"{value:>16d}"
            else:
                line += f"{value:>16.4f}"
        print(line)

    #recall per grade
    for c, grade in enumerate(GRADE_NAMES):
        line = f"{'recall ' + grade:<22}"
        for name in names:
            line += f"{results[name]['per_class_recall'][c]:>16.4f}"
        print(line)


def main(argv=None):
    """Evaluate, returns the results dict (used by tests)."""
    args = parse_args(argv)
    cfg = load_config(args.config, args.override)
    checkpoint_path = args.checkpoint or cfg.eval.checkpoint

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    #build the model from the config inside the checkpoint so it always matches
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    trained_cfg = config_from_dict(state["config"])
    model = build_model(trained_cfg.model)
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    print(f"Loaded {checkpoint_path} (epoch {state.get('epoch')}, "
          f"val QWK {state.get('val_qwk', float('nan')):.4f})")

    #same seed/fractions as training -> same split, only paths come from current config
    cfg.seed = trained_cfg.seed
    cfg.data.val_fraction = trained_cfg.data.val_fraction
    cfg.data.test_fraction = trained_cfg.data.test_fraction
    cfg.model = trained_cfg.model
    set_seed(cfg.seed)

    num_classes = cfg.model.num_classes
    results = {}

    #smoke checkpoint only used the first N images, rebuild that subset
    max_images = None
    if trained_cfg.train.output_dir.endswith("_smoke"):
        max_images = trained_cfg.train.smoke_test_images
    _, _, test_loader = build_dataloaders(cfg, max_images=max_images)
    metrics, labels, preds = evaluate_loader(model, test_loader, device, num_classes)
    results["eyepacs_test"] = metrics

    #Messidor-2, eval only, skipped if not there
    messidor_loader = None
    if os.path.exists(cfg.data.messidor_labels_csv):
        messidor_loader = build_messidor_loader(cfg)
        m_metrics, m_labels, m_preds = evaluate_loader(model, messidor_loader, device,
                                                       num_classes)
        results["messidor2"] = m_metrics
    else:
        print(f"Messidor-2 labels not found at {cfg.data.messidor_labels_csv}; "
              f"skipping the external evaluation")

    print_table(results)

    out_dir = os.path.dirname(cfg.eval.results_json) or "."
    os.makedirs(out_dir, exist_ok=True)
    payload = {"checkpoint": checkpoint_path, "epoch": state.get("epoch"),
               "results": results}
    with open(cfg.eval.results_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_predictions_csv(os.path.join(out_dir, "predictions_eyepacs_test.csv"),
                          test_loader.dataset, labels, preds)
    if messidor_loader is not None:
        write_predictions_csv(os.path.join(out_dir, "predictions_messidor2.csv"),
                              messidor_loader.dataset, m_labels, m_preds)
    print(f"Wrote {cfg.eval.results_json}")
    return results


if __name__ == "__main__":
    main()
