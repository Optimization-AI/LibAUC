"""
Entry point for LibAUC image-classification training.

Usage::

    python -m libauc.trainer.run_image_trainer --config_file config.yaml

CLI flags mirror ``TrainingArguments`` fields and override the YAML config
when supplied.
"""

import argparse
import logging
import sys

import numpy as np
from omegaconf import OmegaConf

from .config import TrainingArguments, parse_defaultconfig
from .core import CLICallback, Trainer
from .data import load_dataset
from .helpers import build_megaconf, apply_cli_overrides, build_metric, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Megaconf defaults
# ---------------------------------------------------------------------------

_MEGACONF_DEFAULTS = {
    "training": {
        "optimizer":              "PESG",
        "optimizer_kwargs":       {},
        "loss":                   "AUCMLoss",
        "loss_kwargs":            {},
        "SEED":                   42,
        "batch_size":             128,
        "eval_batch_size":        128,
        "sampling_rate":          0.5,
        "epochs":                 50,
        "decay_epochs":           [],
        "num_workers":            2,
        "output_path":            "./output",
        "resume_from_checkpoint": True,
        "save_checkpoint_every":  5,
        "project_name":           "libauc",
        "experiment_name":        "run",
        "verbose":                1,
    },
    "dataset": {
        "name":        "",
        "kwargs":      {},
        "eval_splits": ["val"],
    },
    "model": {
        "name":       "resnet18",
        "pretrained": False,
    },
    "metrics":       ["AUROC"],
    "metric_kwargs": [],
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LibAUC image-classification training")

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to a YAML configuration file.",
    )

    # Each flag mirrors a TrainingArguments field; when supplied it overrides
    # the value from the config file.
    parser.add_argument("--epochs",                type=int,   default=None)
    parser.add_argument("--batch_size",            type=int,   default=None)
    parser.add_argument("--eval_batch_size",       type=int,   default=None)
    parser.add_argument("--sampling_rate",         type=float, default=None)
    parser.add_argument("--num_workers",           type=int,   default=None)
    parser.add_argument("--output_path",           type=str,   default=None)
    parser.add_argument("--seed",                  type=int,   default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        action=argparse.BooleanOptionalAction,   # --resume / --no-resume
        default=None,
    )
    parser.add_argument("--save_checkpoint_every", type=int,   default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()

    # 1. Load config (megaconf) and apply CLI overrides
    logger.info(f"Loading config from: {args.config_file}")
    cfg     = build_megaconf(args.config_file, _MEGACONF_DEFAULTS)
    cfg     = apply_cli_overrides(cfg, args)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    training_cfg  = cfg_dict["training"]
    dataset_cfg   = cfg_dict["dataset"]
    model_cfg     = cfg_dict["model"]
    metric_names  = cfg_dict.get("metrics",       ["AUROC"])
    metric_kwargs = cfg_dict.get("metric_kwargs", [])

    # 2. Reproducibility
    set_seed(training_cfg.get("SEED", 42))

    # 3. Load datasets
    dataset_name   = dataset_cfg["name"]
    dataset_kwargs = dataset_cfg.get("kwargs", {})
    eval_splits    = dataset_cfg.get("eval_splits", ["val"])

    logger.info(f"Loading train and {eval_splits} splits of dataset: {dataset_name}")
    train_dataset, eval_datasets = load_dataset(
        dataset_name, splits=eval_splits, **dataset_kwargs
    )

    # 4. Infer task count (binary vs. multi-label)
    labels = np.array(train_dataset.targets).squeeze()
    if labels.ndim == 1:
        num_tasks = len(np.unique(labels))
    else:
        num_tasks = labels.shape[-1]

    logger.info(f"Number of tasks: {num_tasks}")
    multilabel = num_tasks >= 3

    # 5. Build metric function
    metric_fn = build_metric(metric_names, metric_kwargs)

    # 6. Resolve default optimizer / loss configs via parse_defaultconfig
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {})
    loss_kwargs      = training_cfg.get("loss_kwargs",      {})

    default_optimizer_config = parse_defaultconfig(
        training_cfg["optimizer"], multilabel, optimizer_kwargs
    )["optimizer"]
    default_loss_config = parse_defaultconfig(
        training_cfg["loss"], multilabel, loss_kwargs
    )["loss"]

    if multilabel:
        loss_kwargs["num_labels"] = num_tasks

    # 7. Construct TrainingArguments
    train_args = TrainingArguments(
        optimizer              = default_optimizer_config["type"],
        optimizer_kwargs       = optimizer_kwargs,
        loss                   = default_loss_config["type"],
        loss_kwargs            = loss_kwargs,
        SEED                   = training_cfg.get("SEED",                42),
        batch_size             = training_cfg.get("batch_size",          128),
        eval_batch_size        = training_cfg.get("eval_batch_size",     128),
        sampling_rate          = training_cfg.get("sampling_rate",       0.5),
        epochs                 = training_cfg.get("epochs",              50),
        decay_epochs           = training_cfg.get("decay_epochs",        []),
        num_workers            = training_cfg.get("num_workers",         2),
        output_path            = training_cfg.get("output_path",         "./output"),
        num_tasks              = num_tasks,
        resume_from_checkpoint = training_cfg.get("resume_from_checkpoint", True),
        save_checkpoint_every  = training_cfg.get("save_checkpoint_every",  5),
        project_name           = training_cfg.get("project_name",        "libauc"),
        experiment_name        = training_cfg["experiment_name"],
        verbose                = training_cfg.get("verbose",             1),
    )

    # 8. Initialise and run Trainer
    logger.info("Initialising Trainer...")
    trainer = Trainer(
        train_args    = train_args,
        model_cfg     = model_cfg,
        train_dataset = train_dataset,
        eval_dataset  = eval_datasets if eval_datasets else None,
        metric        = metric_fn,
        callbacks     = [CLICallback()],
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
