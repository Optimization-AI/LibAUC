"""
Entry point for GNN training with GNNTrainer.

Usage::

    python -m libauc.trainer.run_trainer --config_file config.yaml

CLI flags mirror ``TrainingArguments`` fields and override the YAML config
when supplied.
"""

import argparse
import logging
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from .config import TrainingArguments, parse_defaultconfig
from .core import CLICallback, GNNTrainer
from .data import load_dataset
from .helpers import build_metric

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

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
        "decay_factor":           10.0,
        "num_workers":            2,
        "output_path":            "./output",
        "resume_from_checkpoint": True,
        "save_checkpoint_every":  5,
        "project_name":           "libauc",
        "experiment_name":        "run_gnn",
        "verbose":                1,
    },
    "dataset": {
        "name":        "",
        "kwargs":      {},
        "eval_splits": ["val"],
    },
    "model": {
        "name":      "gcn",
        "num_tasks": 1,
        "emb_dim":   256,
        "num_layers": 5,
    },
    "metrics":       ["AUROC"],
    "metric_kwargs": [],
}


def _build_megaconf(config_path: str) -> OmegaConf:
    """
    Load a YAML config file and merge it on top of the GNN megaconf defaults.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        OmegaConf DictConfig: Fully resolved configuration.
    """
    base   = OmegaConf.create(_MEGACONF_DEFAULTS)
    user   = OmegaConf.load(config_path)
    merged = OmegaConf.merge(base, user)
    return merged


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="GNN training script")

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to a YAML configuration file.",
    )

    parser.add_argument("--epochs",                type=int,   default=None)
    parser.add_argument("--batch_size",            type=int,   default=None)
    parser.add_argument("--eval_batch_size",       type=int,   default=None)
    parser.add_argument("--sampling_rate",         type=float, default=None)
    parser.add_argument("--num_workers",           type=int,   default=None)
    parser.add_argument("--output_path",           type=str,   default=None)
    parser.add_argument("--seed",                  type=int,   default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--save_checkpoint_every", type=int,   default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLI override helper
# ---------------------------------------------------------------------------

def apply_cli_overrides(cfg: OmegaConf, args) -> OmegaConf:
    """
    Merge CLI-supplied values into the megaconf (mutates in-place).

    The mapping between CLI argument names and config keys is kept in a
    single list so adding a new parameter only requires one entry here.

    Args:
        cfg (OmegaConf): Merged megaconf produced by :func:`_build_megaconf`.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        OmegaConf: The mutated config (same object).
    """
    # (section, cfg_key, args_attr)
    _CLI_MAP: list[tuple[str, str, str]] = [
        ("training", "epochs",                  "epochs"),
        ("training", "batch_size",              "batch_size"),
        ("training", "eval_batch_size",         "eval_batch_size"),
        ("training", "sampling_rate",           "sampling_rate"),
        ("training", "num_workers",             "num_workers"),
        ("training", "output_path",             "output_path"),
        ("training", "SEED",                    "seed"),
        ("training", "resume_from_checkpoint",  "resume_from_checkpoint"),
        ("training", "save_checkpoint_every",   "save_checkpoint_every"),
    ]

    for section, cfg_key, arg_attr in _CLI_MAP:
        value = getattr(args, arg_attr, None)
        if value is not None:
            logger.info(f"CLI override: {section}.{cfg_key} = {value}")
            OmegaConf.update(cfg, f"{section}.{cfg_key}", value)

    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    logger.info(f"Global seed set to {seed}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Load config (megaconf) and apply CLI overrides
    logger.info(f"Loading config from: {args.config_file}")
    cfg = _build_megaconf(args.config_file)
    cfg = apply_cli_overrides(cfg, args)

    # Convert to plain Python dict for downstream compatibility
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    training_cfg  = cfg_dict["training"]
    dataset_cfg   = cfg_dict["dataset"]
    model_cfg     = cfg_dict["model"]
    metric_names  = cfg_dict.get("metrics",       ["AUROC"])
    metric_kwargs = cfg_dict.get("metric_kwargs", [])

    # GNN-specific optional keys (with safe defaults)
    decay_epochs = training_cfg.get("decay_epochs", [])
    decay_factor = training_cfg.get("decay_factor", 10.0)

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

    # 4. Build metric function
    metric_fn = build_metric(metric_names, metric_kwargs)

    # 5. Resolve default optimizer / loss configs via parse_defaultconfig
    #    (mirrors the pattern used in run_trainer.py)
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {})
    loss_kwargs      = training_cfg.get("loss_kwargs",      {})

    # GNN tasks are always treated as binary / single-label for sampler logic
    multilabel = False

    default_optimizer_config = parse_defaultconfig(
        training_cfg["optimizer"], multilabel, optimizer_kwargs
    )["optimizer"]
    default_loss_config = parse_defaultconfig(
        training_cfg["loss"], multilabel, loss_kwargs
    )["loss"]

    # 6. Construct TrainingArguments
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
        decay_epochs           = decay_epochs,
        num_workers            = training_cfg.get("num_workers",         2),
        output_path            = training_cfg.get("output_path",         "./output"),
        num_tasks              = 1,
        resume_from_checkpoint = training_cfg.get("resume_from_checkpoint", True),
        save_checkpoint_every  = training_cfg.get("save_checkpoint_every",  5),
        project_name           = training_cfg.get("project_name",        "libauc"),
        experiment_name        = training_cfg["experiment_name"],
        verbose                = training_cfg.get("verbose",             1),
    )

    # 7. Initialise and run GNNTrainer
    logger.info("Initialising GNNTrainer...")
    trainer = GNNTrainer(
        train_args    = train_args,
        model_cfg     = model_cfg,
        train_dataset = train_dataset,
        eval_dataset  = eval_datasets if eval_datasets else None,
        metric        = metric_fn,
        callbacks     = [CLICallback()],
        decay_epochs  = decay_epochs,
        decay_factor  = decay_factor,
    )

    logger.info("Starting GNN training...")
    train_log = trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
