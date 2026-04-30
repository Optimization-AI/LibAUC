"""
Entry point for AutoMAX training.

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
from .core import CLICallback, Trainer
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
        "num_workers":            2,
        "output_path":            "./output",
        "resume_from_checkpoint": True,
        "save_checkpoint_every":  5,
        "project_name":           "libauc",
        "experiment_name":        "run",
        "verbose":                1,
    },
    "dataset": {
        "name":         "",
        "kwargs":       {},
        "eval_splits":  ["val"],
    },
    "model": {
        "name":        "resnet18",
        "pretrained":  False,
        "num_classes": 1,
        "in_channels": 3,
    },
    "metrics":        ["AUROC"],
    "metric_kwargs":  [],
}


def _build_megaconf(config_path: str) -> OmegaConf:
    """
    Load a YAML config file and merge it on top of the megaconf defaults.

    The merge follows OmegaConf semantics: keys present in the YAML override
    defaults; keys absent in the YAML inherit the default value.

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
    parser = argparse.ArgumentParser(description="AutoMAX training script")

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to a YAML configuration file.",
    )

    # Each flag below mirrors a TrainingArguments field.  When supplied on the
    # CLI it overrides the value coming from the config file / megaconf.
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
# CLI override helper
# ---------------------------------------------------------------------------

def apply_cli_overrides(cfg: OmegaConf, args) -> OmegaConf:
    """
    Merge CLI-supplied values into the megaconf (mutates in-place).

    The mapping between CLI argument names and config keys is kept in a
    single dict so adding a new parameter only requires one line here —
    no logic duplication.  The ``cfg["training"]`` sub-dict is updated for
    training-scoped keys.

    Args:
        cfg (OmegaConf): Merged megaconf produced by :func:`_build_megaconf`.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        OmegaConf: The mutated config (same object).
    """
    # Map: (config section, config key) → argparse attribute name
    _CLI_MAP: list[tuple[str, str, str]] = [
        # section       cfg key                        args attr
        ("training",    "epochs",                      "epochs"),
        ("training",    "batch_size",                  "batch_size"),
        ("training",    "eval_batch_size",             "eval_batch_size"),
        ("training",    "sampling_rate",               "sampling_rate"),
        ("training",    "num_workers",                 "num_workers"),
        ("training",    "output_path",                 "output_path"),
        ("training",    "SEED",                        "seed"),
        ("training",    "resume_from_checkpoint",      "resume_from_checkpoint"),
        ("training",    "save_checkpoint_every",       "save_checkpoint_every"),
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

    # Convert to a plain Python dict for downstream compatibility
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
    if len(labels.shape) == 1:
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
    train_log = trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
