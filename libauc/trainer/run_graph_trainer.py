"""
Entry point for LibAUC graph-classification training with GraphTrainer.

Usage::

    python -m libauc.trainer.run_graph_trainer --config_file config.yaml

CLI flags mirror ``TrainingArguments`` fields and override the YAML config
when supplied.
"""

import argparse
import logging
import sys

from omegaconf import OmegaConf

from .config import TrainingArguments, parse_defaultconfig
from .core import CLICallback, GraphTrainer
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
        "decay_factor":           10.0,
        "num_workers":            2,
        "output_path":            "./output",
        "resume_from_checkpoint": True,
        "save_checkpoint_every":  5,
        "project_name":           "libauc",
        "experiment_name":        "run_graph",
        "verbose":                1,
    },
    "dataset": {
        "name":        "",
        "kwargs":      {},
        "eval_splits": ["val"],
    },
    "model": {
        # num_tasks is inferred from TrainingArguments (always 1 for graph tasks)
        "name":       "gcn",
        "emb_dim":    256,
        "num_layers": 5,
    },
    "metrics":       ["AUROC"],
    "metric_kwargs": [],
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LibAUC graph-classification training")

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
    cfg      = build_megaconf(args.config_file, _MEGACONF_DEFAULTS)
    cfg      = apply_cli_overrides(cfg, args)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    training_cfg  = cfg_dict["training"]
    dataset_cfg   = cfg_dict["dataset"]
    model_cfg     = cfg_dict["model"]
    metric_names  = cfg_dict.get("metrics",       ["AUROC"])
    metric_kwargs = cfg_dict.get("metric_kwargs", [])

    # GNN-specific optional keys
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

    # 5. Resolve default optimizer / loss configs
    #    Graph tasks are always binary — multilabel=False.
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {})
    loss_kwargs      = training_cfg.get("loss_kwargs",      {})

    default_optimizer_config = parse_defaultconfig(
        training_cfg["optimizer"], multilabel=False, kwargs=optimizer_kwargs
    )["optimizer"]
    default_loss_config = parse_defaultconfig(
        training_cfg["loss"], multilabel=False, kwargs=loss_kwargs
    )["loss"]

    # 6. Construct TrainingArguments
    #    num_tasks=1: graph property prediction is always treated as binary.
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

    # 7. Initialise and run GraphTrainer
    logger.info("Initialising GraphTrainer...")
    trainer = GraphTrainer(
        train_args    = train_args,
        model_cfg     = model_cfg,
        train_dataset = train_dataset,
        eval_dataset  = eval_datasets if eval_datasets else None,
        metric        = metric_fn,
        callbacks     = [CLICallback()],
        decay_epochs  = decay_epochs,
        decay_factor  = decay_factor,
    )

    logger.info("Starting graph training...")
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
