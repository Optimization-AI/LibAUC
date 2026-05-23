import logging

import numpy as np
import torch
from omegaconf import OmegaConf

from libauc.metrics import auc_prc_score, auc_roc_score, pauc_roc_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric builder
# ---------------------------------------------------------------------------

def build_metric(metric_names, metric_kwargs):
    """
    Build an evaluation function from a list of metric names.

    The returned callable computes each requested metric after every
    evaluation epoch and returns the results as a flat dict.  It does
    **not** affect the training objective — losses and optimizers are
    configured separately via ``TrainingArguments``.

    Supported metric names (case-insensitive):

    * ``"AUROC"`` — full-ROC AUROC (``libauc.metrics.auc_roc_score``).
      When ``metric_kwargs`` for this entry contains ``max_fpr`` or
      ``min_tpr``, partial AUROC (``libauc.metrics.pauc_roc_score``) is
      computed instead and the result key becomes e.g. ``"PAUROC(max_fpr=0.3)"``.
    * ``"AUPRC"`` — area under the precision-recall curve
      (``libauc.metrics.auc_prc_score``).
    * ``"ACC"`` — accuracy at a fixed decision threshold of 0.5
      (``sklearn.metrics.accuracy_score``).

    Unknown names are skipped with a warning.  The same name may appear
    more than once with different ``metric_kwargs`` entries (e.g. full AUROC
    and partial AUROC simultaneously).

    Args:
        metric_names (list[str]): Ordered list of metric names to compute,
            e.g. ``["AUROC", "AUPRC", "ACC"]``.
        metric_kwargs (list[dict]): Per-metric keyword arguments.
            ``metric_kwargs[i]`` is forwarded to the computation function for
            ``metric_names[i]``.  Missing entries default to ``{}``.

    Returns:
        Callable[[numpy.ndarray, numpy.ndarray], dict[str, float]]:
            A function ``metric_fn(test_true, test_pred) -> results`` where
            ``test_true`` is the 1-D array of ground-truth labels,
            ``test_pred`` is the 1-D array of model output scores, and
            ``results`` maps each metric name to its score.
    """
    from sklearn import metrics as skmetrics

    def metric_fn(test_true, test_pred):
        results = {}
        for idx, name in enumerate(metric_names):
            kwargs = metric_kwargs[idx] if idx < len(metric_kwargs) else {}
            name_upper = name.upper()
            if name_upper == "AUROC":
                if "max_fpr" in kwargs or "min_tpr" in kwargs:
                    args_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())
                    results[f"PAUROC({args_str})"] = pauc_roc_score(test_true, test_pred, **kwargs)
                else:
                    results["AUROC"] = auc_roc_score(test_true, test_pred)
            elif name_upper == "AUPRC":
                results["AUPRC"] = auc_prc_score(test_true, test_pred)
            elif name_upper == "ACC":
                results["ACC"] = skmetrics.accuracy_score(
                    test_true, (test_pred >= 0.5).astype(int)
                )
            else:
                logger.warning(f"Unknown metric '{name}', skipping.")
        return results

    return metric_fn


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set global random seed for NumPy, PyTorch, and cuDNN."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    logger.info(f"Global seed set to {seed}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def build_megaconf(config_path: str, defaults: dict) -> OmegaConf:
    """
    Load a YAML config file and merge it on top of *defaults*.

    The merge follows OmegaConf semantics: keys present in the YAML override
    defaults; keys absent in the YAML inherit the default value.

    Args:
        config_path (str): Path to the YAML configuration file.
        defaults (dict): Default configuration values.

    Returns:
        OmegaConf DictConfig: Fully resolved configuration.
    """
    base   = OmegaConf.create(defaults)
    user   = OmegaConf.load(config_path)
    merged = OmegaConf.merge(base, user)
    return merged


# ---------------------------------------------------------------------------
# CLI override helper
# ---------------------------------------------------------------------------

def apply_cli_overrides(cfg: OmegaConf, args) -> OmegaConf:
    """
    Merge CLI-supplied values into the megaconf (mutates in-place).

    The mapping between CLI argument names and config keys is kept in a
    single list so adding a new parameter only requires one entry here.

    Args:
        cfg (OmegaConf): Merged megaconf produced by :func:`build_megaconf`.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        OmegaConf: The mutated config (same object).
    """
    # (section, cfg_key, args_attr)
    _CLI_MAP: list[tuple[str, str, str]] = [
        ("training", "epochs",                 "epochs"),
        ("training", "batch_size",             "batch_size"),
        ("training", "eval_batch_size",        "eval_batch_size"),
        ("training", "sampling_rate",          "sampling_rate"),
        ("training", "num_workers",            "num_workers"),
        ("training", "output_path",            "output_path"),
        ("training", "SEED",                   "seed"),
        ("training", "resume_from_checkpoint", "resume_from_checkpoint"),
        ("training", "save_checkpoint_every",  "save_checkpoint_every"),
    ]

    for section, cfg_key, arg_attr in _CLI_MAP:
        value = getattr(args, arg_attr, None)
        if value is not None:
            logger.info(f"CLI override: {section}.{cfg_key} = {value}")
            OmegaConf.update(cfg, f"{section}.{cfg_key}", value)

    return cfg
