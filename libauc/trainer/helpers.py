
from libauc.metrics import auc_prc_score, auc_roc_score, pauc_roc_score
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric builder
# ---------------------------------------------------------------------------

def build_metric(metric_names, metric_kwargs):
    """
    Build a metric function from a list of metric name strings.

    Args:
        metric_names: e.g. ["AUROC", "AUPRC", "ACC"]

    Returns:
        Callable (test_true: np.ndarray, test_pred: np.ndarray) -> dict[str, float]
    """
    from sklearn import metrics as skmetrics

    def metric_fn(test_true, test_pred):
        results = {}
        for id, name in enumerate(metric_names):
            if id < len(metric_kwargs):
                kwargs = metric_kwargs[id]
            else:
                kwargs = {}
            name_upper = name.upper()
            if name_upper == "AUROC":
                if "max_fpr" in kwargs.keys() or "min_tpr" in kwargs.keys():
                    args = ', '.join([str(k) + '=' + str(v) for k, v in kwargs.items()])
                    results[f"PAUROC({args})"] = pauc_roc_score(test_true, test_pred, **kwargs)
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