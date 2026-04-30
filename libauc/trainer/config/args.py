import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

class TrainingArguments:
    r"""
    Container for all hyperparameters and settings that govern a single
    training run.

    All fields map one-to-one to keys in the ``training`` section of a YAML
    config file and can be overridden from the CLI via ``apply_cli_overrides``.

    Args:
        optimizer (str): Name of the libauc optimizer class, e.g. ``"PESG"``,
            ``"PDSCA"``, ``"SOAP"``.
        optimizer_kwargs (dict): Extra keyword arguments forwarded verbatim to
            the optimizer constructor (e.g. ``lr``, ``momentum``,
            ``weight_decay``).
        loss (str): Name of the loss-function class, e.g. ``"AUCMLoss"``,
            ``"CompositionalAUCLoss"``.  Looked up first in
            ``libauc.losses``, then ``torch.nn``.
        loss_kwargs (dict): Extra keyword arguments forwarded verbatim to the
            loss constructor.
        SEED (int): Global random seed for NumPy, PyTorch and cuDNN
            (default: ``42``).
        batch_size (int): Mini-batch size for training (default: ``128``).
        eval_batch_size (int): Mini-batch size for evaluation
            (default: ``128``).
        sampling_rate (float): Positive-class sampling rate passed to
            :class:`~libauc.sampler.DualSampler` / ``TriSampler``
            (default: ``0.5``).
        epochs (int): Total number of training epochs (default: ``50``).
        decay_epochs (list): Epoch indices (or fractional multiples of
            ``epochs``) at which the learning-rate / regulariser is decayed.
            Floats are converted to ``int(f * epochs)`` at construction time.
        num_workers (int): Number of DataLoader worker processes
            (default: ``2``).
        output_path (str): Root directory for checkpoints and logs
            (default: ``"./output"``).
        num_tasks (int): Number of output tasks / classes.  ``1`` → binary;
            ``≥ 3`` → multi-label with :class:`~libauc.sampler.TriSampler`.
        resume_from_checkpoint (bool): Whether to resume from the latest
            checkpoint found in ``output_path/experiment_name``
            (default: ``True``).
        save_checkpoint_every (int): Save a checkpoint every *N* epochs
            (default: ``5``).
        project_name (str): Weights & Biases project name
            (default: ``"libauc"``).
        experiment_name (str): Weights & Biases run name; also used as the
            checkpoint sub-directory.
        verbose (int): Verbosity level.  ``0`` = silent; ``1`` = progress bar;
            ``2`` = one line per epoch (default: ``1``).

    Example::

        >>> args = TrainingArguments(
        ...     optimizer="PESG",
        ...     optimizer_kwargs={"lr": 0.1, "momentum": 0.9},
        ...     loss="AUCMLoss",
        ...     loss_kwargs={"margin": 1.0},
        ...     SEED=42,
        ...     batch_size=128,
        ...     eval_batch_size=128,
        ...     sampling_rate=0.5,
        ...     epochs=50,
        ...     decay_epochs=[],
        ...     num_workers=2,
        ...     output_path="./output",
        ...     num_tasks=1,
        ...     resume_from_checkpoint=True,
        ...     save_checkpoint_every=5,
        ...     project_name="libauc",
        ...     experiment_name="my_experiment",
        ...     verbose=1,
        ... )
    """

    def __init__(self, **kwargs):
        # ── Core training settings ───────────────────────────────────────────
        self.optimizer         = kwargs.pop("optimizer")
        self.optimizer_kwargs  = kwargs.pop("optimizer_kwargs")
        self.loss              = kwargs.pop("loss")
        self.loss_kwargs       = kwargs.pop("loss_kwargs")
        self.SEED              = kwargs.pop("SEED")
        self.batch_size        = kwargs.pop("batch_size")
        self.eval_batch_size   = kwargs.pop("eval_batch_size")
        self.sampling_rate     = kwargs.pop("sampling_rate")
        self.epochs            = kwargs.pop("epochs")

        # Convert fractional decay epochs to absolute epoch indices
        self.decay_epochs = kwargs.pop("decay_epochs")
        for i in range(len(self.decay_epochs)):
            if isinstance(self.decay_epochs[i], float):
                self.decay_epochs[i] = int(self.decay_epochs[i] * self.epochs)

        self.num_workers  = kwargs.pop("num_workers")
        self.output_path  = kwargs.pop("output_path")
        self.num_tasks    = kwargs.pop("num_tasks")

        # ── Checkpoint settings ──────────────────────────────────────────────
        self.resume_from_checkpoint = kwargs.pop("resume_from_checkpoint")
        self.save_checkpoint_every  = kwargs.pop("save_checkpoint_every")

        # ── Weights & Biases settings ────────────────────────────────────────
        self.project_name    = kwargs.pop("project_name")
        self.experiment_name = kwargs.pop("experiment_name")

        # ── Logging / display ────────────────────────────────────────────────
        self.verbose = kwargs.pop("verbose")


# ---------------------------------------------------------------------------
# Default config resolver
# ---------------------------------------------------------------------------

def parse_defaultconfig(type_name: str, multilabel: bool = False, kwargs: dict = {}):
    r"""
    Resolve a loss or optimizer name to its canonical ``{optimizer, loss}``
    configuration dict by looking up the corresponding
    :mod:`~trainer.config.spaces` class.

    The mapping covers every loss/optimizer pair supported by libauc:

    +-----------------------------+------------------------------+
    | ``type_name``               | Space class                  |
    +=============================+==============================+
    | ``AUCMLoss`` / ``PESG``     | ``AUCMLossSpace``            |
    |                             | (``MultiLabelAUCMLossSpace`` |
    |                             | when ``multilabel=True``)    |
    +-----------------------------+------------------------------+
    | ``CompositionalAUCLoss`` /  | ``CompositionalAUCLossSpace``|
    | ``PDSCA``                   |                              |
    +-----------------------------+------------------------------+
    | ``APLoss`` / ``SOAP``       | ``APLossSpace``              |
    |                             | (``mAPLossSpace`` when       |
    |                             | ``multilabel=True``)         |
    +-----------------------------+------------------------------+
    | ``pAUC_CVaR_Loss`` /        | ``pAUC_CVaR_LossSpace``      |
    | ``SOPA`` / ``pAUCLoss``     | (``MultiLabel…`` variant)    |
    | mode ``SOPA``               |                              |
    +-----------------------------+------------------------------+
    | ``pAUC_DRO_Loss`` /         | ``pAUC_DRO_LossSpace``       |
    | ``SOPAs`` / ``pAUCLoss``    | (``MultiLabel…`` variant)    |
    | mode ``1w``                 |                              |
    +-----------------------------+------------------------------+
    | ``tpAUC_KL_Loss`` /         | ``tpAUC_KL_LossSpace``       |
    | ``SOTAs`` / ``pAUCLoss``    | (``MultiLabel…`` variant)    |
    | mode ``2w``                 |                              |
    +-----------------------------+------------------------------+
    | ``tpAUC_CVaR_loss`` /       | ``tpAUC_CVaR_lossSpace``     |
    | ``STACO``                   |                              |
    +-----------------------------+------------------------------+
    | ``NDCGLoss`` / ``SONG``     | ``NDCGLossSpace``            |
    +-----------------------------+------------------------------+
    | ``CrossEntropyLoss`` /      | ``SGDSpace``                 |
    | ``SGD``                     |                              |
    +-----------------------------+------------------------------+
    | ``Adam``                    | ``AdamSpace``                |
    +-----------------------------+------------------------------+
    | ``BCELoss``                 | ``BCELossSpace``             |
    +-----------------------------+------------------------------+

    Args:
        type_name (str): Name of the loss or optimizer class.
        multilabel (bool): When ``True``, selects the multi-label variant of
            the space if one exists (default: ``False``).
        kwargs (dict): Additional keyword arguments for the loss/optimizer,
            used to disambiguate ``pAUCLoss`` by its ``mode`` key.

    Returns:
        dict: ``{"optimizer": <optimizer_cfg>, "loss": <loss_cfg>}`` where
        each config is a dict with at least a ``"type"`` key and a ``"space"``
        key containing the hyperparameter search space.

    Raises:
        ValueError: If *type_name* is not recognised.

    Example::

        >>> cfg = parse_defaultconfig("AUCMLoss", multilabel=False)
        >>> cfg["optimizer"]["type"]
        'PESG'
        >>> cfg["loss"]["type"]
        'AUCMLoss'
    """
    if type_name in ('AUCMLoss', 'PESG'):
        if multilabel:
            from .spaces import MultiLabelAUCMLossSpace as Sp
        else:
            from .spaces import AUCMLossSpace as Sp
    elif type_name in ('CompositionalAUCLoss', 'PDSCA'):
        from .spaces import CompositionalAUCLossSpace as Sp
    elif type_name in ('APLoss', 'SOAP'):
        if multilabel:
            from .spaces import mAPLossSpace as Sp
        else:
            from .spaces import APLossSpace as Sp
    elif type_name in ('pAUC_CVaR_Loss', 'SOPA') or (
        type_name == 'pAUCLoss' and kwargs.get('mode') == 'SOPA'
    ):
        if multilabel:
            from .spaces import MultiLabelpAUC_CVaR_LossSpace as Sp
        else:
            from .spaces import pAUC_CVaR_LossSpace as Sp
    elif type_name in ('pAUC_DRO_Loss', 'SOPAs') or (
        type_name == 'pAUCLoss' and kwargs.get('mode') == '1w'
    ):
        if multilabel:
            from .spaces import MultiLabelpAUC_DRO_LossSpace as Sp
        else:
            from .spaces import pAUC_DRO_LossSpace as Sp
    elif type_name in ('tpAUC_KL_Loss', 'SOTAs') or (
        type_name == 'pAUCLoss' and kwargs.get('mode') == '2w'
    ):
        if multilabel:
            from .spaces import MultiLabeltpAUC_KL_LossSpace as Sp
        else:
            from .spaces import tpAUC_KL_LossSpace as Sp
    elif type_name in ('tpAUC_CVaR_loss', 'STACO'):
        from .spaces import tpAUC_CVaR_lossSpace as Sp
    elif type_name in ('NDCGLoss', 'SONG'):
        from .spaces import NDCGLossSpace as Sp
    elif type_name in ('CrossEntropyLoss', 'SGD'):
        from .spaces import SGDSpace as Sp
    elif type_name in ('Adam',):
        from .spaces import AdamSpace as Sp
    elif type_name in ('BCELoss',):
        from .spaces import BCELossSpace as Sp
    else:
        raise ValueError(f"Unsupported loss/optimizer type: '{type_name}'")

    return {"optimizer": Sp.optimizer, "loss": Sp.loss}