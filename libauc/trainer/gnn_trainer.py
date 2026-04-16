import logging
import os
from typing import Callable, List, Mapping, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .trainer import Trainer
from .args import TrainingArguments
from .callbacks import CallbackHandler, TrainerCallback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architectures available in libauc.models for graph learning
# ---------------------------------------------------------------------------
_GNN_REGISTRY = {
    "gcn":        "GCN",
    "gin":        "GIN",
    "gine":       "GINE",
    "graphsage":  "GraphSAGE",
    "gat":        "GAT",
    "mpnn":       "MPNN",
    "deepergcn":  "DeeperGCN",
    "pna":        "PNA",
}

class GNNTrainer(Trainer):

    def __init__(
        self,
        train_args:         TrainingArguments,
        model_cfg:          dict,
        train_dataset,
        eval_dataset:       Optional[List]                               = None,
        metric:             Optional[Callable[..., Mapping[str, float]]] = None,
        callbacks:          Optional[List[TrainerCallback]]              = None,
        decay_epochs:       Optional[List[int]]                          = None,
        decay_factor:       float                                         = 10.0,
        train_eval_dataset                                                = None,
    ):
        """
        Parameters
        ----------
        train_args          : TrainingArguments
        model_cfg           : dict passed to build_gnn_model()
        train_dataset       : PyG-compatible graph dataset (train split)
        eval_dataset        : list of PyG-compatible graph datasets (eval splits)
        metric              : callable (y_true, y_pred) -> dict of floats
        callbacks           : list of TrainerCallback instances
        decay_epochs        : epoch indices at which to decay the LR, e.g. [100, 200]
        decay_factor        : LR divisor applied at each decay epoch (default 10)
        train_eval_dataset  : optional separate dataset for unbiased train-split
                              evaluation; falls back to train_dataset when None
        """
        # Stash GNN-specific state before super().__init__ runs, because the
        # parent calls _get_train_dataloader() which we override below.
        self._model_cfg_gnn      = model_cfg
        self._with_edge_features = False
        self.decay_epochs        = decay_epochs or []
        self.decay_factor        = decay_factor
        self.train_eval_dataset  = train_eval_dataset

        # The parent will call build_model() (CNN path) and produce a
        # placeholder model.  We swap it out at the start of train().
        super().__init__(
            train_args    = train_args,
            model_cfg     = model_cfg,
            train_dataset = train_dataset,
            eval_dataset  = eval_dataset,
            metric        = metric,
            callbacks     = callbacks,
        )

    def _build_model(self, model_cfg: dict):
        """
        Build a GNN model from libauc.models.

        Required keys
        -------------
        name        : one of gcn | gin | gine | graphsage | gat | mpnn | deepergcn | pna
        num_tasks   : number of output tasks  (default 1)
        emb_dim     : node/edge embedding size (default 256)
        num_layers  : number of message-passing layers (default 5)

        Optional keys (forwarded verbatim to the model constructor)
        -----------------------------------------------------------
        graph_pooling, dropout, atom_features_dims, bond_features_dims,
        act, norm, jk, v2 (GAT only),
        aggr / t / learn_t / p / learn_p / block (DeeperGCN only),
        pretrained (bool), pretrained_path (str)

        Returns
        -------
        (model, with_edge_features: bool)
            with_edge_features is inferred from the model's `supports_edge_attr`
            class-level flag, which every model in libauc.models declares.
        """
        name = model_cfg.get("name", "").lower()
        if name not in _GNN_REGISTRY:
            raise ValueError(
                f"Unknown GNN model '{name}'. "
                f"Supported: {list(_GNN_REGISTRY.keys())}"
            )

        # Import the class from libauc.models
        import libauc.models as libauc_models
        cls_name  = _GNN_REGISTRY[name]
        model_cls = getattr(libauc_models, cls_name, None)
        if model_cls is None:
            raise ImportError(
                f"'{cls_name}' not found in libauc.models. "
                "Make sure your libauc version includes GNN support."
            )

        # ---- build constructor kwargs ----------------------------------------
        constructor_kwargs = dict(
            num_tasks  = model_cfg.get("num_tasks",  1),
            emb_dim    = model_cfg.get("emb_dim",    256),
            num_layers = model_cfg.get("num_layers", 5),
        )

        # Optional shared kwargs
        for key in ("graph_pooling", "dropout", "atom_features_dims",
                    "bond_features_dims", "act", "norm", "jk"):
            if key in model_cfg:
                constructor_kwargs[key] = model_cfg[key]

        # DeeperGCN-specific kwargs
        if name == "deepergcn":
            for key in ("aggr", "t", "learn_t", "p", "learn_p", "block"):
                if key in model_cfg:
                    constructor_kwargs[key] = model_cfg[key]

        # GAT-specific kwargs
        if name == "gat" and "v2" in model_cfg:
            constructor_kwargs["v2"] = model_cfg["v2"]

        model = model_cls(**constructor_kwargs).cuda()

        # ---- infer whether this architecture uses edge features ----------------
        # Every BasicGNN subclass in libauc.models declares `supports_edge_attr`
        # as a Final[bool] class attribute.  DeeperGCN doesn't inherit BasicGNN
        # but always takes edge_attr in forward(), so we fall back to a name-based
        # lookup.
        if hasattr(model_cls, "supports_edge_attr"):
            with_edge_features = bool(model_cls.supports_edge_attr)
        else:
            with_edge_features = name in {"deepergcn", "gine", "gat", "mpnn", "pna"}

        # ---- optional warm-start -----------------------------------------------
        if model_cfg.get("pretrained", False):
            pretrained_path = model_cfg.get("pretrained_path")
            if not pretrained_path:
                raise ValueError(
                    "pretrained=True but 'pretrained_path' is not set in model_cfg."
                )
            state_dict = torch.load(pretrained_path, weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            msg = model.load_state_dict(state_dict, strict=False)
            logger.info(f"GNN pretrained weights loaded: {msg}")
            model.graph_pred_linear.reset_parameters()

        logger.info(
            f"Built {cls_name} | emb_dim={constructor_kwargs['emb_dim']} "
            f"| num_layers={constructor_kwargs['num_layers']} "
            f"| with_edge_features={with_edge_features}"
        )
        
        self._with_edge_features  = with_edge_features
        self.model = model

    # ------------------------------------------------------------------
    # DataLoader overrides – use PyG's DataLoader
    # ------------------------------------------------------------------

    def _get_train_dataloader(self, train_args: TrainingArguments):
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from libauc.sampler import DualSampler

        sampler = DualSampler(
            self.train_dataset,
            train_args.batch_size,
            sampling_rate=train_args.sampling_rate,
        )

        loader = PyGDataLoader(
            self.train_dataset,
            batch_size  = train_args.batch_size,
            sampler     = sampler,
            num_workers = train_args.num_workers,
        )
        return sampler, loader

    def _get_eval_dataloader(self, dataset, train_args: TrainingArguments):
        from torch_geometric.loader import DataLoader as PyGDataLoader

        return PyGDataLoader(
            dataset,
            batch_size  = train_args.eval_batch_size,
            shuffle     = False,
            num_workers = train_args.num_workers,
        )

    # ------------------------------------------------------------------
    # GNN forward-pass helper
    # ------------------------------------------------------------------

    def _forward(self, model, batch):
        """
        Run a forward pass using the signature appropriate for this architecture.

            supports_edge_attr=False → model(x, edge_index, batch)
            supports_edge_attr=True  → model(x, edge_index, edge_attr, batch)

        Returns sigmoid probabilities, shape [N, num_tasks].
        """
        if self._with_edge_features:
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        else:
            logits = model(batch.x, batch.edge_index, batch.batch)
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        GNN training loop.

        Steps each epoch:
          1. Optional LR decay if epoch is in decay_epochs.
          2. Forward / backward over the training loader.
          3. Evaluation on the training split (unbiased loader).
          4. Evaluation on all registered eval loaders.
          5. Callbacks and periodic checkpointing.

        Returns
        -------
        list : training log produced by the state / callback system
        """

        self.callback_handler.on_train_begin(self.args, self.state)
        model        = self.model.cuda()
        self.loss_fn = self.loss_fn.cuda()

        # Optional checkpoint resume
        if self.args.resume_from_checkpoint:
            latest = self.get_latest_checkpoint(
                os.path.join(self.args.output_path, self.args.experiment_name)
            )
            if latest:
                self.load_checkpoint(latest)
                logger.info(f"Resuming from epoch {self.state.epoch}")
            else:
                logger.info("No checkpoint found, starting from scratch.")

        for epoch in range(self.state.epoch, self.args.epochs):
            self.callback_handler.on_epoch_begin(self.args, self.state)

            # ── Optional LR decay ───────────────────────────────────────────
            if epoch in self.decay_epochs:
                self.optimizer.update_lr(decay_factor=self.decay_factor)
                logger.info(
                    f"Epoch {epoch}: LR decayed by /{self.decay_factor}. "
                    f"New LR = {self.optimizer.lr:.6f}"
                )

            # ── Training ────────────────────────────────────────────────────
            model.train()
            train_loss = []

            for batch in self.trainloader:
                self.callback_handler.on_step_begin(self.args, self.state)

                data, targets, index = batch
                data = data.cuda()
                targets = targets.cuda()
                index = index.cuda()
                pred    = self._forward(model, data)

                # Compute loss
                if self.args.loss == "CrossEntropyLoss":
                    raise ValueError("CrossEntropyLoss not supported for GNNs.")
                if self.args.loss == "BCELoss":
                    loss = self.loss_fn(pred, targets.reshape(-1,1))
                else:
                    loss = self.loss_fn(pred, targets, index=index)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                self.callback_handler.on_step_end(self.args, self.state)

            # ── Evaluation ──────────────────────────────────────────────────
            model.eval()
            avg_train_loss = float(np.mean(train_loss))

            eval_metrics, test_true, test_pred = self.evaluate_loop(model)

            self.callback_handler.on_epoch_end(
                self.args,
                self.state,
                metrics    = eval_metrics,
                train_loss = avg_train_loss,
                lr         = self.optimizer.lr,
                test_true  = test_true,
                test_pred  = test_pred,
            )

            # ── Checkpointing ───────────────────────────────────────────────
            if (
                (epoch + 1) % self.args.save_checkpoint_every == 0
                or (epoch + 1) == self.args.epochs
            ):
                ckpt = os.path.join(
                    self.args.output_path,
                    self.args.experiment_name,
                    f"epoch_{epoch + 1}.pt",
                )
                self.save_checkpoint(ckpt)

        self.callback_handler.on_train_end(self.args, self.state)
        return self.state.train_log

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _eval_single_loader(self, model, loader):
        """
        Run inference over *loader* and compute the registered metric.

        Returns
        -------
        (metrics_dict, y_true np.ndarray, y_pred np.ndarray)
        """
        pred_list, true_list = [], []

        with torch.no_grad():
            for batch in loader:
                data, targets, index = batch
                data = data.cuda()
                targets = targets.cuda()
                pred    = self._forward(model, data)
                pred_list.append(pred.cpu().detach().numpy())
                true_list.append(targets.cpu().detach().numpy())

        y_true = np.concatenate(true_list)
        y_pred = np.concatenate(pred_list)

        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        if y_true.ndim > 1:
            y_true = y_true.flatten()

        metrics = self.metric(y_true, y_pred) if self.metric else {}
        return metrics, y_true, y_pred

    def evaluate(self, loader, model):
        """
        Override base Trainer.evaluate() to use the GNN forward pass.

        Args
        ----
        loader : PyG DataLoader
        model  : GNN model

        Returns
        -------
        (metrics_dict, y_true, y_pred)
        """
        return self._eval_single_loader(model, loader)