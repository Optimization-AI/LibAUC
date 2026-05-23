import logging
import sys
from typing import List

from ..config.args import TrainingArguments

logger = logging.getLogger(__name__)


class TrainerState:
    """State object to track training progress."""
    
    def __init__(self):
        self.epoch = 0
        self.total_epoch = 0
        self.step = 0
        self.train_log = []
        self.train_summary = {}


class TrainerCallback:
    r"""
    Base class for training lifecycle callbacks.

    Every method is a no-op by default, so subclasses only need to override
    the hooks they care about.  Instances are registered with
    :class:`~trainer.core.callbacks.CallbackHandler`, which calls each hook in
    registration order and forwards a consistent set of keyword arguments
    (``model``, ``optimizer``, ``loss_fn``, plus any extra kwargs the
    :class:`~trainer.core.image_trainer.Trainer` supplies for that event).

    Lifecycle order during a typical training run::

        on_init_end
        on_train_begin
          for each epoch:
            on_epoch_begin
              for each step:
                on_step_begin
                on_step_end
            on_evaluate
            on_epoch_end
          [on_save — called periodically inside the epoch loop]
        on_train_end

    All callback methods are optional and can be overridden in subclasses.
    """
    
    def __init__(self) -> None:
        pass
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called at the end of trainer initialization. No-op in base class."""
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called once before the first epoch. No-op in base class."""
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called once after the last epoch. No-op in base class."""
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called at the start of each epoch. No-op in base class."""
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called at the end of each epoch. No-op in base class."""
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called before each optimizer step. No-op in base class."""
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called after each gradient-accumulation sub-step. No-op in base class."""
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called after each optimizer step. No-op in base class."""
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called after each evaluation pass. No-op in base class."""
        pass

    def on_predict(self, args: TrainingArguments, state: TrainerState, metrics, **kwargs):
        """Called after a prediction pass. No-op in base class."""
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called after a checkpoint is saved. No-op in base class."""
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called after metrics are logged. No-op in base class."""
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Called after each prediction step. No-op in base class."""
        pass


class CallbackHandler(TrainerCallback):
    r"""
    Multiplexer that owns a list of :class:`~TrainerCallback` instances and
    fans out every lifecycle event to each of them in registration order.

    ``CallbackHandler`` itself inherits from :class:`~TrainerCallback` so it
    can be used polymorphically, but its primary role is orchestration rather
    than providing hook implementations of its own.

    Args:
        callbacks (list[TrainerCallback]): Initial callback list.
        model: The model being trained.
        optimizer: The active optimizer.
        loss_fn: The active loss function.
    """
    
    def __init__(self, callbacks: List[TrainerCallback], model, optimizer, loss_fn):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def add_callback(self, callback):
        """Add a callback to the handler."""
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. "
                f"The current list of callbacks is:\n{self.callback_list}"
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        """Remove and return a callback."""
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        """Remove a callback without returning it."""
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        """Get a string representation of all callbacks."""
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_init_end", args, state)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_train_begin", args, state)

    def on_train_end(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_train_end", args, state)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_epoch_begin", args, state)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, metrics, **kwargs):
        return self._call_event("on_epoch_end", args, state, metrics=metrics, **kwargs)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_step_begin", args, state)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_substep_end", args, state)

    def on_step_end(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_step_end", args, state)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_evaluate", args, state)

    def on_predict(self, args: TrainingArguments, state: TrainerState, metrics):
        return self._call_event("on_predict", args, state, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_save", args, state)

    def on_log(self, args: TrainingArguments, state: TrainerState, logs):
        return self._call_event("on_log", args, state, logs=logs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState):
        return self._call_event("on_prediction_step", args, state)

    def _call_event(self, event, args, state, **kwargs):
        """Invoke *event* on every registered callback in order.

        ``model``, ``optimizer``, and ``loss_fn`` are always injected into
        ``**kwargs`` so every callback receives them as ``kwargs["model"]``,
        ``kwargs["optimizer"]``, and ``kwargs["loss_fn"]`` regardless of what
        the caller passes.  Any additional keyword arguments from the caller
        are merged in after.
        """
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                **kwargs,
            )


def _format_metrics(log: dict, skip_keys: tuple = ("epoch", "train_loss", "lr")) -> str:
    """Format metric key-value pairs into a display string, excluding skip_keys."""
    return " | ".join(f"{k}: {v:.4f}" for k, v in log.items() if k not in skip_keys)


def _build_log_dict(
    metrics: list,
    train_loss: float,
    lr: float,
    epoch: int,
) -> dict:
    """
    Build a flat log dict from epoch metrics, suitable for console output and wandb.

    Returns a dict with keys: epoch, train_loss, lr, and one entry per metric per dataset.
    Single-dataset runs use bare metric names; multi-dataset runs prefix with 'ds{N}/'.
    """
    log: dict[str, float] = {
        "epoch":      epoch,
        "train_loss": train_loss,
        "lr":         lr,
    }

    single = len(metrics) == 1
    for ds_idx, ds_metrics in enumerate(metrics):
        if not isinstance(ds_metrics, dict):
            continue
        prefix = "" if single else f"eval_splits{ds_idx + 1}/"
        for k, v in ds_metrics.items():
            if k in ("epoch", "lr", "loss"):
                continue
            try:
                log[f"{prefix}{k}"] = float(v)
            except (ValueError, TypeError):
                pass

    return log


class CLICallback(TrainerCallback):
    r"""
    Console and Weights & Biases logging callback.

    On ``on_train_begin`` it initialises a W&B run (silently falls back to
    console-only when W&B is not installed) and pretty-prints the full
    :class:`~trainer.config.args.TrainingArguments` config.

    On ``on_epoch_end`` it:

    * appends a structured entry to ``state.train_log``;
    * renders a progress bar (``verbose=1``) or a per-epoch line
      (``verbose=2``) to stdout;
    * ships the flat log dict to W&B via ``wandb.log``.

    On ``on_train_end`` it prints a training summary (best validation and test
    scores) and calls ``wandb.finish()``.

    .. note::
        This callback takes no constructor arguments.  All configuration is
        read from :class:`~libauc.trainer.config.args.TrainingArguments` at
        runtime via the ``args`` parameter passed to each lifecycle hook.

    Note:
        W&B logging is silently disabled when ``wandb`` is not installed or
        when ``wandb.log`` raises an exception.
    """

    # Width of the progress bar fill (verbose=1)
    _BAR_WIDTH = 30

    def __init__(self) -> None:
        super().__init__()
        self._use_wandb = True
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wandb_log(self, log: dict, step: int) -> None:
        """Log a dict to wandb, silently skipping if unavailable."""
        if not self._use_wandb:
            return
        try:
            import wandb
            wandb.log(log, step=step)
        except Exception as e:
            logger.warning(f"wandb logging failed: {e}")

    def _render_bar(self, epoch: int, total: int, log: dict) -> str:
        filled   = int(self._BAR_WIDTH * epoch / total) if total else 0
        empty    = self._BAR_WIDTH - filled
        bar      = "█" * filled + "·" * empty
        metrics  = _format_metrics(log)
        metrics_str = f" | {metrics}" if metrics else ""
        return (
            f"\rEpoch [{bar}] {epoch}/{total} | "
            f"Loss: {log.get('train_loss', 0):.4f}"
            f"{metrics_str} | "
            f"LR: {log.get('lr', 0):.6f}"
        )

    # ------------------------------------------------------------------
    # Callback events — what each hook does in CLICallback
    # ------------------------------------------------------------------
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Initialise W&B and print the training config.

        Attempts to start a W&B run using ``args.project_name`` and
        ``args.experiment_name`` with the full ``TrainingArguments`` dict as
        the run config.  If ``wandb`` is not installed the run is silently
        skipped and ``self._use_wandb`` is set to ``False`` so all subsequent
        W&B calls are no-ops.

        When ``args.verbose != 0``, pretty-prints the resolved config between
        two ``====`` separator lines so the user can confirm all hyperparameters
        before training starts.
        """
        try:
            import wandb
            config = {
                k: v for k, v in vars(args).items()
                if not k.startswith("_")
            }
            wandb.init(project=args.project_name, name=args.experiment_name, reinit=True, config=config)
        except ImportError:
            logger.warning("wandb not installed; skipping wandb logging")
            self._use_wandb = False

        if args.verbose == 0:
            return

        import pprint

        config = {k: v for k, v in vars(args).items() if not k.startswith("_")}

        print("=" * 60)
        pprint.pprint(config, indent=2, sort_dicts=False)
        print("=" * 60)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Apply learning-rate / regulariser decay if the epoch is scheduled.

        Checks whether ``state.epoch`` is listed in ``args.decay_epochs``.
        If so, decays by a fixed factor of 10:
        """
        optimizer = kwargs.get("optimizer")
        if optimizer and state.epoch in args.decay_epochs:
            if getattr(optimizer, "model_ref", None) is not None:
                optimizer.update_regularizer(decay_factor=10)
            else:
                optimizer.update_lr(decay_factor=10)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Record metrics, update the console display, and log to W&B.

        1. **Appends** a structured record ``{metrics, epoch, lr, train_loss}``
           to ``state.train_log`` for later retrieval (e.g. by ``on_train_end``).
        2. **Console output** (controlled by ``args.verbose``):

           * ``verbose=1`` — overwrites a single progress-bar line in place
             using ``\\r``, showing a block-character bar, loss, all eval
             metrics, and the current LR.  A newline is printed only on the
             final epoch so the bar persists after training.
           * ``verbose=2`` — prints one pipe-separated line per epoch
             (``Epoch N/T | Loss: X | AUROC: Y | LR: Z``).
           * ``verbose=0`` — no console output.

        3. **W&B** — ships the flat log dict to ``wandb.log`` (no-op if W&B
           is unavailable).
        4. **Increments** ``state.epoch``.
        """
        metrics:    list  = kwargs.get("metrics", [])
        train_loss: float = kwargs.get("train_loss", 0)
        lr:         float = kwargs.get("lr", 0)

        state.train_log.append({
            "metrics":    metrics,
            "epoch":      state.epoch + 1,
            "lr":         lr,
            "train_loss": train_loss,
        })

        log = _build_log_dict(metrics, train_loss, lr, epoch=state.epoch + 1)

        # ---- Console output (mode-dependent) ----------------------------
        if args.verbose == 1:
            # Overwrite the same line with an updated progress bar
            bar_str = self._render_bar(state.epoch + 1, state.total_epoch, log)
            sys.stdout.write(bar_str)
            sys.stdout.flush()
            # Print a newline only on the very last epoch so the bar stays
            # on screen after training ends
            if state.epoch + 1 >= state.total_epoch:
                sys.stdout.write("\n")
                sys.stdout.flush()

        elif args.verbose == 2:
            # One line per epoch (original behaviour)
            display_parts = [
                f"Epoch {state.epoch + 1}/{state.total_epoch}",
                f"Loss: {train_loss:.4f}",
            ]
            display_parts += [
                f"{k}: {v:.4f}"
                for k, v in log.items()
                if k not in ("epoch", "train_loss", "lr")
            ]
            display_parts.append(f"LR: {lr:.6f}")
            print(" | ".join(display_parts))

        # ---- wandb logging (always active unless unavailable) -----------
        self._wandb_log(log, step=state.epoch + 1)

        state.epoch += 1

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Print a training summary and close the W&B run.

        1. Prints a ``"Training complete."`` separator when ``verbose != 0``.
        2. Calls ``wandb.finish()`` to flush and close the W&B run.
        3. Scans ``state.train_log`` to find the best epoch (highest value of
           the first metric on the first eval split) and reports:

           * **1 eval split** — best validation score.
           * **2+ eval splits** — best validation score + average of all
             remaining test split scores at that epoch.

        4. Stores the summary dict under ``state.train_summary``.
        """
        if args.verbose != 0:
            print("-" * 50)
            print("Training complete.")

        if self._use_wandb:
            try:
                import wandb
                wandb.finish()
            except ImportError:
                pass

        train_log = state.train_log
        if not train_log:
            raise ValueError("Training should have at least one evaluation record.")

        # No eval datasets registered — nothing to summarise.
        if not train_log[0]['metrics']:
            logger.info("No evaluation datasets registered; skipping training summary.")
            return

        train_summary = {}
        target    = list(train_log[0]['metrics'][0].keys())[0]
        best_idx  = max(range(len(train_log)), key=lambda i: train_log[i]['metrics'][0][target])
        num_evals = len(train_log[0]['metrics'])

        if num_evals == 1:
            val = train_log[best_idx]['metrics'][0][target]
            logger.info(f"best validation {target}: {val}")
            train_summary["val"] = val
        else:
            val   = train_log[best_idx]['metrics'][0][target]
            score = sum(
                train_log[best_idx]['metrics'][x][target] for x in range(1, num_evals)
            ) / (num_evals - 1)
            logger.info(f"best validation {target}: {val}, best test avg. {target}: {score}")
            train_summary["val"]  = val
            train_summary["test"] = score

        state.train_summary = train_summary

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Increment the global step counter.

        Increments ``state.step`` by 1 after every optimizer update, keeping a
        running total of training steps across all epochs.
        """
        state.step += 1