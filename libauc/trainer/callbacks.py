import logging
import sys
from typing import Any, Dict, List, Optional

from .args import TrainingArguments

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
    """
    Base callback class for training events.
    
    All callback methods are optional and can be overridden in subclasses.
    """
    
    def __init__(self) -> None:
        pass
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of trainer initialization."""
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of training."""
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of training."""
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of an epoch."""
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of an epoch."""
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of a training step."""
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a substep during gradient accumulation."""
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a training step."""
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after an evaluation phase."""
        pass

    def on_predict(self, args: TrainingArguments, state: TrainerState, metrics, **kwargs):
        """Event called after a successful prediction."""
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after a checkpoint save."""
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after logging the last logs."""
        pass

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called after a prediction step."""
        pass


class CallbackHandler(TrainerCallback):
    """
    Handler that manages multiple callbacks and calls them in order.
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
        """Call the specified event on all callbacks."""
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                model=self.model,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                **kwargs,
            )


class DefaultCallback(TrainerCallback):
    """Default callback with basic functionality."""
    
    def __init__(self) -> None:
        super().__init__()
    
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the beginning of an epoch."""
        optimizer = kwargs.get("optimizer")
        if optimizer and state.epoch in args.decay_epochs:
            if getattr(optimizer, "model_ref", None) is not None:
                optimizer.update_regularizer(decay_factor=10)
            else:
                optimizer.update_lr(decay_factor=10)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of an epoch."""
        state.epoch += 1

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a training step."""
        state.step += 1


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
    # Callback events
    # ------------------------------------------------------------------
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, **kwargs):
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
        """Event called at the beginning of an epoch."""
        optimizer = kwargs.get("optimizer")
        if optimizer and state.epoch in args.decay_epochs:
            if getattr(optimizer, "model_ref", None) is not None:
                optimizer.update_regularizer(decay_factor=10)
            else:
                optimizer.update_lr(decay_factor=10)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of an epoch."""
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
        """Event called at the end of training."""
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

        train_summary = {}
        target    = list(train_log[0]['metrics'][0].keys())[0]
        id        = max(range(len(train_log)), key=lambda i: train_log[i]['metrics'][0][target])
        num_evals = len(train_log[0]['metrics'])

        if num_evals == 0:
            raise ValueError("Evaluation should contain at least one dataset split.")
        elif num_evals == 1:
            val = train_log[id]['metrics'][0][target]
            logger.info(f"best validation {target}: {val}")
            train_summary["val"] = val
        elif num_evals == 2:
            val   = train_log[id]['metrics'][0][target]
            score = train_log[id]['metrics'][1][target]
            logger.info(f"best validation {target}: {val}, best test {target}: {score}")
            train_summary["val"]  = val
            train_summary["test"] = score
        else:
            val   = train_log[id]['metrics'][0][target]
            score = sum(
                train_log[id]['metrics'][x][target] for x in range(1, num_evals)
            ) / (num_evals - 1)
            logger.info(f"best validation {target}: {val}, best test avg. {target}: {score}")
            train_summary["val"]  = val
            train_summary["test"] = score

        state.train_summary = train_summary

    def on_step_end(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Event called at the end of a training step."""
        state.step += 1