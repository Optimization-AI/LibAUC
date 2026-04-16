import json
from typing import Any

import logging
logger = logging.getLogger(__name__)

class TrainingArguments:
    """Training configuration arguments."""
    
    def __init__(self, **kwargs):
        # Training args
        self.optimizer = kwargs.pop("optimizer")
        self.optimizer_kwargs = kwargs.pop("optimizer_kwargs")
        self.loss = kwargs.pop("loss")
        self.loss_kwargs = kwargs.pop("loss_kwargs")
        self.SEED = kwargs.pop("SEED")
        self.batch_size = kwargs.pop("batch_size")
        self.eval_batch_size = kwargs.pop("eval_batch_size")
        self.sampling_rate = kwargs.pop("sampling_rate")
        self.epochs = kwargs.pop("epochs")
        self.decay_epochs = kwargs.pop("decay_epochs")
        for i in range(len(self.decay_epochs)):
            if isinstance(self.decay_epochs[i], float):
                self.decay_epochs[i] = int(self.decay_epochs[i] * self.epochs)
        self.num_workers = kwargs.pop("num_workers")
        self.output_path = kwargs.pop("output_path")
        self.num_tasks = kwargs.pop("num_tasks")
        
        # Checkpoint parameters
        self.resume_from_checkpoint = kwargs.pop("resume_from_checkpoint")
        self.save_checkpoint_every = kwargs.pop("save_checkpoint_every")

        # wandb config
        self.project_name = kwargs.pop("project_name")
        self.experiment_name = kwargs.pop("experiment_name")

        # printing
        self.verbose  = kwargs.pop("verbose")