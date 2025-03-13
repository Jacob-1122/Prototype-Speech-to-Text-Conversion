import logging
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class RobustMonitoringCallback(TrainerCallback):
    """
    A custom callback that logs additional training metrics to TensorBoard.
    You can log metrics like the learning rate, loss per step, and even custom metrics.
    """

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def on_step_end(self, args, state, control, **kwargs):
        # This method is called at the end of every training step.
        # The 'logs' dictionary in state contains metrics like loss.
        if state.log_history:
            # Log the most recent log entries
            last_logs = state.log_history[-1]
            step = state.global_step
            for key, value in last_logs.items():
                # Skip if not a scalar
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
                    logger.info(f"Step {step}: {key} = {value}")

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()
