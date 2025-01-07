from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Callable, Optional, Dict

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from tqdm.auto import tqdm

from constants import DataSplit


@dataclass
class TrainingConfig:
    batch_size: int
    context_length: int
    maximum_iterations: int
    eval_intervals: int
    eval_iterations: int
    checkpoint_interval: int
    checkpoint_dir: str
    device: str = 'cpu'
    project_name: str = 'transformer_training'
    experiment_name: str = f'run_{int(time.time())}'


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, callback: Callable = lambda x: x):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_callback = callback

    def save_checkpoint(self,
                        model: Module,
                        optimizer: Optimizer,
                        scheduler: Optional[_LRScheduler],
                        iteration: int,
                        loss_history: Dict,
                        config: TrainingConfig) -> None:
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
            'config': vars(config)
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / f'checkpoint_{iteration}.pt'
        torch.save(checkpoint, checkpoint_path)

        try:
            self._checkpoint_callback(checkpoint, checkpoint_path)
        except Exception as e:
            # We will ignore all exceptions as training must go on
            logging.info(f'Checkpoint callback raised an exception: {e}')

        # Save the latest checkpoint reference
        latest_path = self.checkpoint_dir / 'latest.txt'
        latest_path.write_text(str(checkpoint_path))

    def load_latest_checkpoint(self,
                               model: Module,
                               optimizer: Optimizer,
                               scheduler: Optional[_LRScheduler] = None) -> tuple[int, Dict]:
        latest_path = self.checkpoint_dir / 'latest.txt'
        if not latest_path.exists():
            return 0, {DataSplit.TRAIN.name: [], DataSplit.VALIDATION.name: []}

        checkpoint_path = Path(latest_path.read_text().strip())
        if not checkpoint_path.exists():
            return 0, {DataSplit.TRAIN.name: [], DataSplit.VALIDATION.name: []}

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['iteration'], checkpoint['loss_history']


def setup_logging(config: TrainingConfig) -> None:
    log_dir = Path(config.checkpoint_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'{config.experiment_name}.log'),
            logging.StreamHandler()
        ]
    )


@torch.no_grad()
def estimate_loss(model: Module,
                  eval_iterations: int,
                  get_data: Callable,
                  device: str) -> Dict[str, float]:
    out = {}
    model.eval()

    for split in [DataSplit.TRAIN, DataSplit.VALIDATION]:
        losses = []
        for _ in range(eval_iterations):
            xb, yb = get_data(split)
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type=device, enabled=device.startswith('cuda')):
                logits = model(xb)
                loss = model.calculate_loss(logits, yb)
            losses.append(loss.item())
        out[split.name] = sum(losses) / len(losses)

    model.train()
    return out


def get_batch(data: Tensor,
              batch_size: int,
              context_length: int,
              device: str) -> tuple[Tensor, Tensor]:
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i + context_length] for i in ix])
    y = torch.stack([data[i + 1:i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)


def train_transformer(model: Module,
                      optimizer: Optimizer,
                      scheduler: Optional[_LRScheduler],
                      get_data: Callable,
                      config: TrainingConfig,
                      checkpoint_callback: Callable = lambda x: x,
                      ) -> Dict[str, list]:
    # Setup
    setup_logging(config)
    checkpoint_manager = CheckpointManager(
        config.checkpoint_dir, checkpoint_callback)
    scaler = torch.amp.GradScaler(
        device=config.device, enabled=config.device.startswith('cuda'))

    # Load checkpoint if exists
    start_iter, loss_history = checkpoint_manager.load_latest_checkpoint(
        model, optimizer, scheduler)

    def get_batch_for_split(split: DataSplit) -> tuple[Tensor, Tensor]:
        data = get_data(split)
        return get_batch(data, config.batch_size, config.context_length, config.device)

    # Training loop with progress bar
    pbar = tqdm(
        range(start_iter, config.maximum_iterations),
        desc="Training",
        ncols=1000,
        leave=True
    )

    try:
        for iter_num in pbar:
            # Evaluation
            if iter_num % config.eval_intervals == 0:
                losses = estimate_loss(model, config.eval_iterations,
                                       get_batch_for_split, config.device)

                # Update history
                for k, v in losses.items():
                    loss_history.setdefault(k, []).append(v)

                # Update progress bar
                pbar.set_postfix({
                    'train_loss': f"{losses[DataSplit.TRAIN.name]:.4f}",
                    'val_loss': f"{losses[DataSplit.VALIDATION.name]:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                })

                # Logging
                logging.info(
                    f'Iteration {iter_num}/{config.maximum_iterations} - '
                    f'Train Loss: {losses[DataSplit.TRAIN.name]:.4f}, '
                    f'Val Loss: {losses[DataSplit.VALIDATION.name]:.4f}'
                )

            # Training step
            xb, yb = get_batch_for_split(DataSplit.TRAIN)

            with torch.amp.autocast(device_type=config.device, enabled=config.device.startswith('cuda')):
                logits = model(xb)
                loss = model.calculate_loss(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            # Checkpointing
            if (iter_num + 1) % config.checkpoint_interval == 0:
                checkpoint_manager.save_checkpoint(
                    model, optimizer, scheduler, iter_num, loss_history, config)
                logging.info(f'Saved checkpoint at iteration {iter_num}')

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        # Save checkpoint on interruption
        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, iter_num, loss_history, config)
        logging.info(f'Saved checkpoint at iteration {iter_num}')

    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise e

    # Final evaluation
    final_losses = estimate_loss(model, config.eval_iterations,
                                 get_batch_for_split, config.device)
    logging.info(
        f'Training completed.\n'
        f'Final Train Loss: {final_losses[DataSplit.TRAIN.name]:.4f}\n'
        f'Final Val Loss: {final_losses[DataSplit.VALIDATION.name]:.4f}'
    )

    return loss_history
