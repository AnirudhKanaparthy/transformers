import torch

from torch import Tensor
from torch.nn import Module

from constants import DataSplit
from tqdm import tqdm
from typing import Callable


def get_batch(data: Tensor,
              batch_size: int,
              context_length: int) -> tuple[Tensor, Tensor]:

    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i: i + context_length] for i in ix])
    y = torch.stack([data[i + 1: i + context_length + 1] for i in ix])

    return x, y


@torch.no_grad()
def estimate_loss(model: Module,
                  eval_iterations: int,
                  get_data: Callable) -> dict[str, Tensor]:
    out = {}
    model.eval()

    for split in [DataSplit.TRAIN, DataSplit.VALIDATION]:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            xb, yb = get_data(split)
            logits, loss = model(xb, yb)

            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


def train_transformer(model: Module,
                      optimizer: Module,
                      get_data: Callable,
                      batch_size: int,
                      context_length: int,
                      maximum_iterations: int,
                      eval_intervals: int,
                      eval_iterations: int,
                      device: str = 'cpu',
                      ):
    all_losses = {DataSplit.TRAIN: [], DataSplit.VALIDATION: []}
    t_indent = len(str(maximum_iterations))

    def get_batch_for_split(split: DataSplit) -> tuple[Tensor, Tensor]:
        x, y = get_batch(get_data(split), batch_size, context_length)
        return x.to(device), y.to(device)

    for iter in tqdm(range(maximum_iterations)):
        if iter % eval_intervals == 0:
            losses = estimate_loss(model, eval_iterations, get_batch_for_split)
            [all_losses[k].append(losses[k].item()) for k in losses]

            print(
                f'Iteration[{iter + 1 : >{t_indent}}/{maximum_iterations}], Training Loss: {losses[DataSplit.TRAIN] : .6f}, Validation Loss: {losses[DataSplit.VALIDATION]: .6f}')

        xb, yb = get_batch_for_split(DataSplit.TRAIN)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        optimizer.step()

    print(
        f'Final Loss:\n\tTraining: {losses[DataSplit.TRAIN] : .6f}\n\tValidation: {losses[DataSplit.VALIDATION]: .6f}')

    return all_losses
