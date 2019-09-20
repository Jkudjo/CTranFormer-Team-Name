import torch
import torch.nn as nn
import numpy as np

import src.char_prediction
import src.inputs
import src.utils

import time
import pathlib
import math
from typing import Callable


def train_epoch(model: nn.Module,
                dataset_loader: src.inputs.TeamNameLoader,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                criterion: Callable,
                epoch: int):
    optimizer = scheduler.optimizer
    log_interval = 200
    total_loss = 0.

    start_time = time.time()

    model.train()  # Turn on the train mode
    for i, batch in enumerate(dataset_loader):
        data, targets = batch

        optimizer.zero_grad()
        
        output = model(data, targets)
        loss = criterion(output, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if batch % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f'| epoch {epoch:3d} '
                  f'| batch {i:5d}/{len(dataset_loader):5d} '
                  f'| lr {scheduler.get_lr()[0]:02.2f} '
                  f'| ms/batch {elapsed * 1000 / log_interval:5.2f} '
                  f'| loss {cur_loss:5.2f} '
                  f'| ppl {math.exp(cur_loss):8.2f} |\r')
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model: torch.nn.Module,
             dataset_loader_valid: src.inputs.TeamNameLoader,
             criterion: Callable):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i, batch in dataset_loader_valid:
            data, targets = batch
            output = eval_model(data, targets)
            output_flat = output.view(-1, output.size(-1))
            total_loss += data.size(1) * criterion(output_flat, targets).item()
    return total_loss / len(dataset_loader_valid)


def train(model: torch.nn.Module,
          dataset_loader_train: src.inputs.TeamNameLoader,
          dataset_loader_valid: src.inputs.TeamNameLoader,
          learning_rate: float,
          epochs: int,
          ):

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.3,
                                                           patience=3)
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_epoch(model,
                    dataset_loader_train,
                    scheduler,
                    criterion,
                    epoch)
        val_loss = evaluate(model, dataset_loader_valid)
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} '
              f'| time: {(time.time() - epoch_start_time):5.2f}s '
              f'| valid loss {val_loss:5.2f} '
              f'| valid ppl {math.exp(val_loss):8.2f} |')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()


def main():
    dataset_path = pathlib.Path(R"ctftime_team_names.txt")
    vocabulary_d = src.utils.alphabet_d

    dataset_path_train, dataset_path_valid, dataset_path_test = src.utils.split_dataset(
        dataset_path, (0.7, 0.2, 0.1))

    # --- TRAINING PARAMS ---
    learning_rate = 1e-2
    batch_size = 32
    epochs = 10

    # --- MODEL PARAMS ---
    model_size = 128
    head_n = 4
    encoder_layers_n = 3
    decoder_layers_n = 3
    feedforward_size = 512

    dataset_loader_train = src.inputs.get_dataset(dataset_path_train,
                                                  mask=True,
                                                  batch_size=batch_size,
                                                  drop_last=False)

    dataset_loader_valid = src.inputs.get_dataset(dataset_path_valid,
                                                  mask=True,
                                                  batch_size=batch_size,
                                                  drop_last=False)

    model = src.char_prediction.Model(vocabulary_size=len(vocabulary_d),
                                      model_size=model_size,
                                      head_n=head_n,
                                      encoder_layers_n=encoder_layers_n,
                                      decoder_layers_n=decoder_layers_n,
                                      feedforward_size=feedforward_size)

    train(model,
          dataset_loader_train,
          dataset_loader_valid,
          learning_rate,
          epochs)
    pass


if __name__ == '__main__':
    main()
