import torch
import torch.nn as nn
import numpy as np

import src.char_prediction
import src.inputs
import src.utils

import time
import pathlib
import math
import itertools
from typing import Callable, List


def train_epoch(model: nn.Module,
                dataset_loader: src.inputs.TeamNameLoader,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                criterion: Callable,
                epoch: int):
    optimizer = scheduler.optimizer
    log_interval = 5
    total_loss = 0.

    start_time = time.time()

    model.train()  # Turn on the train mode
    for i, batch in enumerate(dataset_loader):
        data, targets, truth = batch

        optimizer.zero_grad()

        outputs = model(data, targets)

        output_flat = outputs.reshape(-1, model.vocabulary_size).double()
        truth_flat = truth.reshape(-1).long()
        loss = criterion(output_flat, truth_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time

            current_lr = next(
                param_group['lr']
                for param_group in scheduler.optimizer.param_groups)
            print(
                f'| epoch {epoch:3d} '
                f'| batch {i:5d}/{len(dataset_loader)//dataset_loader.batch_size:5d} '
                f'| lr {current_lr:02.5f} '
                f'| ms/batch {elapsed * 1000 / log_interval:6.2f} '
                f'| loss {cur_loss:5.2f} '
                f'| ppl {math.exp(cur_loss):8.2f} |',
                end='\r')
            total_loss = 0.
            start_time = time.time()


def evaluate(eval_model: torch.nn.Module,
             dataset_loader_valid: src.inputs.TeamNameLoader,
             criterion: Callable):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for batch in dataset_loader_valid:
            data, targets, truth = batch

            outputs = eval_model(data, targets)
            output_flat = outputs.reshape(-1,
                                          eval_model.vocabulary_size).double()
            truth_flat = truth.reshape(-1).long()

            loss = criterion(output_flat, truth_flat).detach().item()

            total_loss += data.size(1) * loss
    return total_loss / len(dataset_loader_valid)


def raise_dataset_temperature(dataset_l: List[src.inputs.TeamNameLoader],
                              epoch: int):

    for dataset in dataset_l:
        dataset.temperature = epoch * 0.


def train(
        model: torch.nn.Module,
        dataset_loader_train: src.inputs.TeamNameLoader,
        dataset_loader_valid: src.inputs.TeamNameLoader,
        learning_rate: float,
        epochs: int):

    criterion = nn.NLLLoss(ignore_index=src.utils.alphabet_d[src.utils.PAD])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.3,
                                                           patience=3)
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_epoch(model, dataset_loader_train, scheduler, criterion, epoch)
        val_loss = evaluate(model, dataset_loader_valid, criterion)
        print('-' * 90)
        print(f'| end of epoch {epoch:3d} '
              f'| time: {(time.time() - epoch_start_time):6.2f}s '
              f'| valid loss {val_loss:5.2f} '
              f'| valid ppl {math.exp(val_loss):8.2f} |')
        print('-' * 90)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step(val_loss)
        raise_dataset_temperature(
            [dataset_loader_train, dataset_loader_valid], epoch)


def main():
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    device = 'cpu'

    dataset_path = pathlib.Path(R"ctftime_team_names.txt")
    vocabulary_d = src.utils.alphabet_d

    dataset_path_train, dataset_path_valid, dataset_path_test = src.utils.split_dataset(
        dataset_path, (0.2, 0.1, 0.7))

    # --- TRAINING PARAMS ---
    learning_rate = 5e-3
    batch_size = 128
    epochs = 3

    # --- MODEL PARAMS ---
    model_size = 32
    head_n = 2
    encoder_layers_n = 1
    decoder_layers_n = 1
    feedforward_size = 64

    dataset_loader_train = src.inputs.get_dataset(dataset_path_train,
                                                  mask=True,
                                                  batch_size=batch_size,
                                                  drop_last=True,
                                                  device=device)

    dataset_loader_valid = src.inputs.get_dataset(dataset_path_valid,
                                                  mask=True,
                                                  batch_size=batch_size,
                                                  drop_last=True,
                                                  device=device)

    dataset_loader_test = src.inputs.get_dataset(dataset_path_test,
                                                 mask=True,
                                                 batch_size=1,
                                                 initial_temperature=0.0,
                                                 drop_last=True,
                                                 device=device)

    # * DEMO DATA *
    # dataset_train = src.inputs.DemoDataset(vocab_size=20, size=1000)
    # dataset_loader_train = src.inputs.DemoLoader(dataset_train, 10, device)

    # dataset_valid = src.inputs.DemoDataset(20, 100)
    # dataset_loader_valid = src.inputs.DemoLoader(dataset_valid, 10, device)

    # dataset_test = src.inputs.DemoDataset(20, 100)
    # dataset_loader_test = src.inputs.DemoLoader(dataset_test, 1, device)
    # * END DEMO DATA *

    model = src.char_prediction.Model(vocabulary_size=len(vocabulary_d),
                                      model_size=model_size,
                                      head_n=head_n,
                                      encoder_layers_n=encoder_layers_n,
                                      decoder_layers_n=decoder_layers_n,
                                      feedforward_size=feedforward_size,
                                      dropout_transformer=0.2,
                                      dropout_embedding=0.1,
                                      device=device)

    train(model, dataset_loader_train, dataset_loader_valid, learning_rate,
          epochs)

    n = 100
    model.eval()
    with torch.no_grad():
        for i, batch in itertools.takewhile(lambda x: x[0] < n,
                                            enumerate(dataset_loader_test)):
            data, tgt, truth = batch

            # out = model(data, tgt)

            # out_max = torch.argmax(out, dim=-1).reshape(-1)

            decoded = model(data, tgt)

            data_name = ''.join(src.utils.alphabet_l[i]
                                for i in truth[:-1, :].reshape(-1))
            team_name = ''.join(
                src.utils.alphabet_l[i]
                for i in model.greedy_decode(data))

            print(data_name, team_name)
        pass


if __name__ == '__main__':
    main()
