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
from typing import Callable


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
        data, targets = batch

        optimizer.zero_grad()
        
        outputs = model(data, targets)
        
        output_flat = outputs.reshape(-1, model.vocabulary_size).double()
        target_flat = targets.view(-1).long()
        loss = criterion(output_flat, target_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            
            current_lr = next(param_group['lr'] for param_group in scheduler.optimizer.param_groups)
            print(f'| epoch {epoch:3d} '
                  f'| batch {i:5d}/{len(dataset_loader)//dataset_loader.batch_size:5d} '
                  f'| lr {current_lr:02.5f} '
                  f'| ms/batch {elapsed * 1000 / log_interval:5.2f} '
                  f'| loss {cur_loss:5.2f} '
                  f'| ppl {math.exp(cur_loss):8.2f} |', end='\r')
            total_loss = 0.
            start_time = time.time()


def evaluate(eval_model: torch.nn.Module,
             dataset_loader_valid: src.inputs.TeamNameLoader,
             criterion: Callable):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for batch in dataset_loader_valid:
            data, targets = batch
            
            outputs = eval_model(data, targets)
            output_flat = outputs.reshape(-1, eval_model.vocabulary_size).double()
            target_flat = targets.view(-1).long()
            
            loss = criterion(output_flat, target_flat).detach().item()
            
            total_loss += data.size(1) * loss
    return total_loss / len(dataset_loader_valid)


def train(model: torch.nn.Module,
          dataset_loader_train: src.inputs.TeamNameLoader,
          dataset_loader_valid: src.inputs.TeamNameLoader,
          learning_rate: float,
          epochs: int,
          ):

    criterion = nn.NLLLoss(ignore_index=src.utils.alphabet_d[src.utils.PAD])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.3,
                                                           patience=2)
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_epoch(model,
                    dataset_loader_train,
                    scheduler,
                    criterion,
                    epoch)
        val_loss = evaluate(model, dataset_loader_valid, criterion)
        print('-' * 90)
        print(f'| end of epoch {epoch:3d} '
              f'| time: {(time.time() - epoch_start_time):5.2f}s '
              f'| valid loss {val_loss:5.2f} '
              f'| valid ppl {math.exp(val_loss):8.2f} |')
        print('-' * 90)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step(val_loss)


def main():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dataset_path = pathlib.Path(R"ctftime_team_names.txt")
    vocabulary_d = src.utils.alphabet_d

    dataset_path_train, dataset_path_valid, dataset_path_test = src.utils.split_dataset(
        dataset_path, (0.2, 0.2, 0.6))

    # --- TRAINING PARAMS ---
    learning_rate = 1e-4
    batch_size = 64
    epochs = 10

    # --- MODEL PARAMS ---
    model_size = 16
    head_n = 4
    encoder_layers_n = 2
    decoder_layers_n = 2
    feedforward_size = 64

    dataset_loader_train = src.inputs.get_dataset(dataset_path_train,
                                                  mask=False,
                                                  batch_size=batch_size,
                                                  drop_last=True,
                                                  device=device)

    dataset_loader_valid = src.inputs.get_dataset(dataset_path_valid,
                                                  mask=False,
                                                  batch_size=batch_size,
                                                  drop_last=True,
                                                  device=device)

    dataset_loader_test = src.inputs.get_dataset(dataset_path_test,
                                                  mask=False,
                                                  batch_size=1,
                                                  drop_last=True,
                                                  device=device)

    model = src.char_prediction.Model(vocabulary_size=len(vocabulary_d),
                                      model_size=model_size,
                                      head_n=head_n,
                                      encoder_layers_n=encoder_layers_n,
                                      decoder_layers_n=decoder_layers_n,
                                      feedforward_size=feedforward_size,
                                      device=device)

    train(model,
          dataset_loader_train,
          dataset_loader_valid,
          learning_rate,
          epochs)
    
    n = 10
    with torch.no_grad():
        for i, batch in itertools.takewhile(lambda x: x[0] < n,
                                            enumerate(dataset_loader_test)):
            data, tgt = batch
            
            # out = model(data, tgt)
            
            # out_max = torch.argmax(out, dim=-1).reshape(-1)
            
            decoded = model.greedy_decode(data)
            
            data_name = ''.join(src.utils.alphabet_l[i] for i in data.reshape(-1))
            team_name = ''.join(src.utils.alphabet_l[i] for i in decoded.reshape(-1))
            
            print(data_name, team_name)
        pass


if __name__ == '__main__':
    main()
