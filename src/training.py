import torch
import torch.nn as nn
import numpy as np

import src.char_prediction
import src.inputs
import src.utils

import time
import pathlib
import math

criterion = nn.CrossEntropyLoss()
lr = 1e-2  # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='min',
                                                       factor=0.3,
                                                       patience=3)


def train_epoch(model: nn.Module, dataset_loader: src.inputs.TeamNameLoader,
                optimiser: torch.optim.Optimizer,
                criterion: torch.nn.Function):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch in dataset_loader:
        data, targets = batch
        
        optimizer.zero_grad()
        
        output = model(data)
        loss = criterion(output, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(
                          train_data) // bptt, scheduler.get_lr()[0],
                      elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


best_val_loss = float("inf")
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_epoch()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()


def main():
    dataset_path = pathlib.Path(R"ctftime_team_names.txt")
    vocabulary_d = src.utils.alphabet_d

    # --- TRAINING PARAMS ---
    learning_rate = 1e-2
    batch_size = 32
    epochs = 10

    # --- MODEL PARAMS ---
    model_size = 128
    head_n = 6
    encoder_layers_n = 3
    decoder_layers_n = 3
    feedforward_size = 512

    dataset_loader = src.inputs.get_dataset(dataset_path,
                                            mask=True,
                                            batch_size=batch_size,
                                            drop_last=False)

    model = src.char_prediction.Model(vocabulary_size=len(vocabulary_d),
                                      model_size=model_size,
                                      head_n=head_n,
                                      encoder_layers_n=encoder_layers_n,
                                      decoder_layers_n=decoder_layers_n,
                                      feedforward_size=feedforward_size)
    pass


if __name__ == '__main__':
    main()
