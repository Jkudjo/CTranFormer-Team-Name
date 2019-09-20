#!/usr/bin/env python3

"""
File:   inputs.py
Author: Thibault Douzon
Date:   2019-09-16
        23:40:21
mail:   douzont@gmail.com
"""
import pathlib

import torch
import torch.utils.data as data
import numpy as np

import src.utils

from typing import Dict


class TeamNameDataset(data.Dataset):
    def __init__(self, path: pathlib.Path,
                 vocabulary: Dict[str, int] = src.utils.alphabet_d):
        super().__init__()
        self.path = path
        self.vocabulary = vocabulary
        self._data = list(src.utils.get_team_names(path))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key: int):
        if 0 <= key < len(self._data):
            return src.utils.vectorize(self._data[key],
                                       self.vocabulary)
        else:
            raise IndexError


class TeamNameLoader:
    def __init__(self,
                 dataset: data.Dataset,
                 mask: bool,
                 batch_size: int,
                 drop_last: bool = False):
        super().__init__()
        self._dataset = dataset
        self.mask = mask
        self.batch_size = batch_size
        self.drop_last = drop_last

        # * TODO: A better than random sampler that take into account sample length
        self._sampler = data.BatchSampler(data.RandomSampler(self._dataset,
                                                             replacement=False),
                                          batch_size=self.batch_size,
                                          drop_last=self.drop_last)

    def __iter__(self):
        # * IDEA add temperature level to control no of masks
        pad_item = self._dataset.vocabulary[src.utils.PAD]
        msk_item = self._dataset.vocabulary[src.utils.MSK]
        for next_batch in self._sampler:
            max_seq_len = max(len(self._dataset[i]) for i in next_batch)
            next_target = torch.ones(
                max_seq_len, self.batch_size, dtype=torch.int64) * pad_item
            for i, tensor_idx in enumerate(next_batch):
                tensor = torch.Tensor(self._dataset[tensor_idx])
                next_target[:len(tensor), i] = tensor

                next_input = next_target.clone().detach()
                if self.mask:
                    rnd_mask_idx = torch.randint(1, len(tensor) - 1, (1,))
                    next_input[rnd_mask_idx, i] = msk_item

            yield next_input, next_target


def get_dataset(path: pathlib.Path,
                batch_size: int,
                shuffle: bool = True,
                drop_last: bool = False):
    dataset = TeamNameDataset(path)
    return TeamNameLoader(dataset, mask=True,
                          batch_size=batch_size, drop_last=drop_last)


def main():
    p = pathlib.Path("ctftime_team_names.txt")
    dataset = TeamNameDataset(p)
    dataloader = TeamNameLoader(dataset, True, 2)
    for i, d in enumerate(dataloader):
        if i > 2:
            break

        print(d)


if __name__ == '__main__':
    main()
