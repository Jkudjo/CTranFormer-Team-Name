#!/usr/bin/env python3
"""
File:   utils.py
Author: Thibault Douzon
Date:   2019-09-16
        23:40:35
mail:   douzont@gmail.com
"""

import pathlib
import itertools
import string

import numpy as np
import torch

from typing import Dict, List, Iterable, Tuple
import random

BEG = '__b__'
END = '__e__'
UNK = '__u__'
MSK = '__m__'
PAD = '__p__'

alphabet_l: List[str] = [BEG, END, UNK, MSK, PAD] + list(string.printable)
alphabet_d: Dict[str, int] = {k: v for v, k in enumerate(alphabet_l)}


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_team_names(path: pathlib.Path) -> Iterable[str]:
    with path.open(encoding='utf-8') as file_io:
        for line in file_io.readlines():
            l = line.strip()
            if l:
                yield l


def split_dataset(path: pathlib.Path, proportions: Tuple, split_names=None):
    assert abs(sum(proportions) - 1.) < 1e-3, "Proportions must add up to 1"
    if split_names is None and len(proportions) <= 3:
        split_names = ['train', 'valid', 'test'][:len(proportions)]

    split_dataset_l = [[] for _ in proportions]
    proportion_acc = np.cumsum(proportions)

    for team_name in get_team_names(path):
        rnd = random.random()

        split_idx = next(i for i in range(len(proportions))
                         if rnd < proportion_acc[i])
        split_dataset_l[split_idx].append(team_name)

    split_path_l = []
    for split_suffix, split_data in zip(split_names, split_dataset_l):
        file_path = path.parent / \
            pathlib.Path(path.stem + f'_{split_suffix}' + path.suffix)
        with file_path.open('w', encoding='utf-8') as file_p:
            file_p.write('\n'.join(split_data))
        split_path_l.append(file_path)

    return split_path_l


def vectorize(team_name: str, vocabulary_d: Dict[str, int]) -> List[int]:
    return [
        vocabulary_d[c] if c in vocabulary_d else vocabulary_d[UNK]
        for c in team_name
    ]


def get_padding_mask(x: torch.Tensor, device='cpu') -> torch.ByteTensor:
    """get_padding_mask [summary]

    Args:
        x (torch.Tensor): (S/T, N) tensor

    Returns:
        torch.ByteTensor: (N, S/T)
    """
    return torch.where(x != alphabet_d[PAD],
                       torch.Tensor([False]).to(device),
                       torch.Tensor([True]).to(device)).transpose(1, 0).bool()


def main():
    path = pathlib.Path(R"ctftime_team_names.txt")
    n = 10
    for i, team in itertools.takewhile(lambda x: x[0] < n,
                                       enumerate(get_team_names(path))):
        print(team)
        print(vectorize(team, alphabet_d))

    print(split_dataset(path, (0.7, 0.2, 0.1)))


if __name__ == '__main__':
    main()
