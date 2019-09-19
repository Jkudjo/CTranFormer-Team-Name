#!/usr/bin/env python3
"""
File:   char_prediction.py
Author: Thibault Douzon
Date:   2019-09-16
        23:40:29
mail:   douzont@gmail.com
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        v, m = args
        self.embedding = Embedding(v, m)
        self.transformer = nn.modules.transformer.Transformer()
        print(self.transformer.d_model)
        pass

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        xx = self.embedding(x)
        yy = self.embedding(y)
        zz = self.transformer(xx, yy)

        return zz


class Embedding(nn.Module):
    def __init__(self,
                 vocabulary_size: int,
                 model_size: int,
                 max_sequence_size: int = 256):
        super().__init__()
        self._vocabulary_size = vocabulary_size
        self._model_size = model_size
        self._max_sequence_size = max_sequence_size

        self.embedding = nn.Embedding(vocabulary_size, model_size)
        self.positional_encoding = self._positional_encoding()

    def _positional_encoding(self):
        positional_encoding = torch.zeros(self._max_sequence_size,
                                          self._model_size)

        position = (
            torch.arange(0, self._max_sequence_size).view(-1, 1).float() *
            torch.ones(self._max_sequence_size,
                       self._model_size // 2)).double()
        harmonic = (10000**(
            torch.arange(0, self._model_size, 2).float() *
            torch.ones(self._max_sequence_size, self._model_size // 2) /
            self._model_size)).double()

        positional_encoding[:, ::2] = torch.sin(position * harmonic)
        positional_encoding[:, 1::2] = torch.cos(position * harmonic)

        return positional_encoding.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward [summary]
        Assert input x is batched.
        """
        xx = self.embedding(x[:self._max_sequence_size, ...])
        xx += self.positional_encoding[:min(xx.size(0), self._max_sequence_size), ...]

        return xx


def main():
    model = Model(10, 512)

    v = torch.Tensor([[0], [0], [1], [2]]).long()

    print(model.embedding(v))


if __name__ == '__main__':
    main()
