#!/usr/bin/env python3
"""
File:   char_prediction.py
Author: Thibault Douzon
Date:   2019-09-16
        23:40:29
mail:   douzont@gmail.com
Playing with https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import src.utils


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        v, m = args
        self.embedding = Embedding(v, m)
        self.transformer = nn.modules.transformer.Transformer(
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_mask: torch.Tensor,
                y_mask: torch.Tensor):
        # TODO: Add masks
        xx = self.embedding(x)
        yy = self.embedding(y)

        y_mask = self.transformer.generate_square_subsequent_mask(y.size(0))

        x_padding_mask = self._padding_mask(x)
        y_padding_mask = self._padding_mask(y)
        zz = self.transformer(xx,
                              yy,
                              src_mask=x_mask,
                              tgt_mask=y_mask,
                              src_key_padding_mask=x_padding_mask,
                              tgt_key_padding_mask=y_padding_mask)

        return zz

    @staticmethod
    def _padding_mask(x: torch.Tensor) -> torch.ByteTensor:
        """_padding_mask [summary]
        
        Args:
            x (torch.Tensor): (S/T, N) input tensor
        
        Returns:
            torch.ByteTensor: (N, S/T) byte tensor for padding
        """
        return src.utils.get_padding_mask(x)


class Embedding(nn.Module):
    def __init__(self,
                 vocabulary_size: int,
                 model_size: int,
                 max_sequence_size: int = 16):
        super().__init__()
        self._vocabulary_size = vocabulary_size
        self._model_size = model_size
        self._max_sequence_size = max_sequence_size

        self.embedding = nn.Embedding(vocabulary_size, model_size)
        self._register_positional_encoding()

    def _register_positional_encoding(self) -> None:
        """Computes the positional encoding
        
        Returns:
            None --
        """

        positional_encoding = torch.zeros(self._max_sequence_size,
                                          self._model_size)

        position = (torch.arange(
            0, self._max_sequence_size, dtype=torch.float64).view(-1, 1) *
                    torch.ones(self._max_sequence_size,
                               self._model_size // 2,
                               dtype=torch.float64))

        harmonic = 10000**(
            torch.arange(0, self._model_size, 2, dtype=torch.float64) /
            self._model_size)

        positional_encoding[:, ::2] = torch.sin(position / harmonic)
        positional_encoding[:, 1::2] = torch.cos(position / harmonic)

        positional_encoding.unsqueeze_(1)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """Embedds the tensor and add positional encoding

        Arguments:
            x {torch.LongTensor} -- (S, N) tensor containing character indices

        Returns:
            torch.Tensor -- (S, N, D) embedded input
        """
        xx = self.embedding(x[:self._max_sequence_size, ...])
        xx += self.positional_encoding[:min(xx.size(0), self._max_sequence_size
                                            ), ...]

        return xx


def main():
    model = Model(10, 16)

    v = torch.arange(0, 10).view(-1, 1).long()

    print(model.embedding(v))


if __name__ == '__main__':
    main()
