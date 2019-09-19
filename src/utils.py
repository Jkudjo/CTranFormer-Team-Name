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

from typing import Dict, List, Iterable

BEG = '__b__'
END = '__e__'
UNK = '__u__'
MSK = '__m__'
PAD = '__p__'

alphabet_l: List[str] = [BEG, END, UNK, MSK, PAD] + list(string.printable)
alphabet_d: Dict[str, int] = {k: v for v, k in enumerate(alphabet_l)}


def get_team_names(path: pathlib.Path) -> Iterable[str]:
    with path.open(encoding='utf-8') as file_io:
        for line in file_io.readlines():
            yield line.strip()


def vectorize(team_name: str, vocabulary_d: Dict[str, int]) -> List[int]:
    return ([vocabulary_d[BEG]] + [
        vocabulary_d[c] if c in vocabulary_d else vocabulary_d[UNK]
        for c in team_name
    ] + [vocabulary_d[END]])


def main():
    path = pathlib.Path(R"ctftime_team_names.txt")
    n = 1000
    for i, team in itertools.takewhile(lambda x: x[0] < n,
                                       enumerate(get_team_names(path))):
        print(team)
        print(vectorize(team, alphabet_d))


if __name__ == '__main__':
    main()
