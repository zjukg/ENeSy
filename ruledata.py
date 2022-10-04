import torch
from enum import Enum
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np


class Data():
    def __init__(self, data_path) -> None:
        with open(f'{data_path}/entities.txt') as e, open(f'{data_path}/relations.txt') as r:
            self.ents = [x.strip() for x in e.readlines()]
            self.rels = [x.strip() for x in r.readlines()]
            self.e2id = {i: int(i) for i in self.ents}
            self.r2id = {i: int(i) for i in self.rels}
            rels = self.rels + ['<slf>']
            self.rels_num = len(self.rels)
            self.id2r = {v:k for k, v in self.r2id.items()}
            self.id2e = {v:k for k, v in self.e2id.items()}
        self.data = {}
        with open(f'{data_path}/train.txt') as f:
            train = [item.strip().split('\t') for item in f.readlines()]
            self.data['train'] = list({(int(h), int(r), int(t)) for h, r, t in train})
        with open(f'{data_path}/test.txt') as f:
            test = [item.strip().split('\t') for item in f.readlines()]
            self.data['test'] = list({(int(h), int(r), int(t)) for h, r, t in test})
        with open(f'{data_path}/valid.txt') as f:
            valid = [item.strip().split('\t') for item in f.readlines()]
            self.data['valid'] = list({(int(h), int(r), int(t)) for h, r, t in valid})

        self.nx = {e: defaultdict(list) for e in range(len(self.id2e))}

        indices = [[] for _ in range(self.rels_num)]
        values = [[] for _ in range(self.rels_num)]

        for h, r, t in self.data['train']:
            indices[r].append((h, t))
            values[r].append(1)
            self.nx[h][t].append(r)
        indices = [torch.LongTensor(x).T for x in indices]
        values = [torch.FloatTensor(x) for x in values]
        size = torch.Size([len(self.ents), len(self.ents)])
        self.rel_mat = [torch.sparse.FloatTensor(indices[i], values[i], size).coalesce() for i in range(self.rels_num)]
        self.rel_mat.append(torch.sparse.FloatTensor(torch.LongTensor(
            [[i, i] for i in range(len(self.ents))]).T, torch.ones(len(self.ents)), size).coalesce())

    def getinfo(self):
        return len(self.ents), len(self.rels)

