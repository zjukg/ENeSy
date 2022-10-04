import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def KGE(mode, nentity, nrelation, hidden_dim, gamma, embedding_range):
    if mode == 'TransE':
        entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim))
        relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
        offset_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
    elif mode == 'RotatE':
        entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim * 2))
        relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
        offset_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 2))
    elif mode == 'HAKE':
        entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim * 2))
        relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 3))
        offset_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 2))
    elif mode == 'ComplEx':
        entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim * 2))
        relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 2))
        offset_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim * 2))
    elif mode == 'DistMult':
        entity_embedding = nn.Parameter(torch.zeros(nentity, hidden_dim))
        relation_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
        offset_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))

    epsilon = 2.0
    gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
    nn.init.uniform_(tensor=entity_embedding, a=-embedding_range.item(), b=embedding_range.item())
    nn.init.uniform_(tensor=relation_embedding, a=-embedding_range.item(), b=embedding_range.item())
    nn.init.uniform_(tensor=offset_embedding, a=embedding_range.item()/2, b=embedding_range.item())

    return entity_embedding, relation_embedding, offset_embedding


def KGEcalculate(mode, embedding, rembedding, embedding_range):
    if mode == 'TransE':
        result = embedding + rembedding
        return result
    elif mode == 'RotatE':
        pi = 3.14159262358979323846
        re_head, im_head = torch.chunk(embedding, 2, dim=-1)
        phase_relation = rembedding/(embedding_range.item()/pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_tail = re_head * re_relation - im_head * im_relation
        im_tail = re_head * im_relation + im_head * re_relation

        return torch.cat((re_tail, im_tail), dim=-1)

    elif mode == 'HAKE':
        pi = 3.14159262358979323846
        phase_head, mod_head = torch.chunk(embedding, 2, dim=-1)
        phase_rela, mod_rela, bias_rela = torch.chunk(rembedding, 3, dim=-1)

        phase_head = phase_head / (embedding_range.item() / pi)
        phase_rela = phase_rela / (embedding_range.item() / pi)

        phase_result = (phase_head + phase_rela)
        phase_result = phase_result * (embedding_range.item() / pi)

        mod_rela = torch.abs(mod_rela)
        bias_rela = torch.clamp(bias_rela, max=1)

        indicator = (bias_rela < -mod_rela)
        bias_rela[indicator] = -mod_rela[indicator]

        mod_result = mod_head * ((mod_rela + bias_rela)/(1-bias_rela))

        return torch.cat((phase_result, mod_result), dim=-1)

    elif mode == 'ComplEx':
        re_head, im_head = torch.chunk(embedding, 2, dim=-1)
        re_relation, im_relation = torch.chunk(rembedding, 2, dim=-1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        return torch.cat((re_score, im_score), dim=-1)

    elif mode == 'DistMult':
        return embedding * rembedding


def KGELoss(mode, embedding, target_embedding, gamma, phase_weight=None, modules_weight=None, embedding_range=None):
    if mode == 'TransE':
        score = embedding - target_embedding
        score = gamma - torch.norm(score, p=1, dim=-1)
        return score

    elif mode == 'RotatE':
        re_head, im_head = torch.chunk(embedding, 2, dim=-1)
        re_tail, im_tail = torch.chunk(target_embedding, 2, dim=-1)

        re_score = re_head - re_tail
        im_score = im_head - im_tail

        score = torch.cat([re_score, im_score], dim=-1)
        score = gamma - torch.norm(score, p=1, dim=-1)
        return score

    elif mode == 'HAKE':
        phase_head, mod_head = torch.chunk(embedding, 2, dim=-1)
        phase_tail, mod_tail = torch.chunk(target_embedding, 2, dim=-1)

        pi = 3.14159262358979323846
        phase_head = phase_head / (embedding_range.item() / pi)
        phase_tail = phase_tail / (embedding_range.item() / pi)

        phase_score = phase_head - phase_tail
        r_score = mod_head - mod_tail

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=-1) * phase_weight
        r_score = torch.norm(r_score, dim=-1) * modules_weight
        return gamma - (phase_score + r_score)

    elif mode == 'ComplEx':
        re_head, im_head = torch.chunk(embedding, 2, dim=-1)
        re_tail, im_tail = torch.chunk(target_embedding, 2, dim=-1)

        return torch.sum(re_tail * re_head + im_tail * im_head, dim=-1)

    elif mode == 'DistMult':
        score = embedding * target_embedding
        score = score.sum(dim=-1)
        return score

