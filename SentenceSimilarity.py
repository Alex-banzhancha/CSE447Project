import math
import random

import torch
import torch.nn.functional as f
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


class CrossSimilarity:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', threshold=0.6, rt=0.3):
        self.model = SentenceTransformer(model_name)
        self.thresh = threshold
        self.propose_rate = rt

    def __tokenize(self, text, summary):
        text_embd = torch.tensor(self.model.encode(text))
        summary_embd = torch.tensor(self.model.encode(summary))
        return text_embd, summary_embd

    def __masked_cross_sim(self, text_embd, summary_embd):
        res = torch.zeros(summary_embd.shape[0], text_embd.shape[0])
        for i in range(summary_embd.shape[0]):
            res[i] = f.cosine_similarity(text_embd, summary_embd[i], dim=-1)
        return torch.masked_fill(res, res < self.thresh, 0).float()

    def __sentences_propose(self, masked_sim, text_embd):
        # return a list of proposed sentence, giving sim, text_embd
        sum_size = masked_sim.shape[0]
        text_size = masked_sim.shape[1]
        res = []
        for i in range(sum_size):
            # each summary should propose 2 sentences if exists sentence give enough similarity
            # if propose rate is high or text size is huge, the proposal should contain more
            res.extend(self.__propose(masked_sim[i], text_embd,
                                      math.ceil(text_size * self.propose_rate / sum_size)))
        res_tensor = torch.IntTensor(res)
        res_tensor = torch.unique(res_tensor)
        return res_tensor

    def compute(self, text, summary):
        text = text.split(".")
        summary = summary.split(".")
        text_embd, summary_embd = self.__tokenize(text, summary)
        masked_cross_sim = self.__masked_cross_sim(text_embd, summary_embd)
        proposed = self.__sentences_propose(masked_cross_sim, text_embd)
        final_kr = proposed.shape[0] * 1.0 / text_embd.shape[0]
        res = ""
        for i in proposed.tolist():
            res = res + text[i]
        return res, final_kr

    def __propose(self, target, text_embd, num_propose):
        # return a list of sentence index
        # the length of the return list should be at most num_propose

        # we hope n << len(text_embd)
        n = torch.count_nonzero(target).item()
        cache_local = torch.zeros((n, n))
        if torch.equal(target, torch.zeros(target.shape)):
            return []
        args = torch.flip(target.argsort(), dims=[0])[:n]
        # prepare text cross similarity
        for i in range(n):
            a = args[i]
            for j in range(n):
                b = args[j]
                sim = f.cosine_similarity(text_embd[a], text_embd[b], dim=0)
                cache_local[i, j] = cache_local[j, i] = sim
        chosen = [args[0].item()]
        chose_args = [0]
        for k in range(min(n, num_propose) - 1):
            idx, arg = self.__min_distance(cache_local, chose_args, args)
            chosen.append(arg)
            chose_args.append(idx)
        return chosen

    @staticmethod
    def __min_distance(cache, chose_args, args):
        # use simple sum here, should be modified to use other method
        min_val = float('inf')
        idx = -1
        for i in range(len(args)):
            if i in chose_args:
                continue
            val = cache[i].index_select(0, torch.IntTensor(chose_args)).sum().item()
            if val < min_val:
                min_val = val
                idx = i
        assert idx != -1
        return idx, args[idx].item()