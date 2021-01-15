# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json
import psutil
# data generation
import dlrm_data_pytorch as dp
import gc

# numpy
import numpy as np

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import onnx

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
import torch.multiprocessing as mp
# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

import sklearn.metrics
import threading, queue
from timeit import default_timer as timer
from cache_manager import CacheManagerProcess
from trainer import TrainerProcess

import math

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

exc = getattr(builtins, "IOError", "FileNotFoundError")


def isPrime(n):
    if (n == 1 or n == 2):
        return False

    i = 3

    while i * i < n:
        if n % i == 0:
            return False

        i += 1

    return True


class Embedding_Table_Cache_Group(nn.Module):
    def __init__(self,
                 ndevices,
                 m_spa,
                 ln_emb,
                 max_cache_size,
                 aux_table_size,
                 num_ways):

        super(Embedding_Table_Cache_Group, self).__init__()
        self.ndevices = ndevices
        self.device_ids = range(self.ndevices)
        self.ln_emb = ln_emb
        self.num_ways = num_ways

        self.max_cache_size = self.find_next_prime(max_cache_size)

        self.emb_l, self.cache_sizes = self.create_emb(m_spa, ln_emb, self.max_cache_size, num_ways,
                                                       aux_table_size)  # emb_l[i] is a set of num_ways tables, each corresponding to 1 way. The set would just be the row itself.

        self.occupancy_tables = self.create_occupancy_tables(self.cache_sizes, num_ways)

        self.victim_cache_entries = [None] * len(self.emb_l)

    def find_next_prime(self, max_cache_size):
        for i in range(max_cache_size, 2 * max_cache_size):
            if isPrime(i):
                return i

    def compute_set_indices(self, table_idx, lookup_idxs):
        return torch.remainder(lookup_idxs, self.cache_sizes[table_idx])

    def create_emb(self, m, ln, max_cache_size, num_ways, aux_table_size):
        emb_l = nn.ModuleList()
        cache_sizes = []

        for i in range(0, ln.size):
            n = ln[i]
            num_rows = n if n < max_cache_size else max_cache_size
            cache_sizes.append(num_rows)

            EE = nn.EmbeddingBag(num_ways * num_rows + aux_table_size, m, mode="sum", sparse=True)
            device = torch.device("cuda:" + str(i % self.ndevices))

            EE.to(device)

            emb_l.append(EE)

        return emb_l, cache_sizes

    def create_occupancy_tables(self, cache_sizes, num_ways):
        occupancy_tables = [-1 * torch.ones(cache_sizes[i], num_ways, dtype=torch.int64) for i in
                            range(len(cache_sizes))]
        return occupancy_tables

    def forward(self, lS_o, lS_i, emb_tables):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        ly = []
        per_table_hit_rates = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            device = torch.device("cuda:" + str(k % self.ndevices))

            occupancy_table = self.occupancy_tables[k]

            set_idxs = self.compute_set_indices(k,
                                                sparse_index_group_batch)  # of shape torch.Size([2048]). set_idx[i] is the set_idx that sparse_index_group_batch[i] maps to.
            hit_tensor = (occupancy_table[set_idxs] == sparse_index_group_batch.view(-1, 1)).any(dim=1)
            hit_positions = hit_tensor.nonzero(as_tuple=False).flatten()
            miss_positions = (hit_tensor == False).nonzero(as_tuple=False).flatten()

            hitting_set_idxs = set_idxs[hit_positions]
            hitting_ways = \
            (occupancy_table[hitting_set_idxs] == sparse_index_group_batch[hit_positions].view(-1, 1)).nonzero(
                as_tuple=True)[1]
            hitting_cache_lookup_idxs = self.cache_sizes[k] * hitting_ways + hitting_set_idxs

            missing_sparse_idxs = sparse_index_group_batch[miss_positions]  # Need to fetch from embedding table
            aux_storage_idxs = torch.tensor(
                [self.cache_sizes[k] * self.num_ways + i for i in range(missing_sparse_idxs.shape[0])],
                dtype=torch.long, device=device)
            self.emb_l[k].weight.data[aux_storage_idxs] = emb_tables.emb_l[k].weight.data[missing_sparse_idxs].to(
                device)

            cache_lookup_idxs = torch.empty(sparse_index_group_batch.shape, dtype=torch.long)
            cache_lookup_idxs[hit_positions] = hitting_cache_lookup_idxs

            cache_lookup_idxs = cache_lookup_idxs.to(device)
            cache_lookup_idxs[miss_positions] = aux_storage_idxs

            self.victim_cache_entries[k] = (aux_storage_idxs, missing_sparse_idxs)

            # print(k, hit_positions.shape)

            sparse_offset_group_batch = lS_o[k].to(device)

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector

            # import pdb; pdb.set_trace()

            E = self.emb_l[k]

            V = E(cache_lookup_idxs, sparse_offset_group_batch)  # 2048 x 64 tensor
            ly.append(V)

        # hit_rate = hit_positions.shape[0] / sparse_index_group_batch.shape[0]
        # per_table_hit_rates.append(hit_rate)

        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        return ly  # , sum(per_table_hit_rates) / lS_i.shape[0]


# The main embedding table group, resident on the CPU
class Embedding_Table_Group(nn.Module):
    def __init__(self,
                 ndevices,
                 m_spa=None,
                 ln_emb=None,
                 qr_flag=False,
                 qr_operation="mult",
                 qr_collisions=0,
                 qr_threshold=200,
                 md_flag=False,
                 md_threshold=200):

        super(Embedding_Table_Group, self).__init__()

        self.ndevices = ndevices
        self.device_ids = range(self.ndevices)

        if ((m_spa is not None) and (ln_emb is not None)):
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # create embedding tables
            self.emb_l = self.create_emb(m_spa, ln_emb)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]
            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(n, m, self.qr_collisions,
                                    operation=self.qr_operation, mode="sum", sparse=True)
            elif self.md_flag and n > self.md_threshold:
                _m = m[i]
                base = max(m)
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=False)

            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=False)
                EE.weight.requires_grad = False
            # approach 2
            # EE.weight.data.copy_(torch.tensor(W))
            # approach 3
            # EE.weight = Parameter(torch.tensor(W),requires_grad=True)

            emb_l.append(EE)

        return emb_l

    def fetch_unique_idx_slices(self, lists_of_unique_indices):
        cached_entries_per_table = []
        for k, unique_indices in enumerate(lists_of_unique_indices):
            E = self.emb_l[k]
            cached_entries = E.weight.data[unique_indices]
            cached_entries_per_table.append(cached_entries)

        return cached_entries_per_table

    def forward(self, lS_o, lS_i):
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]
            E = self.emb_l[k]

            V = E(sparse_index_group_batch, sparse_offset_group_batch)
            ly.append(V)

        return ly


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def __init__(
            self,
            m_spa=None,
            ln_emb=None,
            ln_bot=None,
            ln_top=None,
            arch_interaction_op=None,
            arch_interaction_itself=False,
            sigmoid_bot=-1,
            sigmoid_top=-1,
            sync_dense_params=True,
            loss_threshold=0.0,
            ndevices=-1,
            qr_flag=False,
            qr_operation="mult",
            qr_collisions=0,
            qr_threshold=200,
            md_flag=False,
            md_threshold=200,
    ):
        super(DLRM_Net, self).__init__()

        if ((ln_bot is not None) and (ln_top is not None) and (arch_interaction_op is not None)):
            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold

            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, ly):
        if self.ndevices <= 1:
            return self.singlegpu_forward(dense_x, ly)
        else:
            return self.multigpu_forward(dense_x, ly)

    def singlegpu_forward(self, dense_x, ly):
        device = torch.device("cuda:0")

        dense_x = dense_x.to(device)

        x = self.apply_mlp(dense_x, self.bot_l)

        z = self.interact_features(x, ly)
        p = self.apply_mlp(z, self.top_l)

        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def multigpu_forward(self, dense_x, ly):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size)
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)

        dense_x = scatter(dense_x, device_ids, dim=0)
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)

        t_list = []
        for k, _ in enumerate(ly):  # Changed from enumerate(self.emb_l) to enumerate(ly)
            d = torch.device("cuda:" + str(k % self.ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


if __name__ == "__main__":
    ### import packages ###
    import sys
    import argparse
    import os

    mp.set_start_method("forkserver", force=True)

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument("--loss-weights", type=str, default="1.0-1.0")  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--lookahead", type=int, default=2)  # Added
    parser.add_argument("--cache-workers", type=int, default=2)  # Added
    parser.add_argument("--cache-size", type=int, default=10240)
    parser.add_argument("--num-ways", type=int, default=4)  # Added
    parser.add_argument("--average-on-writeback", action="store_true", default=False)  # Added
    parser.add_argument("--evict-victim-cache", action="store_true", default=False)  # Added

    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
    args = parser.parse_args()

    if args.mlperf_logging:
        print('command line args: ', json.dumps(vars(args)))

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if (args.test_mini_batch_size < 0):
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if (args.test_num_workers < 0):
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data
    if (args.data_generation == "dataset"):

        train_data, train_ld, test_data, test_ld, cache_ld = dp.make_criteo_data_and_loaders(args)

        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: x if x < args.max_ind_range else args.max_ind_range,
                ln_emb
            )))
        m_den = train_data.m_den
        ln_bot[0] = m_den
    else:
        # input and target at random
        if args.cache_workers > psutil.cpu_count():
            args.cache_workers = psutil.cpu_count()

        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims
        ).tolist()

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch "
            + str(ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(ln_emb.size)
            + ", with dimensions "
            + str(m_spa)
            + "x:"
        )
        print(ln_emb)

        import pdb;

        pdb.set_trace()

        print("data (inputs and targets):")
        for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
            # early exit if nbatches was set by the user and has been exceeded
            if nbatches > 0 and j >= nbatches:
                break

            print("mini-batch: %d" % j)
            print(X.detach().cpu().numpy())
            # transform offsets to lengths when printing
            print(
                [
                    np.diff(
                        S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                    ).tolist()
                    for i, S_o in enumerate(lS_o)
                ]
            )
            print([S_i.detach().cpu().tolist() for S_i in lS_i])
            print(T.detach().cpu().numpy())

    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)

    cache_group = Embedding_Table_Cache_Group(ndevices,
                                              m_spa,
                                              ln_emb,
                                              max_cache_size=args.cache_size,
                                              aux_table_size=args.mini_batch_size,
                                              num_ways=args.num_ways)

    # import pdb; pdb.set_trace();
    emb_tables = Embedding_Table_Group(ndevices,
                                       m_spa,
                                       ln_emb,
                                       qr_flag=args.qr_flag,
                                       qr_operation=args.qr_operation,
                                       qr_collisions=args.qr_collisions,
                                       qr_threshold=args.qr_threshold,
                                       md_flag=args.md_flag,
                                       md_threshold=args.md_threshold)

    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
    )

    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()
        # if dlrm.ndevices > 1:
        # dlrm.emb_l = dlrm.create_emb(m_spa, ln_emb)

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device('cuda')
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device('cpu'))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_gA = ld_model["train_acc"]
        ld_gL = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_total_accu = ld_model["total_accu"]
        ld_gA_test = ld_model["test_acc"]
        ld_gL_test = ld_model["test_loss"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_gA_test = ld_gA_test
            total_loss = ld_total_loss
            total_accu = ld_total_accu
            k = ld_k  # epochs
            j = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL, ld_gA * 100
            )
        )
        print(
            "Testing state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL_test, ld_gA_test * 100
            )
        )

    print("time/loss/accuracy (if enabled):")
    # batch_fifo = queue.Queue()
    # cm = CacheManager(cache_ld, emb_tables, cache_group, ndevices, batch_fifo, args.mini_batch_size, args.cache_workers, args.lookahead)
    # trainer = Trainer(train_ld, args.mini_batch_size, cache_group, batch_fifo, ndevices, device, args, args.lookahead)
    # cm.start()
    # trainer.start()

    # q = mp.Queue()
    # q.put(cache_ld)
    # print('Finished')

    emb_tables.share_memory()

    # Need to create a bunch of queues
    args_queue = mp.Queue()
    batch_fifo = mp.Manager().Queue(maxsize=5)
    eviction_fifo = mp.Manager().Queue(maxsize=5)
    args_queue.put(args)

    finish_event = mp.Event()

    # p = mp.Process(target=try_emb_table_forward, args=(emb_tables, args_queue))
    # p.start()
    # p.join()

    cm = CacheManagerProcess(args_queue, emb_tables, batch_fifo, eviction_fifo, finish_event, ndevices)
    trainer = TrainerProcess(train_ld, test_ld, cache_group, dlrm, emb_tables, batch_fifo, eviction_fifo, ndevices,
                             device, args)

    this_pid = os.getpid()
    os.system("taskset -p -c %d %d" % (0, this_pid))

    cm.start()
    trainer.start()
    finish_event.set()
    cm.join()

    # import pdb; pdb.set_trace();

    # print('Done training')
