import sys
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter

# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag


class Embedding_Table_Group(nn.Module):
    def __init__(self,
                 m_spa=None,
                 ln_emb=None,
                 qr_flag=False,
                 qr_operation="mult",
                 qr_collisions=0,
                 qr_threshold=200,
                 md_flag=False,
                 md_threshold=200):

        super(Embedding_Table_Group, self).__init__()

        if (m_spa is not None) and (ln_emb is not None):
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
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=False)
                EE.weight.requires_grad = False

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


class Embedding_Table_Cache_Group(nn.Module):
    def __init__(self,
                 m_spa,
                 ln_emb,
                 max_cache_size,
                 aux_table_size,
                 num_ways):

        super(Embedding_Table_Cache_Group, self).__init__()
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

            emb_l.append(EE)

        return emb_l, cache_sizes

    def create_occupancy_tables(self, cache_sizes, num_ways):
        occupancy_tables = [-1 * torch.ones(cache_sizes[i], num_ways, dtype=torch.int64) for i in
                            range(len(cache_sizes))]
        return occupancy_tables

    def forward(self, lS_o, lS_i, emb_tables, rank):
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
        cache_group_idxs = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            occupancy_table = self.occupancy_tables[k]

            set_idxs = self.compute_set_indices(k,
                                                sparse_index_group_batch)  # of shape torch.Size([2048]). set_idx[i] is the set_idx that sparse_index_group_batch[i] maps to.
            hit_tensor = (occupancy_table[set_idxs] == sparse_index_group_batch.view(-1, 1)).any(dim=1)
            hit_positions = hit_tensor.nonzero(as_tuple=False).flatten()
            miss_positions = (hit_tensor == False).nonzero(as_tuple=False).flatten()

            hitting_set_idxs = set_idxs[hit_positions]
            hitting_ways = (occupancy_table[hitting_set_idxs] == sparse_index_group_batch[hit_positions].view(-1, 1)).nonzero(as_tuple=True)[1]
            hitting_cache_lookup_idxs = self.cache_sizes[k] * hitting_ways + hitting_set_idxs

            missing_sparse_idxs = sparse_index_group_batch[miss_positions]  # Need to fetch from embedding table
            aux_storage_idxs = torch.tensor([self.cache_sizes[k] * self.num_ways + i for i in range(missing_sparse_idxs.shape[0])], dtype=torch.long,
                                            device=rank)
            self.emb_l[k].weight.data[aux_storage_idxs] = emb_tables.emb_l[k].weight.data[missing_sparse_idxs].to(rank)

            cache_lookup_idxs = torch.empty(sparse_index_group_batch.shape, dtype=torch.long)
            cache_lookup_idxs[hit_positions] = hitting_cache_lookup_idxs

            cache_lookup_idxs = cache_lookup_idxs.to(rank)
            cache_lookup_idxs[miss_positions] = aux_storage_idxs

            self.victim_cache_entries[k] = (aux_storage_idxs, missing_sparse_idxs)

            # print(k, hit_positions.shape)

            sparse_offset_group_batch = lS_o[k].to(rank)

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector

            # import pdb; pdb.set_trace()

            E = self.emb_l[k]

            V = E(cache_lookup_idxs, sparse_offset_group_batch)  # 2048 x 64 tensor
            ly.append(V)
            cache_group_idxs.append(cache_lookup_idxs.int())

        # hit_rate = hit_positions.shape[0] / sparse_index_group_batch.shape[0]
        # per_table_hit_rates.append(hit_rate)

        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        return ly, cache_group_idxs  # , sum(per_table_hit_rates) / lS_i.shape[0]


class DLRM_Net(nn.Module):
    def __init__(
            self,
            m_spa=None,
            ln_emb=None,
            ln_bot=None,
            ln_top=None,
            arch_interaction_op=None,
            arch_interaction_itself=False,
            max_cache_size=None,
            aux_table_size=None,
            dense_threshold=None,
            num_ways=None,
            sync_dense_params=True,
            sigmoid_bot=-1,
            sigmoid_top=-1,
            loss_threshold=0.0,
    ):
        super(DLRM_Net, self).__init__()

        if (ln_bot is not None) and (ln_top is not None) and (arch_interaction_op is not None):
            # save arguments
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.cpu = torch.device('cpu')

            # Trainable parameters
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        return torch.nn.Sequential(*layers)

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
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)], dtype=torch.long)
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)], dtype=torch.long)
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
        x = self.bot_l(dense_x)
        z = self.interact_features(x, ly)
        p = self.top_l(z)

        if 0.0 < self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z


def isPrime(n):
    if n == 1 or n == 2:
        return False

    i = 3

    while i * i < n:
        if n % i == 0:
            return False

        i += 1

    return True
