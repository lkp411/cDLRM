import argparse
import builtins
import math
import os
import sys
import time
import warnings

import numpy as np
import psutil

import dlrm_data_pytorch as dp
from cache_manager import Prefetcher

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# quotient-remainder trick
# mixed-dimension trick
from tricks.md_embedding_bag import md_solver
from model import Embedding_Table_Group, DLRM_Net

from timeit import default_timer as timer

exc = getattr(builtins, "IOError", "FileNotFoundError")


def ProcessArgs():
    parser = argparse.ArgumentParser(description="Train Deep Learning Recommendation Model (DLRM)")

    ################################### Model Parameters ##################################
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    #######################################################################################

    ################################### Activation and loss ###############################
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument("--loss-weights", type=str, default="1.0-1.0")  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    #######################################################################################

    ######################################## Data #########################################
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--data-generation", type=str, default="random")
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
    #######################################################################################

    ################################# Embedding Table Args ################################
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    #######################################################################################

    ##################################### Training ########################################
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
    #######################################################################################

    ############################### Debugging and profiling ################################
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    ########################################################################################

    ################################## Store/load model ####################################
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    ########################################################################################

    ################################## MLPerf Args #########################################
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
    parser.add_argument("--large-batch", action="store_true", default=False)
    ########################################################################################

    ################################## Distributed training ################################
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--trainer-start-core", type=int, default=7)
    parser.add_argument("--dense-threshold", type=int, default=1000)
    ########################################################################################

    ######################################## Misc ##########################################
    parser.add_argument("--inference-only", action="store_true", default=False)
    parser.add_argument("--save-onnx", action="store_true", default=False)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    ########################################################################################

    return parser.parse_args()


def CacheEmbeddings(cached_entries_per_table, lists_of_unique_idxs, unique_indices_maps, dlrm, eviction_fifo, rank):
    cpu = torch.device("cpu")
    eviction_data = []
    for k, table_cache in enumerate(cached_entries_per_table):
        unique_idxs = lists_of_unique_idxs[k]  # One dimensional tensor of unique ids (original ids)
        map = unique_indices_maps[k]

        set_idxs = dlrm.cache_group.compute_set_indices(k,
                                                        unique_idxs)  # One dimensional tensor of set indices (new ids = row in the cached embedding tables)
        occupancy_table = dlrm.cache_group.occupancy_tables[k]

        # Filter out the hitting indices
        hit_tensor = (occupancy_table[set_idxs] == unique_idxs.view(-1, 1)).any(dim=1)
        hit_positions = hit_tensor.nonzero(as_tuple=False).flatten()
        miss_positions = (hit_tensor == False).nonzero(as_tuple=False).flatten()

        hitting_set_idxs = set_idxs[hit_positions]
        hitting_ways = (occupancy_table[set_idxs] == unique_idxs.view(-1, 1)).nonzero(as_tuple=True)[1]

        necessary_unique_idxs = unique_idxs[miss_positions]  # This is after cache hit evaluation
        necessary_set_idxs = set_idxs[miss_positions]  # This is after cache hit evaluation

        # Compute availability tensor
        avail_tensor_sampler = torch.ones(occupancy_table.shape, dtype=torch.bool)
        avail_tensor_sampler[hitting_set_idxs, hitting_ways] = False
        occupied_sets = (avail_tensor_sampler.any(dim=1) == 0).nonzero(as_tuple=False).flatten()

        # Filter out unique indices that map to sets whose ways are all occupied
        to_be_used_indices = ((necessary_set_idxs.view(-1, 1) == occupied_sets).any(dim=1) == 0).nonzero(
            as_tuple=False).flatten()

        necessary_unique_idxs = necessary_unique_idxs[to_be_used_indices]
        necessary_set_idxs = necessary_set_idxs[to_be_used_indices]

        # Convert to float and sample way assignments
        avail_tensor_sampler = avail_tensor_sampler[necessary_set_idxs].float()
        dist = torch.distributions.Categorical(avail_tensor_sampler)
        ways_assignments = dist.sample()

        ############################################### EVICTION CODE ####################################################

        # Find unique indices being evicted and fetch their embeddings for writeback
        evicting_positions = ((occupancy_table[necessary_set_idxs, ways_assignments] == -1) == False).nonzero(
            as_tuple=False).flatten()
        evicting_set_idxs = necessary_set_idxs[evicting_positions]
        evicting_ways = ways_assignments[evicting_positions]
        evicting_table_idxs = dlrm.cache_group.cache_sizes[k] * evicting_ways + evicting_set_idxs

        evicting_unique_idxs = occupancy_table[evicting_set_idxs, evicting_ways]
        evicting_embeddings = dlrm.cache_group.emb_l[k].weight.data[evicting_table_idxs].to(cpu)

        eviction_data.append((evicting_unique_idxs, evicting_embeddings))
        ###################################################################################################################

        # Finally cache current window embeddings and update occupancy table
        table_idxs = dlrm.cache_group.cache_sizes[k] * ways_assignments + necessary_set_idxs
        occupancy_table[necessary_set_idxs, ways_assignments] = necessary_unique_idxs
        cached_table_idxs = map[necessary_unique_idxs].flatten()
        dlrm.cache_group.emb_l[k].weight.data[table_idxs] = table_cache[cached_table_idxs].to(rank)

    # For now only update with the embeddings from gpu1.
    # In reality, even though all GPUs have the same embeddings as a result of gradient averaging, the randomness
    # in caching could result in each process selecting different ways for the same indices. Can consider later
    if rank == 0:
        eviction_fifo.put(eviction_data)


def process_lookahead_window(dlrm, batch_fifo, eviction_fifo, rank):
    aggregate_large_tables_multigpu(dlrm)
    torch.cuda.synchronize(rank)
    cached_entries_per_table, lists_of_unique_idxs, unique_indices_maps = batch_fifo.get()
    CacheEmbeddings(cached_entries_per_table, lists_of_unique_idxs, unique_indices_maps, dlrm, eviction_fifo, rank)


def loss_fn_wrap(Z, T, loss_fn, args, loss_ws=None):
    if args.loss_function == "mse" or args.loss_function == "bce":
        return loss_fn(Z, T)

    elif args.loss_function == "wbce":
        loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T)
        loss_fn_ = loss_fn(Z, T)

    loss_sc_ = loss_ws_ * loss_fn_
    return loss_sc_.mean()


def aggregate_large_tables_multigpu(model):
    with torch.no_grad():
        for i, table in enumerate(model.cache_group.emb_l):
            if not model.cache_group.undersized_map[i]:
                table.weight /= dist.get_world_size()
                dist.all_reduce_multigpu([table.weight], async_op=True)


def aggregate_gradients(model, req_obj_send, req_objs_recv, recieve_tensors):
    # Aggregate MLPs
    for layer in model.bot_l:
        if isinstance(layer, nn.modules.linear.Linear):
            layer.weight.grad /= dist.get_world_size()
            dist.all_reduce_multigpu([layer.weight.grad], async_op=True)

    for layer in model.top_l:
        if isinstance(layer, nn.modules.linear.Linear):
            layer.weight.grad /= dist.get_world_size()
            dist.all_reduce_multigpu([layer.weight.grad], async_op=True)

    # Wait for broadcasts to finish
    while not req_obj_send.is_completed():
        continue

    for req_obj in req_objs_recv:
        while not req_obj.is_completed():
            continue

    cache_lookups = torch.cat(recieve_tensors, dim=1)

    # Aggregate Small tables
    for i, table in enumerate(model.cache_group.emb_l):
        if model.cache_group.undersized_map[i]:
            unique_idxs = torch.unique(cache_lookups[i], sorted=True)
            grad_silce = table.weight.grad[unique_idxs] / dist.get_world_size()
            dist.all_reduce_multigpu([grad_silce], async_op=False)
            table.weight.grad[unique_idxs] = grad_silce


def train_iteration(j, local_batch_size, X, lS_i, lS_o, T, ddp_model, emb_tables, loss_fn, loss_ws, optimizer, args, rank):
    # Get slice for this process
    X = X[rank * local_batch_size: (rank + 1) * local_batch_size, :]
    lS_i = lS_i[:, rank * local_batch_size: (rank + 1) * local_batch_size]
    lS_o = lS_o[:, :local_batch_size]
    T = T[rank * local_batch_size: (rank + 1) * local_batch_size, :].to(rank)

    # Run forward + backward + param update
    Z, cache_group_idxs = ddp_model(X, lS_o, lS_i, emb_tables, rank)

    # Broadcast cache_group idxs before starting forward and backward
    req_obj_send = dist.broadcast(cache_group_idxs, src=rank, async_op=True)

    # Make broadcast calls to recieve
    recieve_tensors = [torch.zeros_like(cache_group_idxs)] * dist.get_world_size()
    recieve_tensors[rank] = cache_group_idxs

    req_objs_recv = []
    for i in range(dist.get_world_size()):
        if i != rank:
            req_objs_recv.append(dist.broadcast(recieve_tensors[i], src=i, async_op=True))

    L = loss_fn_wrap(Z, T, loss_fn, args, loss_ws)
    L.backward()
    aggregate_gradients(ddp_model.module, req_obj_send, req_objs_recv, recieve_tensors)
    torch.cuda.synchronize(rank)
    optimizer.step()
    optimizer.zero_grad()

    return L, Z, T


def Run(rank, m_spa, ln_emb, ln_bot, ln_top, train_ld, test_ld, batch_fifos, eviction_fifo, interprocess_batch_fifos, emb_tables, args):
    # First pin processes to avoid context switching overhead
    avail_cores = psutil.cpu_count() - args.trainer_start_core
    stride = rank if rank < avail_cores else rank % avail_cores
    new_core = args.trainer_start_core + stride
    this_pid = os.getpid()
    os.system("taskset -p -c %d %d" % (new_core, this_pid))

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    local_batch_size = math.ceil(args.mini_batch_size / args.world_size)

    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        max_cache_size=args.cache_size,
        aux_table_size=local_batch_size,
        dense_threshold=args.dense_threshold,
        num_ways=args.num_ways,
        sync_dense_params=args.sync_dense_params,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        loss_threshold=args.loss_threshold,
    ).to(rank)

    # Create data parallel model
    ddp_model = DDP(dlrm, device_ids=[rank])
    batch_fifo = batch_fifos[rank]

    if args.loss_function == "mse":
        loss_fn = torch.nn.MSELoss(reduction="mean")
        loss_ws = None
    elif args.loss_function == "bce":
        loss_fn = torch.nn.BCELoss(reduction="mean")
        loss_ws = None
    elif args.loss_function == "wbce":
        loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
        loss_fn = torch.nn.BCELoss(reduction="none")

    # Creating optimizer
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=args.learning_rate)

    caching_overhead = 0
    num_cache_steps = 0
    forward_time = 0
    total_iter = 0
    total_loss = 0
    total_accu = 0
    total_samp = 0
    total_loss_world = 0
    total_acc_world = 0

    # Don't sync any params
    with ddp_model.no_sync():
        # Only rank 0 loads training data
        if rank == 0:
            for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
                if j % args.lookahead == 0:
                    start = timer()
                    process_lookahead_window(dlrm, batch_fifo, eviction_fifo, rank)
                    end = timer()
                    caching_overhead += (end - start)
                    num_cache_steps += 1

                for ib_fifo in interprocess_batch_fifos:
                    ib_fifo.put((X, lS_o, lS_i, T))

                start = timer()
                L, Z, T = train_iteration(j, local_batch_size, X, lS_i, lS_o, T, ddp_model, emb_tables, loss_fn, loss_ws, optimizer,
                                          args, rank)
                end = timer()

                L = L.detach().cpu().numpy()
                S = Z.detach().cpu().numpy()
                T = T.detach().cpu().numpy()
                mbs = T.shape[0]
                A = np.sum((np.round(S, 0) == T).astype(np.uint32))

                # Get losses and accuracies from other processes
                for process in range(1, dist.get_world_size()):
                    # Get losses
                    L_t = torch.tensor(L, device=rank)
                    L_p = torch.zeros_like(L_t)
                    dist.broadcast(L_p, src=process)
                    L_p = L_p.detach().cpu().numpy()
                    total_loss_world += L_p * mbs

                    # Get accuracies
                    A_t = torch.tensor([A.item()], device=rank)
                    A_p = torch.zeros_like(A_t)
                    dist.broadcast(A_p, src=process)
                    A_p = A_p.detach().cpu().item()
                    total_acc_world += A_p

                forward_time += (end - start)
                total_loss += ((L * mbs + total_loss_world) / dist.get_world_size())
                total_accu += ((A + total_acc_world) / dist.get_world_size())
                total_samp += mbs
                total_iter += 1
                total_loss_world = 0
                total_acc_world = 0

                if j > 0 and j % args.print_freq == 0:
                    avg_caching_overhead = caching_overhead / (num_cache_steps * args.lookahead)
                    avg_forward_time = forward_time / total_iter
                    avg_loss = total_loss / total_samp
                    avg_acc = total_accu / total_samp

                    # Test
                    test_samp = 0
                    total_test_acc = 0
                    with torch.no_grad():
                        for i, (X, lS_o, lS_i, T) in enumerate(test_ld):
                            X = X.to(rank)
                            Z, _ = dlrm(X, lS_o, lS_i, emb_tables, rank)
                            S = Z.cpu().numpy()
                            T = T.cpu().numpy()
                            test_acc = np.sum((np.round(S, 0) == T).astype(np.uint32))
                            test_samp += T.shape[0]
                            total_test_acc += test_acc

                    print(
                        f'Iter = {j}: '
                        f'Rank = {rank}, '
                        f'Avg overhead = {1000 * avg_caching_overhead}, '
                        f'Avg iter Time = {1000 * avg_forward_time}, '
                        f'Loss = {avg_loss}, Train Acc = {100 * avg_acc}, '
                        f'Test acc = {100 * (total_test_acc / test_samp)}'
                    )

                    forward_time = 0
                    caching_overhead = 0
                    num_cache_steps = 0
                    total_loss = 0
                    total_accu = 0
                    total_iter = 0
                    total_samp = 0

        else:
            ib_fifo = interprocess_batch_fifos[rank - 1]

            for j in range(len(train_ld)):
                if j % args.lookahead == 0:
                    start = timer()
                    process_lookahead_window(dlrm, batch_fifo, eviction_fifo, rank)
                    end = timer()
                    caching_overhead += (end - start)
                    num_cache_steps += 1

                X, lS_o, lS_i, T = ib_fifo.get()

                start = timer()
                L, Z, T = train_iteration(j, local_batch_size, X, lS_i, lS_o, T, ddp_model, emb_tables, loss_fn, loss_ws, optimizer,
                                          args, rank)
                end = timer()

                L = L.detach().cpu().numpy()
                S = Z.detach().cpu().numpy()
                T = T.detach().cpu().numpy()
                mbs = T.shape[0]
                A = np.sum((np.round(S, 0) == T).astype(np.uint32))

                L_t = torch.tensor(L, device=rank)
                A_t = torch.tensor([A.item()], device=rank)
                dist.broadcast(L_t, src=rank)
                dist.broadcast(A_t, src=rank)

                forward_time += (end - start)
                total_iter += 1

                if j > 0 and j % args.print_freq == 0:
                    avg_caching_overhead = caching_overhead / (num_cache_steps * args.lookahead)
                    avg_forward_time = forward_time / total_iter

                    print('Iteration = {}: Rank = {}, Avg Caching Overhead = {}, Avg Iteration Time = {}'.format(j,
                                                                                                                 rank,
                                                                                                                 1000 * avg_caching_overhead,
                                                                                                                 1000 * avg_forward_time,
                                                                                                                 ))

                    forward_time = 0
                    caching_overhead = 0
                    num_cache_steps = 0
                    total_iter = 0


if __name__ == '__main__':
    mp.set_start_method("spawn")  # Cache manager deadlocks with fork as start method. This is paramount.
    args = ProcessArgs()

    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    # region Sanity Checks
    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    if args.data_generation == "dataset":

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
    # endregion

    emb_tables = Embedding_Table_Group(m_spa, ln_emb)
    emb_tables.share_memory()

    # batch_fifos = [mp.Manager().Queue(maxsize=5)] * args.world_size
    batch_fifos = [mp.Manager().Queue(maxsize=5)] * args.world_size
    eviction_fifo = mp.Manager().Queue(maxsize=5)
    interproces_batch_fifos = [mp.Manager().Queue(maxsize=1)] * (args.world_size - 1)
    finish_event = mp.Event()
    barrier = mp.Barrier(args.world_size)

    cm = Prefetcher(args, emb_tables, batch_fifos, eviction_fifo, finish_event, cache_ld)

    # Pin main process
    this_pid = os.getpid()
    os.system("taskset -p -c %d %d" % (0, this_pid))

    cm.start()
    spawn_context = mp.spawn(Run,
                             args=(m_spa, ln_emb, ln_bot, ln_top, train_ld, test_ld,
                                   batch_fifos, eviction_fifo, interproces_batch_fifos,
                                   emb_tables, args),
                             nprocs=args.world_size,
                             join=True)

    finish_event.set()
    cm.join()
