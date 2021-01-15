import threading
import numpy as np
import torch
from torch.nn.parallel import replicate
import time
from timeit import default_timer as timer
import copy
import itertools


# This will run as part of the main process
class TrainerProcess:
    def __init__(self, train_loader,
                 test_loader,
                 cache_group,
                 dlrm,
                 emb_tables,
                 batch_fifo,
                 eviction_fifo,
                 ndevices,
                 main_device,
                 args):

        threading.Thread.__init__(self)

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.cache_group = cache_group

        self.dlrm = dlrm
        start = timer()
        self.dlrm.cache_group_replicas = replicate(self.dlrm.cache_group,
                                                   list(range(ndevices)))  # Replicate caches
        end = timer()
        print(end - start)
        breakpoint()
        self.emb_tables = emb_tables  # On CPU

        self.batch_fifo = batch_fifo
        self.eviction_fifo = eviction_fifo

        self.ndevices = ndevices
        self.device = main_device

        self.args = args
        self.use_gpu = args.use_gpu and torch.cuda.is_available()
        self.nbatches = args.num_batches if args.num_batches > 0 else len(train_loader)

        # Loss function and optimizer
        if args.loss_function == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        elif args.loss_function == "bce":
            self.loss_fn = torch.nn.BCELoss(reduction="mean")
        elif args.loss_function == "wbce":
            self.loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
            self.loss_fn = torch.nn.BCELoss(reduction="none")

        self.optimizer = torch.optim.SGD(self.dlrm.parameters(), lr=args.learning_rate)

    def time_wrap(self):
        if self.use_gpu:
            torch.cuda.synchronize()
        return time.time()

    def loss_fn_wrap(self, Z, T):
        if self.args.loss_function == "mse" or self.args.loss_function == "bce":
            if self.use_gpu:
                return self.loss_fn(Z, T.to(self.device))
            else:
                return self.loss_fn(Z, T)
        elif self.args.loss_function == "wbce":
            if self.use_gpu:
                loss_ws_ = self.loss_ws[T.data.view(-1).long()].view_as(T).to(self.device)
                loss_fn_ = self.loss_fn(Z, T.to(self.device))
            else:
                loss_ws_ = self.loss_ws[T.data.view(-1).long()].view_as(T)
                loss_fn_ = self.loss_fn(Z, T.to(self.device))
            loss_sc_ = loss_ws_ * loss_fn_
            # debug prints
            # print(loss_ws_)
            # print(loss_fn_)
            return loss_sc_.mean()

    def cache_embeddings_with_eviction(self, cached_entries_per_table, lists_of_unique_idxs, unique_indices_maps):
        cpu = torch.device("cpu")
        eviction_data = []
        for k, table_cache in enumerate(cached_entries_per_table):
            unique_idxs = lists_of_unique_idxs[k]  # One dimensional tensor of unique ids (original ids)
            map = unique_indices_maps[k]

            set_idxs = self.dlrm.cache_group.compute_set_indices(k,
                                                            unique_idxs)  # One dimensional tensor of set indices (new ids = row in the cached embedding tables)
            occupancy_table = self.dlrm.cache_group.occupancy_tables[k]

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
            evicting_table_idxs = self.dlrm.cache_group.cache_sizes[k] * evicting_ways + evicting_set_idxs

            evicting_unique_idxs = occupancy_table[evicting_set_idxs, evicting_ways]
            evicting_embeddings = self.dlrm.cache_group_replicas[0].emb_l[k].weight.data[evicting_table_idxs].to(cpu) # Only need to fetch data from one of the replicas during eviction as they are all the same

            eviction_data.append((evicting_unique_idxs, evicting_embeddings))
            ###################################################################################################################

            # Finally cache current window embeddings and update occupancy table
            table_idxs = self.dlrm.cache_group.cache_sizes[k] * ways_assignments + necessary_set_idxs
            occupancy_table[necessary_set_idxs, ways_assignments] = necessary_unique_idxs
            cached_table_idxs = map[necessary_unique_idxs].flatten()

            # Write to cache replicas on each GPU
            for i in range(self.ndevices):
                device = torch.device("cuda:" + str(i))
                self.dlrm.cache_group_replicas[i].emb_l[k].weight.data[table_idxs] = table_cache[cached_table_idxs].to(device)

        self.eviction_fifo.put(eviction_data)

    def evict_victim_cache(self):
        eviction_data = []
        cpu = torch.device("cpu")
        for k in range(len(self.dlrm.cache_group.emb_l)):
            aux_storage_indices, missing_sparse_idxs = self.dlrm.cache_group.victim_cache_entries[k]
            evicting_embeddings = self.dlrm.cache_group.emb_l[k].weight.data[aux_storage_indices].to(cpu)
            eviction_data.append((missing_sparse_idxs, evicting_embeddings))

        self.eviction_fifo.put(eviction_data)

    def start(self):
        best_gA_test = 0
        total_time = 0
        total_loss = 0
        total_accu = 0
        total_iter = 0
        total_samp = 0
        k = 0

        caching_overhead = []
        queue_fetching_overhead = []
        total_overhead = []
        replication_overhead = []
        hit_rates = []

        # time.sleep(60)
        with torch.autograd.profiler.profile(self.args.enable_profiling, self.use_gpu) as prof:
            while k < self.args.nepochs:
                accum_time_begin = self.time_wrap()

                if self.args.mlperf_logging:
                    previous_iteration_time = None

                for j, (X, lS_o, lS_i, T) in enumerate(self.train_loader):
                    print('Started loop')

                    start = timer()
                    self.dlrm.cache_group_replicas = replicate(self.dlrm.cache_group, list(range(self.ndevices))) # Replicate caches
                    end = timer()
                    replication_overhead.append(end - start)

                    print('Finished replication')

                    if j % self.args.lookahead == 0:
                        # Pull from fifo and setup caches
                        start = timer()
                        cached_entries_per_table, lists_of_unique_idxs, unique_indices_maps = self.batch_fifo.get()
                        end = timer()
                        fetching_overhead = end - start
                        queue_fetching_overhead.append(fetching_overhead)

                        start = timer()
                        self.cache_embeddings_with_eviction(cached_entries_per_table, lists_of_unique_idxs, unique_indices_maps)
                        end = timer()
                        caching_over = end - start
                        caching_overhead.append(caching_over)

                        total_overhead.append(fetching_overhead + caching_over)

                    if j >= 1000 and j % 1024 == 0:
                        fo = np.mean(queue_fetching_overhead) / self.args.lookahead
                        co = np.mean(caching_overhead) / self.args.lookahead
                        to = np.mean(total_overhead) / self.args.lookahead
                        ro = np.mean(replication_overhead)
                        print('Replication overhead = {}'.format(ro))
                        print('Fetching overhead = {}, Caching overhead = {}, Total overhead = {}'.format(fo, co, to))
                        queue_fetching_overhead = []
                        caching_overhead = []
                        total_overhead = []

                    # t1 = self.time_wrap()
                    #
                    # # forward pass - might need to change this
                    # lookups = self.cache_group(lS_o, lS_i, self.emb_tables)
                    # # hit_rates.append(avg_table_hit_rate)
                    #
                    # Z = self.dlrm(X, lookups)
                    #
                    # # loss
                    # E = self.loss_fn_wrap(Z, T)
                    #
                    # # compute loss and accuracy
                    # L = E.detach().cpu().numpy()  # numpy array
                    # S = Z.detach().cpu().numpy()  # numpy array
                    # T = T.detach().cpu().numpy()  # numpy array
                    # mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    # A = np.sum((np.round(S, 0) == T).astype(np.uint8))
                    #
                    # if not self.args.inference_only:
                    #     self.optimizer.zero_grad()
                    #     E.backward()
                    #     self.optimizer.step()
                    #
                    #     if self.args.evict_victim_cache:
                    #         self.evict_victim_cache()
                    #
                    # t2 = self.time_wrap()
                    # total_time += t2 - t1
                    # total_accu += A
                    # total_loss += L * mbs
                    # total_iter += 1
                    # total_samp += mbs
                    #
                    # should_print = ((j + 1) % self.args.print_freq == 0) or (j + 1 == self.nbatches)
                    # should_test = (
                    #         (self.args.test_freq > 0)
                    #         and (self.args.data_generation == "dataset")
                    #         and (((j + 1) % self.args.test_freq == 0) or (j + 1 == self.nbatches))
                    # )
                    #
                    # # print time, loss and accuracy
                    # if should_print or should_test:
                    #     gT = 1000.0 * total_time / total_iter if self.args.print_time else -1
                    #     total_time = 0
                    #
                    #     gA = total_accu / total_samp
                    #     total_accu = 0
                    #
                    #     gL = total_loss / total_samp
                    #     total_loss = 0
                    #
                    #     str_run_type = "inference" if self.args.inference_only else "training"
                    #     print(
                    #         "Finished {} it {}/{} of epoch {}, {:.2f} ms/it, ".format(
                    #             str_run_type, j + 1, self.nbatches, k, gT
                    #         )
                    #         + "loss {:.6f}, accuracy {:3.3f} %, ".format(gL, gA * 100)
                    #         + "avg caching overhead = {} ms, ".format(
                    #             1000 * (np.mean(np.array(caching_overhead)) / self.args.lookahead))
                    #         # + "avg overall hit rate = {}".format(np.mean(np.array(hit_rates)))
                    #     )
                    #
                    #     # print("Avg overall hit rate = {}".format(np.mean(np.array(hit_rates))))
                    #     # Uncomment the line below to print out the total time with overhead
                    #     # print("Accumulated time so far: {}" \
                    #     # .format(time_wrap(use_gpu) - accum_time_begin))
                    #     total_iter = 0
                    #     total_samp = 0
                    #     caching_overhead = []
                    #     hit_rates = []
                    #
                    # # testing
                    # if should_test and not self.args.inference_only:
                    #     # don't measure training iter time in a test iteration
                    #     if self.args.mlperf_logging:
                    #         previous_iteration_time = None
                    #
                    #     test_accu = 0
                    #     test_loss = 0
                    #     test_samp = 0
                    #
                    #     accum_test_time_begin = self.time_wrap()
                    #     if self.args.mlperf_logging:
                    #         scores = []
                    #         targets = []
                    #
                    #     with torch.no_grad():
                    #         print('Starting validation')
                    #         for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(self.test_loader):
                    #             # early exit if nbatches was set by the user and was exceeded
                    #             # if self.nbatches > 0 and i >= self.nbatches:
                    #             # break
                    #
                    #             t1_test = self.time_wrap()
                    #
                    #             # forward pass
                    #             lookups_test = self.cache_group(lS_o_test, lS_i_test, self.emb_tables)
                    #             Z_test = self.dlrm(X_test, lookups_test)
                    #             # Z_test = dlrm_wrap(
                    #             #   X_test, lS_o_test, lS_i_test, use_gpu, device
                    #             # )
                    #
                    #             # loss
                    #             E_test = self.loss_fn_wrap(Z_test, T_test)
                    #
                    #             # compute loss and accuracy
                    #             L_test = E_test.detach().cpu().numpy()  # numpy array
                    #             S_test = Z_test.detach().cpu().numpy()  # numpy array
                    #             T_test = T_test.detach().cpu().numpy()  # numpy array
                    #             mbs_test = T_test.shape[0]  # = mini_batch_size except last
                    #             A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
                    #             test_accu += A_test
                    #             test_loss += L_test * mbs_test
                    #             test_samp += mbs_test
                    #
                    #             t2_test = self.time_wrap()
                    #
                    #         gA_test = test_accu / test_samp
                    #         gL_test = test_loss / test_samp
                    #
                    #         is_best = gA_test > best_gA_test
                    #         if is_best:
                    #             best_gA_test = gA_test
                    #             if not (self.args.save_model == ""):
                    #                 print("Saving model to {}".format(self.args.save_model))
                    #                 torch.save(
                    #                     {
                    #                         "epoch": k,
                    #                         "nepochs": self.args.nepochs,
                    #                         "nbatches": self.nbatches,
                    #                         "nbatches_test": nbatches_test,
                    #                         "iter": j + 1,
                    #                         "state_dict": dlrm.state_dict(),
                    #                         "train_acc": gA,
                    #                         "train_loss": gL,
                    #                         "test_acc": gA_test,
                    #                         "test_loss": gL_test,
                    #                         "total_loss": total_loss,
                    #                         "total_accu": total_accu,
                    #                         "opt_state_dict": optimizer.state_dict(),
                    #                     },
                    #                     self.args.save_model,
                    #                 )
                    #
                    #         print(
                    #             "Testing at - {}/{} of epoch {},".format(j + 1, self.nbatches, 0)
                    #             + " loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %".format(
                    #                 gL_test, gA_test * 100, best_gA_test * 100
                    #             )
                    #         )
                    #     # Uncomment the line below to print out the total time with overhead
                    #     # print("Total test time for this group: {}" \
                    #     # .format(time_wrap(use_gpu) - accum_test_time_begin))

                k += 1  # nepochs
