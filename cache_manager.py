import math
import os
import gc

import torch
import torch.multiprocessing as mp

import dlrm_data_pytorch as dp


class Prefetcher:
    def __init__(self, args, emb_tables_cpu, batch_fifos, eviction_fifo):

        # Shared variables
        self.args = args
        self.emb_tables_cpu = emb_tables_cpu
        self.batch_fifos = batch_fifos
        self.eviction_fifo = eviction_fifo

    @staticmethod
    def pin_pool(p):
        this_pid = os.getpid()
        os.system("taskset -p -c %d %d" % (3 + p, this_pid))

        return 1

    @staticmethod
    def process_batch_slice(slice, emb_tables_cpu):
        lists_of_unique_indices = []
        unique_indices_maps = []
        for i in range(len(emb_tables_cpu.emb_l)):
            unique_indices_tensor = torch.unique(slice[i])  # .long()
            lists_of_unique_indices.append(unique_indices_tensor)
            idxs = torch.arange(unique_indices_tensor.shape[0])
            max = torch.max(unique_indices_tensor)
            map = -1 * torch.ones(max + 1, 1, dtype=torch.long)
            map[unique_indices_tensor] = idxs.view(-1, 1)
            unique_indices_maps.append(map)

        cached_entries_per_table = emb_tables_cpu.fetch_unique_idx_slices(lists_of_unique_indices)

        return cached_entries_per_table, lists_of_unique_indices, unique_indices_maps

    @staticmethod
    def eviction_manager(emb_tables, eviction_fifo, average_on_writeback):
        this_pid = os.getpid()
        print('Pinning eviction process...')
        os.system("taskset -p -c %d %d" % (2, this_pid))
        print('Done pinning eviction process')

        try:
            while (True):
                eviction_data = eviction_fifo.get(timeout=1000)
                for k, table_eviction_data in enumerate(eviction_data):
                    idxs = table_eviction_data[0]
                    embeddings = table_eviction_data[1]
                    emb_tables.emb_l[k].weight.data[idxs] = (emb_tables.emb_l[k].weight.data[
                                                                 idxs] + embeddings) / 2 if average_on_writeback else embeddings
        except:
            print('Eviction queue empty longer than expected. Exiting eviction manager...')

    def run(self):
        eviction_process = mp.Process(target=Prefetcher.eviction_manager,
                                      args=(self.emb_tables_cpu, self.eviction_fifo, self.args.average_on_writeback))
        eviction_process.start()

        num_examples_per_process = self.args.lookahead * self.args.mini_batch_size

        _, _, _, _, cache_ld = dp.make_criteo_data_and_loaders(self.args)

        pool = mp.Pool(processes=self.args.cache_workers)
        print('Created pool')

        print('Pinning processes')
        results = [pool.apply_async(Prefetcher.pin_pool, args=(p,)) for p in range(self.args.cache_workers)]
        for res in results:
            res.get()
        print('Done pinning processes. Starting cache manager.')

        for epoch in range(self.args.nepochs):
            for j, (X, lS_o, lS_i, T) in enumerate(cache_ld):
                num_processes_needed = math.ceil(lS_i.shape[1] / num_examples_per_process)

                processed_slices = [pool.apply_async(Prefetcher.process_batch_slice, args=(
                    lS_i[:, p * num_examples_per_process:  (p + 1) * num_examples_per_process], self.emb_tables_cpu)) for p
                                    in range(num_processes_needed)]

                for res in processed_slices:
                    a = res.get()

                    for batch_fifo in self.batch_fifos:
                        batch_fifo.put((a[0], a[1], a[2]))

        pool.close()
        pool.join()
        eviction_process.join()
