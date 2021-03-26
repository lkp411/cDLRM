import math
import os

import torch
import torch.multiprocessing as mp


class Prefetcher(mp.Process):
    def __init__(self, args, emb_tables_cpu, batch_fifo, eviction_fifo, finish_event, cache_ld):
        mp.Process.__init__(self)

        # Shared variables
        self.args = args
        self.emb_tables_cpu = emb_tables_cpu
        self.batch_fifo = batch_fifo
        self.eviction_fifo = eviction_fifo
        self.finish_event = finish_event
        self.cache_ld = cache_ld

    @staticmethod
    def pin_pool(p, core):
        this_pid = os.getpid()
        os.system("taskset -p -c %d %d" % (core + 3 + p, this_pid))

        return 1

    @staticmethod
    def process_batch_slice(slice, emb_tables_cpu):
        lists_of_unique_indices = []
        unique_indices_maps = []
        for i in range(len(emb_tables_cpu.emb_l)):
            unique_indices_tensor = torch.unique(slice[i])  # .long()
            unique_indices_tensor.share_memory_()
            lists_of_unique_indices.append(unique_indices_tensor)

            idxs = torch.arange(unique_indices_tensor.shape[0])
            max = torch.max(unique_indices_tensor)
            map = -1 * torch.ones(max + 1, 1, dtype=torch.long)
            map[unique_indices_tensor] = idxs.view(-1, 1)
            map.share_memory_()

            unique_indices_maps.append(map)

        cached_entries_per_table = emb_tables_cpu.fetch_unique_idx_slices(lists_of_unique_indices)

        return cached_entries_per_table, lists_of_unique_indices, unique_indices_maps

    @staticmethod
    def eviction_manager(emb_tables, eviction_fifo, average_on_writeback, core, timeout):
        this_pid = os.getpid()
        print('Pinning eviction process...')
        os.system("taskset -p -c %d %d" % (core, this_pid))
        print('Done pinning eviction process')

        try:
            while (True):
                eviction_data = eviction_fifo.get(timeout=timeout) if timeout > 0 else eviction_fifo.get()
                for k, table_eviction_data in enumerate(eviction_data):
                    idxs = table_eviction_data[0]
                    embeddings = table_eviction_data[1]
                    emb_tables.emb_l[k].weight.data[idxs] = (emb_tables.emb_l[k].weight.data[
                                                                 idxs] + embeddings) / 2 if average_on_writeback else embeddings
        except:
            print('Eviction queue empty longer than expected. Exiting eviction manager...')

    def run(self):
        this_pid = os.getpid()
        os.system("taskset -p -c %d %d" % (self.args.main_start_core + 1, this_pid))

        eviction_process = mp.Process(target=Prefetcher.eviction_manager,
                                      args=(self.emb_tables_cpu, self.eviction_fifo, self.args.average_on_writeback, self.args.main_start_core + 2,
                                            self.args.eviction_fifo_timeout))
        eviction_process.start()

        num_examples_per_process = self.args.lookahead * self.args.mini_batch_size

        pool = mp.Pool(processes=self.args.cache_workers)

        results = [pool.apply_async(Prefetcher.pin_pool, args=(p, self.args.main_start_core)) for p in range(self.args.cache_workers)]
        for res in results:
            res.get()
        print('Done pinning processes. Starting cache manager.')


        collection_limit = self.args.lookahead * self.args.cache_workers

        for epoch in range(self.args.nepochs):
            lS_i = []
            collected = 0
            for j, (_, _, sparse_idxs, _) in enumerate(self.cache_ld):
                if (j > 0 and collected % collection_limit == 0) or j == len(self.cache_ld) - 1:
                    if j == len(self.cache_ld) - 1:
                        lS_i.append(sparse_idxs)

                    lS_i = torch.cat(lS_i, dim=1)
                    num_processes_needed = math.ceil(lS_i.shape[1] / num_examples_per_process)

                    processed_slices = [pool.apply_async(Prefetcher.process_batch_slice, args=(
                        lS_i[:, p * num_examples_per_process:  (p + 1) * num_examples_per_process], self.emb_tables_cpu)) for p
                                        in range(num_processes_needed)]

                    for res in processed_slices:
                        a = res.get()
                        self.batch_fifo.put((a[0], a[1], a[2]))

                    lS_i = [sparse_idxs]
                    collected = 1
                else:
                    lS_i.append(sparse_idxs)
                    collected += 1

        pool.close()
        pool.join()
        eviction_process.join()
        self.finish_event.wait()
