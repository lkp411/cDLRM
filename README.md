# cDLRM

Enabling pure data parallel training of DLRM via caching and prefetching

Example launch command:

python3 main_no_ddp.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="512-512-256-1" --max-ind-range=-1 --data-generation=dataset --data-set=terabyte --raw-data-file=../../../../datasets/terabyte/day --processed-data-file=../../../../datasets/terabyte/terabyte_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.8 --lr-embeds=0.8 --mini-batch-size=8192 --print-freq=8192 --print-time --test-mini-batch-size=4096 --test-num-workers=16 --test-freq=16384 --memory-map --data-sub-sample-rate=0.875 --cache-workers=4 --lookahead=3000 --cache-size=150000 --num-ways=16 --table-agg-freq=100 --batch-fifo-size=8 --large-batch --world-size=8 --master-port=12345


