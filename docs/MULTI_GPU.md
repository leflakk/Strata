# Strata on two or three GPUs (layer split)

One model can run across several NVIDIA cards in one PC. The layers are split into contiguous ranges, one per GPU:
the first card runs layers 0 to K-1, the next card runs K onward, and so on; the last card also runs the output head
and the draft (MTP) layer. Each card keeps an expert cache for **its own layers only**, so two cards hold about twice
the experts one card holds - for the Coder model on a 16 GB + 24 GB pair, nearly all of them, which is where the
speed comes from (decode then barely touches the CPU pool).

This is pipeline (layer) parallelism, not tensor parallelism: a token crosses from one card to the next once per
verify window (a few hundred KB through pinned RAM), not twice per layer. No NVLink or peer-to-peer access is
needed; cards on x4 or x1 slots work, and the PCIe share of each card is probed on its own link. A `--pcie-frac` you give
is every card's share and skips those probes; there is no per-card setting yet.

## Using it

**Nothing to type.** `START-HERE.bat` (Linux: `./setup.sh`) lists your NVIDIA cards and says for each one whether
Strata can use it:

```
  Your NVIDIA GPUs:
    GPU 0: NVIDIA GeForce RTX 5080, 16 GB VRAM - can be used
    GPU 1: NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM - not supported - older than the RTX 20 series (compute capability 6.1; Strata needs 7.5 or newer)
    GPU 2: NVIDIA GeForce RTX 3090, 24 GB VRAM - can be used
  ...
  1) GPU 0 (NVIDIA GeForce RTX 5080, 16 GB) + GPU 2 (NVIDIA GeForce RTX 3090, 24 GB) together   (recommended)
  2) GPU 2 (NVIDIA GeForce RTX 3090, 24 GB) only
  3) GPU 0 (NVIDIA GeForce RTX 5080, 16 GB) only
Which GPUs? [1]:
```

When two or more cards can share the model, the two best together are recommended (the newest generation first:
it becomes the main card). A model installed on one card asks once, at its next start, whether to use both from
now on; the answer is kept.

**Choosing yourself** (at setup or at any start):

```
--gpus 0,2                 these cards together, as nvidia-smi numbers them; the first is the main one. Remembered.
--gpus all                 every card that can share the model
--gpu 0                    one card (at a start: for that start only)
--layer-split auto         (default) or the first layer of each later card, e.g. 18 or 16,32
```

**Not supported** (setup says so and names the cards that can be used instead):
- a card older than the RTX 20 series (compute capability below 7.5: GTX 10 and older);
- a card with less than 8 GB of VRAM, together with others (each card holds a copy of the dense weights and its
  own prompt buffers) - unless you name it with `--gpus`: then setup says the risk and asks (`--yes` with the named
  cards goes ahead);
- Intel GPUs, and a mix of NVIDIA and AMD cards. (AMD cards share a model among themselves: `./setup.sh --backend
  hip --gpus 1,0`, see [AMD_HIP.md](AMD_HIP.md).)

Or edit an existing config (`strata-*.json`), then restart:

```json
"gpu": [0, 2],
"layer_split": "auto"
```

**Skip the split when the first card holds everything** (opt-in, 0.1.31): `"split_skip_if_fits": true` in the config
(engine flag `--split-skip-if-fits`, with `--layer-split auto`) runs on the first card alone when it holds every
profiled expert plus the context's KV, the draft layer and the reserve, and says so in the log; otherwise the split
stays. On an R9700 32 GB + RX 9070 XT the R9700 holds all of the Coder's experts: with the flag 4K prompts read at
1,776 tok/s instead of 1,244 (split) and decode runs at ~60 tok/s instead of ~51 (16K prompts ~5% slower than split).

**Short prompts on a split (0.1.32, #340).** 0.1.30 gave each card's prompt path a loan from its own expert cache,
refilled after every request; on cards that hold nearly all their experts that cost short prompts up to a third of
their speed. 0.1.32 refills all cards at once, uses a smaller streaming ring on a split, and lets a card with free
VRAM keep its own prompt buffers - the same output as 0.1.31, measured on an R9700 + RX 9070 XT: 2K prompts 993 ->
1,265 tok/s, 16K 1,852 -> 1,950, decode unchanged. `STRATA_SPLIT_OWN=1` (opt-in) gives every card its own buffers:
2K 1,450 and 16K 2,227 tok/s there, but a full card then keeps a different set of experts resident, so the output
differs from the default's (stable and coherent); `STRATA_SPLIT_OWN=auto` does that only where the buffers are at
most 12% of each card's VRAM.

The engine flags behind it: `--layer-split K1[,K2..]|auto` and `--split-device D1[,D2..]` (the later stages'
devices; default the next visible ones). `--layer-split K --split-device 0` runs both stages on one card sharing
everything - the bit-exact check of the hand-off, not a speed mode.

**auto** tries every placement (all of them for two or three cards; proportional to the free VRAM beyond that) and
keeps the one whose caches would hold the most of the expert profile, hottest pairs weighted most; ties go to the
placement that leaves the fullest card the most room. The startup log prints the choice:

```
strata generate: layer split auto: K=19 - the caches hold 11767 of 12288 profiled pairs (fullest device 100%)
strata serve: layer split: layers 0-18 (CUDA0), 19-47 (CUDA1), one hand-off per window
```

## The whole model on the GPUs (three or more cards)

When the cards together hold every expert of the model (the startup log's `expert cache ... N of its N profiled
pairs` on every card says so; with 24 GB cards, the 2-3-bit models on three or four of them, UD-Q4_K_XL on six or
more), the split behaves as a pipeline of GPUs and nothing else: the CPU computes no expert, nothing streams over
PCIe while a prompt is read, and the decode window needs no host step between layers. What the engine does in that
case (each part has a switch for A/B runs; the defaults are on):

- **Balanced layers** (`--layer-split auto`): among the placements where every card holds all the experts of its
  layers, the one whose slowest card has the least work - layers times the card's speed, the last card's draft layer
  counted as half a layer (`STRATA_SPLIT_LAST_EXTRA`) - then the most even one. Any number of cards, not only two or
  three. The log says `every stage holds all the experts of its layers; balanced for prompts`.
  `STRATA_SPLIT_BALANCE=0`: the placement search alone.
- **A real prompt pipeline**: every card reads its own chunk at the same time. Until now a card handed a chunk on and
  waited for all the later cards to finish it, so only the first card ran beside the rest: with three or more cards
  the later ones took turns. `STRATA_SPLIT_PIPELINE=0`: the old hand-off.
- **Hand-offs beside the work**: a card copies a chunk's rows (80 MiB per 2048 tokens) to the next one on a copy
  stream while it reads the next chunk, and the next card receives them while it still reads the chunk before - where
  each card has the VRAM for the buffers (one chunk on a card that hands on, two on one that receives, with the VRAM
  reserve and 1 GiB more left free). The log says `overlapped hand-off copies` or why not. Slow links (x4 or x1
  slots, risers) gain the most. `STRATA_SPLIT_HANDOFF=sync`: the copies on the compute stream, as before.
- **Pipeline chunks**: a prompt is cut into more, smaller chunks so the cards fill up sooner: about
  sqrt(900 x tokens / (cards - 1)) tokens, at least 1024 (`STRATA_PIPE_ALPHA`, `STRATA_PIPE_MIN`; 0 turns it off).
  Only when no card lends cache slots to its prompt path (a lent slot's expert would stream once per chunk).
- **The draft layer's prompt K/V in batches** on the last card, through its prompt path (as one GPU does), instead of
  the drafter's own pass of a few rows at a time. `STRATA_SPLIT_MTP_BATCH=0`: the drafter's own pass.
- **Resident-only verify windows**: a card that holds every expert of its layers (and lends none of its cache to the
  prompt path) plans each layer's experts on the GPU and runs its part of the window as one graph with no host step;
  the next card's graph waits for it on the GPU, not through the host. One card that holds a whole model does the
  same. The log says `resident-only verify windows (no host step per layer) on CUDA0, ...`. `STRATA_RESIDENT_WINDOW=0`:
  the host step on every card; `STRATA_STAGE_CHAIN=0`: each card waited for by the host.
- **Commits without waiting**: after a window every card commits its state on its own stream; the next window follows
  on the same streams. `STRATA_SPLIT_COMMIT_SYNC=1`: each card's commit waited for, as before.
- **Two branches per layer in the window graph** (one card or several): the shared expert runs beside the router and
  the routed experts, and a DeltaNet layer's alpha/beta and z projections beside its qkv projection and convolution -
  the same kernels on the same data, as parallel branches of the graph. `STRATA_VERIFY_FORK=0`: one chain. (The
  `STRATA_VERIFY_PROFILE` columns of the branched kernels then read near zero: their time is under the main chain's.)
- **No row copy in resident-only windows**: every routed expert is a hit there, so the combine reads the grouped
  kernel's rows where it wrote them instead of adding them into zeroed rows first (two kernels less per layer).
  `STRATA_RESIDENT_ROWS=parts`: the copy, as before.

**Long contexts on RTX 20/30/40 cards** (any number of cards): past the register kernel's capacity (a `--max-context`
over ~135K cells, so every decode window of a 262K context), the attention's block selection took a one-CTA kernel
whose histogram increments serialized; on 4x RTX 3090 at a 250K context it cost ~0.4 ms per attention layer and decode
window (~20% of the window). It now runs a 1,024-thread kernel with per-warp histograms and the lanes of one digit
adding once - the same selection rule, so the same cells (RTX 50 cards keep their cluster kernel).
`STRATA_TOPK_WIDE=0`: the previous kernel; `=decode`: the new one for decode windows only. A decode window has only
1-5 queries, so a one-CTA kernel still runs on 1-5 SMs while the rest of the GPU idles: the windows now cut each
query's blocks into 32 slices, one per CTA, and run each radix pass as one launch whose last CTA picks the digit
(then a count and an emit launch) - again the same cells (RTX 20/30/40 cards, a `--max-context` over 64K cells;
`STRATA_TOPK_SPLIT=0`: the one-CTA kernels; `=1`: at any capacity). On 4x RTX 3090 at a 250K context the selection
went from 1.3 to 0.3 ms per card and window, decode from 80 to 95 tokens/s, and a window now costs about what it costs
at 4K. The block scores of a window sum the four indexer
heads with 9 warp shuffles instead of 20, bitwise the same scores (`STRATA_SCORES_RS=0`: the previous kernel).
`qsa_decode_bench` (`cmake --build build --target qsa_decode_bench`) times these kernels alone at a given context and
checks that their scores and cells are the same as the reference kernels'.

**The hyper-connection reads in decode** (every card, any number of them): two per layer, each a norm, a 10240 -> 320
"down" projection and a 320 -> 10240 "up" projection of BF16 weights (6.5 MB each), ~45 us per read on an RTX 3090 and
17% of a decode window's GPU time (nsys, 4x RTX 3090, 250K). Besides the plain read and its two earlier variants
(split, staged), CUDA builds have four more that compute every output with the same operations in the same order:
the down projection with its weights two tiles ahead (`pf`), or on 81 blocks without staging (`direct`), and the up
projection with the next row loaded ahead. At start each card compares every variant with the plain read bit for bit
(1..8 tokens, with and without the pending write), times the ones that agree on a 3-token read, and uses the fastest
(staged unless another is 2% faster); the log lists the times: `strata hc: CUDA0: the hyper-connection read runs as
... us per 3-token read: plain .., split .., staged .., ...`. A variant that differs on a card is never used there.
`STRATA_HC_SPLIT=<digit>` forces one (0 plain, 1 split, 2 staged, 3 direct, 4 pf, 5 staged + pf up, 6 pf down).
(Round 6 made `direct` the default without timing it: exact, but 5% slower in decode on RTX 3090s - hence the timing.)

**The attention's value reads** (opt-in, `STRATA_ATTN_PF=1`, INT8 KV cache): the decode attention's chunk kernel loads
a warp's key rows and its value entries ahead of their use instead of waiting on one row per cell; the same arithmetic
in the same order.

**The draft layer's window**: setup writes `--mtp-window 16384` when every expert is on the GPUs: the draft layer
attends to the last 16K cells instead of 32K. 4x RTX 3090, IQ3_S: decode +2% at 128K and +4% at 250K, the same tokens
(greedy decoding verifies every draft; the window changes only how many are accepted, and it barely did).

**Prompt experts on RTX 30 cards**: when every expert is on the GPUs and every card is compute capability 8.6, setup
writes `STRATA_PF_FUSED=1` into the config's `env`: the prompt's experts run on the fused int8 tensor-core kernels,
grouped on the GPU (no host sort per layer). IQ3_S on 4x RTX 3090 at a 262K context: prompts +13% at 4K, +19% at 32K,
+10% at 128K and 250K; the long-context needle tests (32K/128K/250K at depths 10/50/90%) all found, as with MMQ. The
int8 rounding differs from MMQ's, so the tokens can differ. `"STRATA_PF_FUSED": "0"` in the config keeps MMQ.

**The n-gram table in RAM** (`--ple-io ram`, Linux): the table (28.8 GB) is read once at start and locked, instead of
16 unbuffered SSD reads per token. On 8x RTX 3090 with every expert in VRAM, the first card waited 3.4 s for those
reads over a 32K prompt (the pipeline's slowest stage), and every decode window waited ~0.9 ms on its first card. Setup
turns it on when the GPUs hold every expert and the RAM has room (the table + 16 GB); `--ple-io direct|ram` chooses.

None of these changes the arithmetic of a token: the same rows are computed by the same kernels (the pipeline chunk
can change which GEMM a prompt chunk takes, as `--prefill auto` already does per request).

**Setup** does the same arithmetic before anything is downloaded: when the cards chosen with `--gpus` hold every expert
(each card's VRAM less ~7 GB for its copy of the dense weights and its buffers, ~9 GB with UD-Q4_K_XL, less the
context's KV cache once), the size menu says `fits entirely in the N GPUs`, the experts are mapped from the model
files (`--mmap-experts`: read once at start to fill the caches, nothing kept in RAM, no `experts.bin` copy), UD-Q4_K_XL
gets no RAM budget (a split of it no longer needs ~135 GB of RAM), the KV cache stays in VRAM and the recommended
context is the model's 262K window when the cards still hold everything there. `--low-ram off` keeps the experts in
RAM as before.

`tools/multi_gpu_bench.py` measures a configuration: prompt and decode speed at given lengths through the engine's own
numbers, and the generated tokens for an A/B comparison. Its prompts are cut from this repository's source as it was
at a fixed commit (`--corpus-rev`), so runs of different versions of the code read the same prompts and their tokens
can be compared. `--wrap "nsys profile ..."` runs the engine under a profiler, and `STRATA_CUDA_PROFILE=<cells>:<n>`
limits the capture to n decode rounds from that context on (`--capture-range=cudaProfilerApi
--capture-range-end=stop`):

```
python tools/multi_gpu_bench.py strata-iq3_s.json --gpus all --lengths 4096,32768,131072 --tokens-out new.json
python tools/multi_gpu_bench.py strata-iq3_s.json --gpus all --env STRATA_RESIDENT_WINDOW=0 --tokens-out old.json
python tools/multi_gpu_bench.py --compare new.json old.json
```

## What each card holds

- **every card**: a copy of the dense weights (~3.4 GB for the Coder), its own session state (the KV cache of the full
  context), its verify window and its prompt-path buffers, and an expert cache for its layers filled from the profile;
- **the last card**: also the output head and the draft layer (~0.8 GB);
- **host RAM**: the expert arena once, shared by all cards (the CPU pool computes whatever no card holds).

Prompts are read in chunks that flow through the cards in turn; while a later card reads chunk c, the first card
already reads chunk c+1. Conversation checkpoints save and restore every card's state; the adaptive expert swaps copy
into the card that owns the layer.

## Limits (for now)

- **Works across cards** (bench/results/2026-09-29-layer-split-limits):
  - images (`--vision`): each card keeps its own image-position table;
  - control vectors and the experimental speed projection: each card holds the vector's tables, switched on and
    off per request on all of them;
  - KV streaming (`--kv-resident`): each card streams the KV of its own session;
  - mid-prompt checkpoints (`--prompt-cache-every`): each card saves its part of a checkpoint when it has read that
    chunk;
  - the older helper-GPU caches (`--expert-cache-remote`, docs/SECOND_GPU.md): they take the visible GPUs no stage
    runs on, and hold only experts no stage's cache holds. On the test rig, a 2080 Ti helper made decoding slower,
    as it did without a split: its per-layer round trip costs more than the CPU pool needs for those experts.
- `--mmap-experts` needs a canonical pack (`experts.bin`), with or without a split; a native (IQ) pack says so at
  start.
- The prompt path has its own buffers on every card (1.5 GB each at the default 2048-token chunk; `--prefill 1024`
  halves that) instead of borrowing cache slots as one card does. An explicit `--expert-cache` on the first card is
  capped to leave room for them.
- Under WDDM (Windows, and WSL2) only 8 GiB of the expert arena is pinned (more, mapped into two GPU contexts,
  leaves WDDM refusing allocations); the rest streams through the pinned staging ring. A Linux driver has no such
  limit, so there the whole arena is pinned (since 0.1.31; the cap cost a 4090 + 3060 split two thirds of its
  prompt speed, #253). `STRATA_ARENA_PIN_GIB=N` pins at most N GiB, `0` the whole arena, on any OS.
- Every card needs compute capability 7.5 (RTX 20 or newer). The pre-sm_80 QSA scorer path is fp32 FMAs, so a
  Turing card runs the same kernels instead of the tensor-core prompt attention.

## Measured

The Coder on an RTX 5080 + RTX 3090 (Ryzen 9 9950X3D), 32K context; details in
`bench/results/2026-09-29-layer-split/`:

| | Prompt 16K / 28K tok/s | Decode story / code tok/s |
|---|---|---|
| 5080 alone | 1,726-2,017 / 1,970 | 83-87 / 88-105 |
| 5080 + 3090, best split (K=26) | 2,039 / 2,357 | 84 / 110 |
| 5080 + 3090, auto (K=22) | 2,037 / 2,073 | 80 / 109 |

- **Prompts gain the most** (+18-20%): each card reads its own layers of the chunk while the other reads the next.
- **Decode is on par with the faster card alone**, and ahead on code. Once both caches hold nearly every routed
  expert, the per-layer GPU time decides.
- **Correctness:** one GPU is byte-identical to 0.1.20, and the hand-off itself is bit-exact.

**Which cards and in what order:**
- Put the fastest card first; auto gives it as many layers as its cache allows.
- Leave out a much slower card when two already hold the model. An RTX 2080 Ti as a third card made the 5080 +
  3090 pair slower (68 / 90 tok/s decode): every extra card costs its own round per window.
- More cards pay off when the model's routed experts do not fit the faster ones.
