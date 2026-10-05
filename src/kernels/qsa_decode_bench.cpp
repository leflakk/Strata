// src/kernels/qsa_decode_bench.cpp - the decode window's QSA selection timed alone, at the window's shapes, for `nq`
// consecutive queries at a given context under a given capacity (the engine's --max-context):
//   scores : the multi-query kernel a captured window takes (block_scores_rs_kernel; STRATA_SCORES_RS=0 the one before
//            it), checked BITWISE against the per-(query, block) kernel;
//   top-k  : the 256-thread reference, the default dispatch (the 1,024-thread kernel past the register capacity), and
//            the split kernels (32 CTAs per query, qsa_block_topk_split; sm_70 to sm_89) - ids checked against the
//            reference's;
//   graphs : scores + default top-k, and scores + split top-k, each captured as one graph and replayed as the window
//            replays it.
// GPU, synthetic keys, no model.  Exit code 0 when every check passes.
// Usage: qsa_decode_bench [context=250000] [queries=3] [reps=50] [capacity_cells=262144]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 250000;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 3;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 50;
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : 262144;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / s.idx_block + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    if (ctx > capacity || nq < 1 || nq > 8) {
        std::fprintf(stderr, "context must be <= capacity, queries 1..8\n");
        return 2;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // keys with a shared direction plus noise, so the scores have a spread like a real indexer's (qsa_select_bench)
    std::vector<float> dir(128), pooled((size_t) (max_blocks * 128)), dead(128), q((size_t) (nq * 512));
    for (auto& x : dir) x = nd(rng);
    for (int64_t b = 0; b < max_blocks; ++b) {
        const float a = nd(rng);
        for (int d = 0; d < 128; ++d) pooled[(size_t) (b * 128 + d)] = 0.5f * a * dir[d] + nd(rng);
    }
    for (auto& x : dead) x = nd(rng);
    for (int64_t i = 0; i < nq; ++i)
        for (int d = 0; d < 512; ++d) q[(size_t) (i * 512 + d)] = 0.2f * dir[d % 128] + 0.1f * nd(rng);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {   // qsa_step_fill's arithmetic, the window's consecutive positions
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    const float* d_pooled = up(pooled);
    const float* d_dead = up(dead);
    const float* d_q = up(q);
    const int32_t* d_steps = up(steps);
    float *sc = nullptr, *sc_ref = nullptr;
    int32_t *ids_ref = nullptr, *ids_def = nullptr, *ids_split = nullptr;
    void* scratch = nullptr;
    ck(cudaMalloc(&sc, (size_t) (nq * max_blocks) * 4), "malloc");
    ck(cudaMalloc(&sc_ref, (size_t) (nq * max_blocks) * 4), "malloc");
    ck(cudaMalloc(&ids_ref, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMalloc(&ids_def, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMalloc(&ids_split, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMalloc(&scratch, k::qsa_topk_split_bytes(nq) + 64), "malloc");
    ck(cudaMemset(scratch, 0, k::qsa_topk_split_bytes(nq) + 64), "memset");
    cudaStream_t cs = nullptr;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    // the decode window's calls: no active-block count (a captured graph's context grows after capture)
    auto scores = [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc, cs); };
    // the per-(query, block) kernel (an active count takes it): the arithmetic the multi-query kernels must match
    auto scores_ref = [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc_ref, cs, max_blocks); };
    auto topk_ref = [&] { k::qsa_block_topk_ref(sc, d_steps, nq, max_blocks, cap, s, ids_ref, cs); };
    auto topk_def = [&] { k::qsa_block_topk(sc, d_steps, nq, max_blocks, cap, s, ids_def, cs); };
    bool split_ok = true;
    auto topk_split = [&] { split_ok = k::qsa_block_topk_split(sc, d_steps, nq, max_blocks, cap, s, ids_split, scratch, cs); };
    scores();
    scores_ref();
    topk_ref();
    topk_def();
    topk_split();
    ck(cudaStreamSynchronize(cs), "warm");
    std::vector<float> fa((size_t) (nq * max_blocks)), fb(fa.size());
    ck(cudaMemcpy(fa.data(), sc, fa.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(fb.data(), sc_ref, fb.size() * 4, cudaMemcpyDeviceToHost), "down");
    int64_t sc_same = 0, sc_n = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t nb = steps[(size_t) (i * k::kStepCount + k::kStepNBid)] + 1;
        for (int64_t b = 0; b < nb; ++b, ++sc_n)
            sc_same += std::memcmp(&fa[(size_t) (i * max_blocks + b)], &fb[(size_t) (i * max_blocks + b)], 4) == 0;
    }
    std::vector<int32_t> a((size_t) (nq * cap)), b(a.size()), c(a.size());
    ck(cudaMemcpy(a.data(), ids_ref, a.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(b.data(), ids_def, b.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(c.data(), ids_split, c.size() * 4, cudaMemcpyDeviceToHost), "down");
    int64_t same_def = 0, same_split = 0;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
        bool eq_d = true, eq_s = true;
        for (int64_t j = 0; j < w; ++j) {
            eq_d = eq_d && a[(size_t) (i * cap + j)] == b[(size_t) (i * cap + j)];
            eq_s = eq_s && a[(size_t) (i * cap + j)] == c[(size_t) (i * cap + j)];
        }
        same_def += eq_d;
        same_split += eq_s;
    }
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto timed = [&](auto f) {
        ck(cudaEventRecord(e0, cs), "record");
        for (int r = 0; r < reps; ++r) f();
        ck(cudaEventRecord(e1, cs), "record");
        ck(cudaEventSynchronize(e1), "time");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return ms / (float) reps;
    };
    auto graph_of = [&](auto f) {   // f captured as one graph
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "capture");
        f();
        ck(cudaStreamEndCapture(cs, &graph), "capture end");
        ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        return exec;
    };
    const float t_sc = timed(scores), t_ref = timed(topk_ref), t_def = timed(topk_def);
    const float t_split = split_ok ? timed(topk_split) : 0.0f;
    cudaGraphExec_t g_def = graph_of([&] { scores(); topk_def(); });
    const float t_gdef = timed([&] { ck(cudaGraphLaunch(g_def, cs), "launch"); });
    float t_gsplit = 0.0f;
    if (split_ok) {
        cudaGraphExec_t g_split = graph_of([&] { scores(); topk_split(); });
        t_gsplit = timed([&] { ck(cudaGraphLaunch(g_split, cs), "launch"); });
        // after many replays the scratch must still give the reference's ids
        ck(cudaMemset(ids_split, 0, (size_t) (nq * cap) * 4), "memset");
        ck(cudaGraphLaunch(g_split, cs), "launch");
        ck(cudaStreamSynchronize(cs), "replay");
        ck(cudaMemcpy(c.data(), ids_split, c.size() * 4, cudaMemcpyDeviceToHost), "down");
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
            for (int64_t j = 0; j < w; ++j)
                if (a[(size_t) (i * cap + j)] != c[(size_t) (i * cap + j)]) {
                    --same_split;
                    break;
                }
        }
    }
    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    const char* rs = std::getenv("STRATA_SCORES_RS");
    std::printf("%s, context %lld, capacity %lld (%lld blocks), %lld queries, %d reps\n", prop.name, (long long) ctx,
                (long long) capacity, (long long) max_blocks, (long long) nq, reps);
    std::printf("  scores (%s)               %8.4f ms   bitwise = per-block kernel: %lld/%lld blocks\n",
                rs && std::atoi(rs) == 0 ? "multi kernel  " : "reduce-scatter", t_sc, (long long) sc_same,
                (long long) sc_n);
    std::printf("  top-k, 256-thread reference        %8.4f ms\n", t_ref);
    std::printf("  top-k, default dispatch            %8.4f ms   ids = reference: %lld/%lld queries\n", t_def,
                (long long) same_def, (long long) nq);
    if (split_ok)
        std::printf("  top-k, split (32 CTAs per query)   %8.4f ms   ids = reference: %lld/%lld queries (first call and "
                    "after %d graph replays)\n", t_split, (long long) same_split, (long long) nq, reps);
    else
        std::printf("  top-k, split: not taken here (sm_90+, below sm_70, STRATA_TOPK_SPLIT=0, or a capacity the "
                    "register kernel holds without STRATA_TOPK_SPLIT=1)\n");
    std::printf("  scores + default top-k, one graph  %8.4f ms\n", t_gdef);
    if (split_ok) std::printf("  scores + split top-k, one graph    %8.4f ms\n", t_gsplit);
    const bool pass = sc_same == sc_n && same_def == nq && (!split_ok || same_split == nq);
    std::printf("  %s\n", pass ? "all checks pass" : "CHECK FAILED");
    return pass ? 0 : 1;
}
