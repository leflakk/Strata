// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.
#include "strata/core/emulate.hpp"
#include <cstdlib>
#include <cstring>
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

__device__ __forceinline__ uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_kernel(const float* __restrict__ pooled,
                                                                        const float* __restrict__ dead,
                                                                        const float* __restrict__ q_idx,
                                                                        const int32_t* __restrict__ steps,
                                                                        int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    if (b > n_bid || b >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

__global__ void __launch_bounds__(TOPK_T) block_topk_kernel(const float* __restrict__ scores,
                                                            const int32_t* __restrict__ steps, int64_t max_blocks,
                                                            int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T) hist[i] = 0;
        __syncthreads();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;          // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag; s_b[i] = ae;
            ag += g; ae += e;
        }
    }
    __syncthreads();
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    __syncthreads();
    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) { const int c = s_a[i]; s_a[i] = a; a += c; }
    }
    __syncthreads();
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}


// ---- QSA select on tensor cores (perf-review, after D-1): the block scores of many queries are one GEMM,
// rows (query, indexer head) x columns (blocks), K = 128, with relu per head summed. 3xTF32 (each operand split
// into a TF32 hi and lo part, hi*hi + hi*lo + lo*hi) keeps FP32-level accuracy; the summation order differs from
// the warp kernel, so a score can move in its last bits and a near-tie can select differently (not bitwise).
// Blocks < n_bid only; the tail block n_bid (the `dead` key, +1e9) is scored by the warp kernel's own code.
constexpr int TC_QT = 16;                 // queries per CTA (one m16 tile per indexer head)
constexpr int TC_NB = 32;                 // blocks per tile (4 warps x n8)
constexpr int TC_ITER = 4;                // tiles per CTA (the query tile is loaded once)
constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4;   // query row stride in floats
constexpr int TC_KS = IDX_DIM + 4;               // key row stride

// TF32 conversion and MMA need sm_80: below it they compile to a trap and qsa_block_scores_tc refuses the device
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the warp kernel (below)
#define STRATA_SEL_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_SEL_SM80 1
#else
#define STRATA_SEL_SM80 0
#endif
__device__ __forceinline__ uint32_t tf32_hi(float x) {
#if STRATA_SEL_SM80
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return __float_as_uint(x);
#endif
}
__device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_SEL_SM80
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__global__ void __launch_bounds__(128) block_scores_tc_kernel(const float* __restrict__ pooled,
                                                              const float* __restrict__ q_idx,
                                                              const int32_t* __restrict__ steps, int64_t nq,
                                                              int64_t max_blocks, int64_t reach,
                                                              float* __restrict__ out) {
    extern __shared__ __align__(16) float sm[];
    float* sQ = sm;                                   // [TC_QT][TC_QS]
    float* sK = sm + TC_QT * TC_QS;                   // [TC_NB][TC_KS]
    __shared__ int s_nbid[TC_QT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, gid = lane >> 2, tig = lane & 3;
    const int64_t q0 = (int64_t) blockIdx.y * TC_QT;
    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4), c = i % (IDX_HEADS * IDX_DIM / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + r) * IDX_HEADS * IDX_DIM)[c];
        *reinterpret_cast<float4*>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    int lo_nbid = 0x7fffffff, hi_nbid = 0;
    __syncthreads();
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        lo_nbid = min(lo_nbid, s_nbid[i]);
        hi_nbid = max(hi_nbid, s_nbid[i]);
    }
    for (int it = 0; it < TC_ITER; ++it) {
        const int64_t b0 = ((int64_t) blockIdx.x * TC_ITER + it) * TC_NB;
        if (b0 >= reach || b0 >= hi_nbid) break;      // blocks >= every query's n_bid: nothing to score
        __syncthreads();                              // the previous tile's reads are done
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid) v = reinterpret_cast<const float4*>(pooled + (b0 + r) * IDX_DIM)[c];
            *reinterpret_cast<float4*>(sK + r * TC_KS + c * 4) = v;
        }
        __syncthreads();
        float acc[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
        const float* kr = sK + (warp * 8 + gid) * TC_KS;
#pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig], kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = tf32_hi(kx0);
            bh[1] = tf32_hi(kx1);
            bl[0] = tf32_hi(kx0 - __uint_as_float(bh[0]));
            bl[1] = tf32_hi(kx1 - __uint_as_float(bh[1]));
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sQ + h * IDX_DIM + k0 + tig;
                const float x0 = qa[gid * TC_QS], x1 = qa[(gid + 8) * TC_QS];
                const float x2 = qa[gid * TC_QS + 4], x3 = qa[(gid + 8) * TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = tf32_hi(x0); ah[1] = tf32_hi(x1); ah[2] = tf32_hi(x2); ah[3] = tf32_hi(x3);
                al[0] = tf32_hi(x0 - __uint_as_float(ah[0]));
                al[1] = tf32_hi(x1 - __uint_as_float(ah[1]));
                al[2] = tf32_hi(x2 - __uint_as_float(ah[2]));
                al[3] = tf32_hi(x3 - __uint_as_float(ah[3]));
                mma_tf32(acc[h], al, bh);
                mma_tf32(acc[h], ah, bl);
                mma_tf32(acc[h], ah, bh);
            }
        }
        // relu per head, heads added in order (as the warp kernel), written where the block completed for the query
        const int64_t bc = b0 + warp * 8 + 2 * tig;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int64_t qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int64_t b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

#if defined(__HIPCC__)
// ---- gfx12 (RDNA4: gfx1200 / gfx1201) scorer on v_wmma_f32_16x16x16_bf16 (wave32).  Same GEMM as above: for a tile of 16
// queries, every block's score is sum over the 4 indexer heads of relu(q_h . k), K = 128.  FP32-level accuracy from a
// three-way bf16 split of both operands (x = hi + mid + lo EXACTLY - the top 8, next 8 and last 8 bits of the fp32
// mantissa, so bf16's fp32-sized exponent range needs no scaling) and the six products of order <= 2, smallest first;
// like the TF32 kernel it sums in another order than the warp kernel (not bitwise).  The tail block n_bid is the warp
// kernel's arithmetic (block_scores_tail_kernel).
//   D = A x B with A = 16 key blocks (row l%16 of a lane, k = 8*(l/16)+i), B = 16 queries (column l%16, same k):
//   a lane ends up with 8 consecutive blocks of ONE query, written as one run.  No LDS for keys: a lane reads its
//   32-byte slice of the key row from global (the 16 query tiles of a launch re-read them from L2); the queries are
//   split once per CTA into LDS.
#if defined(__gfx1200__) || defined(__gfx1201__)
#define STRATA_SEL_GFX12 1
#else
#define STRATA_SEL_GFX12 0
#endif
typedef short sel_s8 __attribute__((ext_vector_type(8)));
typedef float sel_f8 __attribute__((ext_vector_type(8)));
typedef uint32_t sel_u4 __attribute__((ext_vector_type(4)));
constexpr int WQT = 16;                    // queries per CTA (the N of one WMMA)
constexpr int WITER = 4;                   // key tiles per warp (the CTA covers 4 warps * WITER * 16 blocks)
constexpr int WQS = IDX_DIM + 8;           // bf16 elements per LDS row: 272 bytes, conflict-free 16-byte reads

__device__ __forceinline__ sel_f8 wmma_bf16(const sel_s8& a, const sel_s8& b, const sel_f8& c) {
#if STRATA_SEL_GFX12
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
#else
    __trap();
    return c;
#endif
}
// the bf16 pieces of x as fp32 bit patterns whose upper 16 bits are the bf16 (lower 16 are zero)
__device__ __forceinline__ void split3(float x, uint32_t& hi, uint32_t& mid, uint32_t& lo) {
    hi = __float_as_uint(x) & 0xffff0000u;
    const float r1 = x - __uint_as_float(hi);                   // exact
    mid = __float_as_uint(r1) & 0xffff0000u;
    lo = __float_as_uint(r1 - __uint_as_float(mid)) & 0xffff0000u;   // exact: at most 8 significant bits left
}
__device__ __forceinline__ uint32_t pack_bf16x2(uint32_t a, uint32_t b) {   // low half = a's bf16, high half = b's
#if STRATA_SEL_GFX12
    return __builtin_amdgcn_perm(b, a, 0x07060302u);
#else
    return (a >> 16) | (b & 0xffff0000u);
#endif
}

__global__ void __launch_bounds__(128) block_scores_wmma_kernel(const float* __restrict__ pooled,
                                                                const float* __restrict__ q_idx,
                                                                const int32_t* __restrict__ steps, int64_t nq,
                                                                int64_t max_blocks, int64_t reach,
                                                                float* __restrict__ out) {
    __shared__ __align__(16) uint16_t sq[3][IDX_HEADS][WQT][WQS];   // hi / mid / lo, [head][query][dim]
    __shared__ int s_nbid[WQT];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5, l16 = lane & 15, g = lane >> 4;
    const int64_t q0 = (int64_t) blockIdx.y * WQT;
    for (int i = t; i < WQT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int row = i / (IDX_DIM / 4), c = i % (IDX_DIM / 4);
        const int qr = row / IDX_HEADS, h = row % IDX_HEADS;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + qr < nq) v = reinterpret_cast<const float4*>(q_idx + (q0 + qr) * IDX_HEADS * IDX_DIM)[h * (IDX_DIM / 4) + c];
        const float x[4] = {v.x, v.y, v.z, v.w};
        uint32_t hb[4], mb[4], lb[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) split3(x[j], hb[j], mb[j], lb[j]);
        *reinterpret_cast<uint2*>(&sq[0][h][qr][c * 4]) = make_uint2(pack_bf16x2(hb[0], hb[1]), pack_bf16x2(hb[2], hb[3]));
        *reinterpret_cast<uint2*>(&sq[1][h][qr][c * 4]) = make_uint2(pack_bf16x2(mb[0], mb[1]), pack_bf16x2(mb[2], mb[3]));
        *reinterpret_cast<uint2*>(&sq[2][h][qr][c * 4]) = make_uint2(pack_bf16x2(lb[0], lb[1]), pack_bf16x2(lb[2], lb[3]));
    }
    if (t < WQT) s_nbid[t] = q0 + t < nq ? steps[(q0 + t) * kStepCount + kStepNBid] : 0;
    __syncthreads();
    int hi_nbid = 0;
#pragma unroll
    for (int i = 0; i < WQT; ++i) hi_nbid = max(hi_nbid, s_nbid[i]);
    const int64_t qi = q0 + l16;
    const int nb_q = s_nbid[l16];
    for (int it = 0; it < WITER; ++it) {
        const int64_t b0 = (((int64_t) blockIdx.x * WITER + it) * 4 + warp) * 16;
        if (b0 >= reach || b0 >= hi_nbid) break;          // warp-uniform; later tiles start higher
        const int64_t row = b0 + l16;
        const bool rv = row < hi_nbid && row < max_blocks;
        const float* kp = pooled + (rv ? row : 0) * IDX_DIM + g * 8;
        // acc: the hi*hi products; cor: the five smaller ones (<= 2^-8 of it). Kept apart and added once at the end: a
        // correction summed into the big accumulator is rounded to ITS ulp at every one of the 48 steps (measured: 8e-7
        // of the score scale; apart, near the warp kernel's)
        sel_f8 acc[IDX_HEADS], cor[IDX_HEADS];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h] = cor[h] = sel_f8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 2
        for (int kk = 0; kk < IDX_DIM / 16; ++kk) {
            float4 k0 = make_float4(0.f, 0.f, 0.f, 0.f), k1 = k0;
            if (rv) {
                k0 = *reinterpret_cast<const float4*>(kp + kk * 16);
                k1 = *reinterpret_cast<const float4*>(kp + kk * 16 + 4);
            }
            const float x[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
            uint32_t hb[8], mb[8], lb[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) split3(x[j], hb[j], mb[j], lb[j]);
            sel_u4 ah, am, al;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                ah[j] = pack_bf16x2(hb[2 * j], hb[2 * j + 1]);
                am[j] = pack_bf16x2(mb[2 * j], mb[2 * j + 1]);
                al[j] = pack_bf16x2(lb[2 * j], lb[2 * j + 1]);
            }
            const sel_s8 Ah = __builtin_bit_cast(sel_s8, ah), Am = __builtin_bit_cast(sel_s8, am),
                         Al = __builtin_bit_cast(sel_s8, al);
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const sel_s8 Bh = *reinterpret_cast<const sel_s8*>(&sq[0][h][l16][kk * 16 + g * 8]);
                const sel_s8 Bm = *reinterpret_cast<const sel_s8*>(&sq[1][h][l16][kk * 16 + g * 8]);
                const sel_s8 Bl = *reinterpret_cast<const sel_s8*>(&sq[2][h][l16][kk * 16 + g * 8]);
                cor[h] = wmma_bf16(Al, Bh, cor[h]);
                cor[h] = wmma_bf16(Ah, Bl, cor[h]);
                cor[h] = wmma_bf16(Am, Bm, cor[h]);
                cor[h] = wmma_bf16(Ah, Bm, cor[h]);
                cor[h] = wmma_bf16(Am, Bh, cor[h]);
                acc[h] = wmma_bf16(Ah, Bh, acc[h]);
            }
        }
        // relu per head, heads added in order (as the warp kernel); this lane: query qi, blocks b0 + 8g .. 8g+7
        if (qi < nq) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int64_t b = b0 + g * 8 + i;
                if (b >= nb_q || b >= max_blocks) continue;
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][i] + cor[h][i];
                    score += d > 0.0f ? d : 0.0f;
                }
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

// the gfx12 kernel needs gfx1200/gfx1201 code objects and a gfx12 device (gfx1100 has WMMA too, with another layout)
bool sel_gfx12_device() {
    static int ok[64] = {};   // per device: 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        cudaDeviceProp prop;
        ok[dev] = (cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                   (std::strncmp(prop.gcnArchName, "gfx1200", 7) == 0 || std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0))
                      ? 1 : 2;
        cudaGetLastError();
    }
    return ok[dev] == 1;
}
#endif  // __HIPCC__

// the tail block n_bid of each query: exactly block_scores_kernel's arithmetic for that block
__global__ void __launch_bounds__(32) block_scores_tail_kernel(const float* __restrict__ dead,
                                                               const float* __restrict__ q_idx,
                                                               const int32_t* __restrict__ steps,
                                                               int64_t max_blocks, float* __restrict__ out) {
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    if (n_bid >= max_blocks) return;
    const int lane = threadIdx.x & 31;
    const float4 k4 = *reinterpret_cast<const float4*>(dead + lane * 4);
    const float* q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (n_kv % R != 0) score += 1e9f;
        out[qi * max_blocks + n_bid] = score;
    }
}

// ---- the same top-k with each query's keys read once: 1,024 threads hold up to TK_PER consecutive blocks' keys in
// registers (contexts up to 4 * 1024 * TK_PER cells), per-warp histograms, block-wide scans. The selection rule is
// block_topk_kernel's (radix threshold, ties to the lowest index, cells ascending): identical ids.
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;
#if defined(__HIPCC__)
// AMD (RDNA, wave32): a 1,024-thread block may use 192 VGPRs per lane, so the keys of 66 blocks per thread fit - contexts
// up to 4 * 1024 * 66 = 270,336 cells (the 262,144 --max-context) keep the register kernel. NVIDIA's 64 registers per
// thread at 1,024 threads allow only TK_PER.
constexpr int TK_PER_MAX = 66;
#else
constexpr int TK_PER_MAX = TK_PER;
#endif

__device__ __forceinline__ int block_excl_scan(int v, int* s_warp, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;               // exclusive per warp
        if (lane == 31) s_warp[32] = z;     // total
    }
    __syncthreads();
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    __syncthreads();
    return r;
}

template <int PER>
__global__ void __launch_bounds__(TK_T) block_topk_reg_kernel(const float* __restrict__ scores,
                                                              const int32_t* __restrict__ steps, int64_t max_blocks,
                                                              int64_t cap, int32_t* __restrict__ ids) {
    __shared__ int hist[TK_T / 32][256];
    __shared__ int s_warp[33];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T;       // <= PER (the caller checks)
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int) (n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1) break;
            const int w = weight(b);
            if (w == 0) continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[warp][(key[j] >> shift) & 255], w);
        }
        __syncthreads();
        if (t < 256) {                                // fold the warps' histograms into warp 0's
            int s = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width) break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) gt += w;
        else if (key[j] == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1) break;
        const int w = weight(b);
        if (w == 0) continue;
        if (key[j] > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (key[j] == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}


#if !defined(__HIPCC__)
// ---- the same top-k past the register kernel's capacity (a --max-context over ~135K cells: every decode window there,
// and the prompt path's queries), where block_topk_kernel ran: one CTA of 256 threads per query making four radix
// passes over the scores with one shared histogram.  A query's scores sit in a few exponents, so the first passes'
// increments hit one or two bins and serialize (8x RTX 3090, a 250K context: ~0.4 ms per QSA layer and decode window,
// ~20% of the window).  Here: 1,024 threads, the radix passes read the scores coalesced, each warp keeps its own
// histogram and the lanes of one digit add as one (__match_any_sync); the count and the writes keep each thread's
// contiguous blocks, as before.  The selection rule is block_topk_kernel's (radix threshold, ties to the lowest index,
// cells ascending): identical ids.  CUDA (sm_70+: __match_any_sync); HIP keeps block_topk_kernel.
constexpr int TW_T = 1024;   // block_excl_scan's 32 warps
constexpr int TW_UNR = 4;    // score loads in flight per thread in a radix pass
__global__ void __launch_bounds__(TW_T) block_topk_wide_kernel(const float* __restrict__ scores,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks,
                                                               int64_t cap, int32_t* __restrict__ ids) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    __shared__ int hist[TW_T / 32][256];
    __shared__ int s_warp[33];
    __shared__ int s_digit, s_above;
    const int64_t qi = blockIdx.x;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {                               // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TW_T) out[j] = (int32_t) j;
        return;
    }
    const float* sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;                      // blocks 0..n_bid, the last possibly empty
    const int w_last = (int) (n_kv - n_bid * R);       // the last block's cells (the others hold R)
    uint32_t prefix = 0;
    int above = 0;                                     // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
        __syncwarp();
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        // every lane of a warp runs every iteration (the match takes them all); block b = base + u * TW_T + t,
        // coalesced, TW_UNR loads in flight per thread before any is used (one at a time left the passes L2-latency
        // bound)
        for (int64_t base = 0; base < nb; base += (int64_t) TW_T * TW_UNR) {
            uint32_t kk[TW_UNR];
#pragma unroll
            for (int u = 0; u < TW_UNR; ++u) {
                const int64_t b = base + (int64_t) u * TW_T + t;
                kk[u] = b < nb ? order_key(sc[b]) : 0u;
            }
#pragma unroll
            for (int u = 0; u < TW_UNR; ++u) {
                const int64_t b = base + (int64_t) u * TW_T + t;
                const uint32_t k = kk[u];
                const bool match = b < nb && (k & hi_mask) == (prefix & hi_mask);
                const unsigned d = (k >> shift) & 255u;
                const bool full = match && b < n_bid;  // R cells; the last block adds its own count below
                const unsigned grp = __match_any_sync(0xffffffffu, full ? d : 256u + (unsigned) lane);
                if (full && lane == __ffs(grp) - 1) atomicAdd(&hist[warp][d], R * __popc(grp));
                if (match && b == n_bid && w_last > 0) atomicAdd(&hist[warp][d], w_last);
            }
        }
        __syncthreads();
        if (t < 256) {                                 // fold the warps' histograms into warp 0's
            int s = 0;
            for (int w2 = 0; w2 < TW_T / 32; ++w2) s += hist[w2][t];
            hist[0][t] = s;
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, dd = 255;
            for (; dd > 0; --dd) {
                if (cum + hist[0][dd] >= width) break;
                cum += hist[0][dd];
            }
            s_digit = dd;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t) s_digit << shift;
        above = s_above;
        __syncthreads();
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;           // cells equal to thr that fit, lowest index first
    const int64_t per = (nb + TW_T - 1) / TW_T;
    const int64_t b0 = (int64_t) t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = b < n_bid ? R : w_last;
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + (int) my_eq;
    int64_t wpos = block_excl_scan(sel, s_warp, tot);
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = b < n_bid ? R : w_last;
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
#else
    (void) scores; (void) steps; (void) max_blocks; (void) cap; (void) ids;
    __trap();
#endif
}
#endif  // !__HIPCC__

// Block scores with every key block read ONCE for all of a call's queries (block_scores_kernel's grid is
// (max_blocks / 8) x nq: ~24,600 mostly-idle blocks per layer at a decode window, each key re-read per query).  A fixed
// grid strides over the blocks; per (block, query) the same arithmetic in the same order as block_scores_kernel.
// qsa_block_scores takes it for every call without an active-block count and at most MQ queries: the captured decode
// window, the uncaptured decode, and prefill's pooled16 call.
constexpr int MQ = 8;
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_multi_kernel(const float* __restrict__ pooled,
                                                                              const float* __restrict__ dead,
                                                                              const float* __restrict__ q_idx,
                                                                              const int32_t* __restrict__ steps, int nq,
                                                                              int64_t max_blocks, float* __restrict__ out) {
    __shared__ __align__(16) float qs[MQ * IDX_HEADS * IDX_DIM];
    __shared__ int64_t s_nkv[MQ], s_nbid[MQ];
    for (int i = threadIdx.x; i < nq * IDX_HEADS * IDX_DIM; i += blockDim.x) qs[i] = q_idx[i];
    if (threadIdx.x < nq) {
        s_nkv[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNKv];
        s_nbid[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNBid];
    }
    __syncthreads();
    int64_t top = 0;
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    const int lane = threadIdx.x & 31;
    const int64_t wstride = (int64_t) gridDim.x * SCORE_WARPS;
    for (int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5); b <= top && b < max_blocks; b += wstride) {
        const float4 kp = *reinterpret_cast<const float4*>(pooled + b * IDX_DIM + lane * 4);
        const float4 kd = *reinterpret_cast<const float4*>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const float4 k4 = (b == n_bid) ? kd : kp;
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
                float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

#if !defined(__HIPCC__)
// block_scores_multi_kernel with the four heads' butterfly sums done as one reduce-scatter: the level-16 exchange
// leaves each lane two heads, the level-8 one a single head (lanes 0-7 head 0, 8-15 head 1, 16-23 head 2, 24-31 head
// 3), the last three levels sum it, and lane 0 gathers the four - 9 shuffles per (block, query) instead of 20.  Every
// head's sum is the same tree with the same operands in the same order (own + partner at each level), the relu'd heads
// are added in the same order: bitwise block_scores_multi_kernel's scores.  The next key block is loaded before the
// current one is scored.  STRATA_SCORES_RS=0: block_scores_multi_kernel.  (CUDA; HIP keeps the kernel above.)
__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_rs_kernel(const float* __restrict__ pooled,
                                                                           const float* __restrict__ dead,
                                                                           const float* __restrict__ q_idx,
                                                                           const int32_t* __restrict__ steps, int nq,
                                                                           int64_t max_blocks, float* __restrict__ out) {
    __shared__ __align__(16) float qs[MQ * IDX_HEADS * IDX_DIM];
    __shared__ int64_t s_nkv[MQ], s_nbid[MQ];
    for (int i = threadIdx.x; i < nq * IDX_HEADS * IDX_DIM; i += blockDim.x) qs[i] = q_idx[i];
    if (threadIdx.x < nq) {
        s_nkv[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNKv];
        s_nbid[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNBid];
    }
    __syncthreads();
    int64_t top = 0;
    for (int q = 0; q < nq; ++q) top = s_nbid[q] > top ? s_nbid[q] : top;
    const int64_t last = top < max_blocks - 1 ? top : max_blocks - 1;   // the last block any query scores
    const int lane = threadIdx.x & 31;
    const bool hi16 = (lane & 16) != 0, hi8 = (lane & 8) != 0;
    const int64_t wstride = (int64_t) gridDim.x * SCORE_WARPS;
    int64_t b = (int64_t) blockIdx.x * SCORE_WARPS + (threadIdx.x >> 5);
    const float4 kd = *reinterpret_cast<const float4*>(dead + lane * 4);
    const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 kp = b <= last ? *reinterpret_cast<const float4*>(pooled + b * IDX_DIM + lane * 4) : zero;
    for (; b <= last; b += wstride) {
        const int64_t bn = b + wstride;
        const float4 kn = bn <= last ? *reinterpret_cast<const float4*>(pooled + bn * IDX_DIM + lane * 4) : zero;
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid) continue;
            const float4 k4 = (b == n_bid) ? kd : kp;
            const float* q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float d[IDX_HEADS];
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = *reinterpret_cast<const float4*>(q + h * IDX_DIM);
                d[h] = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
            }
            float a0 = hi16 ? d[2] : d[0], a1 = hi16 ? d[3] : d[1];   // the two heads this lane keeps
            a0 += __shfl_xor_sync(0xffffffffu, hi16 ? d[0] : d[2], 16);
            a1 += __shfl_xor_sync(0xffffffffu, hi16 ? d[1] : d[3], 16);
            float x = hi8 ? a1 : a0;                                  // the one it keeps
            x += __shfl_xor_sync(0xffffffffu, hi8 ? a0 : a1, 8);
            x += __shfl_xor_sync(0xffffffffu, x, 4);
            x += __shfl_xor_sync(0xffffffffu, x, 2);
            x += __shfl_xor_sync(0xffffffffu, x, 1);
            const float r = x > 0.0f ? x : 0.0f;
            const float r1 = __shfl_sync(0xffffffffu, r, 8);
            const float r2 = __shfl_sync(0xffffffffu, r, 16);
            const float r3 = __shfl_sync(0xffffffffu, r, 24);
            if (lane == 0) {
                float score = 0.0f;
                score += r;
                score += r1;
                score += r2;
                score += r3;
                if (b == n_bid && s_nkv[qi] % R != 0) score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
        kp = kn;
    }
}
#endif  // !__HIPCC__

// ---- the decode top-k on TS_P CTAs per query, for cards without clusters (sm_70 to sm_89).  A decode window has 1-5
// queries, so the one-CTA kernels run it on 1-5 SMs while the rest of the GPU idles: past the register kernel's
// capacity (a --max-context over ~135K) block_topk_wide_kernel makes four radix passes and two scans over up to 65,538
// blocks on one SM (4x RTX 3090, a 250K context: the window's scores + top-k still ~0.4 ms per QSA layer, against
// ~0.05 at 32K).  Here a query's blocks
// are cut into TS_P slices, one per CTA, and each radix pass is one launch: every CTA adds its slice's digit histogram
// into the query's histogram in global memory, and the last CTA through the pass (a counter, after a fence) picks the
// digit for all - block_topk_kernel's rule, written as a suffix scan - and clears the histogram and the counter.  Then
// one launch counts each slice's cells above / at the threshold and one writes them: a slice starts at (cells above
// thr before it) + min(cells at thr before it, eq_budget), the telescoped sum the cluster kernel uses.  Six launches,
// all capturable.  IDENTICAL IDS for the cluster kernel's reason: thr and `above` are pure functions of the (key,
// weight) multiset, and the emit is the reference's rule.  The scratch (a TopkSplitQ per query) must be zero before the
// first call; every call leaves it zero again (the histogram and the counter; the rest is rewritten).
constexpr int TS_T = 256;   // threads per CTA: one histogram bin each in the digit scan
constexpr int TS_P = 32;    // CTAs (slices) per query
struct TopkSplitQ {
    int hist[256];           // the pass's digit histogram over every slice (the last CTA reads and clears it)
    int done;                // CTAs through the pass
    uint32_t prefix;         // the digits fixed so far; after the last pass, the threshold key
    int above;               // cells strictly above them
    int pad;
    int gt[TS_P], eq[TS_P];  // per slice: cells above / at the threshold
};

#if !defined(__HIPCC__)
struct TopkSlice {
    int64_t n_kv, n_bid, width, lo, hi;
    int w_last;
};
// this CTA's blocks [lo, hi) of query qi (hi <= lo: none, the CTA still counts in); false: the query selects every
// cell (n_kv <= width), which the emit kernel writes as the identity
__device__ __forceinline__ bool topk_slice(const int32_t* steps, int64_t qi, int p, TopkSlice& v) {
    const int32_t* st = steps + qi * kStepCount;
    v.n_kv = st[kStepNKv];
    v.n_bid = st[kStepNBid];
    v.width = st[kStepWidth];
    if (v.n_kv <= v.width) return false;
    const int64_t nb = v.n_bid + 1;                    // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TS_P - 1) / TS_P;
    v.lo = (int64_t) p * per;
    v.hi = v.lo + per < nb ? v.lo + per : nb;
    v.w_last = (int) (v.n_kv - v.n_bid * R);           // the last block's cells (the others hold R)
    return true;
}

// one radix pass (shift 24, 16, 8, 0); grid (TS_P, nq)
__global__ void __launch_bounds__(TS_T) topk_split_pass_kernel(const float* __restrict__ scores,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks,
                                                               int shift, TopkSplitQ* __restrict__ qs) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    __shared__ int hist[TS_T / 32][256];
    __shared__ int s_tot[TS_T / 32];
    __shared__ int s_digit, s_last;
    const int64_t qi = blockIdx.y;
    TopkSlice v;
    if (!topk_slice(steps, qi, (int) blockIdx.x, v)) return;
    TopkSplitQ* Q = qs + qi;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const uint32_t prefix = shift == 24 ? 0u : Q->prefix;
    const int above = shift == 24 ? 0 : Q->above;
    const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
    for (int i = lane; i < 256; i += 32) hist[warp][i] = 0;
    __syncwarp();
    // block_topk_wide_kernel's pass over this slice: coalesced, TW_UNR loads in flight, one histogram per warp and the
    // lanes of one digit adding as one (every lane runs every iteration: the match takes them all)
    const float* sc = scores + qi * max_blocks;
    for (int64_t base = v.lo; base < v.hi; base += (int64_t) TS_T * TW_UNR) {
        uint32_t kk[TW_UNR];
#pragma unroll
        for (int u = 0; u < TW_UNR; ++u) {
            const int64_t b = base + (int64_t) u * TS_T + t;
            kk[u] = b < v.hi ? order_key(sc[b]) : 0u;
        }
#pragma unroll
        for (int u = 0; u < TW_UNR; ++u) {
            const int64_t b = base + (int64_t) u * TS_T + t;
            const uint32_t k = kk[u];
            const bool match = b < v.hi && (k & hi_mask) == (prefix & hi_mask);
            const unsigned d = (k >> shift) & 255u;
            const bool full = match && b < v.n_bid;    // R cells; the last block adds its own count below
            const unsigned grp = __match_any_sync(0xffffffffu, full ? d : 256u + (unsigned) lane);
            if (full && lane == __ffs(grp) - 1) atomicAdd(&hist[warp][d], R * __popc(grp));
            if (match && b == v.n_bid && v.w_last > 0) atomicAdd(&hist[warp][d], v.w_last);
        }
    }
    __syncthreads();
    int h = 0;
#pragma unroll
    for (int w = 0; w < TS_T / 32; ++w) h += hist[w][t];
    if (h != 0) atomicAdd(&Q->hist[t], h);
    __threadfence();                                   // this CTA's adds land before it counts in
    __syncthreads();
    if (t == 0) {
        __threadfence();
        s_last = atomicAdd(&Q->done, 1) == TS_P - 1;
    }
    __syncthreads();
    if (!s_last) return;
    // the last CTA through: every slice's counts are in Q->hist (read at L2, and cleared for the next pass)
    __threadfence();
    h = atomicExch(&Q->hist[t], 0);
    // x = S(t), the cells (under the prefix) whose digit is t or more: a suffix scan within the warp, then the warps
    // of the larger digits
    int x = h;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_down_sync(0xffffffffu, x, o);
        if (lane + o < 32) x += y;
    }
    if (lane == 0) s_tot[warp] = x;
    if (t == 0) s_digit = 0;
    __syncthreads();
    for (int w = warp + 1; w < TS_T / 32; ++w) x += s_tot[w];
    // block_topk_kernel walks the digits from 255 down and stops at the first d >= 1 with above + S(d) >= width (none:
    // 0); S does not grow with d, so that is the largest such d
    if (t >= 1 && above + x >= v.width) atomicMax(&s_digit, t);
    __syncthreads();
    if (t == s_digit) {
        Q->prefix = prefix | ((uint32_t) t << shift);
        Q->above = above + x - h;                      // cells strictly above the digit: S(t + 1)
        Q->done = 0;
    }
#else
    (void) scores; (void) steps; (void) max_blocks; (void) shift; (void) qs;
    __trap();
#endif
}

// each slice's cells above / at the threshold; grid (TS_P, nq)
__global__ void __launch_bounds__(TS_T) topk_split_count_kernel(const float* __restrict__ scores,
                                                                const int32_t* __restrict__ steps, int64_t max_blocks,
                                                                TopkSplitQ* __restrict__ qs) {
    __shared__ int s_gt[TS_T / 32], s_eq[TS_T / 32];
    const int64_t qi = blockIdx.y;
    const int p = (int) blockIdx.x;
    TopkSlice v;
    if (!topk_slice(steps, qi, p, v)) return;
    TopkSplitQ* Q = qs + qi;
    const uint32_t thr = Q->prefix;
    const float* sc = scores + qi * max_blocks;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    int gt = 0, eq = 0;
    for (int64_t b = v.lo + t; b < v.hi; b += TS_T) {
        const int w = b < v.n_bid ? R : v.w_last;      // an empty last block adds 0
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        gt += __shfl_xor_sync(0xffffffffu, gt, o);
        eq += __shfl_xor_sync(0xffffffffu, eq, o);
    }
    if (lane == 0) {
        s_gt[warp] = gt;
        s_eq[warp] = eq;
    }
    __syncthreads();
    if (t == 0) {
        int a = 0, e = 0;
        for (int w = 0; w < TS_T / 32; ++w) {
            a += s_gt[w];
            e += s_eq[w];
        }
        Q->gt[p] = a;
        Q->eq[p] = e;
    }
}

// exclusive prefix of v over the CTA's TS_T threads (block_excl_scan's total assumes 32 warps)
__device__ __forceinline__ int split_excl_scan(int v, int* s_warp) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += s_warp[w];
    __syncthreads();                                   // s_warp is reused by the next scan
    return before + x - v;
}

// the cells, ascending; grid (TS_P, nq)
__global__ void __launch_bounds__(TS_T) topk_split_emit_kernel(const float* __restrict__ scores,
                                                               const int32_t* __restrict__ steps, int64_t max_blocks,
                                                               int64_t cap, const TopkSplitQ* __restrict__ qs,
                                                               int32_t* __restrict__ ids) {
    __shared__ int s_warp[TS_T / 32];
    const int64_t qi = blockIdx.y;
    const int p = (int) blockIdx.x, t = threadIdx.x;
    int32_t* out = ids + qi * cap;
    TopkSlice v;
    if (!topk_slice(steps, qi, p, v)) {                // everything is selected: the identity, ascending
        for (int64_t j = (int64_t) p * TS_T + t; j < v.n_kv; j += (int64_t) TS_P * TS_T) out[j] = (int32_t) j;
        return;
    }
    const TopkSplitQ* Q = qs + qi;
    const uint32_t thr = Q->prefix;
    const int64_t eq_budget = v.width - Q->above;      // cells equal to thr that fit, lowest index first
    int64_t gt_off = 0, eq_off = 0;                    // the slices before this one
    for (int r = 0; r < p; ++r) {
        gt_off += Q->gt[r];
        eq_off += Q->eq[r];
    }
    // each thread a contiguous run of the slice, so the cells come out ascending
    const float* sc = scores + qi * max_blocks;
    const int64_t n = v.hi > v.lo ? v.hi - v.lo : 0;
    const int64_t per = (n + TS_T - 1) / TS_T;
    const int64_t b0 = v.lo + (int64_t) t * per, b1 = b0 + per < v.hi ? b0 + per : v.hi;
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = b < v.n_bid ? R : v.w_last;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    const int64_t eq_before = eq_off + split_excl_scan(eq, s_warp);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    int64_t wpos = gt_off + (eq_off < eq_budget ? eq_off : eq_budget) + split_excl_scan(gt + (int) my_eq, s_warp);
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = b < v.n_bid ? R : v.w_last;
        if (w == 0) continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
}
#endif  // !__HIPCC__

#if !defined(__HIPCC__)
// ---- the decode top-k on a thread-block cluster (sm_90+; S19).  One CTA per query (block_topk_reg_kernel, or
// block_topk_kernel above 33,792 blocks: a --max-context over ~135K) makes four radix passes and two scans over up to
// 65,538 blocks on ONE SM while the rest of the GPU idles - a decode window has 1-5 queries.  Here a cluster of CL_N
// CTAs shares a query: CTA r holds the keys of blocks [r * per, (r + 1) * per) in shared memory, builds the digit
// histogram of its keys, and PUSHES it into slot r of every CTA's `hin` (distributed shared memory); after one cluster
// barrier each CTA sums the CL_N slots itself, so all of them take the same digit.  The emit pass needs each CTA's
// cells above / at the threshold: pushed the same way, then each CTA offsets its own by the ranks before it.
// RTX 5070, per call at capacity = context (decode_cluster_parity --bench): 21.9 -> 15.6 us at 32K, 58 -> 18 at 128K,
// 200 -> 22 at 262K (the one-CTA kernels' digit search was serial, too: one thread over 256 bins per pass).
//
// IDENTICAL IDS, by construction rather than by luck: the threshold thr (the width-th largest key, cells counted with
// their weights) and `above` (cells with a larger key) are pure functions of the multiset of (key, weight) - integer
// histograms, summed in any order - and the digit rule is block_topk_kernel's, written as a scan (the largest digit d
// whose cells at or above it reach `need`; 0 when none). The cells emitted are then fixed: every cell of a block with
// key > thr, and the first eq_budget = width - above cells at thr in ascending order (a block at thr may be cut),
// written in ascending cell order. Position of a thread's first cell = (cells above thr before it) + min(cells at thr
// before it, eq_budget): the telescoped sum of the reference's per-thread clamp. NaN keys are 0 (order_key) as there.
//
// Barriers: barrier.cluster arrive / wait (each CTA's threads all take part; it also orders the CTA's own shared
// memory, so it doubles as __syncthreads). Phase 0 is a relaxed arrive at entry, waited before the first remote store
// (a CTA's shared memory exists once it runs). Pushes go out before the arrive that releases them, and nothing reads
// another CTA's memory after the last wait - so a CTA may exit without a closing barrier (a wait counts the threads
// that have not exited). `hin` alternates by pass parity: CTA x reads slot buffer p&1 in pass p before arriving at
// pass p+1's barrier, and nobody writes buffer p&1 again (pass p+2) before waiting on that barrier.
constexpr int CL_N = 8;        // CTAs per query (the portable cluster size)
constexpr int CL_T = 1024;     // threads per CTA
constexpr int CL_MAXQ = 16;    // larger calls (prefill sub-batches) fill the GPU with one CTA per query already
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define STRATA_SEL_CLUSTER 1
#else
#define STRATA_SEL_CLUSTER 0   // an older target's code is a trap; qsa_block_topk_cluster never launches it there
#endif

#if STRATA_SEL_CLUSTER
__device__ __forceinline__ void cl_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }
__device__ __forceinline__ unsigned cl_rank() {
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}
template <typename T> __device__ __forceinline__ T* cl_map(T* p, unsigned rank) {   // p in CTA `rank`'s shared memory
    uint64_t o;
    asm volatile("mapa.u64 %0, %1, %2;\n" : "=l"(o) : "l"((uint64_t) p), "r"(rank));
    return reinterpret_cast<T*>(o);
}
// exclusive block scan of a 64-bit value over CL_T threads; `total` = the CTA's sum
__device__ __forceinline__ unsigned long long cl_excl_scan64(unsigned long long v, unsigned long long* s_warp,
                                                             unsigned long long& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned long long x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned long long y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    if (warp == 0) {
        const unsigned long long w = s_warp[lane];
        unsigned long long z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned long long y = __shfl_up_sync(0xffffffffu, z, o);
            if (lane >= o) z += y;
        }
        s_warp[lane] = z - w;
        if (lane == 31) s_warp[32] = z;
    }
    __syncthreads();
    total = s_warp[32];
    return s_warp[warp] + x - v;
}
#endif

// grid (CL_N, nq), cluster (CL_N, 1, 1), CL_T threads, dynamic shared memory: ceil(max_blocks / CL_N) keys
__global__ void __launch_bounds__(CL_T) block_topk_cluster_kernel(const float* __restrict__ scores,
                                                                  const int32_t* __restrict__ steps, int64_t max_blocks,
                                                                  int64_t cap, int32_t* __restrict__ ids) {
#if STRATA_SEL_CLUSTER
    extern __shared__ uint32_t keys[];                  // this CTA's blocks' keys
    __shared__ __align__(16) int hin[2][CL_N][256];     // the cluster's histograms, slot = the pushing rank
    __shared__ __align__(16) int hloc[2][256];          // this CTA's histogram
    __shared__ unsigned long long cnt[CL_N];            // per rank: its cells above thr (low 32 bits) and at it (high)
    __shared__ unsigned long long s_warp[33];
    __shared__ uint32_t s_prefix;
    __shared__ int s_above;
    const unsigned rank = cl_rank();
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    cl_arrive_relaxed();                                // phase 0: this CTA runs
    const int64_t qi = blockIdx.y;
    const int32_t* st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t* out = ids + qi * cap;
    if (n_kv <= width) {                                // the identity, split over the cluster; no remote access
        for (int64_t j = (int64_t) rank * CL_T + t; j < n_kv; j += (int64_t) CL_N * CL_T) out[j] = (int32_t) j;
        return;
    }
    // the tail block n_bid weighs its cells (0..3); with none it is not a candidate (the reference skips w == 0)
    const int wtail = (int) (n_kv - n_bid * R);
    const int64_t nbe = n_bid + (wtail > 0 ? 1 : 0);
    const int64_t per = (nbe + CL_N - 1) / CL_N;       // <= ceil(max_blocks / CL_N), the keys the host sized
    const int64_t lo = (int64_t) rank * per < nbe ? (int64_t) rank * per : nbe;
    const int64_t hi = lo + per < nbe ? lo + per : nbe;
    const int n = (int) (hi - lo);
    const int tail_i = (wtail > 0 && n_bid >= lo && n_bid < hi) ? (int) (n_bid - lo) : -1;
    const float* sc = scores + qi * max_blocks + lo;
    for (int i = t; i < n; i += CL_T) keys[i] = order_key(sc[i]);
    for (int i = t; i < 2 * 256; i += CL_T) (&hloc[0][0])[i] = 0;
    __syncthreads();
    uint32_t prefix = 0;
    int above = 0;                                      // cells strictly above the digits fixed so far
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        const uint32_t hmask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
        int* hl = hloc[pass & 1];
        // warp-aggregated: the scores share their top bits, so plain atomics would queue on a few bins
        for (int i0 = 0; i0 < n; i0 += CL_T) {
            const int i = i0 + t;
            int bin = -1;
            if (i < n) {
                const uint32_t k = keys[i];
                if ((k & hmask) == prefix) bin = (int) ((k >> shift) & 255);
            }
            const unsigned same = __match_any_sync(0xffffffffu, bin);
            if (bin >= 0 && lane == __ffs(same) - 1) atomicAdd(&hl[bin], __popc(same) * R);
            if (bin >= 0 && i == tail_i) atomicAdd(&hl[bin], wtail - R);
        }
        __syncthreads();
        if (pass == 0) cl_wait();                       // phase 0 done: every CTA's shared memory is there
        for (int i = t; i < CL_N * 64; i += CL_T) {     // the histogram into slot `rank` of every CTA, as int4
            const int dst = i >> 6, c = i & 63;
            *cl_map(reinterpret_cast<int4*>(&hin[pass & 1][rank][0]) + c, (unsigned) dst) =
                reinterpret_cast<const int4*>(hl)[c];
        }
        for (int i = t; i < 256; i += CL_T) hloc[(pass + 1) & 1][i] = 0;
        cl_arrive();
        cl_wait();
        if (warp == 0) {
            // lane L owns digits 255 - 8L down to 248 - 8L; the digit holding the need-th cell from the top
            const int need = (int) width - above;
            int tot[8], part = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int d = 255 - 8 * lane - k;
                int v = 0;
#pragma unroll
                for (int r = 0; r < CL_N; ++r) v += hin[pass & 1][r][d];
                tot[k] = v;
                part += v;
            }
            int incl = part;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int y = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += y;
            }
            const int excl = incl - part;
            const unsigned hit = __ballot_sync(0xffffffffu, excl < need && incl >= need);
            // no hit cannot happen (the cells total n_kv > width); the reference would then take digit 0
            const int who = hit ? __ffs(hit) - 1 : 31;
            if (lane == who) {
                int acc = excl, k = 0;
                for (; k < 7; ++k) {
                    if (acc + tot[k] >= need) break;
                    acc += tot[k];
                }
                s_prefix = prefix | ((uint32_t) (255 - 8 * lane - k) << shift);
                s_above = above + acc;
            }
        }
        __syncthreads();
        prefix = s_prefix;
        above = s_above;
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;           // cells at thr that fit, lowest index first
    // each thread a contiguous run of the CTA's keys; an odd run length keeps the shared reads conflict-free
    int seg = (n + CL_T - 1) / CL_T;
    if ((seg & 1) == 0 && seg > 0) ++seg;
    const int s0 = t * seg < n ? t * seg : n, s1 = s0 + seg < n ? s0 + seg : n;
    uint32_t gt = 0, eq = 0;
    for (int i = s0; i < s1; ++i) {
        const uint32_t k = keys[i], w = i == tail_i ? (uint32_t) wtail : (uint32_t) R;
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    unsigned long long cta;
    const unsigned long long before =
        cl_excl_scan64((unsigned long long) gt | ((unsigned long long) eq << 32), s_warp, cta);
    if (t < CL_N) *cl_map(&cnt[rank], (unsigned) t) = cta;
    cl_arrive();
    cl_wait();
    int64_t gt_off = 0, eq_off = 0;
    for (unsigned r = 0; r < rank; ++r) {
        gt_off += (int64_t) (cnt[r] & 0xffffffffu);
        eq_off += (int64_t) (cnt[r] >> 32);
    }
    const int64_t eq_before = eq_off + (int64_t) (before >> 32);
    int64_t wpos = gt_off + (int64_t) (before & 0xffffffffu) + (eq_before < eq_budget ? eq_before : eq_budget);
    int64_t eq_left = eq_budget - eq_before;
    if (eq_left < 0) eq_left = 0;
    for (int i = s0; i < s1; ++i) {
        const uint32_t k = keys[i];
        const int w = i == tail_i ? wtail : R;
        const int64_t b = lo + i;
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = (int32_t) (b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
        }
    }
#else
    (void) scores; (void) steps; (void) max_blocks; (void) cap; (void) ids;
    __trap();
#endif
}
#endif  // !__HIPCC__
}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    static const bool multi = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
    if (multi && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
#if !defined(__HIPCC__)
        // the heads' sums as one reduce-scatter (bitwise the same scores); STRATA_SCORES_RS=0: the kernel before it
        static const bool rs = [] { const char* v = std::getenv("STRATA_SCORES_RS"); return v == nullptr || std::atoi(v) != 0; }();
        if (rs)
            block_scores_rs_kernel<<<256, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps,
                                                                                      (int) nq, max_blocks, scores);
        else
#endif
        block_scores_multi_kernel<<<256, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, (int) nq,
                                                                                     max_blocks, scores);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores multi: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const dim3 grid((unsigned) ((reach + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned) nq);
    block_scores_kernel<<<grid, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, max_blocks,
                                                                              scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    if (nq <= 0) return true;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535 * TC_QT) return false;
#if defined(__HIPCC__)
    // AMD: the gfx12 (RDNA4) WMMA scorer, opt-in (STRATA_SELECT_WMMA=1): it selects slightly differently from the warp
    // kernel (254/256 queries the same), so the default keeps the warp kernel; every other target keeps it too (false)
    static const bool wmma_on = [] {
        const char* v = std::getenv("STRATA_SELECT_WMMA");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (!wmma_on || !sel_gfx12_device()) return false;
    {
        const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
        const int64_t per = (int64_t) 4 * WITER * 16;
        const dim3 grid((unsigned) ((reach + per - 1) / per), (unsigned) ((nq + WQT - 1) / WQT));
        block_scores_wmma_kernel<<<grid, 128, 0, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, reach, scores);
        block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_tc: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return true;
    }
#else
    {   // sm_80 or newer (TF32 MMA); an older card keeps the warp kernel
        static int cc_major[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
        if (cc_major[dev] == 0) {
            int major = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            // STRATA_QSA_WARP=1|select (an A/B arm): the pre-sm_80 kernels on any card, as RTX 20 runs them
            const char* w = std::getenv("STRATA_QSA_WARP");
            cc_major[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "select")) ? 7 : strata::cc_major_of(major);
        }
        if (cc_major[dev] < 8) return false;
    }
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs it on several)
    int adev = 0;
    cudaGetDevice(&adev);
    const int bytes = (TC_QT * TC_QS + TC_NB * TC_KS) * (int) sizeof(float);
    if (!attr[adev]) {
        if (cudaFuncSetAttribute(block_scores_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[adev] = true;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t per = (int64_t) TC_NB * TC_ITER;
    const dim3 grid((unsigned) ((reach + per - 1) / per), (unsigned) ((nq + TC_QT - 1) / TC_QT));
    block_scores_tc_kernel<<<grid, 128, bytes, (cudaStream_t) stream>>>(pooled, q_idx, steps, nq, max_blocks, reach,
                                                                         scores);
    block_scores_tail_kernel<<<(unsigned) nq, 32, 0, (cudaStream_t) stream>>>(dead, q_idx, steps, max_blocks, scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores_tc: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
#endif
}

void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    block_topk_kernel<<<(unsigned) nq, TOPK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

#if !defined(__HIPCC__)
// Turing and sm_86 have retained model measurements for this CUDA dispatch (sm_86: 4x RTX 3090, IQ3_S under a 262K
// --max-context, prompts +2% at 128K and +1% at 250K, the same tokens). Other CUDA devices keep the capacity rule
// (the RTX 5070 regression below). Cache the properties per calling thread; layer-split device switches are checked.
static bool topk_active_turing_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    static thread_local int cached_device = -1;
    static thread_local bool turing = false;
    if (dev != cached_device) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
        turing = (prop.major == 7 && prop.minor == 5) || (prop.major == 8 && prop.minor == 6);
        cached_device = dev;
    }
    return turing;
}
#endif

bool qsa_block_topk_cluster(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes& s, int32_t* ids, void* stream) {
#if defined(__HIPCC__)
    (void) scores; (void) steps; (void) nq; (void) max_blocks; (void) cap; (void) s; (void) ids; (void) stream;
    return false;
#else
    if (nq <= 0) return true;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s) || nq > 65535 || max_blocks <= 0) return false;
    const size_t smem = (size_t) ((max_blocks + CL_N - 1) / CL_N) * sizeof(uint32_t);
    // Per device (a layer split runs on several): 1 the cluster kernel runs here, 2 it does not. It needs sm_90+ (the
    // card's, or STRATA_EMULATE_CC's) AND code built for it: a build with only older code JIT-compiles their PTX, whose
    // copy of this kernel is a trap - the function's PTX version says which. `opt`: the dynamic shared memory opted in.
    static int ok[64] = {};
    static size_t opt[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        int major = 0;
        cudaFuncAttributes fa{};
        const bool code = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                          strata::cc_major_of(major) >= 9 &&
                          cudaFuncGetAttributes(&fa, block_topk_cluster_kernel) == cudaSuccess && fa.ptxVersion >= 90 &&
                          fa.binaryVersion >= 90;
        cudaGetLastError();
        ok[dev] = code ? 1 : 2;
    }
    if (ok[dev] != 1) return false;
    if (smem > opt[dev]) {   // a larger capacity: opt in, and check that a cluster of CL_N such CTAs can be resident
        int clusters = 0;
        cudaLaunchConfig_t q{};
        cudaLaunchAttribute qa[1];
        qa[0].id = cudaLaunchAttributeClusterDimension;
        qa[0].val.clusterDim.x = CL_N;
        qa[0].val.clusterDim.y = 1;
        qa[0].val.clusterDim.z = 1;
        q.gridDim = dim3(CL_N, 1, 1);
        q.blockDim = dim3(CL_T, 1, 1);
        q.dynamicSmemBytes = smem;
        q.attrs = qa;
        q.numAttrs = 1;
        if (cudaFuncSetAttribute(block_topk_cluster_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) !=
                cudaSuccess ||
            cudaOccupancyMaxActiveClusters(&clusters, block_topk_cluster_kernel, &q) != cudaSuccess || clusters < 1) {
            cudaGetLastError();
            return false;    // this capacity takes the one-CTA kernels; a smaller one may still fit
        }
        opt[dev] = smem;
    }
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CL_N;
    at[0].val.clusterDim.y = 1;
    at[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(CL_N, (unsigned) nq, 1);
    cfg.blockDim = dim3(CL_T, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = (cudaStream_t) stream;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    const cudaError_t e = cudaLaunchKernelEx(&cfg, block_topk_cluster_kernel, scores, steps, max_blocks, cap, ids);
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk cluster: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
#endif
}

size_t qsa_topk_split_bytes(int64_t nq) { return (size_t) (nq > 0 ? nq : 0) * sizeof(TopkSplitQ); }

bool qsa_block_topk_split(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                          const QsaShapes& s, int32_t* ids, void* scratch, void* stream) {
#if defined(__HIPCC__)
    (void) scores; (void) steps; (void) nq; (void) max_blocks; (void) cap; (void) s; (void) ids; (void) scratch;
    (void) stream;
    return false;
#else
    // STRATA_TOPK_SPLIT: unset = a capacity over 64K cells, 1 = at any capacity, 0 = never.  RTX 3090, 3 queries,
    // capacity 128K (the register kernel's domain): register kernel / split 0.029 / 0.034 ms at a 4K or 16K context,
    // 0.036 / 0.035 at 32K, 0.039 / 0.035 at 64K, 0.080 / 0.036 at 120K; past the register capacity (a 262K
    // --max-context) the one-CTA kernel is 7x slower at 250K
    static const int mode = [] {
        const char* v = std::getenv("STRATA_TOPK_SPLIT");
        return !v ? 1 : std::atoi(v) != 0 ? 2 : 0;
    }();
    if (nq <= 0) return true;
    if (mode == 0 || scratch == nullptr || nq > CL_MAXQ || max_blocks <= 0 || s.idx_block != R ||
        cap < qsa_selection_width(kTopkMaxCells, s) || (mode == 1 && max_blocks <= 65536 / R + 2))
        return false;
    // per device: 1 it runs here (sm_70 to sm_89: __match_any_sync, and no clusters - sm_90+ has the cluster kernel),
    // 2 it does not
    static int ok[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (ok[dev] == 0) {
        int major = 0;
        const bool yes = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                         strata::cc_major_of(major) >= 7 && strata::cc_major_of(major) < 9;
        cudaGetLastError();
        ok[dev] = yes ? 1 : 2;
    }
    if (ok[dev] != 1) return false;
    TopkSplitQ* qs = (TopkSplitQ*) scratch;
    const dim3 grid(TS_P, (unsigned) nq);
    const cudaStream_t cs = (cudaStream_t) stream;
    for (int shift = 24; shift >= 0; shift -= 8)
        topk_split_pass_kernel<<<grid, TS_T, 0, cs>>>(scores, steps, max_blocks, shift, qs);
    topk_split_count_kernel<<<grid, TS_T, 0, cs>>>(scores, steps, max_blocks, qs);
    topk_split_emit_kernel<<<grid, TS_T, 0, cs>>>(scores, steps, max_blocks, cap, qs, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk split: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
#endif
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    // keys in registers when every query's blocks fit (contexts up to ~135K cells); the same ids. STRATA_TOPK_OLD=1:
    // the kernel that reads them from memory on every pass
    static const bool old = std::getenv("STRATA_TOPK_OLD") != nullptr;
    if (nq <= 0) return;
#if !defined(__HIPCC__)
    // sm_90+: a cluster of CL_N CTAs per query for the calls of a few queries (decode windows); the same ids.
    // STRATA_QSA_CLUSTER=0: the one-CTA kernels below
    static const bool cluster = [] {
        const char* v = std::getenv("STRATA_QSA_CLUSTER");
        return !v || std::atoi(v) != 0;
    }();
    if (cluster && !old && nq <= CL_MAXQ && qsa_block_topk_cluster(scores, steps, nq, max_blocks, cap, s, ids, stream))
        return;
#endif
    // the blocks a query can have: the call's active count when the caller knows it (the prompt path), else the capacity.
    // Decode (no count) keeps the capacity rule and the original register width: nothing changes there.
#if defined(__HIPCC__)
    const bool counted = active_blocks > 0;
#else
    // Turing: --max-context 262144 makes the stride 65538, even while a 131K prompt's active blocks fit in
    // TK_T * TK_PER registers. Use the prefill bound on sm_75 and sm_86, keeping max_blocks as the score-row stride.
    // Other CUDA devices keep 0.1.32's capacity rule: #337 was measured on RDNA4, and RTX 5070 64K prompts were
    // 1-3% slower. Decode/captured graphs omit the bound and never query the device here.
    static const bool capacity_guard = std::getenv("STRATA_TOPK_CAPACITY_GUARD") != nullptr;
    // STRATA_TOPK_ACTIVE_ANY=1 (tests): the Turing dispatch on any CUDA card, so qsa_topk_active_parity checks it
    // on whatever card runs the tests (the kernels are the same on every architecture)
    static const bool any_card = [] { const char* v = std::getenv("STRATA_TOPK_ACTIVE_ANY"); return v && v[0] == '1'; }();
    const bool counted = !capacity_guard && active_blocks > 0 && active_blocks <= max_blocks &&
                         (any_card || topk_active_turing_device());
#endif
    const int64_t reach = counted && active_blocks < max_blocks ? active_blocks : max_blocks;
    const int64_t fit = (int64_t) TK_T * (counted ? TK_PER_MAX : TK_PER);
#if defined(__HIPCC__)
    constexpr int64_t kRegMinBlocks = 7168;   // gfx1201: below ~28K cells the 1,024-thread kernel's fixed cost loses to the ref
    const bool too_small = counted && reach < kRegMinBlocks;
#else
    const bool too_small = false;
#endif
    if (old || too_small || reach > fit) {
#if !defined(__HIPCC__)
        // past the register kernel's capacity: the 1,024-thread kernel (the same ids) on sm_70+.  STRATA_TOPK_WIDE=0: the
        // 256-thread kernel; =decode: the 1,024-thread one for calls of a few queries only (the decode windows)
        static const int wide = [] {
            const char* v = std::getenv("STRATA_TOPK_WIDE");
            return !v ? 2 : std::strcmp(v, "decode") == 0 ? 1 : std::atoi(v) != 0 ? 2 : 0;
        }();
        static int wide_ok[64] = {};   // per device: 1 sm_70+, 2 older
        int wdev = 0;
        if (!old && wide > 0 && (wide == 2 || nq <= CL_MAXQ) && cudaGetDevice(&wdev) == cudaSuccess && wdev >= 0 &&
            wdev < 64) {
            if (wide_ok[wdev] == 0) {
                int major = 0;
                wide_ok[wdev] = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, wdev) == cudaSuccess &&
                                        strata::cc_major_of(major) >= 7 ? 1 : 2;
                cudaGetLastError();
            }
            if (wide_ok[wdev] == 1 && s.idx_block == R && cap >= qsa_selection_width(kTopkMaxCells, s)) {
                block_topk_wide_kernel<<<(unsigned) nq, TW_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
                const cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk wide: %s\n", cudaGetErrorString(e)); std::exit(1); }
                return;
            }
        }
#endif
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    if (reach <= (int64_t) TK_T * TK_PER)
        block_topk_reg_kernel<TK_PER><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    else
        block_topk_reg_kernel<TK_PER_MAX><<<(unsigned) nq, TK_T, 0, (cudaStream_t) stream>>>(scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace strata::kernels
