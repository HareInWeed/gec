#pragma once
#ifndef GEC_DLP_POLLARD_LAMBDA_HPP
#define GEC_DLP_POLLARD_LAMBDA_HPP

#include <gec/bigint/mixin/random.hpp>
#include <gec/utils/basic.hpp>
#include <gec/utils/misc.hpp>
#include <gec/utils/static_map.hpp>

#include <random>

#ifdef GEC_ENABLE_PTHREADS
#include <pthread.h>
#include <unordered_map>
#endif // GEC_ENABLE_PTHREADS

#ifdef __CUDACC__
#include <thrust/random.h>
#include <unordered_map>
#endif // __CUDACC__

namespace gec {

namespace dlp {

/** @brief pollard lambda algorithm for ECDLP
 *
 * `a` must be strictly less than `b`, otherwise the behaviour is undefined.
 */
template <typename S, typename P, typename Rng, typename Ctx>
__host__ void pollard_lambda(S &GEC_RSTRCT x, S *GEC_RSTRCT sl,
                             P *GEC_RSTRCT pl, const S &GEC_RSTRCT bound,
                             const S &GEC_RSTRCT a, const S &GEC_RSTRCT b,
                             const P &GEC_RSTRCT g, const P &GEC_RSTRCT h,
                             GecRng<Rng> &rng, Ctx &GEC_RSTRCT ctx) {
    using F = typename P::Field;
    auto &ctx_view = ctx.template view_as<P, P, P, S, S, S>();
    auto &p1 = ctx_view.template get<0>();
    auto &p2 = ctx_view.template get<1>();
    auto &temp = ctx_view.template get<2>();
    auto &d = ctx_view.template get<3>();
    auto &idx = ctx_view.template get<4>();
    ctx_view.template get<5>().set_one();
    const auto &one = ctx_view.template get<5>();
    auto &rest_ctx = ctx_view.rest();

    while (true) {
        P *u = &p1, *v = &p2, *tmp = &temp;

        S::sub(x, b, a);
        // with `a` less than `b`, `m` would not underflow
        size_t m = x.most_significant_bit();
        for (size_t i = 0; i < m; ++i) {
            sl[i].array()[0] = typename S::LimbT(i);
        }
        for (size_t i = 0; i < m; ++i) {
            size_t ri = m - 1 - i;
            utils::swap(sl[ri].array()[0], sl[rng.sample(ri)].array()[0]);
        }
        for (size_t i = 0; i < m; ++i) {
            typename F::LimbT e = sl[i].array()[0];
            sl[i].set_pow2(e);
            P::mul(pl[i], sl[i], g, rest_ctx);
        }

        S::sample_inclusive(x, a, b, rng, rest_ctx);
        P::mul(*u, x, g, rest_ctx);
        for (idx.set_zero(); idx < bound; S::add(idx, one)) {
            size_t i = u->x().array()[0] % m;
            S::add(x, sl[i]);
            P::add(*tmp, *u, pl[i], rest_ctx);
            utils::swap(u, tmp);
        }

        d.set_zero();
        *v = h;
        for (idx.set_zero(); idx < bound; S::add(idx, one)) {
            if (P::eq(*u, *v)) {
                S::sub(x, d);
                return;
            }
            size_t idx = v->x().array()[0] % m;
            S::add(d, sl[idx]);
            P::add(*tmp, *v, pl[idx], rest_ctx);
            utils::swap(v, tmp);
        }
    }
}

#ifdef GEC_ENABLE_PTHREADS

namespace _pollard_lambda_ {

template <typename S, typename P>
struct SharedData {
    std::vector<size_t> buckets;
    std::vector<typename P::Hasher::result_type> trap_hashes;
    std::vector<S> xs;
    std::vector<P> traps;
    std::vector<S> sl;
    std::vector<P> pl;
    utils::CHD<> phf;
    pthread_mutex_t x_lock;
    pthread_barrier_t barrier;
    S &x;
    const S &a;
    const S &b;
    const P &g;
    const P &h;
    const S &bound;

    size_t worker_n;
    bool shutdown;

    SharedData(size_t worker_n, S &GEC_RSTRCT x, const S &GEC_RSTRCT a,
               const S &GEC_RSTRCT b, const P &GEC_RSTRCT g,
               const P &GEC_RSTRCT h, const S &GEC_RSTRCT bound, size_t m)
        : buckets(), trap_hashes(), xs(), traps(), sl(m), pl(m),
          phf(nullptr, worker_n), x_lock(PTHREAD_MUTEX_INITIALIZER), barrier(),
          x(x), a(a), b(b), g(g), h(h), bound(bound), worker_n(worker_n),
          shutdown(false) {
        buckets.resize(phf.B);
        trap_hashes.resize(phf.N);
        xs.resize(phf.N);
        traps.resize(phf.N);
        phf.buckets = buckets.data();
        pthread_barrier_init(&barrier, nullptr, (unsigned int)(worker_n));
    }
};

template <typename S, typename P>
struct WorkerData {
    SharedData<S, P> &shared_data;
    size_t seed;
    size_t id;
};

template <typename S, typename P, typename Rng>
void *worker(void *data_ptr) {
    using Data = WorkerData<S, P>;
    using LimbT = typename P::Field::LimbT;

    typename P::Hasher hasher;
    Data &local = *static_cast<Data *>(data_ptr);
    SharedData<S, P> &shared = local.shared_data;
    volatile bool &shutdown = shared.shutdown;

    const size_t m = shared.sl.size();
    P p1, p2;
    P *u = &p1, *tmp = &p2;
    S x, j;
    typename P::template Context<> ctx;
    auto rng = make_gec_rng(Rng((unsigned int)(local.seed)));

    while (true) {
        // calculate jump table
        if (local.id == 0) {
            for (size_t i = 0; i < m; ++i) {
                shared.sl[i].array()[0] = typename S::LimbT(i);
            }
            for (size_t i = 0; i < m; ++i) {
                size_t ri = m - 1 - i;
                utils::swap(shared.sl[ri].array()[0],
                            shared.sl[rng.sample(ri)].array()[0]);
            }
            for (size_t i = 0; i < m; ++i) {
                // TODO: maybe using multithread to generate the jump table?
                LimbT e = shared.sl[i].array()[0];
                shared.sl[i].set_pow2(e);
                P::mul(shared.pl[i], shared.sl[i], shared.g, ctx);
            }
#ifdef GEC_DEBUG
            printf("[worker %03zu]: jump table generated\n", local.id);
#endif // GEC_DEBUG
        }

        pthread_barrier_wait(&shared.barrier);

        // setting traps
        S &t = shared.xs[local.id];
        S::sample_inclusive(t, shared.a, shared.b, rng, ctx);
        P::mul(*u, t, shared.g, ctx);
        for (j.set_zero(); j < shared.bound; S::add(j, 1)) {
            size_t i = u->x().array()[0] % m;
            S::add(t, shared.sl[i]);
            P::add(*tmp, *u, shared.pl[i], ctx);
            utils::swap(u, tmp);
#ifdef GEC_DEBUG
            if (!(utils::LowerKMask<LimbT, 20>::value & j.array()[0])) {
                printf("[worker %03zu]: calculating trap, step ", local.id);
                j.println();
            }
#endif // GEC_DEBUG
        }
        shared.traps[local.id] = *u;
        shared.trap_hashes[local.id] = hasher(*u);
#ifdef GEC_DEBUG
        printf("[worker %03zu]: trap set\n", local.id);
#endif // GEC_DEBUG
        pthread_barrier_wait(&shared.barrier);

        // construct static maps
        if (local.id == 0) {
            size_t placeholder =
                shared.phf.fill_placeholder(shared.trap_hashes.data());
#ifdef GEC_DEBUG
            auto duplicates = shared.phf.build(shared.trap_hashes.data());
            for (auto &dup : duplicates) {
                printf("[worker %03zu]: find hash collision: \n", local.id);
                shared.traps[dup.first].println();
                shared.traps[dup.second].println();
                printf("some traps are omitted");
            }
#else
            shared.phf.build(shared.trap_hashes.data());
#endif // GEC_DEBUG
            shared.phf.rearrange(shared.trap_hashes.data(), placeholder,
                                 shared.xs.data(), shared.traps.data());
        }
        pthread_barrier_wait(&shared.barrier);

        // start searching
        S::sample_inclusive(x, shared.a, shared.b, rng, ctx);
        P::mul(*tmp, x, shared.g, ctx);
        P::add(*u, shared.h, *tmp, ctx);
        for (j.set_zero(); j < shared.bound; S::add(j, 1)) {
            if (shutdown) {
                break;
            }
            auto hash = hasher(*u);
            auto idx = shared.phf.get(hash);
            if (shared.trap_hashes[idx] == hash &&
                P::eq(shared.traps[idx], *u)) {
                pthread_mutex_lock(&shared.x_lock);
                if (!shutdown) {
                    S::sub(shared.x, shared.xs[idx], x);
                    shutdown = true;
                }
                pthread_mutex_unlock(&shared.x_lock);
            }
            size_t i = u->x().array()[0] % m;
            S::add(x, shared.sl[i]);
            P::add(*tmp, *u, shared.pl[i], ctx);
            utils::swap(u, tmp);
#ifdef GEC_DEBUG
            if (!(utils::LowerKMask<LimbT, 20>::value & j.array()[0])) {
                printf("[worker %03zu]: searching, step ", local.id);
                j.println();
            }
#endif // GEC_DEBUG
        }

        pthread_barrier_wait(&shared.barrier);
        // check success

        if (shutdown) {
#ifdef GEC_DEBUG
            printf("[worker %03zu]: collision found, shutting down\n",
                   local.id);
#endif // GEC_DEBUG
            return nullptr;
        }
#ifdef GEC_DEBUG
        printf("[worker %03zu]: collision not found, retry\n", local.id);
#endif // GEC_DEBUG
    }
}

} // namespace _pollard_lambda_

/** @brief multithread pollard lambda algorithm for ECDLP, requires `pthreads`
 *
 * `a` must be strictly less than `b`, otherwise the behaviour is undefined.
 */
template <typename S, typename P, typename Rng>
void multithread_pollard_lambda(S &GEC_RSTRCT x, const S &GEC_RSTRCT bound,
                                unsigned int worker_n, const S &GEC_RSTRCT a,
                                const S &GEC_RSTRCT b, const P &GEC_RSTRCT g,
                                const P &GEC_RSTRCT h, GecRng<Rng> &rng) {
    using namespace _pollard_lambda_;
    using Shared = SharedData<S, P>;
    using Data = WorkerData<S, P>;

    std::vector<pthread_t> workers(worker_n);

    typename P::template Context<> ctx;

    S::sub(x, b, a);
    // with `a` less than `b`, `m` would not underflow
    size_t m = x.most_significant_bit();

    Shared shared(worker_n, x, a, b, g, h, bound, m);

    std::vector<Data> params(worker_n, Data{shared, 0, 0});

    for (size_t i = 0; i < worker_n; ++i) {
        auto &param = params[i];
        param.seed = rng.template sample<size_t>();
        param.id = i;
        pthread_create(&workers[i], nullptr, worker<S, P, Rng>,
                       static_cast<void *>(&param));
    }

    for (size_t i = 0; i < worker_n; ++i) {
        pthread_join(workers[i], nullptr);
    }
}

#endif // GEC_ENABLE_PTHREADS

#ifdef __CUDACC__

namespace _pollard_lambda_ {

template <typename T>
struct MaskZero {
    __host__ __device__ GEC_INLINE static bool call(const T &a, const T &b) {
        return !(a & b);
    }
};

template <typename P>
static __constant__ P cd_g;
template <typename P>
static __constant__ P cd_h;
template <typename F>
static __constant__ F cd_mask;
template <typename S>
static __constant__ S cd_a;
template <typename S>
static __constant__ S cd_bound;

template <typename S, typename P>
__global__ void init_wild_kernel(S *GEC_RSTRCT s, P *GEC_RSTRCT p, size_t n,
                                 size_t step) {
    using uint = unsigned int;
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;

    typename P::template Context<> ctx;
    if (id < n) {
        auto &ctx_view = ctx.template view_as<P, P, S>();
        P &local_p = ctx_view.template get<0>();
        P &natural_p = ctx_view.template get<1>();
        S &local_s = ctx_view.template get<2>();
        auto &rest_ctx = ctx_view.rest();

        local_s.set_zero();
        local_s.array()[0] = id * step;

        P::mul(local_p, local_s, cd_g<P>, rest_ctx);
        P::add(natural_p, local_p, cd_h<P>, rest_ctx);
        s[id] = local_s;
        p[id] = natural_p;
    }
}
template <typename S, typename P>
__global__ void init_tame_kernel(S *GEC_RSTRCT s, P *GEC_RSTRCT p, size_t n,
                                 size_t step) {
    using uint = unsigned int;
    const uint id = blockIdx.x * blockDim.x + threadIdx.x;

    typename P::template Context<> ctx;
    if (id < n) {
        S local_s;
        P local_p;

        local_s = cd_bound<S>;
        local_s.template shift_right<1>();
        S::add(local_s, cd_a<S>);
        S::add(local_s, id * step);
        P::mul(local_p, local_s, cd_g<P>, ctx);

        s[id] = local_s;
        p[id] = local_p;
    }
}

template <typename SizeT>
__global__ void reset_flag_kernel(volatile bool *GEC_RSTRCT done,
                                  SizeT *GEC_RSTRCT len) {
    *done = false;
    *len = SizeT(0);
}
template <typename S, typename P>
__global__ void
searching_kernel(volatile bool *GEC_RSTRCT done, P *p_buffer, S *x_buffer,
                 unsigned int *GEC_RSTRCT len, unsigned int buffer_len,
                 P *GEC_RSTRCT ps, S *GEC_RSTRCT xs, const P *GEC_RSTRCT pl,
                 const S *GEC_RSTRCT sl, size_t l_len, unsigned int thread_n,
                 const unsigned int check_mask) {

    using F = typename P::Field;
    using LT = typename F::LimbT;
    const typename F::LimbT jump_mask = l_len - 1;

    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= thread_n) {
        return;
    }

    typename P::template Context<> ctx;
    typename P::Hasher hasher;

    S x = xs[id];
    P p1 = ps[id], p2;

    bool in_p1 = true;
#define GEC_DEST_ (in_p1 ? p2 : p1)
#define GEC_SRC_ (in_p1 ? p1 : p2)
#define GEC_RELOAD_ (in_p1 = !in_p1)

    for (unsigned int k = 0;; ++k) {
        if (!(k & check_mask) && *done)
            goto shutdown;
        if (utils::VtSeqAll<F::LimbN, LT, MaskZero<LT>>::call(
                GEC_SRC_.x().array(), cd_mask<F>.array())) {
            unsigned int idx = atomicAdd(len, 1);
            if (idx < buffer_len) {
                // if (x.array()[0] == 0xfd8c7f60015dee80llu) {
                //     printf("[device %04llu]: k = %u, [", id, k);
                //     x.print();
                //     printf("]: \n");
                //     GEC_SRC_.println();
                // }
                x_buffer[idx] = x;
                p_buffer[idx] = GEC_SRC_;
            } else {
                *done = true;
                goto shutdown;
            }
        }
        // size_t i = hasher(GEC_SRC_) & 0x1F;
        // size_t i = GEC_SRC_.x().array()[0] & 0x1F;
        // size_t i = GEC_SRC_.x().array()[0] % l_len;
        size_t i = GEC_SRC_.x().array()[0] & jump_mask;
        S::add(x, sl[i]);
        P::add(GEC_DEST_, GEC_SRC_, pl[i], ctx);
        GEC_RELOAD_;
    }
shutdown:
    // if (id == 4480) {
    //     printf("x = ");
    //     x.println();
    //     printf("p = \n");
    //     GEC_SRC_.println();
    // }
    xs[id] = x;
    ps[id] = GEC_SRC_;

#undef GEC_DEST_
#undef GEC_SRC_
#undef GEC_RELOAD_
}

} // namespace _pollard_lambda_

/**
 * @brief CUDA implementation of pollard lambda algorithm
 *
 * Note that this function is not thread safe
 *
 * @tparam S TODO
 * @tparam P TODO
 * @tparam Rng TODO
 * @param x TODO
 * @param bound TODO
 * @param lower TODO
 * @param upper TODO
 * @param g TODO
 * @param h TODO
 * @param seed TODO
 * @param block_num TODO
 * @param thread_num TODO
 */
template <typename S, typename P, typename Rng = std::mt19937>
__host__ cudaError_t cu_pollard_lambda(
    S &GEC_RSTRCT x, const S &GEC_RSTRCT a, const S &GEC_RSTRCT b,
    const P &GEC_RSTRCT g, const P &GEC_RSTRCT h,
    const typename P::Field &candidate_mask, size_t seed,
    unsigned int block_num, unsigned int thread_num,
    unsigned int buffer_size = 0x1000, unsigned int check_mask = 0x3FF) {

    using namespace _pollard_lambda_;
    using uint = unsigned int;
    using FieldT = typename P::Field;
    using VS = std::vector<S>;
    using VP = std::vector<P>;
    using Map = std::unordered_multimap<P, S, typename P::Hasher>;

    cudaError_t cu_err = cudaSuccess;
#ifdef GEC_DEBUG
#define _CUDA_CHECK_TO_(code, label)                                           \
    do {                                                                       \
        cu_err = (code);                                                       \
        if (cu_err != cudaSuccess) {                                           \
            printf("%s(%d): %s, %s", __FILE__, __LINE__,                       \
                   cudaGetErrorName(cu_err), cudaGetErrorString(cu_err));      \
            goto label;                                                        \
        }                                                                      \
    } while (0)
#else
#define _CUDA_CHECK_TO_(code, label)                                           \
    do {                                                                       \
        cu_err = (code);                                                       \
        if (cu_err != cudaSuccess)                                             \
            goto label;                                                        \
    } while (0)
#endif // GEC_DEBUG
#define _CUDA_CHECK_(code) _CUDA_CHECK_TO_((code), clean_up)
    const size_t thread_n = block_num * thread_num;
    const bool true_literal = true;

    S::sub(x, b, a);
    const size_t l_len =
        1 << utils::most_significant_bit(x.most_significant_bit());

    const size_t tame_n = thread_n;
    const size_t wild_n = thread_n - 1;
    const size_t step_size = tame_n * wild_n;

    VS sl(l_len);
    S *d_sl = nullptr;

    VP pl(l_len);
    P *d_pl = nullptr;

    typename P::template Context<> ctx;

    S *d_wxs = nullptr;
    P *d_wilds = nullptr;
    S *d_wx_buf = nullptr;
    P *d_wild_buf = nullptr;
    VS wx_buf(buffer_size);
    VP wild_buf(buffer_size);

    S *d_txs = nullptr;
    P *d_tames = nullptr;
    S *d_tx_buf = nullptr;
    P *d_tame_buf = nullptr;
    VS tx_buf(buffer_size);
    VP tame_buf(buffer_size);

    uint *d_len;

    Map wild_tracks;
    Map tame_tracks;

    GecRng<Rng> rng = make_gec_rng(Rng(uint(seed)));
    bool *d_done = nullptr;

    cudaStream_t s_exec, s_wild, s_tame;
    cudaEvent_t wild_copy, tame_copy;
    _CUDA_CHECK_TO_(cudaStreamCreate(&s_exec), clean_s_exec);
    _CUDA_CHECK_TO_(cudaStreamCreate(&s_wild), clean_s_wild);
    _CUDA_CHECK_TO_(cudaStreamCreate(&s_tame), clean_s_tame);
    _CUDA_CHECK_TO_(cudaEventCreate(&wild_copy), clean_wild_copy);
    _CUDA_CHECK_TO_(cudaEventCreate(&tame_copy), clean_tame_copy);

    _CUDA_CHECK_(cudaMalloc(&d_sl, sizeof(S) * l_len));
    _CUDA_CHECK_(cudaMalloc(&d_pl, sizeof(P) * l_len));
    _CUDA_CHECK_(cudaMalloc(&d_wxs, sizeof(S) * wild_n));
    _CUDA_CHECK_(cudaMalloc(&d_wilds, sizeof(P) * wild_n));
    _CUDA_CHECK_(cudaMalloc(&d_wx_buf, sizeof(S) * buffer_size));
    _CUDA_CHECK_(cudaMalloc(&d_wild_buf, sizeof(P) * buffer_size));
    _CUDA_CHECK_(cudaMalloc(&d_txs, sizeof(S) * tame_n));
    _CUDA_CHECK_(cudaMalloc(&d_tames, sizeof(P) * tame_n));
    _CUDA_CHECK_(cudaMalloc(&d_tx_buf, sizeof(S) * buffer_size));
    _CUDA_CHECK_(cudaMalloc(&d_tame_buf, sizeof(P) * buffer_size));
    _CUDA_CHECK_(cudaMalloc(&d_len, sizeof(uint)));
    _CUDA_CHECK_(cudaMalloc(&d_done, sizeof(bool)));

    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(cd_a<S>, &a, sizeof(S), 0,
                                         cudaMemcpyHostToDevice, s_exec));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(cd_bound<S>, &x, sizeof(S), 0,
                                         cudaMemcpyHostToDevice, s_exec));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(cd_mask<FieldT>, &candidate_mask,
                                         sizeof(FieldT), 0,
                                         cudaMemcpyHostToDevice, s_exec));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(cd_g<P>, &g, sizeof(P), 0,
                                         cudaMemcpyHostToDevice, s_exec));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(cd_h<P>, &h, sizeof(P), 0,
                                         cudaMemcpyHostToDevice, s_exec));

    init_wild_kernel<S, P>
        <<<block_num, thread_num, 0, s_exec>>>(d_wxs, d_wilds, wild_n, tame_n);
    init_tame_kernel<S, P>
        <<<block_num, thread_num, 0, s_exec>>>(d_txs, d_tames, tame_n, wild_n);

    // calculate jump table
    {
        for (size_t i = 0; i < l_len; ++i) {
            sl[i].array()[0] = typename S::LimbT(i);
        }
        for (size_t i = 0; i < l_len; ++i) {
            size_t ri = l_len - 1 - i;
            utils::swap(sl[ri].array()[0], sl[rng.sample(ri)].array()[0]);
        }
        for (size_t i = 0; i < l_len; ++i) {
            size_t e = size_t(sl[i].array()[0]);
            sl[i].array()[0] = typename S::LimbT(step_size);
            sl[i].shift_left(e);
            P::mul(pl[i], sl[i], g, ctx);
        }
    }

    _CUDA_CHECK_(cudaMemcpyAsync(d_sl, sl.data(), sizeof(S) * l_len,
                                 cudaMemcpyHostToDevice, s_wild));
    _CUDA_CHECK_(cudaMemcpyAsync(d_pl, pl.data(), sizeof(P) * l_len,
                                 cudaMemcpyHostToDevice, s_wild));

    _CUDA_CHECK_(cudaDeviceSynchronize());
    _CUDA_CHECK_(cudaGetLastError());

    {
        P *d_p_buf = d_wild_buf, *vd_p_buf = d_tame_buf;
        S *d_x_buf = d_wx_buf, *vd_x_buf = d_tx_buf;
        S *x_buf = wx_buf.data(), *v_x_buf = tx_buf.data();
        P *p_buf = wild_buf.data(), *v_p_buf = tame_buf.data();
        P *ps = d_wilds, *v_ps = d_tames;
        S *xs = d_wxs, *v_xs = d_txs;
        uint n = uint(wild_n), v_n = uint(tame_n);
        Map *tracks = &wild_tracks, *v_tracks = &tame_tracks;
        cudaEvent_t *copy_evt = &wild_copy, *v_copy_evt = &tame_copy;
        cudaStream_t *s_data = &s_wild, *v_s_data = &s_tame;

        for (bool collect = false, wild = true;;) {
            reset_flag_kernel<uint><<<1, 1, 0, s_exec>>>(d_done, d_len);
            searching_kernel<S, P><<<block_num, thread_num, 0, s_exec>>>(
                d_done, d_p_buf, d_x_buf, d_len, buffer_size, ps, xs, d_pl,
                d_sl, l_len, n, check_mask);
            _CUDA_CHECK_(cudaEventRecord(*copy_evt, s_exec));
            _CUDA_CHECK_(cudaStreamWaitEvent(*s_data, *copy_evt));
            _CUDA_CHECK_(cudaMemcpyAsync(p_buf, d_p_buf,
                                         sizeof(P) * buffer_size,
                                         cudaMemcpyDeviceToHost, *s_data));
            _CUDA_CHECK_(cudaMemcpyAsync(x_buf, d_x_buf,
                                         sizeof(S) * buffer_size,
                                         cudaMemcpyDeviceToHost, *s_data));

            utils::swap(d_p_buf, vd_p_buf);
            utils::swap(d_x_buf, vd_x_buf);
            utils::swap(x_buf, v_x_buf);
            utils::swap(p_buf, v_p_buf);
            utils::swap(ps, v_ps);
            utils::swap(xs, v_xs);
            utils::swap(n, v_n);
            utils::swap(tracks, v_tracks);
            utils::swap(copy_evt, v_copy_evt);
            utils::swap(s_data, v_s_data);

            if (collect) {
                _CUDA_CHECK_(cudaStreamSynchronize(*s_data));
                _CUDA_CHECK_(cudaGetLastError());
                for (size_t k = 0; k < buffer_size; ++k) {
                    // P expected;
                    // if (wild) {
                    //     P tmp1;
                    //     P::mul(tmp1, x_buf[k], g, ctx);
                    //     P::add(expected, h, tmp1, ctx);
                    // } else {
                    //     P::mul(expected, x_buf[k], g, ctx);
                    // }
                    // if (!P::on_curve(p_buf[k], ctx)) {
                    //     printf("[host] in %zuth result: \n", k);
                    //     p_buf[k].print();
                    //     printf("not on curve\n");
                    // }
                    // if (!P::eq(expected, p_buf[k])) {
                    //     printf("[host] in %zuth result: \n", k);
                    //     if (wild) {
                    //         h.print();
                    //         printf(" + ");
                    //     }
                    //     printf("[");
                    //     x_buf[k].print();
                    //     printf("]\n");
                    //     g.print();
                    //     printf(" == ");
                    //     expected.print();
                    //     printf(" != ");
                    //     p_buf[k].print();
                    //     printf("\n");
                    // }

                    auto range = v_tracks->equal_range(p_buf[k]);
                    for (auto it = range.first; it != range.second; ++it) {
                        if (it->second == x_buf[k]) {
                            continue;
                        }
                        _CUDA_CHECK_(
                            cudaMemcpyAsync(d_done, &true_literal, sizeof(bool),
                                            cudaMemcpyHostToDevice, *s_data));
                        if (wild) {
                            S::sub(x, it->second, x_buf[k]);
                        } else {
                            S::sub(x, x_buf[k], it->second);
                        }
                        goto done;
                    }
                    tracks->insert(std::make_pair(p_buf[k], x_buf[k]));
                }
                wild = !wild;
#ifdef GEC_DEBUG
                printf("[host] %s track size: %zu\n", wild ? "wild" : "tame",
                       tracks->size());
#endif // GEC_DEBUG
            } else {
                collect = true;
            }
        }
    }

done:
    _CUDA_CHECK_(cudaDeviceSynchronize());
    _CUDA_CHECK_(cudaGetLastError());
clean_up:
    cudaFree(d_sl);
    cudaFree(d_pl);
    cudaFree(d_wxs);
    cudaFree(d_wilds);
    cudaFree(d_wx_buf);
    cudaFree(d_wild_buf);
    cudaFree(d_txs);
    cudaFree(d_tames);
    cudaFree(d_tx_buf);
    cudaFree(d_tame_buf);
    cudaFree(d_len);
    cudaFree(d_done);

    cudaEventDestroy(tame_copy);
clean_tame_copy:
    cudaEventDestroy(wild_copy);
clean_wild_copy:
    cudaStreamDestroy(s_tame);
clean_s_tame:
    cudaStreamDestroy(s_wild);
clean_s_wild:
    cudaStreamDestroy(s_exec);
clean_s_exec:
    return cu_err;

#undef _CUDA_CHECK_
#undef _CUDA_CHECK_TO_
}

#endif // __CUDACC__

} // namespace dlp

} // namespace gec

#endif // GEC_DLP_POLLARD_LAMBDA_HPP
