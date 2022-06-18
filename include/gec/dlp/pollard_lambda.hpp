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
        size_t m = x.most_significant_bit() - 1;
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
    S x, j, one(1);
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
        for (j.set_zero(); j < shared.bound; S::add(j, one)) {
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
        for (j.set_zero(); j < shared.bound; S::add(j, one)) {
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
    size_t m = x.most_significant_bit() - 1;

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

template <typename Rng>
__global__ void init_rng_kernel(GecRng<Rng> *GEC_RSTRCT rng, size_t seed) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    hash::hash_combine(seed, id);
    rng[id] = make_gec_rng(Rng((unsigned int)(seed)));
}
template <typename S, typename Rng>
__global__ void sampling_scaler_kernel(S *GEC_RSTRCT s, const S &GEC_RSTRCT a,
                                       const S &GEC_RSTRCT b,
                                       GecRng<Rng> *GEC_RSTRCT rng) {
    typename S::template Context<> ctx;
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    S tmp;
    auto r = rng[id];
    S::sample_inclusive(tmp, a, b, r, ctx);
    s[id] = tmp;
}
template <typename S, typename P>
__global__ void generate_traps_kernel(
    P *GEC_RSTRCT traps, typename P::Hasher::result_type *GEC_RSTRCT hashes,
    S *GEC_RSTRCT txs, const S *GEC_RSTRCT sl, const P *GEC_RSTRCT pl,
    size_t l_len, const P &GEC_RSTRCT g, const S &GEC_RSTRCT bound) {

    typename P::template Context<> ctx;
    typename P::Hasher hasher;

    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    P p1, p2;
    P *u = &p1, *tmp = &p2;
    S t = txs[id], j, one(1);

    P::mul(*u, t, g, ctx);
    for (j.set_zero(); j < bound; S::add(j, one)) {
        size_t i = u->x().array()[0] % l_len;
        S::add(t, sl[i]);
        P::add(*tmp, *u, pl[i], ctx);
        utils::swap(u, tmp);
#ifdef GEC_DEBUG
        using LimbT = typename P::Field::LimbT;
        if (!(utils::LowerKMask<LimbT, 20>::value & j.array()[0])) {
            printf("[worker %03zu]: calculating trap, step ", id);
            j.println();
        }
#endif // GEC_DEBUG
    }

    txs[id] = t;
    hashes[id] = hasher(*u);
    traps[id] = *u;
}

template <typename S, typename P>
__global__ void
searching_kernel(volatile bool *GEC_RSTRCT done, size_t *GEC_RSTRCT found_id,
                 typename P::Hasher::result_type *GEC_RSTRCT trap_hashes,
                 P *GEC_RSTRCT traps, S *GEC_RSTRCT txs, S *GEC_RSTRCT xs,
                 utils::CHD<> phf, const S *GEC_RSTRCT sl,
                 const P *GEC_RSTRCT pl, size_t l_len, const P &GEC_RSTRCT g,
                 const P &GEC_RSTRCT h, const S &GEC_RSTRCT bound,
                 typename P::Field::LimbT check_mask) {
    typename P::template Context<> ctx;
    typename P::Hasher hasher;

    const size_t thread_n = gridDim.x * blockDim.x;
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    P p1, p2;
    P *u = &p1, *tmp = &p2;

    S x = xs[id], j, one(1);

    P::mul(*tmp, x, g, ctx);
    P::add(*u, h, *tmp, ctx);
    for (j.set_zero(); j < bound; S::add(j, one)) {
        if (!(j.array()[0] & check_mask) && *done)
            return;
        auto hash = hasher(*u);
        auto idx = phf.get(hasher(*u));
        if (trap_hashes[idx] == hash && P::eq(traps[idx], *u)) {
            atomicCAS(found_id, thread_n, id);
            *done = true;
            // FIXME: names of reused variables are confusing
            j = txs[idx];
            S::sub(one, j, x);
            xs[idx] = one;
            return;
        }
        size_t i = u->x().array()[0] % l_len;
        S::add(x, sl[i]);
        P::add(*tmp, *u, pl[i], ctx);
        utils::swap(u, tmp);
    }
}

template <typename S>
static __constant__ S cd_a;
template <typename S>
static __constant__ S cd_b;
template <typename S>
static __constant__ S cd_bound;
template <typename P>
static __constant__ P cd_g;
template <typename P>
static __constant__ P cd_h;

} // namespace _pollard_lambda_

/**
 * @brief CUDA implementation of pollard lambda algorithm
 *
 * Note that this function is not thread safe
 *
 * @tparam S: TODO
 * @tparam P: TODO
 * @tparam Rng: TODO
 * @tparam cuRng: TODO
 * @param x: TODO
 * @param bound: TODO
 * @param lower: TODO
 * @param upper: TODO
 * @param g: TODO
 * @param h: TODO
 * @param seed: TODO
 * @param block_num: TODO
 * @param thread_num: TODO
 */
template <typename S, typename P, typename Rng = std::mt19937,
          typename cuRng = thrust::random::ranlux24>
__host__ cudaError_t cu_pollard_lambda(
    S &GEC_RSTRCT x, const S &GEC_RSTRCT bound, const S &GEC_RSTRCT a,
    const S &GEC_RSTRCT b, const P &GEC_RSTRCT g, const P &GEC_RSTRCT h,
    size_t seed, unsigned int block_num, unsigned int thread_num,
    typename P::Field::LimbT check_mask = typename P::Field::LimbT(0xFF)) {

    using namespace _pollard_lambda_;
    using LimbT = typename P::Field::LimbT;

    cudaError_t cu_err = cudaSuccess;
#define _CUDA_CHECK_TO_(code, label)                                           \
    do {                                                                       \
        cu_err = (code);                                                       \
        if (cu_err != cudaSuccess)                                             \
            goto label;                                                        \
    } while (0)
#define _CUDA_CHECK_(code) _CUDA_CHECK_TO_((code), clean_up)

    const bool false_literal = false;

    const size_t thread_n = block_num * thread_num;

    utils::CHD<> phf(nullptr, thread_n);

    std::vector<size_t> buckets(phf.B);
    size_t *d_buckets = nullptr;

    using HashT = typename P::Hasher::result_type;
    std::vector<HashT> hashes;
    HashT *d_hashes = nullptr;

    const S &d_a = cd_a<S>, &d_b = cd_b<S>, &d_bound = cd_bound<S>;
    const P &d_g = cd_g<P>, &d_h = cd_h<P>;

    S::sub(x, b, a);
    size_t l_len = x.most_significant_bit() - 1;

    std::vector<S> sl(l_len);
    S *d_sl = nullptr;

    std::vector<P> pl(l_len);
    P *d_pl = nullptr;

    typename P::template Context<> ctx;

    GecRng<cuRng> *d_rng;

    std::vector<S> txs(phf.N);
    S *d_txs = nullptr;
    S *d_xs = nullptr;
    std::vector<P> traps(phf.N);
    P *d_traps = nullptr;

    GecRng<Rng> rng = make_gec_rng(Rng((unsigned int)(seed)));

    cudaStream_t s_exec, s_data;

    size_t found_id;
    size_t *d_found_id = nullptr;

    bool *d_done;

    _CUDA_CHECK_TO_(cudaStreamCreate(&s_exec), clean_s_exec);
    _CUDA_CHECK_TO_(cudaStreamCreate(&s_data), clean_s_data);

    _CUDA_CHECK_(cudaMalloc(&d_buckets, sizeof(size_t) * phf.B));
    _CUDA_CHECK_(cudaMalloc(&d_hashes, sizeof(HashT) * phf.N));
    _CUDA_CHECK_(cudaMalloc(&d_sl, sizeof(S) * l_len));
    _CUDA_CHECK_(cudaMalloc(&d_pl, sizeof(P) * l_len));
    _CUDA_CHECK_(cudaMalloc(&d_rng, sizeof(GecRng<cuRng>) * thread_n));
    _CUDA_CHECK_(cudaMalloc(&d_xs, sizeof(S) * thread_n));
    _CUDA_CHECK_(cudaMalloc(&d_txs, sizeof(S) * phf.N));
    _CUDA_CHECK_(cudaMalloc(&d_traps, sizeof(P) * phf.N));
    _CUDA_CHECK_(cudaMalloc(&d_found_id, sizeof(size_t)));
    _CUDA_CHECK_(cudaMalloc(&d_done, sizeof(bool)));

    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(d_a, &a, sizeof(S), 0,
                                         cudaMemcpyHostToDevice, s_exec));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(d_b, &b, sizeof(S), 0,
                                         cudaMemcpyHostToDevice, s_exec));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(d_bound, &bound, sizeof(S), 0,
                                         cudaMemcpyHostToDevice, s_data));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(d_g, &g, sizeof(P), 0,
                                         cudaMemcpyHostToDevice, s_data));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(d_h, &h, sizeof(P), 0,
                                         cudaMemcpyHostToDevice, s_data));

    init_rng_kernel<cuRng><<<block_num, thread_num, 0, s_exec>>>(d_rng, seed);

    for (;;) {
        sampling_scaler_kernel<S, cuRng>
            <<<block_num, thread_num, 0, s_exec>>>(d_txs, d_a, d_b, d_rng);
        sampling_scaler_kernel<S, cuRng>
            <<<block_num, thread_num, 0, s_exec>>>(d_xs, d_a, d_b, d_rng);

        // calculate jump table
        for (size_t i = 0; i < l_len; ++i) {
            sl[i].array()[0] = typename S::LimbT(i);
        }
        for (size_t i = 0; i < l_len; ++i) {
            size_t ri = l_len - 1 - i;
            utils::swap(sl[ri].array()[0], sl[rng.sample(ri)].array()[0]);
        }
        for (size_t i = 0; i < l_len; ++i) {
            LimbT e = sl[i].array()[0];
            sl[i].set_pow2(e);
            P::mul(pl[i], sl[i], g, ctx);
        }

        _CUDA_CHECK_(cudaMemcpyAsync(d_sl, sl.data(), sizeof(S) * l_len,
                                     cudaMemcpyHostToDevice, s_data));
        _CUDA_CHECK_(cudaMemcpyAsync(d_pl, pl.data(), sizeof(P) * l_len,
                                     cudaMemcpyHostToDevice, s_data));

        _CUDA_CHECK_(cudaDeviceSynchronize());
        _CUDA_CHECK_(cudaGetLastError());

        // setting traps
        generate_traps_kernel<S, P><<<block_num, thread_num, 0, s_exec>>>(
            d_traps, d_hashes, d_txs, d_sl, d_pl, l_len, d_g, d_bound);
        _CUDA_CHECK_(cudaMemcpyAsync(txs.data(), d_txs, sizeof(S) * thread_n,
                                     cudaMemcpyDeviceToHost, s_exec));
        _CUDA_CHECK_(cudaMemcpyAsync(traps.data(), d_traps,
                                     sizeof(P) * thread_n,
                                     cudaMemcpyDeviceToHost, s_exec));
        _CUDA_CHECK_(cudaMemcpyAsync(hashes.data(), d_hashes,
                                     sizeof(HashT) * thread_n,
                                     cudaMemcpyDeviceToHost, s_exec));
        _CUDA_CHECK_(cudaDeviceSynchronize());
        _CUDA_CHECK_(cudaGetLastError());

        // construct static maps
        phf.buckets = buckets.data();
        size_t placeholder = phf.fill_placeholder(hashes.data());
#ifdef GEC_DEBUG
        auto duplicates =
#endif // GEC_DEBUG
            phf.build(hashes.data());
#ifdef GEC_DEBUG
        for (auto &dup : duplicates) {
            printf("[host]: find hash collision: \n");
            traps[dup.first].println();
            traps[dup.second].println();
            printf("some traps are omitted");
        }
#endif // GEC_DEBUG
        phf.rearrange(hashes.data(), placeholder, txs.data(), traps.data());

        // start searching
        phf.buckets = d_buckets;
        cudaMemcpyAsync(d_done, &false_literal, sizeof(bool),
                        cudaMemcpyHostToDevice, s_exec);
        _CUDA_CHECK_(cudaMemcpyAsync(d_buckets, buckets.data(),
                                     sizeof(size_t) * phf.B,
                                     cudaMemcpyHostToDevice, s_exec));
        _CUDA_CHECK_(cudaMemcpyAsync(d_txs, txs.data(), sizeof(S) * phf.N,
                                     cudaMemcpyHostToDevice, s_exec));
        _CUDA_CHECK_(cudaMemcpyAsync(d_traps, traps.data(), sizeof(P) * phf.N,
                                     cudaMemcpyHostToDevice, s_exec));
        _CUDA_CHECK_(cudaMemcpyAsync(d_hashes, hashes.data(),
                                     sizeof(HashT) * phf.N,
                                     cudaMemcpyHostToDevice, s_exec));
        searching_kernel<S, P><<<block_num, thread_num, 0, s_exec>>>(
            d_done, d_found_id, d_hashes, d_traps, d_txs, d_xs, phf, d_sl, d_pl,
            l_len, d_g, d_h, d_bound, check_mask);
        _CUDA_CHECK_(cudaDeviceSynchronize());

        switch (cudaGetLastError()) {
        case cudaErrorLaunchFailure: // success
            cudaMemcpy(&found_id, d_found_id, sizeof(size_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(&x, d_xs + found_id, sizeof(S), cudaMemcpyDeviceToHost);
            goto clean_up;
        case cudaSuccess: // failed
            break;
        default: // error
            goto clean_up;
        }
    }

clean_up:
    cudaFree(d_buckets);
    cudaFree(d_hashes);
    cudaFree(d_sl);
    cudaFree(d_pl);
    cudaFree(d_rng);
    cudaFree(d_xs);
    cudaFree(d_txs);
    cudaFree(d_traps);
    cudaFree(d_found_id);
    cudaFree(d_done);

    cudaStreamDestroy(s_data);
clean_s_data:
    cudaStreamDestroy(s_exec);
clean_s_exec:
    return cu_err;
#undef CUDA_CHECK
}

#endif // __CUDACC__

} // namespace dlp

} // namespace gec

#endif // GEC_DLP_POLLARD_LAMBDA_HPP