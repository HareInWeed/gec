#pragma once
#ifndef GEC_DLP_POLLARD_RHO_HPP
#define GEC_DLP_POLLARD_RHO_HPP

#include <gec/bigint/mixin/random.hpp>
#include <gec/utils/basic.hpp>
#include <gec/utils/misc.hpp>

#include <random>

#ifdef GEC_ENABLE_PTHREADS
#include <gec/utils/sequence.hpp>

#include <pthread.h>

#include <unordered_map>
#include <vector>

#endif // GEC_ENABLE_PTHREADS

#ifdef __CUDACC__
#include <gec/utils/hash.hpp>
#endif // __CUDACC__

namespace gec {

namespace dlp {

template <typename S, typename P, typename Ctx, typename Rng>
__host__ __device__ void pollard_rho(S &c,
                                     S &d2, // TODO: require general int inv
                                     size_t l, S *al, S *bl, P *pl, const P &g,
                                     const P &h, GecRng<Rng> &rng, Ctx &ctx) {
    auto &ctx_view = ctx.template view_as<P, P, P, P, P, S, S>();

    auto &ag = ctx_view.template get<0>();
    auto &bh = ctx_view.template get<1>();
    auto &temp = ctx_view.template get<2>();
    auto &x1 = ctx_view.template get<3>();
    auto &x2 = ctx_view.template get<4>();

    auto &c2 = ctx_view.template get<5>();
    auto &d = ctx_view.template get<6>();
    auto &rest_ctx = ctx_view.rest();
    P *tmp = &temp, *x = &x1;

    do {
        for (size_t k = 0; k < l; ++k) {
            S::sample(al[k], rng);
            S::sample(bl[k], rng);
            P::mul(ag, al[k], g, rest_ctx);
            P::mul(bh, bl[k], h, rest_ctx);
            P::add(pl[k], ag, bh, rest_ctx);
        }

        S::sample(c, rng);
        S::sample(d, rng);
        P::mul(ag, c, g, rest_ctx);
        P::mul(bh, d, h, rest_ctx);
        P::add(*x, ag, bh, rest_ctx);
        c2 = c;
        d2 = d;
        x2 = *x;
        size_t i;

        do {
            i = x->x().array()[0] % l;
            S::add(c, al[i]);
            S::add(d, bl[i]);
            P::add(*tmp, *x, pl[i], rest_ctx);
            utils::swap(x, tmp);

            i = x2.x().array()[0] % l;
            S::add(c2, al[i]);
            S::add(d2, bl[i]);
            P::add(*tmp, x2, pl[i], rest_ctx);

            i = tmp->x().array()[0] % l;
            S::add(c2, al[i]);
            S::add(d2, bl[i]);
            P::add(x2, *tmp, pl[i], rest_ctx);
        } while (!P::eq(*x, x2, rest_ctx));

    } while (d == d2);

    S::sub(c, c2);
    S::sub(d2, d);
}

#ifdef GEC_ENABLE_PTHREADS

namespace _pollard_rho_ {

template <typename T>
struct MaskZero {
    __host__ __device__ GEC_INLINE static bool call(const T &a, const T &b) {
        return !(a & b);
    }
};

template <typename S>
struct Coefficient {
    S x;
    S y;
};

template <typename S, typename P>
struct SharedData {
    std::vector<S> al;
    std::vector<S> bl;
    std::vector<P> pl;
    std::unordered_multimap<P, Coefficient<S>, typename P::Hasher> candidates;
    pthread_mutex_t candidates_mutex;
    const typename P::Field &mask;
    const P &g;
    const P &h;
    S &c;
    S &d2;
    size_t workers;
    bool done;

    SharedData(size_t l, size_t workers, const typename P::Field &mask,
               const P &g, const P &h, S &c, S &d2)
        : al(l), bl(l), pl(l), candidates(),
          candidates_mutex(PTHREAD_MUTEX_INITIALIZER), mask(mask), g(g), h(h),
          c(c), d2(d2), workers(workers), done(false) {}
};

template <typename S, typename P>
struct WorkerData {
    SharedData<S, P> &shared;
    unsigned int seed;
    size_t id;
};

template <typename S, typename P, typename Rng>
void *worker(void *data_ptr) {
    using F = typename P::Field;
    using LT = typename F::LimbT;

    WorkerData<S, P> &local = *static_cast<WorkerData<S, P> *>(data_ptr);
    SharedData<S, P> &shared = local.shared;
    volatile bool &done = shared.done;
    auto rng = make_gec_rng(Rng(local.seed));
    typename P::template Context<> ctx;
    Coefficient<S> coeff;
    auto &ctx_view = ctx.template view_as<P, P, P, P>();
    auto &p1 = ctx_view.template get<0>();
    auto &p2 = ctx_view.template get<1>();
    auto &xg = ctx_view.template get<2>();
    auto &yh = ctx_view.template get<3>();
    P *p = &p1, *tmp = &p2;

    S::sample(coeff.x, rng);
    S::sample(coeff.y, rng);
    P::mul(xg, coeff.x, shared.g, ctx_view.rest());
    P::mul(yh, coeff.y, shared.h, ctx_view.rest());
    P::add(*p, xg, yh, ctx_view.rest());

    size_t l = shared.pl.size();
    int i;
    while (true) {
        if (done) {
            return nullptr;
        }
        if (utils::VtSeqAll<F::LimbN, LT, MaskZero<LT>>::call(
                p->x().array(), shared.mask.array())) {
            pthread_mutex_lock(&shared.candidates_mutex);
            if (done) {
                pthread_mutex_unlock(&shared.candidates_mutex);
                return nullptr;
            }
            auto range = shared.candidates.equal_range(*p);
            for (auto it = range.first; it != range.second; ++it) {
                const auto &p0 = it->first;
                const auto &coeff0 = it->second;
                if (P::eq(p0, *p) && coeff0.y != coeff.y) {
                    S::sub(shared.c, coeff0.x, coeff.x);
                    S::sub(shared.d2, coeff.y, coeff0.y);
                    done = true;
                    pthread_mutex_unlock(&shared.candidates_mutex);
                    return nullptr;
                }
            }
            shared.candidates.insert(std::make_pair(*p, coeff));
            pthread_mutex_unlock(&shared.candidates_mutex);
        }
        i = p->x().array()[0] % l;
        S::add(coeff.x, shared.al[i]);
        S::add(coeff.y, shared.bl[i]);
        P::add(*tmp, *p, shared.pl[i], ctx.template view_as<P, P>().rest());
        utils::swap(p, tmp);
    }
}

} // namespace _pollard_rho_

template <typename S, typename P, typename Rng>
void multithread_pollard_rho(S &c,
                             S &d2, // TODO: require general int inv
                             size_t l, size_t worker_n,
                             const typename P::Field &mask, const P &g,
                             const P &h, GecRng<Rng> &rng) {
    using namespace _pollard_rho_;
    using Shared = SharedData<S, P>;
    using Local = WorkerData<S, P>;

    Shared shared(l, worker_n, mask, g, h, c, d2);

    typename P::template Context<> ctx;

    P ag, bh;
    for (size_t k = 0; k < l; ++k) {
        S::sample(shared.al[k], rng);
        S::sample(shared.bl[k], rng);
        P::mul(ag, shared.al[k], g, ctx);
        P::mul(bh, shared.bl[k], h, ctx);
        P::add(shared.pl[k], ag, bh, ctx);
    }

    std::vector<pthread_t> workers(worker_n);
    std::vector<Local> params(worker_n, {shared, 0, 0});

    for (size_t k = 0; k < worker_n; ++k) {
        auto &data = params[k];
        data.id = k;
        data.seed = rng.template sample<unsigned int>();
        pthread_create(&workers[k], nullptr, worker<S, P, Rng>,
                       static_cast<void *>(&data));
    }

    for (size_t k = 0; k < worker_n; ++k) {
        pthread_join(workers[k], nullptr);
    }
}

#endif // GEC_ENABLE_PTHREADS

#ifdef __CUDACC__

namespace _pollard_rho_ {

template <typename P>
static __constant__ P cd_g;
template <typename P>
static __constant__ P cd_h;
template <typename F>
static __constant__ F cd_mask;

template <typename Rng>
__global__ void init_rng_kernel(GecRng<Rng> *GEC_RSTRCT rng, size_t seed) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    hash::hash_combine(seed, id);
    rng[id] = make_gec_rng(Rng((unsigned int)(seed)));
}
template <typename S, typename Rng>
__global__ void sampling_scaler_kernel(S *GEC_RSTRCT s,
                                       GecRng<Rng> *GEC_RSTRCT rng) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    S tmp;
    auto r = rng[id];
    S::sample(tmp, r);
    s[id] = tmp;
}
template <typename S, typename P>
__global__ void init_ps_kernel(P *GEC_RSTRCT init_ps,
                               const S *GEC_RSTRCT init_xs,
                               const S *GEC_RSTRCT init_ys,
                               const P &GEC_RSTRCT g, const P &GEC_RSTRCT h) {

    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    typename P::template Context<> ctx;
    auto ctx_view = ctx.template view_as<P, P, P, S, S>();
    auto &p = ctx_view.template get<0>();
    auto &xg = ctx_view.template get<1>();
    auto &yh = ctx_view.template get<2>();
    auto &x = ctx_view.template get<3>();
    auto &y = ctx_view.template get<4>();
    auto &rest_ctx = ctx_view.rest();

    x = init_xs[id];
    y = init_ys[id];
    p = init_ps[id];

    P::mul(xg, x, g, rest_ctx);
    P::mul(yh, y, h, rest_ctx);
    P::add(p, xg, yh, rest_ctx);

    init_ps[id] = p;
}
template <typename S, typename P>
__global__ void searching_kernel(volatile bool *GEC_RSTRCT done,
                                 P *GEC_RSTRCT candidate, S *GEC_RSTRCT xs,
                                 S *GEC_RSTRCT ys, unsigned int *GEC_RSTRCT len,
                                 unsigned int max_len, S *GEC_RSTRCT init_xs,
                                 S *GEC_RSTRCT init_ys, P *GEC_RSTRCT init_ps,
                                 const S *GEC_RSTRCT al, const S *GEC_RSTRCT bl,
                                 const P *GEC_RSTRCT pl, size_t l,
                                 const typename P::Field &GEC_RSTRCT mask,
                                 const unsigned int check_mask) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    using F = typename P::Field;
    using LT = typename F::LimbT;

    typename P::template Context<> ctx;
    auto &ctx_view = ctx.template view_as<P, P>();
    auto &p1 = ctx_view.template get<0>();
    auto &p2 = ctx_view.template get<1>();
    auto &rest_ctx = ctx_view.rest();
    P *p = &p1, *tmp = &p2;

    S x = init_xs[id], y = init_ys[id];
    *p = init_ps[id];

    int i;
    for (unsigned int k = 0;; ++k) {
        if (!(k & check_mask) && *done)
            goto shutdown;
        if (utils::VtSeqAll<F::LimbN, LT, MaskZero<LT>>::call(p->x().array(),
                                                              mask.array())) {
            size_t idx = atomicAdd(len, 1);
            if (idx >= max_len) {
                *done = true;
                candidate[idx] = *p;
                xs[idx] = x;
                ys[idx] = y;
                goto shutdown;
            }
        }
        i = p->x().array()[0] % l;
        S::add(x, al[i]);
        S::add(y, bl[i]);
        P::add(*tmp, *p, pl[i], rest_ctx);
        utils::swap(p, tmp);
    }

shutdown:
    init_xs[id] = x;
    init_ys[id] = y;
    init_ps[id] = *p;
    return;
}

} // namespace _pollard_rho_

template <typename S, typename P, typename Rng = std::mt19937,
          typename cuRng = thrust::random::ranlux24>
cudaError_t
cu_pollard_rho(S &c,
               S &d2, // TODO: require general int inv
               size_t l, const typename P::Field &mask, const P &g, const P &h,
               size_t seed, unsigned int block_num, unsigned int thread_num,
               unsigned int buffer_size = 0, unsigned int check_mask = 0xFF) {

    using namespace _pollard_rho_;
    using F = typename P::Field;

    cudaError_t cu_err = cudaSuccess;
#define _CUDA_CHECK_TO_(code, label)                                           \
    do {                                                                       \
        cu_err = (code);                                                       \
        if (cu_err != cudaSuccess)                                             \
            goto label;                                                        \
    } while (0)
#define _CUDA_CHECK_(code) _CUDA_CHECK_TO_((code), clean_up)

    const bool false_literal = false;
    const unsigned int zero_literal = 0;

    const P &d_g = cd_g<P>, &d_h = cd_h<P>;
    const F &d_mask = cd_mask<F>;

    const size_t thread_n = block_num * thread_num;

    GecRng<Rng> rng = make_gec_rng(Rng((unsigned int)(seed)));

    std::vector<S> al(l);
    S *d_al = nullptr;
    std::vector<S> bl(l);
    S *d_bl = nullptr;
    std::vector<P> pl(l);
    P *d_pl = nullptr;

    typename P::template Context<> ctx;

    GecRng<cuRng> *d_rng = nullptr;
    S *d_init_xs = nullptr, *d_init_ys = nullptr;
    P *d_init_ps = nullptr;

    const unsigned int buf_size = buffer_size ? buffer_size : thread_n;
    std::unordered_multimap<P, Coefficient<S>, typename P::Hasher>
        candidates_map;
    std::vector<P> candidates(buf_size);
    std::vector<S> xs(buf_size);
    std::vector<S> ys(buf_size);
    unsigned int *d_buf_cursor;
    P *d_candidate = nullptr;
    S *d_xs = nullptr, *d_ys = nullptr;

    P ag, bh;

    bool *d_done = nullptr;

    cudaStream_t stream1, stream2, stream3;
    _CUDA_CHECK_TO_(cudaStreamCreate(&stream1), clean_stream1);
    _CUDA_CHECK_TO_(cudaStreamCreate(&stream2), clean_stream2);
    _CUDA_CHECK_TO_(cudaStreamCreate(&stream3), clean_stream3);

    _CUDA_CHECK_(cudaMalloc(&d_al, sizeof(S) * l));
    _CUDA_CHECK_(cudaMalloc(&d_bl, sizeof(S) * l));
    _CUDA_CHECK_(cudaMalloc(&d_pl, sizeof(P) * l));
    _CUDA_CHECK_(cudaMalloc(&d_init_xs, sizeof(S) * thread_n));
    _CUDA_CHECK_(cudaMalloc(&d_init_ys, sizeof(S) * thread_n));
    _CUDA_CHECK_(cudaMalloc(&d_init_ps, sizeof(P) * thread_n));
    _CUDA_CHECK_(cudaMalloc(&d_candidate, sizeof(P) * buf_size));
    _CUDA_CHECK_(cudaMalloc(&d_xs, sizeof(S) * buf_size));
    _CUDA_CHECK_(cudaMalloc(&d_ys, sizeof(S) * buf_size));
    _CUDA_CHECK_(cudaMalloc(&d_buf_cursor, sizeof(unsigned int)));
    _CUDA_CHECK_(cudaMalloc(&d_done, sizeof(bool)));

    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(d_g, &g, sizeof(P), 0,
                                         cudaMemcpyHostToDevice, stream1));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(d_h, &h, sizeof(P), 0,
                                         cudaMemcpyHostToDevice, stream1));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(d_mask, &mask, sizeof(F), 0,
                                         cudaMemcpyHostToDevice, stream1));

    init_rng_kernel<cuRng><<<block_num, thread_num, 0, stream2>>>(d_rng, seed);
    sampling_scaler_kernel<S, cuRng>
        <<<block_num, thread_num, 0, stream2>>>(d_init_xs, d_rng);
    sampling_scaler_kernel<S, cuRng>
        <<<block_num, thread_num, 0, stream2>>>(d_init_ys, d_rng);

    init_ps_kernel<S, P>
        <<<block_num, thread_num>>>(d_init_ps, d_init_xs, d_init_ys, g, h);

    for (size_t k = 0; k < l; ++k) {
        S::sample(al[k], rng);
        S::sample(bl[k], rng);
        P::mul(ag, al[k], g, ctx);
        P::mul(bh, bl[k], h, ctx);
        P::add(pl[k], ag, bh, ctx);
    }

    _CUDA_CHECK_(cudaDeviceSynchronize());
    _CUDA_CHECK_(cudaGetLastError());

    for (;;) {
        cudaMemcpyAsync(d_done, &false_literal, sizeof(bool),
                        cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_buf_cursor, &zero_literal, sizeof(unsigned int),
                        cudaMemcpyHostToDevice, stream1);
        searching_kernel<S, P><<<block_num, thread_num, 0, stream1>>>(
            d_done, d_candidate, d_xs, d_ys, d_buf_cursor, buf_size, d_init_xs,
            d_init_ys, d_init_ps, d_al, d_bl, d_pl, l, d_mask, check_mask);
        cudaMemcpyAsync(candidates.data(), d_candidate, sizeof(P) * buf_size,
                        cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(xs.data(), d_xs, sizeof(S) * buf_size,
                        cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(ys.data(), d_ys, sizeof(S) * buf_size,
                        cudaMemcpyDeviceToHost, stream1);
        _CUDA_CHECK_(cudaDeviceSynchronize());
        _CUDA_CHECK_(cudaGetLastError());
        for (unsigned int k = 0; k < buf_size; ++k) {
            auto &p = candidates[k];
            auto &x = xs[k];
            auto &y = ys[k];
            auto range = candidates_map.equal_range(p);
            for (auto it = range.first; it != range.second; ++it) {
                const auto &p0 = it->first;
                const auto &coeff0 = it->second;
                if (P::eq(p0, p) && coeff0.y != y) {
                    S::sub(c, coeff0.x, x);
                    S::sub(d2, y, coeff0.y);
                    goto clean_up;
                }
            }
            candidates_map.insert(std::make_pair(p, Coefficient<S>{x, y}));
        }
    }

clean_up:
    cudaFree(d_al);
    cudaFree(d_bl);
    cudaFree(d_pl);
    cudaFree(d_init_xs);
    cudaFree(d_init_ys);
    cudaFree(d_init_ps);
    cudaFree(d_candidate);
    cudaFree(d_xs);
    cudaFree(d_ys);
    cudaFree(d_buf_cursor);
    cudaFree(d_done);

    cudaStreamDestroy(stream3);
clean_stream3:
    cudaStreamDestroy(stream2);
clean_stream2:
    cudaStreamDestroy(stream1);
clean_stream1:
    return cu_err;

#undef _CUDA_CHECK_
#undef _CUDA_CHECK_TO_
}

#endif // __CUDACC__

} // namespace dlp

} // namespace gec

#endif // !GEC_DLP_POLLARD_RHO_HPP