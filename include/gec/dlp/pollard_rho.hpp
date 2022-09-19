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

template <typename S, typename P, typename Rng>
GEC_HD void pollard_rho(S &c,
                        S &d2, // TODO: require general int inv
                        size_t l, S *al, S *bl, P *pl, const P &g, const P &h,
                        GecRng<Rng> &rng) {
    P ag, bh, temp, x1, x2;

    S c2, d;
    P *tmp = &temp, *x = &x1;

    do {
        for (size_t k = 0; k < l; ++k) {
            S::sample(al[k], rng);
            S::sample(bl[k], rng);
            P::mul(ag, al[k], g);
            P::mul(bh, bl[k], h);
            P::add(pl[k], ag, bh);
        }

        S::sample(c, rng);
        S::sample(d, rng);
        P::mul(ag, c, g);
        P::mul(bh, d, h);
        P::add(*x, ag, bh);
        c2 = c;
        d2 = d;
        x2 = *x;
        size_t i;

        do {
            i = x->x().array()[0] % l;
            S::add(c, al[i]);
            S::add(d, bl[i]);
            P::add(*tmp, *x, pl[i]);
            utils::swap(x, tmp);

            i = x2.x().array()[0] % l;
            S::add(c2, al[i]);
            S::add(d2, bl[i]);
            P::add(*tmp, x2, pl[i]);

            i = tmp->x().array()[0] % l;
            S::add(c2, al[i]);
            S::add(d2, bl[i]);
            P::add(x2, *tmp, pl[i]);
        } while (!P::eq(*x, x2));

    } while (d == d2);

    S::sub(c, c2);
    S::sub(d2, d);
}

namespace _pollard_rho_ {

template <typename T>
struct MaskZero {
    GEC_HD GEC_INLINE static bool call(const T &a, const T &b) {
        return !(a & b);
    }
};

template <typename S>
struct Coefficient {
    S x;
    S y;
};

} // namespace _pollard_rho_

#ifdef GEC_ENABLE_PTHREADS

namespace _pollard_rho_ {

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
    Coefficient<S> coeff;

    P p1, p2, xg, yh;
    P *p = &p1, *tmp = &p2;

    S::sample(coeff.x, rng);
    S::sample(coeff.y, rng);
    P::mul(xg, coeff.x, shared.g);
    P::mul(yh, coeff.y, shared.h);
    P::add(*p, xg, yh);

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
        P::add(*tmp, *p, shared.pl[i]);
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

    P ag, bh;
    for (size_t k = 0; k < l; ++k) {
        S::sample(shared.al[k], rng);
        S::sample(shared.bl[k], rng);
        P::mul(ag, shared.al[k], g);
        P::mul(bh, shared.bl[k], h);
        P::add(shared.pl[k], ag, bh);
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
__global__ void sampling_scalar_kernel(S *GEC_RSTRCT s,
                                       GecRng<Rng> *GEC_RSTRCT rng) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    S tmp;
    auto r = rng[id];
    S::sample(tmp, r);
    s[id] = tmp;
    rng[id] = r;
}
template <typename S, typename P>
__global__ void init_ps_kernel(P *GEC_RSTRCT init_ps,
                               const S *GEC_RSTRCT init_xs,
                               const S *GEC_RSTRCT init_ys) {

    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    S x(init_xs[id]), y(init_ys[id]);

    P p, xg, yh;
    P::mul(xg, x, cd_g<P>);
    P::mul(yh, y, cd_h<P>);
    P::add(p, xg, yh);

    init_ps[id] = p;
}

template <typename SizeT>
__global__ static void reset_flags_kernel(bool *done, SizeT *buf_cursor) {
    *done = false;
    *buf_cursor = SizeT(0);
}
template <typename S, typename P>
__global__ void searching_kernel(volatile bool *GEC_RSTRCT done,
                                 P *GEC_RSTRCT candidate, S *GEC_RSTRCT xs,
                                 S *GEC_RSTRCT ys, unsigned int *GEC_RSTRCT len,
                                 unsigned int max_len, S *GEC_RSTRCT init_xs,
                                 S *GEC_RSTRCT init_ys, P *GEC_RSTRCT init_ps,
                                 const S *GEC_RSTRCT al, const S *GEC_RSTRCT bl,
                                 const P *GEC_RSTRCT pl, size_t l,
                                 const unsigned int check_mask) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    using F = typename P::Field;
    using LT = typename F::LimbT;

    S x(init_xs[id]), y(init_ys[id]);
    P p1(init_ps[id]), p2;

    bool in_p1 = true;
#define GEC_DEST_ (in_p1 ? p2 : p1)
#define GEC_SRC_ (in_p1 ? p1 : p2)
#define GEC_RELOAD_ (in_p1 = !in_p1)

    int i;
    for (unsigned int k = 0;; ++k) {
        if (!(k & check_mask) && *done)
            goto shutdown;
        if (utils::VtSeqAll<F::LimbN, LT, MaskZero<LT>>::call(
                GEC_SRC_.x().array(), cd_mask<F>.array())) {
            size_t idx = atomicAdd(len, 1);
            if (idx < max_len) {
                candidate[idx] = GEC_SRC_;
                xs[idx] = x;
                ys[idx] = y;
            } else {
                *done = true;
                goto shutdown;
            }
        }
        i = GEC_SRC_.x().array()[0] % l;
        S::add(x, al[i]);
        S::add(y, bl[i]);
        P::add(GEC_DEST_, GEC_SRC_, pl[i]);
        GEC_RELOAD_;
    }

shutdown:
    init_xs[id] = x;
    init_ys[id] = y;
    init_ps[id] = GEC_SRC_;
    return;

#undef GEC_DEST_
#undef GEC_SRC_
#undef GEC_RELOAD_
}

} // namespace _pollard_rho_

template <typename S, typename P, typename Rng = std::mt19937,
          typename cuRng = thrust::random::ranlux24>
cudaError_t cu_pollard_rho(S &c,
                           S &d2, // TODO: require general int inv
                           size_t l, const typename P::Field &mask, const P &g,
                           const P &h, size_t seed, unsigned int block_num,
                           unsigned int thread_num,
                           unsigned int buffer_size = 0x1000,
                           unsigned int check_mask = 0xFF) {

    using namespace _pollard_rho_;
    using uint = unsigned int;
    using F = typename P::Field;

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

    GecRng<Rng> rng = make_gec_rng(Rng((uint)(seed)));

    std::vector<S> al(l);
    S *d_al = nullptr;
    std::vector<S> bl(l);
    S *d_bl = nullptr;
    std::vector<P> pl(l);
    P *d_pl = nullptr;

    GecRng<cuRng> *d_rng = nullptr;
    S *d_init_xs = nullptr, *d_init_ys = nullptr;
    P *d_init_ps = nullptr;

    std::unordered_multimap<P, Coefficient<S>, typename P::Hasher>
        candidates_map;
    std::vector<P> candidates(buffer_size);
    std::vector<S> xs(buffer_size);
    std::vector<S> ys(buffer_size);
    uint *d_buf_cursor;
    P *d_candidate = nullptr, *d_candidate1 = nullptr, *d_candidate2 = nullptr;
    S *d_xs = nullptr, *d_xs1 = nullptr, *d_xs2 = nullptr;
    S *d_ys = nullptr, *d_ys1 = nullptr, *d_ys2 = nullptr;

    P ag, bh;

    bool *d_done = nullptr;

    cudaStream_t *stream = nullptr, stream1, stream2;
    cudaEvent_t *done_evt = nullptr, done_evt1, done_evt2;
    _CUDA_CHECK_TO_(cudaStreamCreate(&stream1), clean_stream1);
    _CUDA_CHECK_TO_(cudaStreamCreate(&stream2), clean_stream2);
    _CUDA_CHECK_TO_(cudaEventCreate(&done_evt1), clean_done_evt1);
    _CUDA_CHECK_TO_(cudaEventCreate(&done_evt2), clean_done_evt2);

    _CUDA_CHECK_(cudaMalloc(&d_al, sizeof(S) * l));
    _CUDA_CHECK_(cudaMalloc(&d_bl, sizeof(S) * l));
    _CUDA_CHECK_(cudaMalloc(&d_pl, sizeof(P) * l));
    _CUDA_CHECK_(cudaMalloc(&d_rng, sizeof(GecRng<cuRng>) * thread_n));
    _CUDA_CHECK_(cudaMalloc(&d_init_xs, sizeof(S) * thread_n));
    _CUDA_CHECK_(cudaMalloc(&d_init_ys, sizeof(S) * thread_n));
    _CUDA_CHECK_(cudaMalloc(&d_init_ps, sizeof(P) * thread_n));
    _CUDA_CHECK_(cudaMalloc(&d_candidate1, sizeof(P) * buffer_size));
    _CUDA_CHECK_(cudaMalloc(&d_candidate2, sizeof(P) * buffer_size));
    _CUDA_CHECK_(cudaMalloc(&d_xs1, sizeof(S) * buffer_size));
    _CUDA_CHECK_(cudaMalloc(&d_xs2, sizeof(S) * buffer_size));
    _CUDA_CHECK_(cudaMalloc(&d_ys1, sizeof(S) * buffer_size));
    _CUDA_CHECK_(cudaMalloc(&d_ys2, sizeof(S) * buffer_size));
    _CUDA_CHECK_(cudaMalloc(&d_buf_cursor, sizeof(uint)));
    _CUDA_CHECK_(cudaMalloc(&d_done, sizeof(bool)));

    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(cd_g<P>, &g, sizeof(P), 0,
                                         cudaMemcpyHostToDevice, stream1));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(cd_h<P>, &h, sizeof(P), 0,
                                         cudaMemcpyHostToDevice, stream1));
    _CUDA_CHECK_(cudaMemcpyToSymbolAsync(cd_mask<F>, &mask, sizeof(F), 0,
                                         cudaMemcpyHostToDevice, stream1));

    init_rng_kernel<cuRng><<<block_num, thread_num, 0, stream2>>>(d_rng, seed);
    sampling_scalar_kernel<S, cuRng>
        <<<block_num, thread_num, 0, stream2>>>(d_init_xs, d_rng);
    sampling_scalar_kernel<S, cuRng>
        <<<block_num, thread_num, 0, stream2>>>(d_init_ys, d_rng);

    init_ps_kernel<S, P>
        <<<block_num, thread_num>>>(d_init_ps, d_init_xs, d_init_ys);

    for (size_t k = 0; k < l; ++k) {
        S::sample(al[k], rng);
        S::sample(bl[k], rng);
        P::mul(ag, al[k], g);
        P::mul(bh, bl[k], h);
        P::add(pl[k], ag, bh);
    }

    _CUDA_CHECK_(cudaMemcpyAsync(d_al, al.data(), sizeof(S) * l,
                                 cudaMemcpyHostToDevice, stream1));
    _CUDA_CHECK_(cudaMemcpyAsync(d_bl, bl.data(), sizeof(S) * l,
                                 cudaMemcpyHostToDevice, stream1));
    _CUDA_CHECK_(cudaMemcpyAsync(d_pl, pl.data(), sizeof(P) * l,
                                 cudaMemcpyHostToDevice, stream1));

    _CUDA_CHECK_(cudaDeviceSynchronize());
    _CUDA_CHECK_(cudaGetLastError());

    d_candidate = d_candidate1;
    d_xs = d_xs1;
    d_ys = d_ys1;
    stream = &stream1;
    done_evt = &done_evt1;
    for (bool round = true, first = true;; round = !round) {
        if (!first) {
            _CUDA_CHECK_(
                cudaStreamWaitEvent(*stream, round ? done_evt2 : done_evt1));
        }
        reset_flags_kernel<<<1, 1, 0, *stream>>>(d_done, d_buf_cursor);
        searching_kernel<S, P><<<block_num, thread_num, 0, *stream>>>(
            d_done, d_candidate, d_xs, d_ys, d_buf_cursor, buffer_size,
            d_init_xs, d_init_ys, d_init_ps, d_al, d_bl, d_pl, l, check_mask);
        _CUDA_CHECK_(cudaEventRecord(*done_evt, *stream));

        d_candidate = round ? d_candidate2 : d_candidate1;
        d_xs = round ? d_xs2 : d_xs1;
        d_ys = round ? d_ys2 : d_ys1;
        stream = round ? &stream2 : &stream1;
        done_evt = round ? &done_evt2 : &done_evt1;

        if (!first) {
            _CUDA_CHECK_(cudaMemcpyAsync(candidates.data(), d_candidate,
                                         sizeof(P) * buffer_size,
                                         cudaMemcpyDeviceToHost, *stream));
            _CUDA_CHECK_(cudaMemcpyAsync(xs.data(), d_xs,
                                         sizeof(S) * buffer_size,
                                         cudaMemcpyDeviceToHost, *stream));
            _CUDA_CHECK_(cudaMemcpyAsync(ys.data(), d_ys,
                                         sizeof(S) * buffer_size,
                                         cudaMemcpyDeviceToHost, *stream));
            _CUDA_CHECK_(cudaStreamSynchronize(*stream));
            _CUDA_CHECK_(cudaGetLastError());
            for (uint k = 0; k < buffer_size; ++k) {
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
#ifdef GEC_DEBUG
            printf("[host] number of candidates: %zu\n", candidates_map.size());
#endif // GEC_DEBUG
        } else {
            first = false;
        }
    }

clean_up:
    cudaFree(d_al);
    cudaFree(d_bl);
    cudaFree(d_pl);
    cudaFree(d_rng);
    cudaFree(d_init_xs);
    cudaFree(d_init_ys);
    cudaFree(d_init_ps);
    cudaFree(d_candidate1);
    cudaFree(d_candidate2);
    cudaFree(d_xs1);
    cudaFree(d_xs2);
    cudaFree(d_ys1);
    cudaFree(d_ys2);
    cudaFree(d_buf_cursor);
    cudaFree(d_done);

    cudaEventDestroy(done_evt2);
clean_done_evt2:
    cudaEventDestroy(done_evt1);
clean_done_evt1:
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