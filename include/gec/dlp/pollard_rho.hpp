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
struct WorkerData {
    const P &g;
    const P &h;
    const std::vector<S> &al;
    const std::vector<S> &bl;
    const std::vector<P> &pl;
    std::unordered_multimap<P, Coefficient<S>, typename P::Hasher> &candidates;
    const typename P::Field &mask;
    pthread_mutex_t *candidates_mutex;
    volatile bool &done;
    S &c;
    S &d2;
    std::random_device::result_type seed;
    size_t workers;
    size_t id;
};

template <typename S, typename P, typename Rng>
void *worker(void *data_ptr) {
    using F = typename P::Field;
    using LT = typename F::LimbT;

    WorkerData<S, P> &data = *static_cast<WorkerData<S, P> *>(data_ptr);
    auto rng = make_gec_rng(Rng(data.seed));
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
    P::mul(xg, coeff.x, data.g, ctx_view.rest());
    P::mul(yh, coeff.y, data.h, ctx_view.rest());
    P::add(*p, xg, yh, ctx_view.rest());

    size_t l = data.pl.size();
    int i;
    while (true) {
        if (data.done) {
            return nullptr;
        }
        if (utils::VtSeqAll<F::LimbN, LT, MaskZero<LT>>::call(
                p->x().array(), data.mask.array())) {
            pthread_mutex_lock(data.candidates_mutex);
            if (data.done) {
                pthread_mutex_unlock(data.candidates_mutex);
                return nullptr;
            }
            auto range = data.candidates.equal_range(*p);
            for (auto it = range.first; it != range.second; ++it) {
                const auto &p0 = it->first;
                const auto &coeff0 = it->second;
                if (P::eq(p0, *p) && coeff0.y != coeff.y) {
                    S::sub(data.c, coeff0.x, coeff.x);
                    S::sub(data.d2, coeff.y, coeff0.y);
                    data.done = true;
                    pthread_mutex_unlock(data.candidates_mutex);
                    return nullptr;
                }
            }
            data.candidates.insert(std::make_pair(*p, coeff));
            pthread_mutex_unlock(data.candidates_mutex);
        }
        i = p->x().array()[0] % l;
        S::add(coeff.x, data.al[i]);
        S::add(coeff.y, data.bl[i]);
        P::add(*tmp, *p, data.pl[i], ctx.template view_as<P, P>().rest());
        utils::swap(p, tmp);
    }
}

template <typename S, typename P, typename Rng>
void multithread_pollard_rho(S &c,
                             S &d2, // TODO: require general int inv
                             size_t l, size_t worker_n,
                             const typename P::Field &mask, const P &g,
                             const P &h, GecRng<Rng> &rng) {
    using Data = WorkerData<S, P>;

    std::vector<S> al(l), bl(l);
    std::vector<P> pl(l);
    typename P::template Context<> ctx;

    P ag, bh;
    for (size_t k = 0; k < l; ++k) {
        S::sample(al[k], rng);
        S::sample(bl[k], rng);
        P::mul(ag, al[k], g, ctx);
        P::mul(bh, bl[k], h, ctx);
        P::add(pl[k], ag, bh, ctx);
    }

    bool done = false;

    std::vector<pthread_t> workers(worker_n);
    pthread_mutex_t candidates_mutex = PTHREAD_MUTEX_INITIALIZER;
    std::unordered_multimap<P, Coefficient<S>, typename P::Hasher> candidates;
    std::vector<Data> workers_data(
        worker_n, {g, h, al, bl, pl, candidates, mask, &candidates_mutex, done,
                   c, d2, /* seed */ 0, worker_n, /* id */ 0});

    for (size_t k = 0; k < worker_n; ++k) {
        auto &data = workers_data[k];
        data.id = k;
        data.seed = rng.template sample<std::random_device::result_type>();
        pthread_create(&workers[k], nullptr, worker<S, P, Rng>,
                       static_cast<void *>(&data));
    }

    for (size_t k = 0; k < worker_n; ++k) {
        pthread_join(workers[k], nullptr);
    }
}

} // namespace _pollard_rho_

using _pollard_rho_::multithread_pollard_rho; // NOLINT(misc-unused-using-decls)

#endif // GEC_ENABLE_PTHREADS

} // namespace dlp

} // namespace gec

#endif // !GEC_DLP_POLLARD_RHO_HPP