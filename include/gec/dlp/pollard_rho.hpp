#pragma once
#ifndef GEC_DLP_POLLARD_RHO_HPP
#define GEC_DLP_POLLARD_RHO_HPP

#include <gec/utils/basic.hpp>

#include <utility>

#ifdef GEC_ENABLE_PTHREAD
#include <gec/utils/sequence.hpp>

#include <pthread.h>

#include <random>
#include <unordered_map>
#include <vector>

#endif // GEC_ENABLE_PTHREAD

namespace gec {

namespace dlp {

template <typename S, typename P, typename Rng, typename P_CTX, typename S_CTX>
__host__ __device__ void pollard_rho(S &c,
                                     S &d2, // TODO: require general int inv
                                     size_t l, S *al, S *bl, P *pl, const P &g,
                                     const P &h, Rng &rng, P_CTX &ctx,
                                     S_CTX &s_ctx) {
    using F = typename P::Field;
    // TODO: a safe way to get a scaler from context P_CTX
    S &c2 = s_ctx.template get<0>();
    S &d = s_ctx.template get<1>();
    F &f1 = ctx.template get<0>();
    F &f2 = ctx.template get<1>();
    P &ag = ctx.template get_p<0>();
    P &bh = ctx.template get_p<1>();
    P &temp = ctx.template get_p<2>();
    P &x1 = ctx.template get_p<3>();
    P &x2 = ctx.template get_p<4>();
    P *tmp = &temp, *x = &x1;
    auto &rest_ctx = ctx.template rest<2, 5>();

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
            std::swap(x, tmp);

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

#ifdef GEC_ENABLE_PTHREAD

template <typename T>
struct MaskZero {
    GEC_INLINE static bool call(const T &a, const T &b) { return !(a & b); }
};

template <typename S>
struct Coefficient {
    S x;
    S y;
};

template <typename S, typename P>
struct PollardRhoData {
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
    size_t seed;
    size_t workers;
    size_t id;
};

template <typename S, typename P>
void *multithread_pollard_rho_worker(void *data_ptr) {
    using F = typename P::Field;
    using LT = typename F::LimbT;

    PollardRhoData<S, P> &data = *static_cast<PollardRhoData<S, P> *>(data_ptr);
    std::mt19937 rng(data.seed);
    typename P::template Context<> ctx;
    Coefficient<S> coeff;
    P p1, p2;
    P *p = &p1, *tmp = &p2;

    S::sample(coeff.x, rng);
    S::sample(coeff.y, rng);
    P &xg = ctx.template get_p<0>();
    P &yh = ctx.template get_p<1>();
    P::mul(xg, coeff.x, data.g, ctx.template rest<0, 2>());
    P::mul(yh, coeff.y, data.h, ctx.template rest<0, 2>());
    P::add(*p, xg, yh, ctx.template rest<0, 2>());

    int l = data.pl.size();
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
        P::add(*tmp, *p, data.pl[i], ctx);
        std::swap(p, tmp);
    }
}

template <typename S, typename P>
void multithread_pollard_rho(S &c,
                             S &d2, // TODO: require general int inv
                             size_t l, size_t worker_n,
                             const typename P::Field &mask, const P &g,
                             const P &h) {
    using Payload = PollardRhoData<S, P>;
    std::random_device rd;
    std::mt19937 rng(rd());

    std::vector<S> al(l), bl(l);
    std::vector<P> pl(l);
    typename P::template Context<> ctx;

    // TODO: a safe way to get a scaler from context
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
    std::vector<Payload> workers_data(
        worker_n, {g, h, al, bl, pl, candidates, mask, &candidates_mutex, done,
                   c, d2, /* seed */ 0, worker_n, /* id */ 0});

    for (size_t k = 0; k < worker_n; ++k) {
        auto &data = workers_data[k];
        data.id = k;
        data.seed = rd();
        pthread_create(&workers[k], nullptr,
                       multithread_pollard_rho_worker<S, P>,
                       static_cast<void *>(&data));
    }

    for (size_t k = 0; k < worker_n; ++k) {
        pthread_join(workers[k], nullptr);
    }
}
#endif // GEC_ENABLE_PTHREAD

} // namespace dlp

} // namespace gec

#endif // !GEC_DLP_POLLARD_RHO_HPP