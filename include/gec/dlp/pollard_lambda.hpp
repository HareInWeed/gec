#pragma once
#ifndef GEC_DLP_POLLARD_LAMBDA_HPP
#define GEC_DLP_POLLARD_LAMBDA_HPP

#include <gec/utils/basic.hpp>

#include <random>
#include <utility>

#ifdef GEC_ENABLE_PTHREADS
#include <pthread.h>
#include <unordered_map>
#endif // GEC_ENABLE_PTHREADS
namespace gec {

namespace dlp {

/** @brief pollard lambda algorithm for ECDLP
 *
 * `a` must be strictly less than `b`, otherwise the behaviour is undefined.
 */
template <typename S, typename P, typename Rng, typename Ctx>
void pollard_lambda(S &GEC_RSTRCT x, S *GEC_RSTRCT sl, P *GEC_RSTRCT pl,
                    const S &GEC_RSTRCT bound, const S &GEC_RSTRCT a,
                    const S &GEC_RSTRCT b, const P &GEC_RSTRCT g,
                    const P &GEC_RSTRCT h, Rng &GEC_RSTRCT rng,
                    Ctx &GEC_RSTRCT ctx) {
    using F = typename P::Field;
    auto &ctx_view = ctx.template view_as<P, P, P, S, S, S>();
    auto &p1 = ctx_view.template get<0>();
    auto &p2 = ctx_view.template get<1>();
    auto &temp = ctx_view.template get<2>();
    auto &d = ctx_view.template get<3>();
    auto &i = ctx_view.template get<4>();
    ctx_view.template get<5>().set_one();
    const auto &one = ctx_view.template get<5>();
    auto &rest_ctx = ctx_view.rest();

    while (true) {
        P *u = &p1, *v = &p2, *tmp = &temp;

        S::sub(x, b, a);
        // with `a` less than `b`, `m` would not underflow
        size_t m = x.most_significant_bit() - 1;
        for (size_t i = 0; i < m; ++i) {
            sl[i].array()[0] = i;
        }
        for (size_t i = 0; i < m; ++i) {
            size_t ri = m - 1 - i;
            std::uniform_int_distribution<size_t> gen(0, ri);
            std::swap(sl[ri].array()[0], sl[gen(rng)].array()[0]);
        }
        for (size_t i = 0; i < m; ++i) {
            typename F::LimbT e = sl[i].array()[0];
            sl[i].set_pow2(e);
            P::mul(pl[i], sl[i], g, rest_ctx);
        }

        S::sample_inclusive(x, a, b, rng, rest_ctx);
        P::mul(*u, x, g, rest_ctx);
        for (i.set_zero(); i < bound; S::add(i, one)) {
            size_t i = u->x().array()[0] % m;
            S::add(x, sl[i]);
            P::add(*tmp, *u, pl[i], rest_ctx);
            std::swap(u, tmp);
        }

        d.set_zero();
        *v = h;
        for (i.set_zero(); i < bound; S::add(i, one)) {
            if (P::eq(*u, *v)) {
                S::sub(x, d);
                return;
            }
            size_t i = v->x().array()[0] % m;
            S::add(d, sl[i]);
            P::add(*tmp, *v, pl[i], rest_ctx);
            std::swap(v, tmp);
        }
    }
}

#ifdef GEC_ENABLE_PTHREADS

namespace pollard_lambda_ {

template <typename S, typename P>
struct WorkerData {
    S &x;
    const S &a;
    const S &b;
    const P &g;
    const P &h;
    std::vector<S> &sl;
    std::vector<P> &pl;
    const S &bound;

    pthread_barrier_t *barrier;

    pthread_mutex_t *traps_lock;
    std::unordered_map<P, S, typename P::Hasher> &traps;

    volatile bool &shutdown;

    size_t seed;
    size_t worker_n;
    size_t id;
};

template <typename S, typename P, typename Rng>
void *worker(void *data_ptr) {
    using Data = WorkerData<S, P>;
    using LimbT = typename P::Field::LimbT;

    Data &data = *static_cast<Data *>(data_ptr);
    const size_t m = data.sl.size();
    P p1, p2;
    P *u = &p1, *tmp = &p2;
    S x, i, one(1);
    typename P::template Context<> ctx;
    Rng rng(data.seed);

    while (true) {
        // calculate jump table
        if (data.id == 0) {
            for (size_t i = 0; i < m; ++i) {
                data.sl[i].array()[0] = i;
            }
            for (size_t i = 0; i < m; ++i) {
                size_t ri = m - 1 - i;
                std::uniform_int_distribution<size_t> gen(0, ri);
                std::swap(data.sl[ri].array()[0], data.sl[gen(rng)].array()[0]);
            }
            for (size_t i = 0; i < m; ++i) {
                LimbT e = data.sl[i].array()[0];
                data.sl[i].set_pow2(e);
                P::mul(data.pl[i], data.sl[i], data.g, ctx);
            }
        }

        pthread_barrier_wait(data.barrier);
        // setting traps

        S::sample_inclusive(x, data.a, data.b, rng, ctx);
        P::mul(*u, x, data.g, ctx);
        for (i.set_zero(); i < data.bound; S::add(i, one)) {
            size_t i = u->x().array()[0] % m;
            S::add(x, data.sl[i]);
            P::add(*tmp, *u, data.pl[i], ctx);
            std::swap(u, tmp);
        }
        pthread_mutex_lock(data.traps_lock);
        data.traps.insert(std::make_pair(*u, x));
        pthread_mutex_unlock(data.traps_lock);

        pthread_barrier_wait(data.barrier);
        // start searching

        S::sample_inclusive(x, data.a, data.b, rng, ctx);
        P::mul(*tmp, x, data.g, ctx);
        P::add(*u, data.h, *tmp, ctx);
        for (i.set_zero(); i < data.bound; S::add(i, one)) {
            if (data.shutdown) {
                break;
            }
            auto it = data.traps.find(*u);
            if (it != data.traps.end() && it->second != x) {
                S::sub(data.x, it->second, x);
                data.shutdown = true;
                break;
            }
            size_t i = u->x().array()[0] % m;
            S::add(x, data.sl[i]);
            P::add(*tmp, *u, data.pl[i], ctx);
            std::swap(u, tmp);
        }

        pthread_barrier_wait(data.barrier);
        // check success

        if (data.shutdown) {
            return nullptr;
        }
    }
}

/** @brief multithread pollard lambda algorithm for ECDLP, requires `pthreads`
 *
 * `a` must be strictly less than `b`, otherwise the behaviour is undefined.
 */
template <typename S, typename P, typename Rng = std::mt19937>
void multithread_pollard_lambda(S &GEC_RSTRCT x, const S &GEC_RSTRCT bound,
                                size_t worker_n, const S &GEC_RSTRCT a,
                                const S &GEC_RSTRCT b, const P &GEC_RSTRCT g,
                                const P &GEC_RSTRCT h, size_t seed) {
    Rng rng(seed);

    using Data = WorkerData<S, P>;

    std::vector<pthread_t> workers(worker_n);

    typename P::template Context<> ctx;

    S::sub(x, b, a);
    // with `a` less than `b`, `m` would not underflow
    size_t m = x.most_significant_bit() - 1;

    std::vector<S> sl(m);
    std::vector<P> pl(m);

    pthread_mutex_t traps_lock = PTHREAD_MUTEX_INITIALIZER;
    std::unordered_map<P, S, typename P::Hasher> traps(worker_n);

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, worker_n);

    bool shutdown = false;

    std::vector<Data> params(worker_n,
                             Data{x, a, b, g, h, sl, pl, bound, &barrier,
                                  &traps_lock, traps, shutdown,
                                  /* seed */ 0, worker_n,
                                  /* id */ 0});

    for (size_t i = 0; i < worker_n; ++i) {
        auto &param = params[i];
        param.seed = rng();
        param.id = i;
        pthread_create(&workers[i], nullptr, worker<S, P, Rng>,
                       static_cast<Data *>(&param));
    }

    for (size_t i = 0; i < worker_n; ++i) {
        pthread_join(workers[i], nullptr);
    }
}

} // namespace pollard_lambda_

// NOLINTNEXTLINE(misc-unused-using-decls)
using pollard_lambda_::multithread_pollard_lambda;

#endif // GEC_ENABLE_PTHREADS

} // namespace dlp

} // namespace gec

#endif // GEC_DLP_POLLARD_LAMBDA_HPP