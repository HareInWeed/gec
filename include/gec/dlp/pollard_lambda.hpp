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
    pthread_mutex_t traps_lock;
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
          phf(nullptr, worker_n), x_lock(PTHREAD_MUTEX_INITIALIZER),
          traps_lock(PTHREAD_MUTEX_INITIALIZER), barrier(), x(x), a(a), b(b),
          g(g), h(h), bound(bound), worker_n(worker_n), shutdown(false) {
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
    auto rng = make_gec_rng(std::mt19937((unsigned int)(local.seed)));

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
            data.phf.build(shared.trap_hashes.data());
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
            auto idx = shared.phf.get(hasher(*u));
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

#ifdef GEC_ENABLE_CUDA

#endif // GEC_ENABLE_CUDA

} // namespace dlp

} // namespace gec

#endif // GEC_DLP_POLLARD_LAMBDA_HPP