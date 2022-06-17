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

} // namespace _pollard_rho_

using _pollard_rho_::multithread_pollard_rho; // NOLINT(misc-unused-using-decls)

#endif // GEC_ENABLE_PTHREADS

} // namespace dlp

} // namespace gec

#endif // !GEC_DLP_POLLARD_RHO_HPP