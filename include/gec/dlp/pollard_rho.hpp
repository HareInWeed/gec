#pragma once
#ifndef GEC_DLP_POLLARD_RHO_HPP
#define GEC_DLP_POLLARD_RHO_HPP

#include <gec/utils/basic.hpp>

#include <utility>

namespace gec {

namespace dlp {

template <typename S, typename P, typename Rng, typename P_CTX>
__host__ __device__ void pollard_rho(S &c,
                                     S &d2, // TODO: require general int inv
                                     size_t l, S *al, S *bl, P *pl, const P &g,
                                     const P &h, Rng &rng, P_CTX &ctx) {
    // TODO: a safe way to get a scaler from context
    S &c2 = reinterpret_cast<S &>(ctx.template get<0>());
    S &d = reinterpret_cast<S &>(ctx.template get<1>());
    P &ag = ctx.template get_p<0>();
    P &bh = ctx.template get_p<1>();
    P &temp = ctx.template get_p<2>();
    P &x1 = ctx.template get_p<3>();
    P &x2 = ctx.template get_p<4>();
    auto &rest_ctx = ctx.template rest<2, 5>();
    P *tmp = &temp, *x = &x1;

    for (int k = 0; k < l; ++k) {
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
        std::swap(tmp, x);

        i = x2.x().array()[0] % l;
        S::add(c2, al[i]);
        S::add(d2, bl[i]);
        P::add(*tmp, x2, pl[i], rest_ctx);

        i = tmp->x().array()[0] % l;
        S::add(c2, al[i]);
        S::add(d2, bl[i]);
        P::add(x2, *tmp, pl[i], rest_ctx);
    } while (!P::eq(*x, x2, rest_ctx));

    S::sub(c, c2);
    S::sub(d2, d);
}

} // namespace dlp

} // namespace gec

#endif // !GEC_DLP_POLLARD_RHO_HPP