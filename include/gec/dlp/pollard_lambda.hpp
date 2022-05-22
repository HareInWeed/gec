#pragma once
#ifndef GEC_DLP_POLLARD_LAMBDA_HPP
#define GEC_DLP_POLLARD_LAMBDA_HPP

#include <gec/utils/basic.hpp>

#include <random>
#include <utility>

namespace gec {

namespace dlp {

/** @brief pollard lambda algorithm for ECDLP
 *
 * `a` must be strictly less than `b`, otherwise the behaviour is undefined.
 */
template <typename S, typename P, typename Rng, typename CTX>
void pollard_lambda(S &GEC_RSTRCT x, S *GEC_RSTRCT sl, P *GEC_RSTRCT pl,
                    const S &GEC_RSTRCT bound, const S &GEC_RSTRCT a,
                    const S &GEC_RSTRCT b, const P &GEC_RSTRCT g,
                    const P &GEC_RSTRCT h, Rng &GEC_RSTRCT rng,
                    CTX &GEC_RSTRCT ctx) {
    using F = typename P::Field;
    auto &ctx_view = ctx.template view_as<P, P, P, S, S, S, S>();
    auto &p1 = ctx_view.template get<0>();
    auto &p2 = ctx_view.template get<1>();
    auto &temp = ctx_view.template get<2>();
    auto &c = ctx_view.template get<3>();
    auto &d = ctx_view.template get<4>();
    auto &i = ctx_view.template get<5>();
    ctx_view.template get<6>().set_one();
    const auto &one = ctx_view.template get<6>();
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
        c.set_zero();
        for (i.set_zero(); i < bound; S::add(i, one)) {
            size_t i = u->x().array()[0] % m;
            S::add(c, sl[i]);
            P::add(*tmp, *u, pl[i], rest_ctx);
            std::swap(u, tmp);
        }

        d.set_zero();
        *v = h;
        for (i.set_zero(); i < bound; S::add(i, one)) {
            if (P::eq(*u, *v)) {
                S::add(x, c);
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

} // namespace dlp

} // namespace gec

#endif // GEC_DLP_POLLARD_LAMBDA_HPP