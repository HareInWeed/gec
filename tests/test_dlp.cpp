#include "common.hpp"
#include "utils.hpp"

#include <gec/dlp.hpp>

#include "configured_catch.hpp"
#include "curve.hpp"

using namespace gec;
using namespace dlp;

TEST_CASE("pollard_rho", "[dlp][pollard_rho]") {
    using C = Dlp2CurveJ;
    using F = Dlp2Field;
    using S = Dlp2Scaler;

    std::random_device rd;
    std::mt19937 rng(rd());

    C::Context<> ctx;

    C g;
    F::mul(g.x(), {4023}, F::r_sqr());
    F::mul(g.y(), {6036}, F::r_sqr());
    C::from_affine(g);

    C h;
    F::mul(h.x(), {4135}, F::r_sqr());
    F::mul(h.y(), {3169}, F::r_sqr());
    C::from_affine(h);

    constexpr size_t l = 32;
    S al[l], bl[l];
    C pl[l];

    S c, d;

    pollard_rho(c, d, l, al, bl, pl, g, h, rng, ctx);

    cout << c << "/" << d << endl;
}