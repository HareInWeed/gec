#include "cuda_common.cuh"

#define GEC_DEBUG
#include <curve.hpp>

#include <gec/dlp/pollard_rho.hpp>

using namespace gec;
using namespace dlp;

TEST_CASE("cu_pollard_rho", "[dlp][pollard_rho][multithread]") {
    using C = Dlp3CurveA;
    const C &g = Dlp3Gen1;
    using S = Dlp3G1Scaler;
    using F = C::Field;

    std::random_device rd;
    auto data_seed = rd();
    auto seed = rd();
    CAPTURE(data_seed, seed);
    auto rng = make_gec_rng(std::mt19937(data_seed));

    C::Context<> ctx;

    C h;
    REQUIRE(C::on_curve(g, ctx));

    S x;
    S::sample_non_zero(x, rng);

    C::mul(h, x, g, ctx);
    CAPTURE(h);
    REQUIRE(C::on_curve(h, ctx));

    size_t l = 32;
    S c, d, mon_c, mon_d;

    F mask(0xF0000000, 0, 0, 0, 0, 0, 0, 0);

    // the grid size here are for test only, typical grid size should be much
    // larger
    CUDA_REQUIRE(cu_pollard_rho(c, d, l, mask, g, h, seed, 4, 32, 0x1000));
    S::to_montgomery(mon_c, c);
    S::to_montgomery(mon_d, d);
    S::inv(mon_d, ctx);
    S::mul(d, mon_c, mon_d);
    S::from_montgomery(c, d);

    C xg;
    C::mul(xg, c, g, ctx);
    CAPTURE(c, xg, h);
    REQUIRE(C::eq(xg, h));
}
