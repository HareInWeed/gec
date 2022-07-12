#include "cuda_common.cuh"

#define GEC_DEBUG
#include <curve.hpp>

#include <gec/dlp.hpp>

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

    C::mul(h, S::mod(), g, ctx);
    CAPTURE(h);
    REQUIRE(h.is_inf());

    S x;
    S::sample_non_zero(x, rng);

    C::mul(h, x, g, ctx);
    REQUIRE(C::on_curve(h, ctx));

    size_t l = 32;
    S c, d, mon_c, mon_d;

    F mask(0x80000000, 0, 0, 0, 0, 0, 0, 0);

    CUDA_REQUIRE(cu_pollard_rho(c, d, l, mask, g, h, seed, 60, 128));
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

TEST_CASE("cu_pollard_lambda", "[dlp][pollard_lambda][cuda]") {
    using C = Dlp3CurveA;
    const C &g = Dlp3Gen1;
    using S = Dlp3G1Scaler;
    using F = C::Field;

    std::random_device rd;
    auto data_seed = rd();
    auto seed = rd();
    // auto data_seed = 1882371591;
    // auto seed = 399752428;
    CAPTURE(data_seed, seed);
    auto rng = make_gec_rng(std::mt19937(data_seed));

    C::Context<> ctx;

    C h;
    REQUIRE(C::on_curve(g, ctx));

    {
        C::mul(h, S::mod(), g, ctx);
        CAPTURE(h);
        REQUIRE(h.is_inf());
    }

    S x0, lower(1 << 3), upper((1 << 3) + (1 << 15));
    S::sample_inclusive(x0, lower, upper, rng, ctx);

    C::mul(h, x0, g, ctx);
    REQUIRE(C::on_curve(h, ctx));

    CAPTURE(g, lower, upper, x0, h);

    S x;
    F mask(0xFFFF0000, 0, 0, 0, 0, 0, 0, 0);
    // the grid size here are for test only, typical grid size should be much
    // larger
    CUDA_REQUIRE(cu_pollard_lambda(x, lower, upper, g, h, mask, seed, 80, 32));

    C xg;
    C::mul(xg, x, g, ctx);
    CAPTURE(x, xg);
    REQUIRE(C::eq(xg, h));
}