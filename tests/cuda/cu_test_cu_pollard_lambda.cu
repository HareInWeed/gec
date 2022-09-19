#include "cuda_common.cuh"

#define GEC_DEBUG
#include <curve.hpp>

#include <gec/dlp/pollard_lambda.hpp>

using namespace gec;
using namespace dlp;

TEST_CASE("cu_pollard_lambda", "[dlp][pollard_lambda][cuda]") {
    using C = Dlp3CurveA;
    const C &g = Dlp3Gen1;
    using S = Dlp3G1Scalar;
    using F = C::Field;

    std::random_device rd;
    auto data_seed = rd();
    auto seed = rd();
    // auto data_seed = 1882371591;
    // auto seed = 399752428;
    CAPTURE(data_seed, seed);
    auto rng = make_gec_rng(std::mt19937(data_seed));

    C h;
    REQUIRE(C::on_curve(g));

    {
        C::mul(h, S::mod(), g);
        CAPTURE(h);
        REQUIRE(h.is_inf());
    }

    S x0, lower(1 << 3), upper((1 << 3) + (1 << 15));
    S::sample_inclusive(x0, lower, upper, rng);

    C::mul(h, x0, g);
    REQUIRE(C::on_curve(h));

    CAPTURE(g, lower, upper, x0, h);

    S x;
    F mask(0xF0000000, 0, 0, 0, 0, 0, 0, 0);
    // the grid size here are for test only, typical grid size should be much
    // larger
    CUDA_REQUIRE(
        cu_pollard_lambda(x, lower, upper, g, h, mask, seed, 6, 4, 32, 0x1000));

    C xg;
    C::mul(xg, x, g);
    CAPTURE(x, xg);
    REQUIRE(C::eq(xg, h));
}