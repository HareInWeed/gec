#include <common.hpp>

#include <gec/bigint.hpp>
#include <gec/curve.hpp>

#include "cuda_common.cuh"
#include <curve.hpp>

#include <configured_catch.hpp>

#include <cstring>

using namespace gec;
using namespace bigint;
using namespace curve;
using namespace gec::bigint::literal;

__global__ static void test_point_kernel(Point<Field160, 2> *dest) {
    using F = Field160;
    Point<F, 2> p(F(0x0u, 0x0u, 0x0u, 0x0u, 0x1u),
                  F(0x1u, 0x0u, 0x0u, 0x0u, 0x0u));
    memcpy(dest, &p, sizeof(Point<F, 2>));
}

TEST_CASE("cuda point", "[curve][cuda]") {
    using F = Field160;
    using P = Point<F, 2>;

    P p;
    P *d_p;
    CUDA_REQUIRE(cudaMalloc(&d_p, sizeof(P)));
    test_point_kernel<<<1, 1>>>(d_p);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(&p, d_p, sizeof(P), cudaMemcpyDeviceToHost));

    CAPTURE(p.x(), p.y());
    REQUIRE(p.x().array()[0] == 1);
    for (size_t k = 1; k < F::LimbN; ++k) {
        CAPTURE(k);
        REQUIRE(p.x().array()[k] == 0);
    }
    REQUIRE(p.y().array()[F::LimbN - 1] == 1);
    for (size_t k = 0; k < F::LimbN - 1; ++k) {
        CAPTURE(k);
        REQUIRE(p.y().array()[k] == 0);
    }

    CUDA_REQUIRE(cudaFree(d_p));
}

__global__ static void test_affine_kernel(CurveA *sum, CurveA *p1, CurveA *p2) {
    using C = CurveA;

    C sum_l, p1_l = *p1, p2_l = *p2;
    C::add(sum_l, p1_l, p2_l);
    *sum = sum_l;
}

TEST_CASE("cuda affine", "[curve][affine][cuda]") {
    using C = CurveA;
    using F = Field160;

    C test(F(1), F(1));
    REQUIRE(!C::on_curve(test));

    C p1, *d_p1 = nullptr;
    F::to_montgomery(p1.x(), //
                     0x0ee27967'5de1bde5'faf553e9'2185fec7'43e7dd56_int);
    F::to_montgomery(p1.y(), //
                     0xa43c088f'a471d05c'3d1bed80'b89428be'84e54fae_int);
    CAPTURE(p1);
    REQUIRE(C::on_curve(p1));

    C p2, *d_p2 = nullptr;
    F::to_montgomery(p2.x(), //
                     0x16b60634'e1d3e896'2879d7aa'2c1672ab'de0252bb_int);
    F::to_montgomery(p2.y(), //
                     0x99056d94'e6864afa'a034f181'd8b4192f'1cbedd98_int);
    CAPTURE(p2);
    REQUIRE(C::on_curve(p2));

    CUDA_REQUIRE(cudaMalloc(&d_p1, sizeof(C)));
    CUDA_REQUIRE(cudaMalloc(&d_p2, sizeof(C)));
    CUDA_REQUIRE(cudaMemcpy(d_p1, &p1, sizeof(C), cudaMemcpyHostToDevice));
    CUDA_REQUIRE(cudaMemcpy(d_p2, &p2, sizeof(C), cudaMemcpyHostToDevice));

    C sum, *d_sum, expected;
    CUDA_REQUIRE(cudaMalloc(&d_sum, sizeof(C)));

    test_affine_kernel<<<1, 1>>>(d_sum, d_p1, d_p2);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(&sum, d_sum, sizeof(C), cudaMemcpyDeviceToHost));
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));
    F::to_montgomery(expected.x(), //
                     0x506c783f'82e6ba2f'323ddc50'ffe966bf'41cb4178_int);
    F::to_montgomery(expected.y(), //
                     0x8fc3cd04'2e78553e'b84d4c96'196151fe'e3bd209b_int);
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected));
    REQUIRE(C::eq(expected, sum));

    test_affine_kernel<<<1, 1>>>(d_sum, d_p1, d_p1);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(&sum, d_sum, sizeof(C), cudaMemcpyDeviceToHost));
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));
    F::to_montgomery(expected.x(), //
                     0x6b52f5f8'836d4559'4eb4f96f'11b16271'b9194d96_int);
    F::to_montgomery(expected.y(), //
                     0x1fd6f136'cd8ecae6'bec3bb77'a5bdc183'842648be_int);
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected));
    REQUIRE(C::eq(expected, sum));

    test_affine_kernel<<<1, 1>>>(d_sum, d_p2, d_p2);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(&sum, d_sum, sizeof(C), cudaMemcpyDeviceToHost));
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));
    F::to_montgomery(expected.x(), //
                     0x34aabf2e'f06c1194'bd316d0a'3a407ef7'850f874e_int);
    F::to_montgomery(expected.y(), //
                     0x1870fd80'e627d83b'7af69418'ad073ee5'ba3606e5_int);
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected));
    REQUIRE(C::eq(expected, sum));

    p2.set_inf();
    CUDA_REQUIRE(cudaMemcpy(d_p2, &p2, sizeof(C), cudaMemcpyHostToDevice));
    test_affine_kernel<<<1, 1>>>(d_sum, d_p1, d_p2);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(&sum, d_sum, sizeof(C), cudaMemcpyDeviceToHost));
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));
    REQUIRE(C::eq(p1, sum));

    p1.set_inf();
    CUDA_REQUIRE(cudaMemcpy(d_p1, &p1, sizeof(C), cudaMemcpyHostToDevice));
    test_affine_kernel<<<1, 1>>>(d_sum, d_p2, d_p2);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(&sum, d_sum, sizeof(C), cudaMemcpyDeviceToHost));
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum));
    REQUIRE(sum.is_inf());

    CUDA_REQUIRE(cudaFree(d_p1));
    CUDA_REQUIRE(cudaFree(d_p2));
    CUDA_REQUIRE(cudaFree(d_sum));
}

TEST_CASE("cuda affine scalar_mul", "[curve][affine][scalar_mul]") {
    using C = Dlp1CurveA;
    using F = Dlp1Field;
    using S = Dlp1Scalar;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    C p;
    F::to_montgomery(p.x(), //
                     0x1e3b0742'ebf7d73f'f1a78116'4c46739a'153663f3_int);
    F::to_montgomery(p.y(), //
                     0x16a8c9aa'c4ad5fdf'58163ef3'9de531f5'e9cb1575_int);
    REQUIRE(C::on_curve(p));
    CAPTURE(p);

    C prod1, prod2, sum;

    C::mul(prod1, 0, p);
    CAPTURE(prod1);
    REQUIRE(prod1.is_inf());

    C::mul(prod1, 1, p);
    CAPTURE(prod1);
    REQUIRE(prod1.x() == p.x());
    REQUIRE(prod1.y() == p.y());

    C::mul(prod1, S::mod(), p);
    CAPTURE(prod1);
    REQUIRE(C::on_curve(prod1));
    REQUIRE(prod1.is_inf());

    S s1, s2;
    for (int k = 0; k < 100; ++k) {
        S::sample(s1, rng);
        S::neg(s2, s1);

        C::mul(prod1, s1, p);
        CAPTURE(prod1);
        C::mul(prod2, s2, p);
        CAPTURE(prod2);
        C::add(sum, prod1, prod2);
        CAPTURE(sum);
        REQUIRE(sum.is_inf());

        S::add(s1, s2, 1);
        C::mul(prod2, s1, p);
        CAPTURE(prod2);
        C::add(sum, prod1, prod2);
        CAPTURE(sum);
        REQUIRE(C::eq(sum, p));
    }
}