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

    C::Context<> ctx;
    C sum_l, p1_l = *p1, p2_l = *p2;
    C::add(sum_l, p1_l, p2_l, ctx);
    *sum = sum_l;
}

TEST_CASE("cuda affine", "[curve][affine][cuda]") {
    using C = CurveA;
    using F = Field160;

    C::Context<> ctx;

    C test(F(1), F(1));
    REQUIRE(!C::on_curve(test, ctx));

    C p1, *d_p1 = nullptr;
    F::to_montgomery(
        p1.x(), //
        {0x0ee27967u, 0x5de1bde5u, 0xfaf553e9u, 0x2185fec7u, 0x43e7dd56u});
    F::to_montgomery(
        p1.y(), //
        {0xa43c088fu, 0xa471d05cu, 0x3d1bed80u, 0xb89428beu, 0x84e54faeu});
    CAPTURE(p1);
    REQUIRE(C::on_curve(p1, ctx));

    C p2, *d_p2 = nullptr;
    F::to_montgomery(
        p2.x(), //
        {0x16b60634u, 0xe1d3e896u, 0x2879d7aau, 0x2c1672abu, 0xde0252bbu});
    F::to_montgomery(
        p2.y(), //
        {0x99056d94u, 0xe6864afau, 0xa034f181u, 0xd8b4192fu, 0x1cbedd98u});
    CAPTURE(p2);
    REQUIRE(C::on_curve(p2, ctx));

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
    REQUIRE(C::on_curve(sum, ctx));
    F::to_montgomery(
        expected.x(), //
        {0x506c783fu, 0x82e6ba2fu, 0x323ddc50u, 0xffe966bfu, 0x41cb4178u});
    F::to_montgomery(
        expected.y(), //
        {0x8fc3cd04u, 0x2e78553eu, 0xb84d4c96u, 0x196151feu, 0xe3bd209bu});
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected, ctx));
    REQUIRE(C::eq(expected, sum));

    test_affine_kernel<<<1, 1>>>(d_sum, d_p1, d_p1);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(&sum, d_sum, sizeof(C), cudaMemcpyDeviceToHost));
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum, ctx));
    F::to_montgomery(
        expected.x(), //
        {0x6b52f5f8u, 0x836d4559u, 0x4eb4f96fu, 0x11b16271u, 0xb9194d96u});
    F::to_montgomery(
        expected.y(), //
        {0x1fd6f136u, 0xcd8ecae6u, 0xbec3bb77u, 0xa5bdc183u, 0x842648beu});
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected, ctx));
    REQUIRE(C::eq(expected, sum));

    test_affine_kernel<<<1, 1>>>(d_sum, d_p2, d_p2);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(&sum, d_sum, sizeof(C), cudaMemcpyDeviceToHost));
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum, ctx));
    F::to_montgomery(
        expected.x(), //
        {0x34aabf2eu, 0xf06c1194u, 0xbd316d0au, 0x3a407ef7u, 0x850f874eu});
    F::to_montgomery(
        expected.y(), //
        {0x1870fd80u, 0xe627d83bu, 0x7af69418u, 0xad073ee5u, 0xba3606e5u});
    CAPTURE(expected);
    REQUIRE(C::on_curve(expected, ctx));
    REQUIRE(C::eq(expected, sum));

    p2.set_inf();
    CUDA_REQUIRE(cudaMemcpy(d_p2, &p2, sizeof(C), cudaMemcpyHostToDevice));
    test_affine_kernel<<<1, 1>>>(d_sum, d_p1, d_p2);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(&sum, d_sum, sizeof(C), cudaMemcpyDeviceToHost));
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum, ctx));
    REQUIRE(C::eq(p1, sum));

    p1.set_inf();
    CUDA_REQUIRE(cudaMemcpy(d_p1, &p1, sizeof(C), cudaMemcpyHostToDevice));
    test_affine_kernel<<<1, 1>>>(d_sum, d_p2, d_p2);
    CUDA_REQUIRE(cudaDeviceSynchronize());
    CUDA_REQUIRE(cudaGetLastError());
    CUDA_REQUIRE(cudaMemcpy(&sum, d_sum, sizeof(C), cudaMemcpyDeviceToHost));
    CAPTURE(sum);
    REQUIRE(C::on_curve(sum, ctx));
    REQUIRE(sum.is_inf());

    CUDA_REQUIRE(cudaFree(d_p1));
    CUDA_REQUIRE(cudaFree(d_p2));
    CUDA_REQUIRE(cudaFree(d_sum));
}

TEST_CASE("cuda affine scaler_mul", "[curve][affine][scaler_mul]") {
    using C = Dlp1CurveA;
    using F = Dlp1Field;
    using S = Dlp1Scaler;

    std::random_device rd;
    auto seed = rd();
    INFO("seed: " << seed);
    auto rng = make_gec_rng(std::mt19937(seed));

    C::Context<> ctx;

    C p;
    F::to_montgomery(
        p.x(), //
        {0x1e3b0742u, 0xebf7d73fu, 0xf1a78116u, 0x4c46739au, 0x153663f3u});
    F::to_montgomery(
        p.y(), //
        {0x16a8c9aau, 0xc4ad5fdfu, 0x58163ef3u, 0x9de531f5u, 0xe9cb1575u});
    REQUIRE(C::on_curve(p, ctx));
    CAPTURE(p);

    C prod1, prod2, sum;

    C::mul(prod1, 0, p, ctx);
    CAPTURE(prod1);
    REQUIRE(prod1.is_inf());

    C::mul(prod1, 1, p, ctx);
    CAPTURE(prod1);
    REQUIRE(prod1.x() == p.x());
    REQUIRE(prod1.y() == p.y());

    C::mul(prod1, S::mod(), p, ctx);
    CAPTURE(prod1);
    REQUIRE(C::on_curve(prod1, ctx));
    REQUIRE(prod1.is_inf());

    S s1, s2;
    for (int k = 0; k < 100; ++k) {
        S::sample(s1, rng);
        S::neg(s2, s1);

        C::mul(prod1, s1, p, ctx);
        CAPTURE(prod1);
        C::mul(prod2, s2, p, ctx);
        CAPTURE(prod2);
        C::add(sum, prod1, prod2, ctx);
        CAPTURE(sum);
        REQUIRE(sum.is_inf());

        S::add(s1, s2, 1);
        C::mul(prod2, s1, p, ctx);
        CAPTURE(prod2);
        C::add(sum, prod1, prod2, ctx);
        CAPTURE(sum);
        REQUIRE(C::eq(sum, p, ctx));
    }
}